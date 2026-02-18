import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from xgboost.callback import EarlyStopping
from loguru import logger
import yaml

# ================= SETTINGS ================= #
MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)

logger.info(f"Models will be saved to: {os.path.abspath(MODELS_DIR)}")

# ================= SYNTHETIC DATA ================= #
logger.info("Generating synthetic data...")

n_samples = 2000

ages = np.random.randint(18, 61, n_samples)
weights_kg = np.random.uniform(45, 101, n_samples)
heights_cm = np.random.uniform(150, 196, n_samples)

bmis = weights_kg / ((heights_cm / 100) ** 2)

activity_levels = np.random.randint(0, 3, n_samples)
calorie_intake = np.random.uniform(1500, 3500, n_samples)
goals = np.random.randint(0, 3, n_samples)
lstm_workout_intensity = np.random.randint(0, 3, n_samples)

lstm_target = (
    0.3 * bmis
    + 0.2 * activity_levels
    + 0.1 * (calorie_intake / 1000)
    + 0.2 * ages
    + 0.1 * lstm_workout_intensity
    + 0.1 * goals
    + np.random.normal(0, 5, n_samples)
)

lstm_target = MinMaxScaler().fit_transform(
    lstm_target.reshape(-1, 1)
)

lstm_data = pd.DataFrame({
    "age": ages,
    "weight_kg": weights_kg,
    "height_cm": heights_cm,
    "bmi": bmis,
    "activity_level": activity_levels,
    "calorie_intake": calorie_intake,
    "goal": goals,
    "workout_intensity": lstm_workout_intensity,
    "fitness_score": lstm_target.flatten()
})

workout_duration = np.random.randint(15, 121, n_samples)
workout_intensity_xgb = np.random.randint(0, 3, n_samples)

xgb_target = (
    0.6 * workout_duration
    + 0.4 * workout_intensity_xgb * 10
    + np.random.normal(0, 10, n_samples)
)

xgb_target = MinMaxScaler().fit_transform(
    xgb_target.reshape(-1, 1)
)

xgboost_data = pd.DataFrame({
    "workout_duration_minutes": workout_duration,
    "workout_intensity": workout_intensity_xgb,
    "preference_score": xgb_target.flatten()
})

logger.info("Synthetic data generation complete.")

# ================= LSTM TRAINING ================= #
logger.info("Training LSTM...")

lstm_features = lstm_data[
    [
        "age",
        "weight_kg",
        "height_cm",
        "bmi",
        "activity_level",
        "calorie_intake",
        "goal",
        "workout_intensity",
    ]
].values

lstm_labels = lstm_data["fitness_score"].values

scaler_lstm = MinMaxScaler()
lstm_features_scaled = scaler_lstm.fit_transform(lstm_features)

timesteps = 1
n_features = lstm_features_scaled.shape[1]

X_lstm = lstm_features_scaled.reshape(
    n_samples, timesteps, n_features
)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, lstm_labels, test_size=0.2, random_state=42
)

lstm_model = Sequential([
    Input(shape=(timesteps, n_features)),
    LSTM(100, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

loss, lstm_mae = lstm_model.evaluate(
    X_test_lstm, y_test_lstm, verbose=0
)

logger.info(f"LSTM MAE: {lstm_mae:.4f}")

lstm_model_path = os.path.join(MODELS_DIR, "lstm_model.h5")
lstm_model.save(lstm_model_path)
logger.info("LSTM model saved.")

# ================= XGBOOST TRAINING ================= #
logger.info("Training XGBoost...")

xgb_features = xgboost_data[
    ["workout_duration_minutes", "workout_intensity"]
].copy()

xgb_features["bmi"] = lstm_data["bmi"]
xgb_features["activity_level"] = lstm_data["activity_level"]
xgb_features["goal"] = lstm_data["goal"]

xgb_labels = xgboost_data["preference_score"].values

scaler_xgb = MinMaxScaler()
xgb_features_scaled = scaler_xgb.fit_transform(xgb_features)

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    xgb_features_scaled,
    xgb_labels,
    test_size=0.2,
    random_state=42
)

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
)

# ⭐ NEW XGBOOST SAFE TRAINING
xgb_model.fit(
    X_train_xgb,
    y_train_xgb
)

pred = xgb_model.predict(X_test_xgb)

xgb_rmse = np.sqrt(np.mean((pred - y_test_xgb) ** 2))
xgb_mae = np.mean(np.abs(pred - y_test_xgb))
xgb_r2 = r2_score(y_test_xgb, pred)

logger.info(
    f"XGB RMSE={xgb_rmse:.4f}, MAE={xgb_mae:.4f}, R2={xgb_r2:.4f}"
)

xgb_path = os.path.join(MODELS_DIR, "xgboost_model.json")
xgb_model.save_model(xgb_path)
logger.info("XGBoost model saved.")

# ================= UPDATE ENSEMBLE WEIGHTS ================= #
logger.info("Updating ensemble weights...")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

total_inv = (1 / lstm_mae) + (1 / xgb_mae)

config["models"]["weighted_ensemble"]["weights"]["lstm"] = float(
    (1 / lstm_mae) / total_inv
)

config["models"]["weighted_ensemble"]["weights"]["xgboost"] = float(
    (1 / xgb_mae) / total_inv
)

with open("config.yaml", "w") as f:
    yaml.safe_dump(config, f)

logger.info("Ensemble weights updated.")

logger.info("ALL MODELS TRAINED SUCCESSFULLY 🚀")
