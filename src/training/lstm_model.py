import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import yaml
import os
import numpy as np
import pandas as pd
from loguru import logger
 
class LSTMModel:
    """
    A class to define, train, and manage an LSTM model for progress prediction.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.model_params = self.config['model_training']['lstm']
        self.model = None
        self.model_save_path = self.model_params['model_path']
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        logger.info(f"LSTMModel initialized. Model will be saved to: {self.model_save_path}")
 
    def _load_config(self, config_path):
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
 
    def build_model(self, input_shape):
        """
        Builds the LSTM model architecture.
        :param input_shape: Tuple, shape of the input data (timesteps, features).
        """
        self.model = Sequential([
            LSTM(units=self.model_params.get('lstm_units', 50),
                 return_sequences=True,
                 input_shape=input_shape),
            Dropout(self.model_params.get('dropout_rate', 0.2)),
            LSTM(units=self.model_params.get('lstm_units', 50)),
            Dropout(self.model_params.get('dropout_rate', 0.2)),
            Dense(units=self.model_params.get('dense_units', 25), activation='relu'),
            Dense(units=1)  # Output layer for regression (e.g., predicting weight change)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_params.get('learning_rate', 0.001)),
                           loss='mean_squared_error')
        logger.info("LSTM model built successfully.")
        self.model.summary(print_fn=lambda x: logger.info(x))
        return self.model
 
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Trains the LSTM model.
        :param X_train: Training input data (numpy array).
        :param y_train: Training target data (numpy array).
        :param X_val: Validation input data (numpy array).
        :param y_val: Validation target data (numpy array).
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() first.")
            return
 
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.model_params.get('patience', 10), restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.model_save_path, monitor='val_loss', save_best_only=True, mode='min')
 
        logger.info(f"Starting LSTM model training for {self.model_params.get('epochs', 50)} epochs...")
        history = self.model.fit(X_train, y_train,
                                 epochs=self.model_params.get('epochs', 50),
                                 batch_size=self.model_params.get('batch_size', 32),
                                 validation_data=(X_val, y_val),
                                 callbacks=[early_stopping, model_checkpoint],
                                 verbose=1)
        logger.info("LSTM model training finished.")
        return history
 
    def predict(self, X_new):
        """
        Makes predictions using the trained LSTM model.
        :param X_new: New input data for prediction (numpy array).
        :return: Predicted values.
        """
        if self.model is None:
            self.load_model() # Attempt to load if not already loaded
            if self.model is None:
                logger.error("Model not built or loaded. Cannot make predictions.")
                return None
        logger.info("Making predictions with LSTM model.")
        return self.model.predict(X_new)
 
    def save_model(self):
        """
        Saves the trained LSTM model to the specified path.
        """
        if self.model:
            self.model.save(self.model_save_path)
            logger.info(f"LSTM model saved to {self.model_save_path}")
        else:
            logger.warning("No model to save.")
 
    def load_model(self):
        """
        Loads a pre-trained LSTM model from the specified path.
        """
        if os.path.exists(self.model_save_path):
            try:
                self.model = tf.keras.models.load_model(self.model_save_path)
                logger.info(f"LSTM model loaded from {self.model_save_path}")
                return self.model
            except Exception as e:
                logger.error(f"Error loading LSTM model from {self.model_save_path}: {e}")
                self.model = None
                return None
        else:
            logger.warning(f"No LSTM model found at {self.model_save_path}. Model not loaded.")
            self.model = None
            return None
 
# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Create a dummy config.yaml for testing if it doesn't exist
    if not os.path.exists("config.yaml"):
        dummy_config_content = """
# Global Configuration
project_name: "AI Workout & Diet Planner"
version: "1.0.0"
 
# Data Paths
data_paths:
  raw: "data/raw"
  external: "data/external"
  processed: "data/processed"
  features: "data/features"
  models: "data/models"
 
# Model Training
model_training:
  lstm:
    epochs: 5
    batch_size: 16
    learning_rate: 0.001
    lstm_units: 32
    dense_units: 16
    dropout_rate: 0.2
    patience: 2
    model_path: "data/models/lstm_model.h5"
  xgboost:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 5
    model_path: "data/models/xgboost_model.json"
        """
        with open("config.yaml", "w") as f:
            f.write(dummy_config_content)
 
    # Create dummy data for testing LSTM
    # Assuming input_shape = (timesteps, features)
    timesteps = 10
    features = 5
    num_samples = 100
 
    X_dummy = np.random.rand(num_samples, timesteps, features)
    y_dummy = np.random.rand(num_samples, 1)
 
    # Split into train and validation
    split_idx = int(0.8 * num_samples)
    X_train_dummy, X_val_dummy = X_dummy[:split_idx], X_dummy[split_idx:]
    y_train_dummy, y_val_dummy = y_dummy[:split_idx], y_dummy[split_idx:]
 
    lstm_trainer = LSTMModel()
 
    # Build model
    lstm_trainer.build_model(input_shape=(timesteps, features))
 
    # Train model
    lstm_trainer.train_model(X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy)
 
    # Make predictions
    predictions = lstm_trainer.predict(X_val_dummy[:5])
    logger.info(f"Dummy predictions for first 5 validation samples: {predictions.flatten()}")
 
    # Save and load model test
    lstm_trainer.save_model()
    loaded_model = LSTMModel()
    loaded_model.load_model()
 
    if loaded_model.model:
        loaded_predictions = loaded_model.predict(X_val_dummy[:5])
        logger.info(f"Loaded model predictions for first 5 validation samples: {loaded_predictions.flatten()}")
        # Verify if predictions are similar (they should be if model loaded correctly)
        assert np.allclose(predictions, loaded_predictions), "Loaded model predictions differ from original model predictions!"
        logger.info("Loaded model predictions match original model predictions.")
 
    # Clean up dummy config and model file
    # os.remove("config.yaml")
    # os.remove(lstm_trainer.model_save_path)
    # os.rmdir("data/models")
    # os.rmdir("data")
