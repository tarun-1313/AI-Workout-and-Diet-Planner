import yaml
import os
import joblib # For saving/loading ensemble weights if needed
from tensorflow.keras.models import load_model # For loading LSTM model
import xgboost as xgb
from loguru import logger
 
# Assuming LSTMModel and XGBoostModel classes are available for loading
# For simplicity, we'll directly load the models here.
# In a more complex scenario, we might import and use the classes.
 
class WeightedEnsembleModel:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.ensemble_params = self.config['model_training']['ensemble']
        self.lstm_model_path = self.config['model_training']['lstm']['model_path']
        self.xgboost_model_path = self.config['model_training']['xgboost']['model_path']
        self.ensemble_save_path = self.ensemble_params['model_path']
        self.weights = self.ensemble_params['weights']
 
        self.lstm_model = None
        self.xgboost_model = None
        logger.info("WeightedEnsembleModel initialized.")
 
    def _load_config(self, config_path):
        """Loads configuration from a YAML file."""
        script_dir = os.path.dirname(__file__)
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        config_full_path = os.path.join(base_dir, config_path)
        try:
            with open(config_full_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_full_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
 
    def load_base_models(self):
        """Loads the pre-trained LSTM and XGBoost models."""
        try:
            self.lstm_model = load_model(self.lstm_model_path)
            logger.info(f"LSTM model loaded from {self.lstm_model_path}")
        except Exception as e:
            logger.error(f"Error loading LSTM model from {self.lstm_model_path}: {e}")
            self.lstm_model = None
 
        try:
            self.xgboost_model = xgb.XGBRegressor()
            self.xgboost_model.load_model(self.xgboost_model_path)
            logger.info(f"XGBoost model loaded from {self.xgboost_model_path}")
        except Exception as e:
            logger.error(f"Error loading XGBoost model from {self.xgboost_model_path}: {e}")
            self.xgboost_model = None
 
    def predict(self, lstm_input, xgboost_input):
        """
        Makes predictions using the ensemble model.
        Assumes lstm_input and xgboost_input are preprocessed and ready for their respective models.
        """
        if self.lstm_model is None or self.xgboost_model is None:
            self.load_base_models()
            if self.lstm_model is None or self.xgboost_model is None:
                logger.error("One or both base models could not be loaded. Cannot make predictions.")
                raise RuntimeError("One or both base models could not be loaded.")
 
        logger.info("Making predictions with Weighted Ensemble model.")
        lstm_pred = self.lstm_model.predict(lstm_input)
        xgboost_pred = self.xgboost_model.predict(xgboost_input)
 
        # Ensure predictions are 1D arrays for weighted average
        lstm_pred = lstm_pred.flatten()
        xgboost_pred = xgboost_pred.flatten()
 
        # Apply weights
        ensemble_pred = (self.weights['lstm'] * lstm_pred +
                         self.weights['xgboost'] * xgboost_pred)
        return ensemble_pred
 
    def _save_ensemble_config(self):
        """Saves the ensemble configuration (weights)."""
        os.makedirs(os.path.dirname(self.ensemble_save_path), exist_ok=True)
        try:
            joblib.dump(self.weights, self.ensemble_save_path)
            logger.info(f"Ensemble configuration saved to {self.ensemble_save_path}")
        except Exception as e:
            logger.error(f"Error saving ensemble configuration to {self.ensemble_save_path}: {e}")
            raise
 
    def _load_ensemble_config(self):
        """Loads the ensemble configuration (weights)."""
        if os.path.exists(self.ensemble_save_path):
            try:
                self.weights = joblib.load(self.ensemble_save_path)
                logger.info(f"Ensemble configuration loaded from {self.ensemble_save_path}")
            except Exception as e:
                logger.error(f"Error loading ensemble configuration from {self.ensemble_save_path}: {e}")
                raise
        else:
            logger.warning(f"No ensemble configuration found at {self.ensemble_save_path}. Using default weights.")
