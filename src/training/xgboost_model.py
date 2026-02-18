import xgboost as xgb
import yaml
import os
from loguru import logger
 
class XGBoostModel:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.model_params = self.config['model_training']['xgboost']
        self.model_save_path = self.model_params['model_path']
        self.model = None
        logger.info(f"XGBoostModel initialized. Model will be saved to: {self.model_save_path}")
 
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
 
    def build_model(self):
        """Builds the XGBoost Regressor model."""
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.model_params.get('n_estimators', 100),
            learning_rate=self.model_params.get('learning_rate', 0.1),
            max_depth=self.model_params.get('max_depth', 5),
            random_state=42
        )
        logger.info("XGBoost model built successfully.")
 
    def train_model(self, X_train, y_train):
        """Trains the XGBoost model."""
        if self.model is None:
            self.build_model()
        logger.info("Starting XGBoost model training...")
        self.model.fit(X_train, y_train)
        self._save_model()
        logger.info("XGBoost model training finished.")
 
    def predict(self, X):
        """Makes predictions using the trained XGBoost model."""
        if self.model is None:
            self._load_model()
        if self.model is not None:
            logger.info("Making predictions with XGBoost model.")
            return self.model.predict(X)
        else:
            logger.error("Model not trained or loaded. Cannot make predictions.")
            raise RuntimeError("Model not trained or loaded.")
 
    def _save_model(self):
        """Saves the trained XGBoost model."""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        try:
            self.model.save_model(self.model_save_path)
            logger.info(f"XGBoost model saved to {self.model_save_path}")
        except Exception as e:
            logger.error(f"Error saving XGBoost model to {self.model_save_path}: {e}")
            raise
 
    def _load_model(self):
        """Loads a pre-trained XGBoost model."""
        if os.path.exists(self.model_save_path):
            try:
                self.model = xgb.XGBRegressor()
                self.model.load_model(self.model_save_path)
                logger.info(f"XGBoost model loaded from {self.model_save_path}")
            except Exception as e:
                logger.error(f"Error loading XGBoost model from {self.model_save_path}: {e}")
                self.model = None
        else:
            logger.warning(f"No XGBoost model found at {self.model_save_path}. Model not loaded.")
            self.model = None
