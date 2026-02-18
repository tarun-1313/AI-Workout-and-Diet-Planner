import yaml
from pathlib import Path
from loguru import logger
import numpy as np
import xgboost as xgb

from tensorflow.keras.models import load_model


class ModelServer:

    def __init__(self, config_path="config.yaml"):

        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        self.config_full_path = PROJECT_ROOT / config_path

        self.config = self._load_config()
        self.models = {}

        self._load_all_models()

    # ---------- LOAD CONFIG ---------- #
    def _load_config(self):
        with open(self.config_full_path, "r") as f:
            logger.info(f"Loading configuration from {self.config_full_path}")
            return yaml.safe_load(f)

    # ---------- GET MODEL PATH ---------- #
    def _get_model_path(self, model_name):

        model_config = self.config["models"].get(model_name)

        if not model_config:
            raise ValueError(f"Model {model_name} not found")

        # ⭐ IMPORTANT FIX (ensemble has no file)
        if "model_file" not in model_config:
            return None

        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        return PROJECT_ROOT / model_config["model_file"]

    # ---------- LOAD ALL MODELS ---------- #
    def _load_all_models(self):

        logger.info("Loading all models...")

        for model_name, model_info in self.config["models"].items():

            model_type = model_info["type"]
            model_path = self._get_model_path(model_name)

            # ⭐ config-only models
            if model_path is None:
                self.models[model_name] = model_info
                logger.info(f"Loaded config-only model: {model_name}")
                continue

            try:

                if model_type == "xgboost":
                    model = xgb.XGBRegressor()
                    model.load_model(str(model_path))
                    self.models[model_name] = model
                    logger.info(f"Loaded XGBoost: {model_name}")

                elif model_type == "lstm":
                    # ⭐ FIX keras loading issue
                    self.models[model_name] = load_model(
                        model_path,
                        compile=False
                    )
                    logger.info(f"Loaded LSTM: {model_name}")

                elif model_type == "weighted_ensemble":
                    self.models[model_name] = model_info

            except Exception as e:
                logger.error(f"Failed loading {model_name}: {e}")

    # ---------- PREDICT ---------- #
    def predict(self, model_name, features):

        model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"{model_name} not loaded")

        model_type = self.config["models"][model_name]["type"]

        if model_type == "xgboost":
            return model.predict(features)

        elif model_type == "lstm":
            return model.predict(features)

        elif model_type == "weighted_ensemble":

            lstm_model = self.models.get("lstm_model")
            xgb_model = self.models.get("xgboost_model")

            lstm_pred = lstm_model.predict(features["lstm"])
            xgb_pred = xgb_model.predict(features["xgboost"])

            weights = model.get("weights", {})

            lstm_w = weights.get("lstm", 0.5)
            xgb_w = weights.get("xgboost", 0.5)

            final_pred = (
                lstm_w * lstm_pred.flatten()[0]
                + xgb_w * xgb_pred.flatten()[0]
            )

            return [final_pred]

        else:
            raise ValueError("Unsupported model type")
