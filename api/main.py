from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import yaml
import os
import numpy as np
from loguru import logger

from src.model_serving.model_server import ModelServer
from src.gemini_integration.gemini_service import (
    generate_ai_suggestions,
    chat_with_ai
)

# ================= FASTAPI APP ================= #
app = FastAPI(
    title="AI Workout & Diet Planner API",
    description="API for personalized workout and diet recommendations",
    version="1.0.0"
)

# ================= REQUEST MODELS ================= #
class PredictionRequest(BaseModel):
    user_data: list[float]      # [age, weight, height]
    workout_data: list[float]   # [duration, intensity]


class ChatRequest(BaseModel):
    user_message: str
    chat_history: list = []


# ================= LOAD CONFIG ================= #
def load_config(config_path="config.yaml"):

    script_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    config_full_path = os.path.join(base_dir, config_path)

    with open(config_full_path, "r") as f:
        logger.info(f"Loading config from {config_full_path}")
        return yaml.safe_load(f)


config = load_config()

# ================= MODEL SERVER ================= #
model_server_instance = None


@app.on_event("startup")
async def startup_event():
    global model_server_instance

    logger.info("Initializing ModelServer...")
    model_server_instance = ModelServer()
    logger.info("Models loaded successfully.")


# ================= ROUTES ================= #
@app.get("/")
async def root():
    return {"message": "API running successfully"}


@app.get("/health")
async def health():
    if model_server_instance is None:
        return {"status": "unhealthy"}
    return {"status": "healthy"}


# ================= PREDICT ================= #
@app.post("/predict")
async def predict(request: PredictionRequest):

    if model_server_instance is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:

        # -------- BASIC INPUT -------- #
        age = request.user_data[0]
        weight = request.user_data[1]
        height = request.user_data[2]

        workout_duration = request.workout_data[0]
        workout_intensity = request.workout_data[1]

        # -------- BUILD FULL LSTM FEATURES (8) -------- #
        bmi = weight / ((height / 100) ** 2)

        activity_level = 1      # default
        calorie_intake = 2000   # default
        goal = 1                # default
        intensity = workout_intensity

        lstm_features = [
            age,
            weight,
            height,
            bmi,
            activity_level,
            calorie_intake,
            goal,
            intensity
        ]

        # -------- XGBOOST FEATURES -------- #
        xgb_features = [
           workout_duration,
           workout_intensity,
           bmi,
           activity_level,
           goal
        ]

        # -------- MODEL INPUT SHAPES -------- #
        lstm_input = np.array(lstm_features).reshape(1, 1, -1)
        xgb_input = np.array(xgb_features).reshape(1, -1)

        # -------- MODEL PREDICTION -------- #
        prediction = model_server_instance.predict(
            "weighted_ensemble",
            {
                "lstm": lstm_input,
                "xgboost": xgb_input
            }
        )

        # -------- GEMINI SUGGESTIONS -------- #
        try:
            ai_suggestions = generate_ai_suggestions(
                {
                    "age": age,
                    "weight_kg": weight,
                    "height_cm": height,
                    "goal": "fitness"
                },
                {"prediction": float(prediction[0])}
            )
        except Exception as e:
            logger.error(f"Gemini suggestion failed: {e}")
            ai_suggestions = "AI suggestions currently unavailable."

        return {
            "prediction": [float(prediction[0])],
            "workout_plan": [
                "Pushups - 3 sets",
                "Squats - 3 sets",
                "Plank - 60 sec"
            ],
            "diet_plan": [
                "High protein breakfast",
                "Balanced lunch",
                "Light dinner"
            ],
            "ai_suggestions": ai_suggestions
        }

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================= CHATBOT ================= #
@app.post("/chat")
async def chat(request: ChatRequest):

    try:
        response = chat_with_ai(
            request.user_message,
            request.chat_history
        )
        return {"response": response}

    except Exception as e:
        logger.exception(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================= MAIN ================= #
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"]
    )
