from pydantic import BaseModel
from typing import List, Optional

class UserData(BaseModel):
    age: int
    weight_kg: float
    height_cm: float
    gender: str
    activity_level: str
    goal: str

class WorkoutData(BaseModel):
    workout_duration_minutes: int
    workout_intensity: str

class PredictionRequest(BaseModel):
    user_data: UserData
    workout_data: WorkoutData
