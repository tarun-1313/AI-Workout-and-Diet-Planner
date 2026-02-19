import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import requests
import numpy as np
from datetime import date
from loguru import logger
import pandas as pd
import joblib

from src.diet_recommendation.diet_model_trainer import (
    generate_diet_plan,
    load_data,
    prepare_food_data
)
from src.evaluation.evaluator import Evaluator
from src.data_collection.user_progress_tracker import UserProgressTracker


# ================= DEPLOYED BACKEND ================= #
BASE_API_URL = "https://ai-workout-and-diet-planner-4.onrender.com"

logger.info("Dashboard application started.")


# ================= INITIALIZE OBJECTS ================= #
evaluator = Evaluator()
progress_tracker = UserProgressTracker()


# ================= LOAD DIET MODEL ================= #
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")
DIET_MODEL_PATH = os.path.join(MODEL_DIR, "diet_model.pkl")
DIET_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "diet_preprocessor.pkl")
FOOD_DATA_PATH = os.path.join(PROJECT_ROOT, "Indian_Food_Nutrition_Processed.csv")

try:
    diet_model = joblib.load(DIET_MODEL_PATH)
    diet_preprocessor = joblib.load(DIET_PREPROCESSOR_PATH)
    food_df_raw = load_data(FOOD_DATA_PATH)
    food_df_diet = prepare_food_data(food_df_raw.copy())
    logger.info("Diet model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading diet model: {e}")
    diet_model = None
    diet_preprocessor = None
    food_df_diet = None


# ================= STREAMLIT PAGE CONFIG ================= #
st.set_page_config(
    page_title="AI Workout & Diet Planner",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("AI Workout & Diet Planner")
st.write("Your personalized AI Workout & Diet Planner")


# ================= USER INPUT ================= #
st.header("User Data Input")

with st.form("user_input_form"):

    st.subheader("Personal Information")

    age = st.number_input("Age", 1, 100, 25)
    weight_kg = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    height_cm = st.number_input("Height (cm)", 100.0, 250.0, 170.0)

    gender = st.selectbox("Gender", ["Male", "Female"])

    activity_level = st.selectbox(
        "Activity Level",
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Super Active"]
    )

    goal = st.selectbox(
        "Your Primary Goal",
        ["Maintain Weight", "Lose Weight", "Gain Muscle"]
    )

    st.subheader("Workout Preferences")

    workout_duration_minutes = st.slider(
        "Preferred Workout Duration (minutes)", 15, 120, 45
    )

    workout_intensity = st.selectbox(
        "Preferred Workout Intensity", ["Low", "Medium", "High"]
    )

    st.subheader("AI Diet Recommendation")

    calorie_target = st.number_input(
        "Daily Calorie Target", 1000, 5000, 2000
    )

    submitted = st.form_submit_button("Generate Plan")

    if submitted:

        st.info("Generating your personalized plan...")

        # Model features
        lstm_features = [age, weight_kg, height_cm]

        xgboost_features = [
            workout_duration_minutes,
            0 if workout_intensity == "Low"
            else 1 if workout_intensity == "Medium"
            else 2
        ]

        # ================= DIET PLAN ================= #
        generated_diet_plan = "No diet plan generated."

        if diet_model and diet_preprocessor and food_df_diet is not None:
            try:
                user_data_for_diet = {
                    "age": age,
                    "weight_kg": weight_kg,
                    "height_cm": height_cm,
                    "gender": gender,
                    "activity_level": activity_level,
                    "goal": goal,
                    "calorie_target": calorie_target
                }

                generated_diet_plan = generate_diet_plan(
                    user_data_for_diet,
                    food_df_diet,
                    diet_model,
                    diet_preprocessor
                )
            except Exception as e:
                st.error(f"Diet plan error: {e}")

        # ================= WORKOUT API CALL ================= #
        try:
            api_url = f"{BASE_API_URL}/predict"

            response = requests.post(
                api_url,
                json={
                    "user_data": lstm_features,
                    "workout_data": xgboost_features
                },
                timeout=60
            )

            response.raise_for_status()
            prediction_result = response.json()

            st.success("Plan Generated Successfully!")

            # Workout Plan
            st.subheader("Workout Plan")

            workout_plan = prediction_result.get(
                "workout_plan",
                "No workout plan generated."
            )

            if isinstance(workout_plan, list):
                for item in workout_plan:
                    st.write(f"- {item}")
            else:
                st.write(workout_plan)

            # Diet Plan
            st.subheader("Diet Plan 🍽️")

            final_diet_plan = (
                generated_diet_plan.get("diet_plan", generated_diet_plan)
                if isinstance(generated_diet_plan, dict)
                else generated_diet_plan
            )

            if isinstance(final_diet_plan, dict):
                for meal_type, dishes in final_diet_plan.items():
                    if meal_type == "daily_summary":
                        st.subheader("Daily Macro Summary")
                        for k, v in dishes.items():
                            st.write(f"**{k.replace('_',' ').title()}**: {v}")
                    else:
                        st.markdown(f"### {meal_type.capitalize()}")
                        if isinstance(dishes, list):
                            for dish in dishes:
                                st.write(f"• {dish}")
                        else:
                            st.write(dishes)
            else:
                st.write(final_diet_plan)

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")


# ================= PROGRESS TRACKING ================= #
st.header("Track Your Progress")

user_id = "demo_user"

with st.form("progress_tracking_form"):

    progress_date = st.date_input("Date", value=date.today())

    current_weight_kg = st.number_input(
        "Current Weight (kg)",
        30.0,
        200.0,
        70.0
    )

    progress_submitted = st.form_submit_button("Save Progress")

    if progress_submitted:

        progress_data = {
            "date": str(progress_date),
            "weight_kg": current_weight_kg
        }

        all_progress = progress_tracker.load_progress(user_id)

        if user_id not in all_progress:
            all_progress[user_id] = []

        all_progress[user_id].append(progress_data)
        progress_tracker.save_progress(user_id, all_progress)

        st.success("Progress saved successfully!")


# ================= HISTORICAL DATA ================= #
st.subheader("Your Historical Progress")

loaded_progress = progress_tracker.load_progress(user_id)

if user_id in loaded_progress and loaded_progress[user_id]:

    progress_df = pd.DataFrame(loaded_progress[user_id])
    progress_df['date'] = pd.to_datetime(progress_df['date'])
    progress_df = progress_df.sort_values(by='date')

    st.dataframe(progress_df)
    st.line_chart(progress_df.set_index('date')['weight_kg'])

else:
    st.info("No progress data found yet.")


# ================= AI CHATBOT ================= #
st.header("AI Fitness Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about fitness..."):

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        chat_api_url = f"{BASE_API_URL}/chat"

        response = requests.post(
            chat_api_url,
            json={
                "user_message": prompt,
                "chat_history": st.session_state.chat_history
            },
            timeout=60
        )

        response.raise_for_status()

        ai_response = response.json().get(
            "response",
            "Error: Could not get response."
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": ai_response}
        )

        with st.chat_message("assistant"):
            st.markdown(ai_response)

    except requests.exceptions.RequestException as e:
        st.error(f"Chatbot API Error: {e}")
