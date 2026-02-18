import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import yaml
import requests
import numpy as np
from datetime import date
from loguru import logger
import pandas as pd
import joblib
from src.diet_recommendation.diet_model_trainer import generate_diet_plan, load_data, prepare_food_data, get_preprocessor

from src.evaluation.evaluator import Evaluator
from src.data_collection.user_progress_tracker import UserProgressTracker


# ---------------- LOAD CONFIG ---------------- #
def load_config(config_path="config.yaml"):
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    config_full_path = os.path.join(base_dir, config_path)

    try:
        with open(config_full_path, 'r') as f:
            logger.info(f"Loading configuration from {config_full_path}")
            return yaml.safe_load(f)

    except FileNotFoundError:
        logger.error(f"Config file not found at {config_full_path}")
        st.error(f"Configuration file not found: {config_path}")
        st.stop()

    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        st.error(f"Config parsing error: {e}")
        st.stop()


config = load_config()
logger.info("Dashboard application started.")


# ---------------- INITIALIZE OBJECTS ---------------- #
evaluator = Evaluator()
progress_tracker = UserProgressTracker()

# --- Load Diet Model and Preprocessor ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")
DIET_MODEL_PATH = os.path.join(MODEL_DIR, "diet_model.pkl")
DIET_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "diet_preprocessor.pkl")
FOOD_DATA_PATH = os.path.join(PROJECT_ROOT, "Indian_Food_Nutrition_Processed.csv")

try:
    diet_model = joblib.load(DIET_MODEL_PATH)
    diet_preprocessor = joblib.load(DIET_PREPROCESSOR_PATH)
    food_df_raw = load_data(FOOD_DATA_PATH)
    food_df_diet = prepare_food_data(food_df_raw.copy())
    logger.info("Diet model, preprocessor, and food data loaded successfully.")
except FileNotFoundError:
    logger.error("Diet model, preprocessor, or food data file not found. Please run diet_model_trainer.py first.")
    st.error("Diet recommendation system not fully set up. Please run the diet model trainer.")
    diet_model = None
    diet_preprocessor = None
    food_df_diet = None
except Exception as e:
    logger.error(f"Error loading diet model or preprocessor: {e}")
    st.error(f"Error loading diet recommendation system: {e}")
    diet_model = None
    diet_preprocessor = None
    food_df_diet = None


# ---------------- STREAMLIT PAGE CONFIG ---------------- #
st.set_page_config(
    page_title=config['dashboard']['title'],
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title(config['dashboard']['title'])
st.write("Your personalized AI Workout & Diet Planner")


# ================= USER INPUT SECTION ================= #
st.header("User Data Input")

with st.form("user_input_form"):

    st.subheader("Personal Information")

    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

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

    st.write("More detailed preferences will be added here.")

    st.subheader("AI Diet Recommendation")
    calorie_target = st.number_input("Daily Calorie Target", min_value=1000, max_value=5000, value=2000)

    submitted = st.form_submit_button("Generate Plan")

    # -------- FORM SUBMISSION -------- #
    if submitted:

        logger.info("User submitted form. Generating plan...")
        st.write("Generating your personalized plan...")

        st.session_state['goal'] = goal

        # Features for models
        lstm_features = [age, weight_kg, height_cm]

        xgboost_features = [
            workout_duration_minutes,
            0 if workout_intensity == "Low"
            else 1 if workout_intensity == "Medium"
            else 2
        ]

        # Generate Diet Plan
        generated_diet_plan = "No diet plan generated."
        if diet_model and diet_preprocessor and food_df_diet is not None:
            user_data_for_diet = {
                "age": age,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "gender": gender,
                "activity_level": activity_level,
                "goal": goal,
                "calorie_target": calorie_target
            }
            try:
                generated_diet_plan = generate_diet_plan(user_data_for_diet, food_df_diet, diet_model, diet_preprocessor)
                logger.info("Diet plan generated successfully.")
            except Exception as e:
                logger.error(f"Error generating diet plan: {e}")
                st.error(f"Error generating diet plan: {e}")
        else:
            st.warning("Diet recommendation system not fully loaded. Cannot generate diet plan.")

        api_host = config['api']['host']
        api_port = config['api']['port']
        api_url = f"http://{api_host}:{api_port}/predict"

        logger.info(f"Calling API at {api_url}")

        try:
            response = requests.post(api_url, json={
                "user_data": lstm_features,
                "workout_data": xgboost_features
            })

            response.raise_for_status()
            prediction_result = response.json()

            st.success("Plan Generated Successfully!")

            st.write("### Your Personalized Recommendation")

            workout_plan = prediction_result.get(
                "workout_plan", "No workout plan generated."
            )
            diet_plan = prediction_result.get(
                "diet_plan", "No diet plan generated."
            )

            st.subheader("Workout Plan")
            if isinstance(workout_plan, list):
                for item in workout_plan:
                    st.write(f"- {item}")
            else:
                st.write(workout_plan)

            st.subheader("Diet Plan 🍽️")

            # ⭐ Extract real diet plan
            final_diet_plan = generated_diet_plan.get(
                "diet_plan",
                generated_diet_plan
            ) if isinstance(generated_diet_plan, dict) else generated_diet_plan

            if isinstance(final_diet_plan, dict):

                for meal_type, dishes in final_diet_plan.items():
                    if meal_type == "daily_summary":
                        st.subheader("Daily Macro Summary")
                        st.write(f"**Target Calories:** {dishes.get('target_calories')} kcal")
                        st.write(f"**Consumed Calories:** {dishes.get('consumed_calories')} kcal")
                        st.write(f"**Target Protein:** {dishes.get('target_protein_g')} g")
                        st.write(f"**Consumed Protein:** {dishes.get('consumed_protein_g')} g")
                        st.write(f"**Target Carbs:** {dishes.get('target_carbs_g')} g")
                        st.write(f"**Consumed Carbs:** {dishes.get('consumed_carbs_g')} g")
                        st.write(f"**Target Fats:** {dishes.get('target_fats_g')} g")
                        st.write(f"**Consumed Fats:** {dishes.get('consumed_fats_g')} g")
                    else:
                        st.markdown(f"### {meal_type.capitalize()}")

                        if isinstance(dishes, list):
                            for dish in dishes:
                                st.write(f"• {dish}")
                        else:
                            st.write(dishes)

            else:
                st.write(final_diet_plan)

            # ---------- EVALUATION ---------- #
            y_pred = prediction_result.get("prediction", [0.0])[0]

            if goal == "Lose Weight":
                y_true = weight_kg * 0.95
            elif goal == "Gain Muscle":
                y_true = weight_kg * 1.02
            else:
                y_true = weight_kg

            user_data_for_evaluator = {
                "actual_progress": y_pred,
                "target_goal": y_true,
                "actual_calorie_intake": 2000,
                "target_calorie_intake": 2100,
                "recommendation_impact": 0.7,
                "user_health_risk_factors": ["none"]
            }

            

        except requests.exceptions.ConnectionError:
            logger.error("API connection failed.")
            st.error(f"Could not connect to API: {api_url}")

        except requests.exceptions.RequestException as e:
            logger.exception(f"API error: {e}")
            st.error(f"Error calling API: {e}")

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            st.error(f"Unexpected error: {e}")


# ================= DASHBOARD SECTION ================= #
st.header("Your Progress Dashboard")
st.line_chart([10, 20, 15, 25, 30])


# ================= PROGRESS TRACKING ================= #
st.header("Track Your Progress")

user_id = "demo_user"

with st.form("progress_tracking_form"):

    st.subheader("Log Your Progress")

    progress_date = st.date_input("Date", value=date.today())

    current_weight_kg = st.number_input(
        "Current Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        key="current_weight_input"
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
    st.info("No progress data found yet. Log your first progress above!")


# ================= AI CHATBOT SECTION ================= #
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

    api_host = config['api']['host']
    api_port = config['api']['port']
    chat_api_url = f"http://{api_host}:{api_port}/chat"

    try:
        response = requests.post(chat_api_url, json={
            "user_message": prompt,
            "chat_history": st.session_state.chat_history
        })
        response.raise_for_status()
        ai_response = response.json().get("response", "Error: Could not get response from AI.")

        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)

    except requests.exceptions.ConnectionError:
        logger.error("API connection failed for chatbot.")
        st.error(f"Could not connect to AI Chatbot API: {chat_api_url}")
        st.session_state.chat_history.append({"role": "assistant", "content": "Error: Could not connect to AI Chatbot."})
    except requests.exceptions.RequestException as e:
        logger.exception(f"AI Chatbot API error: {e}")
        st.error(f"Error calling AI Chatbot API: {e}")
        st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
    except Exception as e:
        logger.exception(f"Unexpected error in chatbot: {e}")
        st.error(f"Unexpected error in chatbot: {e}")
        st.session_state.chat_history.append({"role": "assistant", "content": f"Unexpected error: {e}"})
