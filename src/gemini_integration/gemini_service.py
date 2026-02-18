import streamlit as st
import google.generativeai as genai
from loguru import logger

# ================= LOAD ENV ================= #

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in Streamlit secrets.")

genai.configure(api_key=GEMINI_API_KEY)

# ⭐ FREE MODEL
MODEL_NAME = "gemini-3-flash-preview"

suggestion_model = genai.GenerativeModel(MODEL_NAME)
chatbot_model = genai.GenerativeModel(MODEL_NAME)


# ================= PROMPT ================= #
def _generate_prompt_for_suggestions(user_data, prediction):

    return f"""
    You are a friendly AI fitness coach.

    User info:
    Age: {user_data.get("age")}
    Weight: {user_data.get("weight_kg")}
    Height: {user_data.get("height_cm")}
    Goal: {user_data.get("goal")}
    Prediction: {prediction}

    Give:
    1. Workout tips
    2. Diet suggestions
    3. Recovery advice
    4. Motivational message

    Keep it simple and safe.
    """


# ================= AI SUGGESTIONS ================= #
def generate_ai_suggestions(user_data, prediction_result):

    try:
        prompt = _generate_prompt_for_suggestions(
            user_data,
            prediction_result
        )

        response = suggestion_model.generate_content(prompt)

        return response.text

    except Exception as e:
        logger.error(f"Gemini suggestion error: {e}")
        return "AI suggestions currently unavailable."


# ================= CHATBOT ================= #
def chat_with_ai(user_message, chat_history):

    try:

        # ⭐ CONVERT STREAMLIT HISTORY → GEMINI FORMAT
        gemini_history = []

        for msg in chat_history:
            gemini_history.append({
                "role": msg.get("role", "user"),
                "parts": [{"text": msg.get("content", "")}]
            })

        # start chat with converted history
        chat = chatbot_model.start_chat(history=gemini_history)

        response = chat.send_message(user_message)

        return response.text

    except Exception as e:
        logger.error(f"Gemini chat error: {e}")
        return "AI assistant is currently unavailable."
