# AI Workout & Diet Planner

This project provides a personalized AI-powered system for generating workout and diet plans, integrated into an interactive Streamlit dashboard. It leverages machine learning models for diet recommendations and an AI chatbot for fitness-related queries.

## Features

-   **Personalized Diet Plans**: Generates daily diet plans based on user's age, weight, height, gender, activity level, and fitness goals (Lose Weight, Gain Muscle, Maintain Weight).
    -   **Macro-Nutrient Targeting**: Diet plans are tailored to meet specific calorie, protein, carbohydrate, and fat targets based on the user's goal.
    -   **Rule-Based Filtering**: Filters out non-meal items (e.g., spices, powders) and prioritizes foods based on suitability scores.
    -   **ML-Powered Ranking**: Uses a RandomForestRegressor model to rank food items for optimal suitability.
-   **Workout Plan Generation**: Provides workout recommendations based on preferred duration and intensity.
-   **Interactive Streamlit Dashboard**: A user-friendly interface for inputting personal data, viewing generated plans, and tracking progress.
-   **AI Fitness Chatbot**: An integrated chatbot powered by Gemini-2.5-Flash to answer fitness and nutrition-related questions.
-   **User Progress Tracking**: Allows users to log and visualize their weight progress over time.
-   **Modular Architecture**: Separates concerns into data collection, diet recommendation, workout generation, API services, and UI.

## Technologies Used

-   **Python**: Core programming language.
-   **Streamlit**: For building the interactive web dashboard.
-   **FastAPI**: For the backend API services (workout plan generation, AI chatbot).
-   **Scikit-learn**: For machine learning models (RandomForestRegressor, MinMaxScaler, OneHotEncoder, ColumnTransformer) in diet recommendation.
-   **Pandas & NumPy**: For data manipulation and numerical operations.
-   **Joblib**: For serializing and deserializing machine learning models.
-   **Requests**: For making HTTP requests to the FastAPI backend.
-   **PyYAML**: For configuration management.
-   **Loguru**: For robust logging.
-   **Google Gemini-2.5-Flash**: Powers the AI chatbot.

## Project Structure

```
.
├── api/                          # FastAPI backend services
│   ├── main.py                   # Main FastAPI application
│   └── ...
├── dashboard/                    # Streamlit frontend application
│   └── dashboard.py              # Main Streamlit app
├── data/                         # Data files and trained models
│   ├── models/                   # Saved ML models and preprocessors
│   └── ...
├── src/                          # Source code for various modules
│   ├── data_collection/          # User progress tracking
│   ├── diet_recommendation/      # Diet model training and generation logic
│   │   └── diet_model_trainer.py # Script to train and save the diet model
│   ├── evaluation/               # Evaluation logic
│   ├── gemini_integration/       # Gemini API integration
│   └── ...
├── Indian_Food_Nutrition_Processed.csv # Dataset for diet recommendations
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # Project README
```

## Setup Instructions

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "AI Workout & Diet Planner" # Or whatever your project folder is named
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Navigate to the project root directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

You will need a Google Gemini API key for the AI Chatbot functionality.
1.  Go to the [Google AI Studio](https://aistudio.google.com/app/apikey) to get your API key.
2.  Create a `.streamlit` directory in your project root if it doesn't exist.
3.  Inside the `.streamlit` directory, create a file named `secrets.toml`.
4.  Add your Gemini API key to `secrets.toml` like this:
    ```toml
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    Replace `"YOUR_GEMINI_API_KEY"` with your actual Gemini API key.
    
    **Important**: Do not commit `secrets.toml` to your version control system. Add it to your `.gitignore` file.

### 5. Train the Diet Recommendation Model

Before running the dashboard, you need to train the diet recommendation model. Navigate to the `src/diet_recommendation` directory and run the `diet_model_trainer.py` script:

```bash
cd src/diet_recommendation
python diet_model_trainer.py
# Or on some systems: py diet_model_trainer.py
```
This will save the trained model and preprocessor to the `data/models` directory.

### 6. Start the FastAPI Backend

Navigate back to the project root directory and start the FastAPI application:

```bash
cd ../.. # If you are in src/diet_recommendation
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Keep this terminal running.

### 7. Run the Streamlit Dashboard

Open a new terminal, navigate to the project root directory, and run the Streamlit dashboard:

```bash
streamlit run dashboard/dashboard.py
```

This will open the Streamlit application in your web browser.

## Usage

1.  **Input User Data**: On the Streamlit dashboard, fill in your personal information, workout preferences, and diet preferences.
2.  **Generate Plan**: Click "Generate Plan" to receive personalized workout and diet recommendations.
3.  **Track Progress**: Use the "Track Your Progress" section to log your weight over time and visualize your progress.
4.  **AI Chatbot**: Interact with the AI Fitness Chatbot for any fitness or nutrition-related questions.

## Contributing

Feel free to fork the repository, open issues, and submit pull requests.

## License
 MIT License