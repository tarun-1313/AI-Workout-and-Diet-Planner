import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# --- Configuration ---
DATA_PATH = "Indian_Food_Nutrition_Processed.csv"
MODEL_DIR = "data/models"
MODEL_FILENAME = "diet_model.pkl"
PREPROCESSOR_FILENAME = "diet_preprocessor.pkl"

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Data Loading and Initial Inspection ---
def load_data(data_path):
    """Loads the dataset and renames columns for easier access."""
    df = pd.read_csv(data_path)
    # Rename columns to be more Python-friendly
    df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True).str.lower()
    df = df.rename(columns={
        'calories_kcal': 'calories',
        'carbohydrates_g': 'carbohydrates',
        'protein_g': 'protein',
        'fats_g': 'fats',
        'free_sugar_g': 'free_sugar',
        'fibre_g': 'fibre',
        'sodium_mg': 'sodium',
        'calcium_mg': 'calcium',
        'iron_mg': 'iron',
        'vitamin_c_mg': 'vitamin_c',
        'folate_g': 'folate'
    })
    return df

# --- 2. Data Preparation (before ColumnTransformer) ---
def prepare_food_data(df):
    """Handles missing values and adds necessary features like suitability_score."""
    # Add meal_type and diet_type if not present (assuming they will be added or inferred)
    if 'meal_type' not in df.columns:
        df['meal_type'] = 'unknown' # Placeholder, ideally this comes from data
    if 'diet_type' not in df.columns:
        df['diet_type'] = 'general' # Placeholder, ideally this comes from data

    # Filter out non-meal items (spices, powders, etc.)
    # Use a regex pattern for case-insensitive filtering
    filter_pattern = "masala|powder|spice|paste|seasoning|pickle|chutney"
    df = df[~df["dish_name"].str.contains(filter_pattern, case=False, na=False)]

    # Handle missing values
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')

    # Feature Engineering: Create a synthetic 'suitability_score' for ranking
    df['suitability_score'] = (
        df['protein'] * 0.45 +      # Increased emphasis on protein
        df['carbohydrates'] * 0.25 + # Added carbohydrates
        df['fibre'] * 0.2 +         # Maintained fiber importance
        df['calories'] * 0.05 -     # Slight increase in calorie contribution
        df['free_sugar'] * 0.25 -   # Increased penalty for free sugar
        df['fats'] * 0.15           # Increased penalty for fats
    )
    return df

# --- 2. Data Preprocessing (ColumnTransformer part) ---
def get_preprocessor(df):
    """Defines and returns the ColumnTransformer for numerical and categorical features."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Remove 'suitability_score' from numerical_cols if it's there, as it's the target
    if 'suitability_score' in numerical_cols:
        numerical_cols.remove('suitability_score')

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    # Remove 'dish_name' from categorical_cols if it's there
    if 'dish_name' in categorical_cols:
        categorical_cols.remove('dish_name')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor

# --- 3. Model Training ---
def train_model(df_prepared, preprocessor):
    """Trains a RandomForestRegressor model for food ranking."""
    X = df_prepared.drop(columns=['suitability_score', 'dish_name']) # dish_name is not a feature for training
    y = df_prepared['suitability_score']

    # Apply preprocessing to X
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    # This part is tricky with ColumnTransformer and remainder='passthrough'
    # For simplicity, let's assume we only use transformed features for now
    # If 'dish_name' was passed through, it would be at the end.
    # We need to ensure X_processed is a DataFrame for model training if we want feature names
    # For RandomForest, a numpy array is fine.

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_processed, y)
    return model

# Helper function to calculate macronutrient targets
def calculate_macro_targets(calorie_target, goal):
    """Calculates target macronutrients based on calorie target and goal."""
    # Default balanced ratios (e.g., for maintenance)
    protein_ratio = 0.25
    carb_ratio = 0.50
    fat_ratio = 0.25

    if goal == 'Lose Weight':
        protein_ratio = 0.35 # Higher protein for satiety and muscle preservation
        carb_ratio = 0.40
        fat_ratio = 0.25
    elif goal == 'Gain Muscle':
        protein_ratio = 0.30 # High protein for muscle synthesis
        carb_ratio = 0.50 # Sufficient carbs for energy
        fat_ratio = 0.20
    # For 'Maintain Weight', use default balanced ratios

    # 1 gram of protein = 4 calories
    # 1 gram of carbohydrates = 4 calories
    # 1 gram of fat = 9 calories

    target_protein_g = (calorie_target * protein_ratio) / 4
    target_carbs_g = (calorie_target * carb_ratio) / 4
    target_fats_g = (calorie_target * fat_ratio) / 9

    return {
        'protein': target_protein_g,
        'carbs': target_carbs_g,
        'fats': target_fats_g
    }

# --- 4. Diet Plan Generation Function ---
def generate_diet_plan(user_data, food_df, model, preprocessor):
    """
    Generates a personalized diet plan based on user information.

    Args:
        user_data (dict): Contains user's age, weight, height, goal, activity_level, calorie_target.
        food_df (pd.DataFrame): The preprocessed food dataset.
        model: The trained RandomForestRegressor model.
        preprocessor: The fitted ColumnTransformer for preprocessing.

    Returns:
        dict: A structured daily diet plan.
    """
    age = user_data['age']
    weight_kg = user_data['weight_kg']
    height_cm = user_data['height_cm']
    gender = user_data['gender']
    activity_level = user_data['activity_level']
    goal = user_data['goal']
    calorie_target = user_data.get('calorie_target') # Use provided target or calculate

    # Calculate BMR (Basal Metabolic Rate) - Mifflin-St Jeor Equation
    if gender == 'Male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else: # Female
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    # Activity Multipliers
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Super Active": 1.9
    }
    tdee = bmr * activity_multipliers.get(activity_level, 1.2)

    # Adjust calorie target based on goal
    if calorie_target is None:
        if goal == 'Lose Weight':
            calorie_target = tdee - 500 # 500 kcal deficit for weight loss
        elif goal == 'Gain Muscle':
            calorie_target = tdee + 300 # 300 kcal surplus for muscle gain
        else: # Maintain Weight
            calorie_target = tdee

    # Calculate macro targets
    macro_targets = calculate_macro_targets(calorie_target, goal)

    # --- Rule-based Filtering ---
    filtered_foods = food_df.copy()

    # Example rule: For weight loss, prioritize lower calorie and lower sugar foods
    if goal == 'Lose Weight':
        filtered_foods = filtered_foods[filtered_foods['calories'] < 500]
        filtered_foods = filtered_foods[filtered_foods['free_sugar'] < 10]
    # Example rule: For muscle gain, prioritize higher protein foods
    elif goal == 'Gain Muscle':
        filtered_foods = filtered_foods[filtered_foods['protein'] > 5]

    # --- ML Ranking ---
    # Prepare filtered foods for prediction
    X_filtered = filtered_foods.drop(columns=['suitability_score', 'dish_name'])
    X_filtered_processed = preprocessor.transform(X_filtered)
    filtered_foods['predicted_suitability'] = model.predict(X_filtered_processed)

    # Sort by predicted suitability score
    filtered_foods = filtered_foods.sort_values(by='predicted_suitability', ascending=False).reset_index(drop=True)

    # --- Build Structured Daily Diet Plan ---
    diet_plan = {
        "breakfast": [],
        "lunch": [],
        "dinner": [],
        "snacks": []
    }

    # Calorie distribution for meals (can be customized)
    meal_calorie_distribution = {
        "breakfast": 0.25,
        "lunch": 0.35,
        "dinner": 0.30,
        "snacks": 0.10
    }

    # Track daily macros consumed
    daily_protein_consumed = 0
    daily_carbs_consumed = 0
    daily_fats_consumed = 0
    daily_calories_consumed = 0

    # Keep track of used dishes to avoid duplicates
    used_dishes = set()

    # Assign foods to meals, prioritizing macro targets
    for meal_type in ["breakfast", "lunch", "dinner", "snacks"]:
        meal_calorie_budget = calorie_target * meal_calorie_distribution[meal_type]
        current_meal_calories = 0
        meal_protein_consumed = 0
        meal_carbs_consumed = 0
        meal_fats_consumed = 0

        # Filter foods suitable for this meal type or general
        available_foods = filtered_foods[
            (filtered_foods['meal_type'] == meal_type) |
            (filtered_foods['meal_type'].isin(['unknown', 'general']))
        ].copy()

        # Remove already used dishes
        available_foods = available_foods[~available_foods['dish_name'].isin(used_dishes)]

        # Sort by predicted suitability
        available_foods = available_foods.sort_values(by='predicted_suitability', ascending=False)

        for _, food in available_foods.iterrows():
            # Check if adding this food exceeds meal calorie budget
            if current_meal_calories + food['calories'] > meal_calorie_budget:
                continue

            # Check if adding this food would excessively over/under-shoot daily macro targets
            # This is a simplified check; a more advanced one would use optimization
            temp_protein = daily_protein_consumed + food['protein']
            temp_carbs = daily_carbs_consumed + food['carbohydrates']
            temp_fats = daily_fats_consumed + food['fats']

            # Simple heuristic: avoid adding if it pushes a macro too far over target
            # Or if it's a critical macro (like protein for muscle gain/weight loss) and we're already low
            if (goal == 'Gain Muscle' or goal == 'Lose Weight') and temp_protein > macro_targets['protein'] * 1.2: # Don't exceed protein too much
                 # If we are already close to protein target and this food has a lot of protein, skip
                if daily_protein_consumed > macro_targets['protein'] * 0.8 and food['protein'] > macro_targets['protein'] * 0.1:
                    continue
            
            if (goal == 'Lose Weight') and temp_fats > macro_targets['fats'] * 1.2: # Don't exceed fats too much for weight loss
                if daily_fats_consumed > macro_targets['fats'] * 0.8 and food['fats'] > macro_targets['fats'] * 0.1:
                    continue

            # Add the food
            diet_plan[meal_type].append(food['dish_name'])
            used_dishes.add(food['dish_name'])
            current_meal_calories += food['calories']
            meal_protein_consumed += food['protein']
            meal_carbs_consumed += food['carbohydrates']
            meal_fats_consumed += food['fats']

            daily_calories_consumed += food['calories']
            daily_protein_consumed += food['protein']
            daily_carbs_consumed += food['carbohydrates']
            daily_fats_consumed += food['fats']

            # Stop adding to this meal if calorie budget is reasonably met
            if current_meal_calories >= meal_calorie_budget * 0.8:
                break
        
        # If meal is still empty or very low on calories, try to fill with any remaining suitable food
        # This is a fallback to ensure meals are not empty
        if not diet_plan[meal_type] or current_meal_calories < meal_calorie_budget * 0.5:
            fallback_foods = filtered_foods[~filtered_foods['dish_name'].isin(used_dishes)].copy()
            fallback_foods = fallback_foods.sort_values(by='predicted_suitability', ascending=False)
            for _, food in fallback_foods.iterrows():
                if current_meal_calories + food['calories'] <= meal_calorie_budget:
                    diet_plan[meal_type].append(food['dish_name'])
                    used_dishes.add(food['dish_name'])
                    current_meal_calories += food['calories']
                    meal_protein_consumed += food['protein']
                    meal_carbs_consumed += food['carbohydrates']
                    meal_fats_consumed += food['fats']

                    daily_calories_consumed += food['calories']
                    daily_protein_consumed += food['protein']
                    daily_carbs_consumed += food['carbohydrates']
                    daily_fats_consumed += food['fats']
                    break # Add only one fallback food to avoid overfilling

    # Add a summary of daily macros
    diet_plan['daily_summary'] = {
        'target_calories': round(calorie_target),
        'consumed_calories': round(daily_calories_consumed),
        'target_protein_g': round(macro_targets['protein']),
        'consumed_protein_g': round(daily_protein_consumed),
        'target_carbs_g': round(macro_targets['carbs']),
        'consumed_carbs_g': round(daily_carbs_consumed),
        'target_fats_g': round(macro_targets['fats']),
        'consumed_fats_g': round(daily_fats_consumed)
    }

    return {"diet_plan": diet_plan}

# --- Main Execution for Training and Saving ---
if __name__ == "__main__":
    # Load data
    df = load_data(DATA_PATH)

    # Preprocess data and get preprocessor
    df_prepared = prepare_food_data(df.copy()) # Use a copy to avoid modifying original df
    preprocessor = get_preprocessor(df_prepared)

    # Train model
    model = train_model(df_prepared, preprocessor)

    # Save model and preprocessor
    joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILENAME))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, PREPROCESSOR_FILENAME))
    print(f"Model saved to {os.path.join(MODEL_DIR, MODEL_FILENAME)}")
    print(f"Preprocessor saved to {os.path.join(MODEL_DIR, PREPROCESSOR_FILENAME)}")

    # --- Example Prediction ---
    print("\n--- Example Diet Plan Generation ---")
    # Load the saved model and preprocessor for inference
    loaded_model = joblib.load(os.path.join(MODEL_DIR, MODEL_FILENAME))
    loaded_preprocessor = joblib.load(os.path.join(MODEL_DIR, PREPROCESSOR_FILENAME))

    # Example user data
    example_user_data = {
        'age': 30,
        'weight_kg': 70,
        'height_cm': 175,
        'gender': 'Male',
        'activity_level': 'Moderately Active',
        'goal': 'Lose Weight',
        'calorie_target': None # Let the system calculate
    }

    diet_recommendation = generate_diet_plan(example_user_data, df_prepared, loaded_model, loaded_preprocessor)
    import json
    print(json.dumps(diet_recommendation, indent=2))

    example_user_data_muscle = {
        'age': 25,
        'weight_kg': 80,
        'height_cm': 180,
        'gender': 'Male',
        'activity_level': 'Very Active',
        'goal': 'Gain Muscle',
        'calorie_target': 2800 # Specific target
    }
    print("\n--- Example Diet Plan Generation (Muscle Gain) ---")
    diet_recommendation_muscle = generate_diet_plan(example_user_data_muscle, df_prepared, loaded_model, loaded_preprocessor)
    print(json.dumps(diet_recommendation_muscle, indent=2))
