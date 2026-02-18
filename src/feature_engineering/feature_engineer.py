import pandas as pd
import yaml
import os
from loguru import logger
 
class FeatureEngineer:
    """
    Handles the creation of features from raw user and other collected data.
    This includes calculations for BMI, BMR, TDEE, and macro splits.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        logger.info("FeatureEngineer initialized.")
 
    def _load_config(self, config_path):
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
 
    def calculate_bmi(self, weight_kg, height_cm):
        """
        Calculates Body Mass Index (BMI).
        BMI = weight (kg) / (height (m))^2
        """
        if height_cm <= 0:
            logger.error("Height must be greater than 0 to calculate BMI.")
            return None
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        logger.info(f"Calculated BMI: {bmi:.2f} for weight {weight_kg}kg and height {height_cm}cm.")
        return round(bmi, 2)
 
    def calculate_bmr(self, weight_kg, height_cm, age, gender):
        """
        Calculates Basal Metabolic Rate (BMR) using the Mifflin-St Jeor Equation.
        For men: BMR = (10 * weight in kg) + (6.25 * height in cm) - (5 * age in years) + 5
        For women: BMR = (10 * weight in kg) + (6.25 * height in cm) - (5 * age in years) - 161
        """
        if gender.lower() == 'male':
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        elif gender.lower() == 'female':
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        else:
            logger.error(f"Invalid gender '{gender}' for BMR calculation. Use 'male' or 'female'.")
            return None
        logger.info(f"Calculated BMR: {bmr:.2f} for user (gender: {gender}, age: {age}, weight: {weight_kg}kg, height: {height_cm}cm).")
        return round(bmr, 2)
 
    def calculate_tdee(self, bmr, activity_level):
        """
        Calculates Total Daily Energy Expenditure (TDEE) based on BMR and activity level.
        Activity factors:
        - Sedentary: 1.2 (little or no exercise)
        - Light: 1.375 (light exercise/sports 1-3 days/week)
        - Moderate: 1.55 (moderate exercise/sports 3-5 days/week)
        - Very Active: 1.725 (hard exercise/sports 6-7 days a week)
        - Extra Active: 1.9 (very hard exercise/physical job)
        """
        activity_factors = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "very_active": 1.725,
            "extra_active": 1.9
        }
        factor = activity_factors.get(activity_level.lower())
        if factor is None:
            logger.error(f"Invalid activity level '{activity_level}'. "
                         f"Choose from {list(activity_factors.keys())}.")
            return None
        tdee = bmr * factor
        logger.info(f"Calculated TDEE: {tdee:.2f} for BMR {bmr} and activity level {activity_level}.")
        return round(tdee, 2)
 
    def calculate_macro_split(self, tdee, goal, dietary_preference="none"):
        """
        Calculates macronutrient split (Protein, Carbs, Fats) based on TDEE, goal, and dietary preference.
        Returns percentages for P, C, F.
        """
        # Default splits (percentages)
        # Protein, Carbs, Fats
        macro_splits = {
            "lose_weight": {"protein": 0.35, "carbs": 0.30, "fat": 0.35},
            "gain_muscle": {"protein": 0.30, "carbs": 0.40, "fat": 0.30},
            "maintain": {"protein": 0.25, "carbs": 0.45, "fat": 0.30},
        }
 
        split = macro_splits.get(goal.lower())
        if split is None:
            logger.error(f"Invalid goal '{goal}'. Choose from {list(macro_splits.keys())}.")
            return None
 
        # Adjust for dietary preferences if necessary (simplified example)
        if dietary_preference.lower() == "vegetarian":
            # Might slightly increase carbs/fats if protein sources are limited
            split["protein"] = min(split["protein"] * 0.9, 0.30) # Slightly less protein
            split["carbs"] = min(split["carbs"] * 1.05, 0.50) # Slightly more carbs
            split["fat"] = 1 - split["protein"] - split["carbs"] # Adjust fat
        elif dietary_preference.lower() == "keto":
            split = {"protein": 0.25, "carbs": 0.05, "fat": 0.70} # High fat, very low carb
 
        # Calculate grams
        protein_grams = (tdee * split["protein"]) / 4  # 4 calories per gram of protein
        carbs_grams = (tdee * split["carbs"]) / 4     # 4 calories per gram of carbohydrates
        fat_grams = (tdee * split["fat"]) / 9         # 9 calories per gram of fat
 
        logger.info(f"Calculated macro split for goal '{goal}' and dietary preference '{dietary_preference}': "
                    f"Protein: {protein_grams:.2f}g, Carbs: {carbs_grams:.2f}g, Fat: {fat_grams:.2f}g.")
        return {
            "protein_grams": round(protein_grams, 2),
            "carbs_grams": round(carbs_grams, 2),
            "fat_grams": round(fat_grams, 2),
            "protein_percent": round(split["protein"], 2),
            "carbs_percent": round(split["carbs"], 2),
            "fat_percent": round(split["fat"], 2)
        }
 
    def engineer_features(self, user_data):
        """
        Main method to engineer all features for a given user.
        `user_data` is expected to be a dictionary or a pandas Series
        containing 'weight_kg', 'height_cm', 'age', 'gender', 'goal', 'activity_level', 'dietary_preference'.
        """
        weight_kg = user_data.get('weight_kg')
        height_cm = user_data.get('height_cm')
        age = user_data.get('age')
        gender = user_data.get('gender')
        goal = user_data.get('goal')
        activity_level = user_data.get('activity_level')
        dietary_preference = user_data.get('dietary_preference', 'none')
 
        if not all([weight_kg, height_cm, age, gender, goal, activity_level]):
            logger.error("Missing essential user data for feature engineering.")
            return None
 
        features = {}
 
        # BMI
        bmi = self.calculate_bmi(weight_kg, height_cm)
        if bmi is not None:
            features['bmi'] = bmi
 
        # BMR
        bmr = self.calculate_bmr(weight_kg, height_cm, age, gender)
        if bmr is not None:
            features['bmr'] = bmr
 
        # TDEE
        if bmr is not None:
            tdee = self.calculate_tdee(bmr, activity_level)
            if tdee is not None:
                features['tdee'] = tdee
 
                # Macro Split
                macro_split = self.calculate_macro_split(tdee, goal, dietary_preference)
                if macro_split is not None:
                    features.update(macro_split)
 
        logger.info(f"Engineered features for user: {features}")
        return features
 
# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Create a dummy config.yaml for testing if it doesn't exist
    if not os.path.exists("config.yaml"):
        dummy_config_content = """
# Global Configuration
project_name: "AI Workout & Diet Planner"
version: "1.0.0"
 
# Data Paths
data_paths:
  raw: "data/raw"
  external: "data/external"
  processed: "data/processed"
  features: "data/features"
  models: "data/models"
 
# Feature Engineering Configuration (example)
feature_engineering:
  macro_split_ratios:
    lose_weight:
      protein: 0.35
      carbs: 0.30
      fat: 0.35
    gain_muscle:
      protein: 0.30
      carbs: 0.40
      fat: 0.30
    maintain:
      protein: 0.25
      carbs: 0.45
      fat: 0.30
        """
        with open("config.yaml", "w") as f:
            f.write(dummy_config_content)
 
    engineer = FeatureEngineer()
 
    # Test BMI calculation
    bmi_val = engineer.calculate_bmi(weight_kg=70, height_cm=175)
    print(f"Test BMI: {bmi_val}")
 
    # Test BMR calculation
    bmr_val = engineer.calculate_bmr(weight_kg=70, height_cm=175, age=30, gender="male")
    print(f"Test BMR (male): {bmr_val}")
 
    bmr_val_female = engineer.calculate_bmr(weight_kg=60, height_cm=165, age=25, gender="female")
    print(f"Test BMR (female): {bmr_val_female}")
 
    # Test TDEE calculation
    if bmr_val:
        tdee_val = engineer.calculate_tdee(bmr_val, activity_level="moderate")
        print(f"Test TDEE: {tdee_val}")
 
        # Test Macro Split calculation
        if tdee_val:
            macro_split_val = engineer.calculate_macro_split(tdee_val, goal="lose_weight", dietary_preference="none")
            print(f"Test Macro Split (lose_weight, none): {macro_split_val}")
 
            macro_split_vegetarian = engineer.calculate_macro_split(tdee_val, goal="gain_muscle", dietary_preference="vegetarian")
            print(f"Test Macro Split (gain_muscle, vegetarian): {macro_split_vegetarian}")
 
            macro_split_keto = engineer.calculate_macro_split(tdee_val, goal="maintain", dietary_preference="keto")
            print(f"Test Macro Split (maintain, keto): {macro_split_keto}")
 
    # Test engineer_features method
    user_data_example = {
        "weight_kg": 75,
        "height_cm": 180,
        "age": 35,
        "gender": "male",
        "goal": "gain_muscle",
        "activity_level": "very_active",
        "dietary_preference": "none"
    }
    engineered_features = engineer.engineer_features(user_data_example)
    print(f"Engineered Features: {engineered_features}")
 
    # Clean up dummy config
    # os.remove("config.yaml")
