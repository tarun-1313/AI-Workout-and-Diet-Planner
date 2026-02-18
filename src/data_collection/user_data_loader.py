import pandas as pd
import yaml
import os
from loguru import logger
 
class UserDataLoader:
    """
    Handles the collection and loading of user fitness and diet data.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.user_data_file = self.config['data_collection']['user_data_file']
        self.raw_data_path = self.config['data_paths']['raw']
        self.user_data_filepath = os.path.join(self.raw_data_path, self.user_data_file)
        logger.info(f"UserDataLoader initialized. User data will be loaded from: {self.user_data_filepath}")
 
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
 
    def get_user_input(self, user_profile=None):
        """
        Collects user input for fitness and diet planning.
        In a Streamlit application, this would interact with UI elements.
        For now, it can return a dummy profile or take a dictionary.
        """
        if user_profile:
            logger.info("Using provided user profile data.")
            return user_profile
        else:
            logger.warning("No user profile provided. Returning dummy data. "
                           "Implement Streamlit UI for actual user input.")
            # Dummy data for initial development
            return {
                "age": 30,
                "weight_kg": 70,
                "height_cm": 175,
                "gender": "male",
                "goal": "lose_weight", # lose_weight, gain_muscle, maintain
                "activity_level": "moderate", # sedentary, light, moderate, very_active, extra_active
                "dietary_preference": "vegetarian", # vegetarian, vegan, paleo, keto, none
                "health_conditions": "none", # diabetes, heart_disease, none
                "steps": 8000,
                "heart_rate": 75,
                "sleep_hours": 7,
                "calories_burned_activity": 500
            }
 
    def load_user_data(self):
        """
        Loads existing user data from a CSV file.
        If the file does not exist, it returns an empty DataFrame.
        """
        if os.path.exists(self.user_data_filepath):
            try:
                df = pd.read_csv(self.user_data_filepath)
                logger.info(f"Successfully loaded user data from {self.user_data_filepath}")
                return df
            except Exception as e:
                logger.error(f"Error loading user data from {self.user_data_filepath}: {e}")
                return pd.DataFrame()
        else:
            logger.warning(f"User data file not found at {self.user_data_filepath}. Returning empty DataFrame.")
            return pd.DataFrame()
 
    def save_user_data(self, user_data_df):
        """
        Saves user data to a CSV file.
        """
        os.makedirs(self.raw_data_path, exist_ok=True)
        try:
            user_data_df.to_csv(self.user_data_filepath, index=False)
            logger.info(f"User data successfully saved to {self.user_data_filepath}")
        except Exception as e:
            logger.error(f"Error saving user data to {self.user_data_filepath}: {e}")
 
# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Ensure the data/raw directory exists for testing
    os.makedirs("data/raw", exist_ok=True)
 
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
 
# Data Collection
data_collection:
  user_data_file: "user_data.csv"
  nutrition_data_file: "nutrition_database.csv"
  workout_data_file: "workout_database.csv"
  calorie_data_file: "calorie_dataset.csv"
  tracker_data_files:
    - "steps_tracker.csv"
    - "heart_rate_tracker.csv"
    - "sleep_tracker.csv"
    - "calories_burned_tracker.csv"
        """
        with open("config.yaml", "w") as f:
            f.write(dummy_config_content)
 
    loader = UserDataLoader()
 
    # Test collecting user input
    user_profile_data = loader.get_user_input()
    logger.info(f"Collected user profile: {user_profile_data}")
 
    # Convert to DataFrame and save
    user_df = pd.DataFrame([user_profile_data])
    loader.save_user_data(user_df)
 
    # Test loading user data
    loaded_df = loader.load_user_data()
    logger.info(f"Loaded user data:\n{loaded_df}")
 
    # Clean up dummy config and data file
    # os.remove("config.yaml")
    # os.remove(loader.user_data_filepath)
    # os.rmdir("data/raw")
    # os.rmdir("data")
