import pandas as pd
import yaml
import os
from loguru import logger
 
class NutritionDataLoader:
    """
    Handles loading of nutrition-related datasets.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.nutrition_data_file = self.config['data_collection']['nutrition_data_file']
        self.raw_data_path = self.config['data_paths']['raw']
        self.nutrition_data_filepath = os.path.join(self.raw_data_path, self.nutrition_data_file)
        logger.info(f"NutritionDataLoader initialized. Nutrition data will be loaded from: {self.nutrition_data_filepath}")
 
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
 
    def load_nutrition_database(self, directory_path=None):
        """
        Loads nutrition data from a specified directory containing multiple CSV files.
        If no directory is provided, it will look for a single file specified in config.
        """
        if directory_path:
            all_nutrition_data = []
            for filename in os.listdir(directory_path):
                if filename.endswith(".csv"):
                    filepath = os.path.join(directory_path, filename)
                    try:
                        df = pd.read_csv(filepath)
                        all_nutrition_data.append(df)
                        logger.info(f"Successfully loaded {filename} from {directory_path}")
                    except Exception as e:
                        logger.warning(f"Could not load {filename} from {directory_path}: {e}")
            if all_nutrition_data:
                # Attempt to concatenate, handling potential schema differences
                try:
                    combined_df = pd.concat(all_nutrition_data, ignore_index=True, sort=False)
                    logger.info(f"Successfully combined {len(all_nutrition_data)} nutrition CSVs.")
                    return combined_df
                except Exception as e:
                    logger.error(f"Error combining nutrition dataframes: {e}")
                    # Return the list of dataframes if concatenation fails
                    return all_nutrition_data
            else:
                logger.warning(f"No CSV files found or loaded from {directory_path}.")
                return pd.DataFrame()
        else:
            # Fallback to loading a single file if directory_path is not provided
            if os.path.exists(self.nutrition_data_filepath):
                try:
                    df = pd.read_csv(self.nutrition_data_filepath)
                    logger.info(f"Successfully loaded nutrition data from {self.nutrition_data_filepath}")
                    return df
                except Exception as e:
                    logger.error(f"Error loading nutrition data from {self.nutrition_data_filepath}: {e}")
                    return pd.DataFrame()
            else:
                logger.warning(f"Nutrition data file not found at {self.nutrition_data_filepath}. Returning empty DataFrame.")
                return pd.DataFrame()
 
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
 
    loader = NutritionDataLoader()
 
    # Example of loading from a specific directory (replace with your actual path)
    # For demonstration, let's assume you have a 'test_nutrition_data' folder with some CSVs
    # You would replace this with the actual path to your FoodData_Central_foundation_food_csv_2025-04-24 directory
    test_nutrition_dir = "d:/Internship project/backend/FoodData_Central_foundation_food_csv_2025-04-24"
    if os.path.exists(test_nutrition_dir):
        nutrition_df = loader.load_nutrition_database(directory_path=test_nutrition_dir)
        if isinstance(nutrition_df, pd.DataFrame):
            logger.info(f"Loaded combined nutrition data with {len(nutrition_df)} rows and {len(nutrition_df.columns)} columns.")
            logger.info(f"Nutrition data head:\n{nutrition_df.head()}")
        elif isinstance(nutrition_df, list):
            logger.info(f"Loaded {len(nutrition_df)} nutrition dataframes (could not combine).")
            for i, df in enumerate(nutrition_df):
                logger.info(f"DataFrame {i} head:\n{df.head()}")
    else:
        logger.warning(f"Test nutrition directory not found: {test_nutrition_dir}. Skipping directory load test.")
 
    # Example of loading a single nutrition file (if specified in config and exists)
    # For this to work, you'd need a 'nutrition_database.csv' in 'data/raw'
    single_nutrition_df = loader.load_nutrition_database()
    if not single_nutrition_df.empty:
        logger.info(f"Loaded single nutrition data with {len(single_nutrition_df)} rows.")
        logger.info(f"Single nutrition data head:\n{single_nutrition_df.head()}")
