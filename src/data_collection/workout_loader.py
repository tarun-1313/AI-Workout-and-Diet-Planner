import pandas as pd
import yaml
import os
from loguru import logger
 
class WorkoutDataLoader:
    """
    Handles loading of workout-related datasets.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.workout_data_file = self.config['data_collection']['workout_data_file']
        self.raw_data_path = self.config['data_paths']['raw']
        self.workout_data_filepath = os.path.join(self.raw_data_path, self.workout_data_file)
        logger.info(f"WorkoutDataLoader initialized. Workout data will be loaded from: {self.workout_data_filepath}")
 
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
 
    def load_workout_data(self, directory_path=None):
        """
        Loads workout data from a specified directory containing multiple CSV files.
        If no directory is provided, it will look for a single file specified in config.
        """
        if directory_path:
            all_workout_data = []
            for filename in os.listdir(directory_path):
                if filename.endswith(".csv"):
                    filepath = os.path.join(directory_path, filename)
                    try:
                        df = pd.read_csv(filepath)
                        all_workout_data.append(df)
                        logger.info(f"Successfully loaded {filename} from {directory_path}")
                    except Exception as e:
                        logger.warning(f"Could not load {filename} from {directory_path}: {e}")
            if all_workout_data:
                # Attempt to concatenate, handling potential schema differences
                try:
                    combined_df = pd.concat(all_workout_data, ignore_index=True, sort=False)
                    logger.info(f"Successfully combined {len(all_workout_data)} workout CSVs.")
                    return combined_df
                except Exception as e:
                    logger.error(f"Error combining workout dataframes: {e}")
                    # Return the list of dataframes if concatenation fails
                    return all_workout_data
            else:
                logger.warning(f"No CSV files found or loaded from {directory_path}.")
                return pd.DataFrame()
        else:
            # Fallback to loading a single file if directory_path is not provided
            if os.path.exists(self.workout_data_filepath):
                try:
                    df = pd.read_csv(self.workout_data_filepath)
                    logger.info(f"Successfully loaded workout data from {self.workout_data_filepath}")
                    return df
                except Exception as e:
                    logger.error(f"Error loading workout data from {self.workout_data_filepath}: {e}")
                    return pd.DataFrame()
            else:
                logger.warning(f"Workout data file not found at {self.workout_data_filepath}. Returning empty DataFrame.")
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
 
    loader = WorkoutDataLoader()
 
    # Create a dummy workout CSV for testing single file load
    dummy_workout_data = pd.DataFrame({
        'exercise': ['Push-ups', 'Squats', 'Plank'],
        'muscle_group': ['Chest', 'Legs', 'Core'],
        'equipment': ['None', 'None', 'None'],
        'difficulty': ['Beginner', 'Beginner', 'Beginner']
    })
    dummy_workout_data.to_csv(loader.workout_data_filepath, index=False)
    logger.info(f"Created dummy workout data at {loader.workout_data_filepath}")
 
    # Test loading a single workout file
    single_workout_df = loader.load_workout_data()
    if not single_workout_df.empty:
        logger.info(f"Loaded single workout data with {len(single_workout_df)} rows.")
        logger.info(f"Single workout data head:\n{single_workout_df.head()}")
 
    # Example of loading from a specific directory (replace with your actual path)
    # For demonstration, let's assume you have a 'test_workout_data' folder with some CSVs
    # You would replace this with the actual path to your workout data directory
    test_workout_dir = "d:/Internship project/backend/workout_data" # Placeholder for actual directory
    if os.path.exists(test_workout_dir):
        workout_df = loader.load_workout_data(directory_path=test_workout_dir)
        if isinstance(workout_df, pd.DataFrame):
            logger.info(f"Loaded combined workout data with {len(workout_df)} rows and {len(workout_df.columns)} columns.")
            logger.info(f"Workout data head:\n{workout_df.head()}")
        elif isinstance(workout_df, list):
            logger.info(f"Loaded {len(workout_df)} workout dataframes (could not combine).")
            for i, df in enumerate(workout_df):
                logger.info(f"DataFrame {i} head:\n{df.head()}")
    else:
        logger.warning(f"Test workout directory not found: {test_workout_dir}. Skipping directory load test.")
 
    # Clean up dummy config and data file
    # os.remove("config.yaml")
    # os.remove(loader.workout_data_filepath)
    # os.rmdir("data/raw")
    # os.rmdir("data")
