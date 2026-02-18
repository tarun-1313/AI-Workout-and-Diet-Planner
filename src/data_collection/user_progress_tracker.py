import json
import os
from loguru import logger

class UserProgressTracker:
    def __init__(self, data_dir="data/progress"):
        script_dir = os.path.dirname(__file__)
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        self.progress_dir = os.path.join(base_dir, data_dir)
        os.makedirs(self.progress_dir, exist_ok=True)
        logger.info(f"User progress data will be stored in: {self.progress_dir}")

    def _get_user_progress_file(self, user_id):
        return os.path.join(self.progress_dir, f"user_{user_id}_progress.json")

    def save_progress(self, user_id, progress_data):
        file_path = self._get_user_progress_file(user_id)
        try:
            with open(file_path, 'w') as f:
                json.dump(progress_data, f, indent=4)
            logger.info(f"Progress saved for user {user_id} to {file_path}")
        except IOError as e:
            logger.error(f"Error saving progress for user {user_id}: {e}")

    def load_progress(self, user_id):
        file_path = self._get_user_progress_file(user_id)
        if not os.path.exists(file_path):
            logger.info(f"No progress found for user {user_id} at {file_path}. Returning empty data.")
            return {}
        try:
            with open(file_path, 'r') as f:
                progress_data = json.load(f)
            logger.info(f"Progress loaded for user {user_id} from {file_path}")
            return progress_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for user {user_id} progress: {e}")
            return {}
        except IOError as e:
            logger.error(f"Error loading progress for user {user_id}: {e}")
            return {}
