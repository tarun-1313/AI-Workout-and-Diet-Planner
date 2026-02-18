import yaml
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger
 
class Evaluator:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.metrics_to_calculate = self.config['evaluation']['metrics']
        self.fitness_metrics_to_calculate = self.config['evaluation']['fitness_metrics']
        # For regression accuracy, define a tolerance
        self.accuracy_tolerance = 0.1 # e.g., within 10% of the true value
        logger.info("Evaluator initialized.")
 
    def _load_config(self, config_path):
        """Loads configuration from a YAML file."""
        script_dir = os.path.dirname(__file__)
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        config_full_path = os.path.join(base_dir, config_path)
        try:
            with open(config_full_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_full_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
 
    def evaluate(self, y_true, y_pred, user_data=None):
        """
        Evaluates model predictions against true values using configured metrics.
        Args:
            y_true (np.array): True target values.
            y_pred (np.array): Predicted target values.
        Returns:
            dict: A dictionary of calculated metrics.
        """
        results = {}
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        logger.info("Starting model evaluation.")
 
        for metric in self.metrics_to_calculate:
            if metric == "mae":
                results["mae"] = self._calculate_mae(y_true, y_pred)
            elif metric == "rmse":
                results["rmse"] = self._calculate_rmse(y_true, y_pred)
            elif metric == "r2":
                results["r2"] = self._calculate_r2_score(y_true, y_pred)
            else:
                logger.warning(f"Unknown metric '{metric}' specified in config.yaml")
        
        for fitness_metric in self.fitness_metrics_to_calculate:
            if user_data is None:
                logger.warning(f"User data not provided for fitness metric calculation. Skipping {fitness_metric}.")
                continue

            if fitness_metric == "goal_adherence_score":
                actual_progress = user_data.get('actual_progress')
                target_goal = user_data.get('target_goal')
                if actual_progress is not None and target_goal is not None:
                    results["goal_adherence_score"] = self._calculate_goal_adherence_score(actual_progress, target_goal)
                else:
                    logger.warning("Missing 'actual_progress' or 'target_goal' in user_data for goal_adherence_score.")
                    results["goal_adherence_score"] = 0.0
            elif fitness_metric == "calorie_balance_score":
                actual_calorie_intake = user_data.get('actual_calorie_intake')
                target_calorie_intake = user_data.get('target_calorie_intake')
                if actual_calorie_intake is not None and target_calorie_intake is not None:
                    results["calorie_balance_score"] = self._calculate_calorie_balance_score(actual_calorie_intake, target_calorie_intake)
                else:
                    logger.warning("Missing 'actual_calorie_intake' or 'target_calorie_intake' in user_data for calorie_balance_score.")
                    results["calorie_balance_score"] = 0.0
            elif fitness_metric == "risk_aware_recommendation_score":
                recommendation_impact = user_data.get('recommendation_impact')
                user_health_risk_factors = user_data.get('user_health_risk_factors')
                if recommendation_impact is not None and user_health_risk_factors is not None:
                    results["risk_aware_recommendation_score"] = self._calculate_risk_aware_recommendation_score(recommendation_impact, user_health_risk_factors)
                else:
                    logger.warning("Missing 'recommendation_impact' or 'user_health_risk_factors' in user_data for risk_aware_recommendation_score.")
                    results["risk_aware_recommendation_score"] = 0.0
            else:
                logger.warning(f"Unknown fitness metric '{fitness_metric}' specified in config.yaml")
        logger.info("Model evaluation finished.")
        return results
 
    def _calculate_mae(self, y_true, y_pred):
        """Calculates Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
 
    def _calculate_rmse(self, y_true, y_pred):
        """Calculates Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
 
    def _calculate_r2_score(self, y_true, y_pred):
        """Calculates R-squared (R²) score."""
        return r2_score(y_true, y_pred)
 
    def _calculate_goal_adherence_score(self, actual_progress, target_goal):
        """
        Calculates a score indicating how well a user is adhering to their goals.
        This is a placeholder and will need actual implementation based on goal tracking.
        """
        # Example: simple percentage of goal achieved
        if target_goal == 0:
            return 0.0
        return max(0.0, min(1.0, actual_progress / target_goal))
 
    def _calculate_calorie_balance_score(self, actual_calorie_intake, target_calorie_intake):
        """
        Calculates a score for calorie balance, indicating if intake is within a healthy range
        relative to the target. This is a placeholder.
        """
        # Example: score based on deviation from target
        deviation = abs(actual_calorie_intake - target_calorie_intake)
        # A simple inverse relationship with deviation, capped at 1.0
        return max(0.0, 1.0 - (deviation / target_calorie_intake)) if target_calorie_intake > 0 else 0.0
 
    def _calculate_risk_aware_recommendation_score(self, recommendation_impact, user_health_risk_factors):
        """
        Calculates a score that considers the potential impact of a recommendation
        against a user's health risk factors.
        
        Args:
            recommendation_impact (float): A numerical value representing the positive/negative impact
                                           of a recommendation (e.g., -1 to 1).
            user_health_risk_factors (list): A list of strings indicating health conditions or risks.
        
        Returns:
            float: A score between 0 and 1, where higher is better (lower risk, higher positive impact).
        """
        base_score = 0.5 # Start with a neutral score
        
        # Adjust score based on recommendation impact
        # Assuming recommendation_impact is between -1 and 1
        score = base_score + (recommendation_impact * 0.5) # Scale impact to affect score by up to 0.5
        
        # Apply penalties for each health risk factor
        if user_health_risk_factors:
            for risk_factor in user_health_risk_factors:
                # Example: simple penalty for each risk factor
                # More complex logic could involve different penalties for different risks
                score -= 0.1 # Subtract 0.1 for each identified risk factor
                logger.info(f"Applied penalty for risk factor: {risk_factor}. Current score: {score:.2f}")
        
        # Ensure the score is within a valid range [0, 1]
        final_score = max(0.0, min(1.0, score))
        logger.info(f"Calculated risk-aware recommendation score: {final_score:.2f} "
                    f"(Impact: {recommendation_impact}, Risks: {user_health_risk_factors})")
        return final_score

