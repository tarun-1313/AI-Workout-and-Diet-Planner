import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Ensure the data/models directory exists
os.makedirs('data/models', exist_ok=True)

# 1. Generate Synthetic Data for Calorie Prediction
# Inputs: Age, Gender (0:Female, 1:Male), Height (cm), Weight (kg), Activity Level (0-3)
# Output: Calories
np.random.seed(42)
n_samples = 1000

ages = np.random.randint(18, 60, n_samples)
genders = np.random.randint(0, 2, n_samples)
heights = np.random.normal(170, 10, n_samples)
weights = np.random.normal(70, 15, n_samples)
activity_levels = np.random.randint(0, 4, n_samples) # 0: Sedentary, 1: Light, 2: Moderate, 3: Active

# Harris-Benedict Approximation for Ground Truth + Noise
bmr = 10 * weights + 6.25 * heights - 5 * ages + np.where(genders==1, 5, -161)
multipliers = [1.2, 1.375, 1.55, 1.725]
activity_factors = np.array([multipliers[i] for i in activity_levels])
calories = bmr * activity_factors + np.random.normal(0, 50, n_samples)

data = pd.DataFrame({
    'age': ages,
    'gender': genders,
    'height': heights,
    'weight': weights,
    'activity_level': activity_levels,
    'calories': calories
})

# 2. Train Model
X = data[['age', 'gender', 'height', 'weight', 'activity_level']]
y = data['calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model Score: {model.score(X_test, y_test)}")

# 3. Save Model
joblib.dump(model, 'data/models/calorie_model.pkl')
print("Model saved to data/models/calorie_model.pkl")

# 4. Create dummy workout dataset for filtering logic (since we use CSV/DB for content filtering)
# This part is just to ensure we have data to filter
workouts = [
    {"name": "Home HIIT", "difficulty": "Intermediate", "equipment": "None", "location": "Home", "type": "Cardio"},
    {"name": "Gym Strength", "difficulty": "Advanced", "equipment": "Weights", "location": "Gym", "type": "Strength"},
    {"name": "Yoga Flow", "difficulty": "Beginner", "equipment": "Mat", "location": "Home", "type": "Flexibility"},
    {"name": "Park Run", "difficulty": "Beginner", "equipment": "None", "location": "Outdoor", "type": "Cardio"},
    {"name": "Dumbbell Press", "difficulty": "Intermediate", "equipment": "Dumbbells", "location": "Home", "type": "Strength"}
]
pd.DataFrame(workouts).to_csv('workouts.csv', index=False)
print("Created workouts.csv")
