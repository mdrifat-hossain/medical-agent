"""
generate_sample_data.py

Generates small SYNTHETIC CSVs that mirror the column schemas of the three
real Kaggle datasets. This is only so you can test the full pipeline
end-to-end before/without downloading the real data.

⚠️ For your actual submission, replace these with the REAL Kaggle CSVs:
    - https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
    - https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset
    - https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Run:
    python generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(42)
N = 300

# ---------- Heart Disease (johnsmith88 schema) ----------
heart = pd.DataFrame({
    "age": rng.integers(29, 78, N),
    "sex": rng.integers(0, 2, N),
    "cp": rng.integers(0, 4, N),
    "trestbps": rng.integers(94, 200, N),
    "chol": rng.integers(126, 564, N),
    "fbs": rng.integers(0, 2, N),
    "restecg": rng.integers(0, 3, N),
    "thalach": rng.integers(71, 202, N),
    "exang": rng.integers(0, 2, N),
    "oldpeak": np.round(rng.uniform(0, 6.2, N), 1),
    "slope": rng.integers(0, 3, N),
    "ca": rng.integers(0, 5, N),
    "thal": rng.integers(0, 4, N),
    "target": rng.integers(0, 2, N),
})
heart.to_csv("data/heart.csv", index=False)

# ---------- Cancer Prediction (rabieelkharoua schema) ----------
cancer = pd.DataFrame({
    "Age": rng.integers(20, 90, N),
    "Gender": rng.integers(0, 2, N),
    "BMI": np.round(rng.uniform(15, 40, N), 1),
    "Smoking": rng.integers(0, 2, N),
    "GeneticRisk": rng.integers(0, 3, N),
    "PhysicalActivity": np.round(rng.uniform(0, 10, N), 1),
    "AlcoholIntake": np.round(rng.uniform(0, 5, N), 1),
    "CancerHistory": rng.integers(0, 2, N),
    "Diagnosis": rng.integers(0, 2, N),
})
cancer.to_csv("data/cancer.csv", index=False)

# ---------- Diabetes (iammustafatz/diabetes-prediction-dataset schema) ----------
diabetes = pd.DataFrame({
    "gender": rng.choice(["Male", "Female"], N),
    "age": np.round(rng.uniform(1, 80, N), 1),
    "hypertension": rng.integers(0, 2, N),
    "heart_disease": rng.integers(0, 2, N),
    "smoking_history": rng.choice(
        ["never", "No Info", "current", "former", "ever", "not current"], N
    ),
    "bmi": np.round(rng.uniform(15, 50, N), 2),
    "HbA1c_level": np.round(rng.uniform(3.5, 9.0, N), 1),
    "blood_glucose_level": rng.integers(70, 300, N),
    "diabetes": rng.integers(0, 2, N),
})
diabetes.to_csv("data/diabetes.csv", index=False)

print("✅ Sample CSVs written to ./data/: heart.csv, cancer.csv, diabetes.csv")
print("   Now run: python build_databases.py")
