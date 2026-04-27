from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

from src.utils.config import MODELS_DIR

app = FastAPI(title="MLOps API")
model = joblib.load(MODELS_DIR / "model.pkl")


@app.get("/")
def root():
    return {"message": "API attiva"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: list[float]):
    arr = np.array(data).reshape(1, -1)
    pred = model.predict(arr)

    result = int(pred[0])

    return {
        "prediction": result,
        "label": "Leave" if result == 1 else "Stay"
    }


# Expected input format, already preprocessed with no categorical, example: 
"""
    [
        [36,1299,27,3,1,13,3,94,...],
        [42,800,10,2,4,20,2,65,...]
    ]
    
"""
@app.post("/drift/check")
def drift_check(data: list[list[float]]):


    current = np.array(data)

    baseline_means = np.array([
        37,     # age
        800,    # daily_rate
        9,      # distance_from_home
        3,      # education
        2,      # environment_satisfaction
        65,     # hourly_rate
        3,      # job_involvement
        2,      # job_level
        3,      # job_satisfaction
        6500,   # monthly_income
        14000,  # monthly_rate
        3,      # num_companies_worked
        15,     # percent_salary_hike
        3,      # performance_rating
        3,      # relationship_rating
        1,      # stock_option_level
        11,     # total_working_years
        3,      # training_times_last_year
        3,      # work_life_balance
        7,      # years_at_company
        4,      # years_in_current_role
        2,      # years_since_last_promotion
        4       # years_with_current_manager
    ])

    current_means = current.mean(axis=0)

    pct_diff = np.abs((current_means - baseline_means) / baseline_means)
    threshold = 0.25
    drift_flags = pct_diff > threshold

    feature_names = [
        "age",
        "daily_rate",
        "distance_from_home",
        "education",
        "environment_satisfaction",
        "hourly_rate",
        "job_involvement",
        "job_level",
        "job_satisfaction",
        "monthly_income",
        "monthly_rate",
        "num_companies_worked",
        "percent_salary_hike",
        "performance_rating",
        "relationship_rating",
        "stock_option_level",
        "total_working_years",
        "training_times_last_year",
        "work_life_balance",
        "years_at_company",
        "years_in_current_role",
        "years_since_last_promotion",
        "years_with_current_manager"
    ]

    drifted_features = [
        feature_names[i]
        for i in range(len(feature_names))
        if drift_flags[i]
    ]

    return {
        "drift_detected": bool(len(drifted_features) > 0),
        "drifted_features": drifted_features,
        "current_means": current_means.tolist(),
        "percent_difference": pct_diff.tolist()
    }
@app.get("/model/info")
def model_info():
    return {
        # TODO da prendere il nome usato davvero
        "model_type": type(model).__name__,
        "model_path": str(MODELS_DIR / "model.pkl")
    }