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


# da collegare a streamlit
@app.post("/predict")
def predict(data: list[float]):
    arr = np.array(data).reshape(1, -1)
    pred = model.predict(arr)

    result = int(pred[0])

    return {
        "prediction": result,
        "label": "Leave" if result == 1 else "Stay"
    }

# simple mean comparison drift
# TODO da cambiare
@app.post("/drift/check")
def drift_check(data: list[list[float]]):

    current = np.array(data)

    current_means = current.mean(axis=0).tolist()

    drift_detected = any(abs(x) > 1000 for x in current_means)

    return {
        "drift_detected": drift_detected,
        "feature_means": current_means
    }

@app.get("/model/info")
def model_info():
    return {
        # TODO da prendere il nome usato davvero
        "model_type": type(model).__name__,
        "model_path": str(MODELS_DIR / "model.pkl")
    }