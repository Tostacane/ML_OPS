import joblib
from pathlib import Path

from src.utils.config import MODELS_DIR


def save_model(model, filename: str = "model.pkl") -> Path:
    MODELS_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / filename
    joblib.dump(model, model_path)

    return model_path


def load_model(filename: str = "model.pkl"):
    model_path = MODELS_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)
