from typing import Any, Dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, DropHighPSIFeatures
from feature_engine.encoding import OneHotEncoder

from src.utils.config import load_config


_MODEL_CONFIG = load_config("model")["model"]


def _build_classifier(name: str, params: Dict[str, Any]):
    if name == "logistic_regression":
        return LogisticRegression(**params)
    raise ValueError(f"Unsupported model type: {name}")


def build_model() -> Pipeline:
    model_name = _MODEL_CONFIG["name"]
    model_params = _MODEL_CONFIG.get(model_name, {})

    classifier = _build_classifier(model_name, model_params)

    pipeline = Pipeline([
        ("drop_constants", DropConstantFeatures(tol=0.999)),
        ("drop_duplicates", DropDuplicateFeatures()),
        ("drop_low_variation", DropHighPSIFeatures(threshold=0.99)),
        ("encode", OneHotEncoder(drop_last=True)),
        ("scale", RobustScaler()),
        ("model", classifier)
    ])

    return pipeline
