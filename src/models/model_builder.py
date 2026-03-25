from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, DropHighPSIFeatures
from feature_engine.encoding import OneHotEncoder

from src.utils.config import load_config


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "knn": KNeighborsClassifier
}

def _build_classifier(name: str, params: Dict[str, Any]):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {name}")
    return MODEL_REGISTRY[name](**params)

def build_model(model_config: Dict[str, Any] | None = None) -> Pipeline:
    if model_config is None:
        model_config = load_config("model")["model"]

    model_name = model_config["name"]
    model_params = model_config.get(model_name, {})

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