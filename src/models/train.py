from typing import Tuple
import pandas as pd

from sklearn.model_selection import train_test_split

from data.loaders import load_dataset
from data.preprocess import preprocess
from models.model_builder import build_model
from utils.config import load_config


def train(
    dataset_name: str,
    test_size: float = 0.2,
    random_state: int = 101,
    model_config: dict | None = None
) -> Tuple[object, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    
    preprocess_config = load_config("preprocess")["preprocess"]
    target_column = preprocess_config["target"]["name"]

    df = load_dataset(dataset_name)
    df = preprocess(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Config dinamica
    model = build_model(model_config)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_pred, y_proba