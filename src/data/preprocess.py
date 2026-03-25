from typing import Dict, List, Any
import pandas as pd

from src.utils.config import load_config


_PREPROCESS_CONFIG = load_config("preprocess")["preprocess"]


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.drop(columns=columns, errors="ignore")


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def convert_binary_target(df: pd.DataFrame, column: str, positive_value: Any) -> pd.DataFrame:
    df = df.copy()
    df[column] = (df[column] == positive_value).astype(int)
    return df


def cast_columns(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    for col, dtype in schema.items():
        if col not in df.columns:
            continue

        if dtype == "int":
            df[col] = df[col].astype(int)
        elif dtype == "float":
            df[col] = df[col].astype(float)
        elif dtype == "bool":
            df[col] = df[col].astype(str).str.lower().map({"yes": True, "no": False}).astype(int)
        else:
            df[col] = df[col].astype(dtype)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = drop_columns(df, _PREPROCESS_CONFIG["drop_columns"])

    df = drop_missing(df)

    df = cast_columns(df, _PREPROCESS_CONFIG.get("cast_columns", {}))

    target_cfg = _PREPROCESS_CONFIG.get("target")
    if target_cfg:
        df = convert_binary_target(df, target_cfg["name"], target_cfg["positive_value"])

    return df
