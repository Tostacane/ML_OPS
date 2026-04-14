import pytest
import numpy as np
import pandas as pd

from src.data.preprocess import (
    drop_columns,
    drop_missing,
    convert_binary_target,
    cast_columns
)


@pytest.fixture
def df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "firstname": ["Davide", "Julian", "Vanni"],
            "lastname": ["Villa", "Cummaudo", "Galli"],
            "age": ["23", "22", None],
            "role": ["student", None, "teacher"],
            "task_done": ["Yes", "Yes", "No"],
        }
    )
    return df


def test_drop_columns(df : pd.DataFrame) -> None:
    new_df = drop_columns(df, ["age"])

    assert new_df.equals(df) == False
    assert len(new_df.columns) == 4


def test_drop_missing(df: pd.DataFrame) -> None:
    new_df = drop_missing(df)

    assert new_df.equals(df) == False
    assert len(new_df) == 1


def test_convert_binary_target(df: pd.DataFrame) -> None:
    new_df = convert_binary_target(df, "task_done", "Yes")

    assert new_df.equals(df) == False
    assert np.array_equal(
        new_df["task_done"].values,
        np.array([True, True, False])
    )


def test_cast_columns(df: pd.DataFrame) -> None:
    new_df = drop_missing(df)
    new_df = cast_columns(new_df, {"age": "int"})

    assert new_df.equals(df) == False
    assert new_df["age"].dtype == int
