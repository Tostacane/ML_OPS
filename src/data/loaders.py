import pandas as pd
from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_config, PROJECT_ROOT


_DATA_CONFIG = load_config("data")


def _get_base_dir(stage: str) -> Path:
    paths_cfg = _DATA_CONFIG["paths"]

    mapping = {
        "raw": Path(paths_cfg["raw_dir"]),
        #"processed": Path(paths_cfg.get("processed_dir", "")),
        #"interim": Path(paths_cfg.get("interim_dir", "")),
        #"external": Path(paths_cfg.get("external_dir", "")),
    }

    if stage not in mapping:
        raise ValueError(f"Unknown stage: {stage!r}. Expected one of: {list(mapping.keys())}")

    return mapping[stage]


def _get_dataset_config(name: str) -> Dict[str, Any]:
    try:
        return _DATA_CONFIG["datasets"][name]
    except KeyError:
        raise ValueError(
            f"Dataset {name!r} not found in data.yaml. "
            f"Available datasets: {list(_DATA_CONFIG['datasets'].keys())}"
        )


def load_dataset(name: str) -> pd.DataFrame:
    ds_cfg = _get_dataset_config(name)

    stage = ds_cfg.get("stage", "raw")
    fmt = ds_cfg.get("format", "csv")
    filename = ds_cfg["filename"]
    read_options = ds_cfg.get("read_options", {})

    base_dir = PROJECT_ROOT / _get_base_dir(stage)
    file_path = base_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if fmt == "csv":
        return pd.read_csv(file_path, **read_options)
    elif fmt == "parquet":
        return pd.read_parquet(file_path, **read_options)
    else:
        raise ValueError(f"Unsupported format: {fmt!r} for dataset {name!r}")
