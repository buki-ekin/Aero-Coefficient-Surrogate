"""Dataset loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = ["camber", "camber_position", "thickness", "alpha_deg", "reynolds"]
TARGET_COLUMNS = ["cl", "cd", "cm"]

def save_dataset(dataset: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a dataset as CSV, creating parent folders as needed."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset and validate the required columns."""

    dataset = pd.read_csv(path)
    required = set(["naca", *FEATURE_COLUMNS, *TARGET_COLUMNS])
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return dataset
