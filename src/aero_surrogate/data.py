"""Dataset loading, validation, and training-domain metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["camber", "camber_position", "thickness", "alpha_deg", "reynolds"]
TARGET_COLUMNS = ["cl", "cd", "cm"]
OPERATING_POINT_COLUMNS = ["naca", "reynolds", "alpha_deg"]

# Broad physical plausibility checks catch corrupt imports without pretending to
# define the much narrower domain in which the fitted surrogate is valid.
PLAUSIBLE_RANGES = {
    "camber": (0.0, 0.09),
    "camber_position": (0.0, 0.9),
    "thickness": (0.01, 0.40),
    "alpha_deg": (-90.0, 90.0),
    "reynolds": (1.0, float("inf")),
    "cl": (-3.0, 3.0),
    "cd": (0.0, 1.0),
    "cm": (-1.0, 1.0),
}


@dataclass(frozen=True)
class TrainingDomain:
    """Observed minimum and maximum for every model input."""

    feature_ranges: dict[str, tuple[float, float]]

    @classmethod
    def from_dataset(cls, dataset: pd.DataFrame) -> "TrainingDomain":
        """Build a domain description from a validated dataset."""

        return cls(
            {
                feature: (
                    float(dataset[feature].min()),
                    float(dataset[feature].max()),
                )
                for feature in FEATURE_COLUMNS
            }
        )

    def violations(self, values: dict[str, float]) -> list[str]:
        """Return human-readable messages for inputs outside observed ranges."""

        messages = []
        for feature, (lower, upper) in self.feature_ranges.items():
            if feature not in values:
                continue
            value = float(values[feature])
            if not lower <= value <= upper:
                messages.append(
                    f"{feature}={value:g} is outside the training range "
                    f"[{lower:g}, {upper:g}]"
                )
        return messages


def save_dataset(dataset: pd.DataFrame, output_path: str | Path) -> Path:
    """Validate and save a dataset as CSV."""

    validated = validate_dataset(dataset)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(path, index=False)
    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and apply scientific integrity checks."""

    dataset = pd.read_csv(path)
    return validate_dataset(dataset)


def validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of a model-ready flow5 dataset.

    Validation rejects missing or non-finite numerical values, duplicate
    operating points, empty NACA identifiers, and values outside broad physical
    plausibility limits. These checks protect the workflow from corrupt inputs;
    they are not a statement of surrogate-model validity.
    """

    required = {"naca", *FEATURE_COLUMNS, *TARGET_COLUMNS}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    validated = dataset.copy()
    if validated.empty:
        raise ValueError("Dataset must contain at least one operating point.")

    naca = validated["naca"].astype("string")
    if naca.isna().any() or naca.str.strip().eq("").any():
        raise ValueError("Column 'naca' contains missing or empty identifiers.")
    validated["naca"] = naca.str.strip()

    numeric_columns = [*FEATURE_COLUMNS, *TARGET_COLUMNS]
    for column in numeric_columns:
        converted = pd.to_numeric(validated[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted.to_numpy(dtype=float))
        if invalid.any():
            rows = validated.index[invalid].tolist()[:5]
            raise ValueError(
                f"Column '{column}' contains missing or non-finite values "
                f"at rows {rows}."
            )
        validated[column] = converted.astype(float)

    duplicates = validated.duplicated(OPERATING_POINT_COLUMNS, keep=False)
    if duplicates.any():
        sample = validated.loc[duplicates, OPERATING_POINT_COLUMNS].head(3)
        raise ValueError(
            "Dataset contains duplicate operating points, for example: "
            f"{sample.to_dict(orient='records')}"
        )

    for column, (lower, upper) in PLAUSIBLE_RANGES.items():
        outside = ~validated[column].between(lower, upper, inclusive="both")
        if outside.any():
            sample = validated.loc[outside, column].head(5).tolist()
            raise ValueError(
                f"Column '{column}' contains values outside the plausible range "
                f"[{lower:g}, {upper:g}]: {sample}"
            )

    return validated.reset_index(drop=True)
