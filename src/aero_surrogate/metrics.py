"""Evaluation utilities for surrogate-model verification and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aero_surrogate.data import FEATURE_COLUMNS, TARGET_COLUMNS


def grouped_train_test_split(
    dataset: pd.DataFrame,
    test_fraction: float = 0.2,
    random_seed: int = 42,
    group_column: str = "naca",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split complete airfoil groups into deterministic train and test sets."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    if group_column not in dataset:
        raise ValueError(f"Dataset is missing group column: {group_column}")

    groups = np.array(sorted(dataset[group_column].astype(str).unique()))
    if len(groups) < 2:
        raise ValueError("Grouped validation needs at least two airfoils.")

    rng = np.random.default_rng(random_seed)
    rng.shuffle(groups)
    test_size = min(len(groups) - 1, max(1, int(round(len(groups) * test_fraction))))
    test_groups = sorted(groups[:test_size].tolist())
    is_test = dataset[group_column].astype(str).isin(test_groups)
    train = dataset.loc[~is_test].reset_index(drop=True)
    test = dataset.loc[is_test].reset_index(drop=True)
    return train, test, test_groups


def evaluate_predictions(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    target_columns: tuple[str, ...] = tuple(TARGET_COLUMNS),
) -> dict[str, dict[str, float]]:
    """Calculate regression metrics for each target column."""

    metrics: dict[str, dict[str, float]] = {}
    for column in target_columns:
        observed = truth.loc[:, column].to_numpy(dtype=float)
        predicted = prediction.loc[:, column].to_numpy(dtype=float)
        residual = observed - predicted
        mse = float(np.mean(residual**2))
        mae = float(np.mean(np.abs(residual)))
        bias = float(np.mean(residual))
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator else float("nan")
        metrics[column] = {
            "bias": bias,
            "mae": mae,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": r2,
        }
    return metrics


def prediction_table(
    inputs: pd.DataFrame,
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Create a flat table with inputs, truth, predictions, and residuals."""

    columns = [column for column in ("naca", *FEATURE_COLUMNS) if column in inputs]
    table = inputs.loc[:, columns].reset_index(drop=True).copy()
    table["split"] = split_name
    for column in TARGET_COLUMNS:
        table[f"true_{column}"] = truth.loc[:, column].to_numpy(dtype=float)
        table[f"pred_{column}"] = prediction.loc[:, column].to_numpy(dtype=float)
        table[f"residual_{column}"] = table[f"true_{column}"] - table[f"pred_{column}"]
    return table
