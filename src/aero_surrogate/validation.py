"""Repeated grouped validation and baseline-model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from aero_surrogate.data import FEATURE_COLUMNS, TARGET_COLUMNS, validate_dataset
from aero_surrogate.metrics import evaluate_predictions
from aero_surrogate.sklearn_surrogate import RandomForestSurrogate


@dataclass(frozen=True)
class GroupedFold:
    """One train/test partition containing complete NACA groups."""

    repeat: int
    fold: int
    held_out_airfoils: tuple[str, ...]
    train: pd.DataFrame
    test: pd.DataFrame


def repeated_grouped_folds(
    dataset: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_seed: int = 42,
    group_column: str = "naca",
) -> Iterator[GroupedFold]:
    """Yield deterministic repeated folds that never split one airfoil group."""

    validated = validate_dataset(dataset)
    groups = np.array(sorted(validated[group_column].astype(str).unique()))
    if not 2 <= n_splits <= len(groups):
        raise ValueError(
            f"n_splits must be between 2 and the number of groups ({len(groups)})."
        )
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    for repeat in range(n_repeats):
        rng = np.random.default_rng(random_seed + repeat)
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        for fold, held_out in enumerate(np.array_split(shuffled, n_splits), start=1):
            held_out_airfoils = tuple(sorted(held_out.tolist()))
            is_test = validated[group_column].isin(held_out_airfoils)
            yield GroupedFold(
                repeat=repeat + 1,
                fold=fold,
                held_out_airfoils=held_out_airfoils,
                train=validated.loc[~is_test].reset_index(drop=True),
                test=validated.loc[is_test].reset_index(drop=True),
            )


def evaluate_model_systematics(
    dataset: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_seed: int = 42,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare Random Forest, linear, and mean baselines on repeated folds."""

    rows: list[dict[str, object]] = []
    for grouped_fold in repeated_grouped_folds(
        dataset,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_seed=random_seed,
    ):
        train = grouped_fold.train
        test = grouped_fold.test
        x_train = train.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train.loc[:, TARGET_COLUMNS].to_numpy(dtype=float)
        x_test = test.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)

        models = {
            "random_forest": RandomForestSurrogate(
                n_estimators=n_estimators,
                random_state=random_seed,
                min_samples_leaf=min_samples_leaf,
            ).fit(train),
            "linear_regression": LinearRegression().fit(x_train, y_train),
            "mean_baseline": DummyRegressor(strategy="mean").fit(x_train, y_train),
        }

        for model_name, model in models.items():
            if isinstance(model, RandomForestSurrogate):
                prediction = model.predict(
                    test,
                    warn_outside_domain=False,
                )
            else:
                prediction = pd.DataFrame(
                    model.predict(x_test),
                    columns=TARGET_COLUMNS,
                    index=test.index,
                )
            metrics = evaluate_predictions(test, prediction)
            for target, values in metrics.items():
                rows.append(
                    {
                        "repeat": grouped_fold.repeat,
                        "fold": grouped_fold.fold,
                        "model": model_name,
                        "target": target,
                        "held_out_airfoils": ",".join(
                            grouped_fold.held_out_airfoils
                        ),
                        "n_train": len(train),
                        "n_test": len(test),
                        **values,
                    }
                )

    fold_metrics = pd.DataFrame(rows)
    summary = _summarize_fold_metrics(
        fold_metrics,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_seed=random_seed,
    )
    return fold_metrics, summary


def _summarize_fold_metrics(
    fold_metrics: pd.DataFrame,
    *,
    n_splits: int,
    n_repeats: int,
    random_seed: int,
) -> dict[str, object]:
    results: dict[str, dict[str, dict[str, float]]] = {}
    for (model, target), group in fold_metrics.groupby(["model", "target"]):
        model_results = results.setdefault(str(model), {})
        target_results: dict[str, float] = {}
        for metric in ("rmse", "mae", "bias", "r2"):
            values = group[metric].to_numpy(dtype=float)
            standard_error = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            target_results.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_std": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                    f"{metric}_ci95": 1.96 * standard_error,
                    f"{metric}_min": float(values.min()),
                    f"{metric}_max": float(values.max()),
                }
            )
        model_results[str(target)] = target_results

    return {
        "strategy": "repeated_grouped_kfold_by_naca",
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_evaluations_per_model": n_splits * n_repeats,
        "random_seed": random_seed,
        "models": results,
        "interpretation": (
            "Intervals describe variation across held-out airfoil folds. "
            "They quantify split sensitivity against flow5, not experimental "
            "or epistemic uncertainty."
        ),
    }
