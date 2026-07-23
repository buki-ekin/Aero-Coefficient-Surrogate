"""Scientific figures generated from recorded validation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aero_surrogate.data import TARGET_COLUMNS, load_dataset
from aero_surrogate.reproducibility import write_json

TARGET_LABELS = {"cl": r"$C_L$", "cd": r"$C_D$", "cm": r"$C_M$"}
TARGET_COLORS = {"cl": "#176B55", "cd": "#315F9E", "cm": "#A34F43"}
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "linear_regression": "Linear regression",
    "mean_baseline": "Mean baseline",
}
MODEL_COLORS = {
    "random_forest": "#176B55",
    "linear_regression": "#D99A2B",
    "mean_baseline": "#8A8F98",
}


def generate_scientific_figures(
    *,
    dataset_path: str | Path,
    predictions_path: str | Path,
    cross_validation_path: str | Path,
    output_dir: str | Path,
    expected_alpha_points: int = 21,
) -> dict[str, Any]:
    """Generate reproducible PNG figures for reports and presentations."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Scientific figures require matplotlib. Install project dependencies."
        ) from exc

    dataset = load_dataset(dataset_path)
    predictions = pd.read_csv(predictions_path)
    cv_metrics = pd.read_csv(cross_validation_path)
    test = predictions.loc[predictions["split"] == "test"].copy()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    figures: dict[str, str] = {}
    figures["measured_vs_predicted"] = _measured_vs_predicted(
        plt, test, destination / "measured_vs_predicted.png"
    )
    figures["residuals_vs_alpha"] = _residuals_vs_alpha(
        plt, test, destination / "residuals_vs_alpha.png"
    )
    figures["rmse_by_airfoil"] = _rmse_by_airfoil(
        plt, test, destination / "rmse_by_airfoil.png"
    )
    figures["model_comparison"] = _model_comparison(
        plt, cv_metrics, destination / "grouped_cv_model_comparison.png"
    )
    figures["data_coverage"] = _data_coverage(
        plt,
        dataset,
        destination / "data_coverage.png",
        expected_alpha_points=expected_alpha_points,
    )

    manifest = {
        "figures": figures,
        "source_predictions": Path(predictions_path).as_posix(),
        "source_cross_validation": Path(cross_validation_path).as_posix(),
        "source_dataset": Path(dataset_path).as_posix(),
        "test_rows": len(test),
        "expected_grid_points": (
            dataset["naca"].nunique()
            * dataset["reynolds"].nunique()
            * expected_alpha_points
        ),
        "converged_grid_points": len(dataset),
        "missing_grid_points": (
            dataset["naca"].nunique()
            * dataset["reynolds"].nunique()
            * expected_alpha_points
            - len(dataset)
        ),
    }
    write_json(destination / "figure_manifest.json", manifest)
    return manifest


def _style_axes(axes: Any) -> None:
    for axis in np.atleast_1d(axes):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, alpha=0.2, linewidth=0.7)


def _measured_vs_predicted(plt: Any, test: pd.DataFrame, path: Path) -> str:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for axis, target in zip(axes, TARGET_COLUMNS):
        observed = test[f"true_{target}"]
        predicted = test[f"pred_{target}"]
        limits = [
            float(min(observed.min(), predicted.min())),
            float(max(observed.max(), predicted.max())),
        ]
        axis.scatter(
            observed,
            predicted,
            s=18,
            alpha=0.68,
            color=TARGET_COLORS[target],
            edgecolors="none",
        )
        axis.plot(limits, limits, color="#20262B", linestyle="--", linewidth=1.3)
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("flow5 reference")
        axis.set_ylabel("surrogate prediction")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
    _style_axes(axes)
    figure.suptitle("Held-out airfoils: prediction follows the flow5 reference")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path.as_posix()


def _residuals_vs_alpha(plt: Any, test: pd.DataFrame, path: Path) -> str:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for axis, target in zip(axes, TARGET_COLUMNS):
        axis.axhline(0.0, color="#20262B", linewidth=1.0)
        for reynolds, group in test.groupby("reynolds"):
            axis.scatter(
                group["alpha_deg"],
                group[f"residual_{target}"],
                s=19,
                alpha=0.62,
                label=f"Re {reynolds / 1e6:g}M",
            )
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel(r"angle of attack $\alpha$ [deg]")
        axis.set_ylabel("flow5 - prediction")
    _style_axes(axes)
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Residuals expose operating conditions hidden by aggregate metrics")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path.as_posix()


def _rmse_by_airfoil(plt: Any, test: pd.DataFrame, path: Path) -> str:
    grouped = test.groupby("naca")
    rmse = pd.DataFrame(
        {
            target: grouped[f"residual_{target}"].apply(
                lambda values: float(np.sqrt(np.mean(values.to_numpy() ** 2)))
            )
            for target in TARGET_COLUMNS
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, target in zip(axes, TARGET_COLUMNS):
        values = rmse[target].sort_values()
        axis.barh(
            values.index.str.replace("NACA ", "", regex=False),
            values,
            color=TARGET_COLORS[target],
        )
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("RMSE")
    _style_axes(axes)
    figure.suptitle("Error differs by held-out geometry")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path.as_posix()


def _model_comparison(plt: Any, metrics: pd.DataFrame, path: Path) -> str:
    summary = (
        metrics.groupby(["model", "target"])["rmse"]
        .agg(["mean", "std"])
        .reset_index()
    )
    order = ["mean_baseline", "linear_regression", "random_forest"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, target in zip(axes, TARGET_COLUMNS):
        target_rows = summary.loc[summary["target"] == target].set_index("model")
        means = [target_rows.loc[model, "mean"] for model in order]
        errors = [target_rows.loc[model, "std"] for model in order]
        axis.bar(
            [MODEL_LABELS[model] for model in order],
            means,
            yerr=errors,
            capsize=4,
            color=[MODEL_COLORS[model] for model in order],
        )
        axis.set_title(TARGET_LABELS[target])
        axis.set_ylabel("grouped CV RMSE")
        axis.tick_params(axis="x", rotation=24)
    _style_axes(axes)
    figure.suptitle("Repeated grouped validation supports the Random Forest choice")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path.as_posix()


def _data_coverage(
    plt: Any,
    dataset: pd.DataFrame,
    path: Path,
    *,
    expected_alpha_points: int,
) -> str:
    coverage = dataset.pivot_table(
        index="naca",
        columns="reynolds",
        values="alpha_deg",
        aggfunc="count",
        fill_value=0,
    ).sort_index()
    figure, axis = plt.subplots(figsize=(7.5, 8.2), constrained_layout=True)
    image = axis.imshow(
        coverage.to_numpy(),
        aspect="auto",
        cmap="YlGn",
        vmin=0,
        vmax=expected_alpha_points,
    )
    axis.set_yticks(range(len(coverage.index)))
    axis.set_yticklabels(
        coverage.index.str.replace("NACA ", "", regex=False), fontsize=8
    )
    axis.set_xticks(range(len(coverage.columns)))
    axis.set_xticklabels([f"{value / 1e6:g}M" for value in coverage.columns])
    axis.set_xlabel("Reynolds number")
    axis.set_ylabel("NACA airfoil")
    axis.set_title(
        f"Converged flow5 points per condition (requested: {expected_alpha_points})"
    )
    for row in range(coverage.shape[0]):
        for column in range(coverage.shape[1]):
            value = int(coverage.iloc[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > expected_alpha_points * 0.72 else "#16201B",
            )
    figure.colorbar(image, ax=axis, label="converged angle-of-attack points")
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path.as_posix()
