"""Training, validation, and deployment workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aero_surrogate.data import FEATURE_COLUMNS, TARGET_COLUMNS, load_dataset
from aero_surrogate.metrics import (
    evaluate_predictions,
    grouped_train_test_split,
    prediction_table,
)
from aero_surrogate.reproducibility import (
    environment_metadata,
    input_manifest,
    sha256,
    write_json,
)
from aero_surrogate.sklearn_surrogate import RandomForestSurrogate
from aero_surrogate.surrogate import save_model


@dataclass(frozen=True)
class TrainingRunConfig:
    dataset_path: str
    raw_data_dir: str | None = None
    test_fraction: float = 0.2
    random_seed: int = 42
    n_estimators: int = 300
    min_samples_leaf: int = 2
    run_root: str = "runs"
    run_id: str | None = None
    deployment_model_path: str | None = None


def make_run_id(random_seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_aerosurrogate_random-forest_seed{random_seed}"


def run_training_workflow(config: TrainingRunConfig) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    run_id = config.run_id or make_run_id(config.random_seed)
    run_dir = Path(config.run_root) / run_id
    outputs_dir = run_dir / "outputs"
    reports_dir = run_dir / "reports"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_path)
    train, test, held_out_airfoils = grouped_train_test_split(
        dataset,
        test_fraction=config.test_fraction,
        random_seed=config.random_seed,
    )

    validation_model = _new_model(config).fit(train)
    train_prediction = validation_model.predict(train)
    test_prediction = validation_model.predict(test)
    train_metrics = evaluate_predictions(train, train_prediction)
    test_metrics = evaluate_predictions(test, test_prediction)

    metrics_payload = {
        "split_strategy": "grouped_by_naca",
        "held_out_airfoils": held_out_airfoils,
        "n_train": len(train),
        "n_test": len(test),
        "train": train_metrics,
        "test": test_metrics,
    }

    config_payload = asdict(config) | {
        "run_id": run_id,
        "split_strategy": "grouped_by_naca",
    }
    config_path = write_json(run_dir / "config.json", config_payload)
    meta_path = write_json(run_dir / "meta.json", environment_metadata(Path.cwd()))
    manifest_path = write_json(
        run_dir / "manifest.json",
        input_manifest(_input_files(dataset_path, config.raw_data_dir)),
    )
    metrics_path = write_json(reports_dir / "metrics.json", metrics_payload)

    validation_model_path = save_model(
        validation_model,
        outputs_dir / "validation_model.pkl",
    )
    deployment_model = _new_model(config).fit(dataset)
    deployment_model_path = save_model(
        deployment_model,
        outputs_dir / "deployment_model.pkl",
    )
    if config.deployment_model_path:
        save_model(deployment_model, config.deployment_model_path)

    predictions = pd.concat(
        [
            prediction_table(train, train, train_prediction, "train"),
            prediction_table(test, test, test_prediction, "test"),
        ],
        ignore_index=True,
    )
    predictions_path = reports_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    deployment_manifest_path = write_json(
        outputs_dir / "deployment_manifest.json",
        {
            "dataset_path": dataset_path.as_posix(),
            "dataset_sha256": sha256(dataset_path),
            "model_path": deployment_model_path.as_posix(),
            "model_sha256": sha256(deployment_model_path),
            "training_rows": len(dataset),
            "features": FEATURE_COLUMNS,
            "targets": TARGET_COLUMNS,
            "random_seed": config.random_seed,
            "n_estimators": config.n_estimators,
            "min_samples_leaf": config.min_samples_leaf,
        },
    )

    summary_path = write_json(
        reports_dir / "summary.json",
        {
            "run_id": run_id,
            "data_source": "flow5 polar exports",
            "model": "RandomForestSurrogate",
            "split_strategy": "grouped_by_naca",
            "held_out_airfoils": held_out_airfoils,
            "features": FEATURE_COLUMNS,
            "targets": TARGET_COLUMNS,
            "test_rmse": {
                target: values["rmse"] for target, values in test_metrics.items()
            },
            "deployment_training_rows": len(dataset),
            "main_outputs": {
                "validation_model": validation_model_path.as_posix(),
                "deployment_model": deployment_model_path.as_posix(),
                "deployment_manifest": deployment_manifest_path.as_posix(),
                "metrics": metrics_path.as_posix(),
                "predictions": predictions_path.as_posix(),
            },
        },
    )

    log_path = run_dir / "run.log"
    _write_run_log(
        log_path,
        run_id=run_id,
        dataset_path=dataset_path,
        n_rows=len(dataset),
        n_train=len(train),
        n_test=len(test),
        held_out_airfoils=held_out_airfoils,
        test_metrics=test_metrics,
    )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "config": config_path,
        "meta": meta_path,
        "manifest": manifest_path,
        "validation_model": validation_model_path,
        "deployment_model": deployment_model_path,
        "deployment_manifest": deployment_manifest_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "summary": summary_path,
        "log": log_path,
        "held_out_airfoils": held_out_airfoils,
        "test_metrics": test_metrics,
    }


def _new_model(config: TrainingRunConfig) -> RandomForestSurrogate:
    return RandomForestSurrogate(
        n_estimators=config.n_estimators,
        random_state=config.random_seed,
        min_samples_leaf=config.min_samples_leaf,
    )


def _input_files(dataset_path: Path, raw_data_dir: str | None) -> list[Path]:
    files = [dataset_path]
    if raw_data_dir:
        raw_root = Path(raw_data_dir)
        if not raw_root.exists():
            raise FileNotFoundError(f"Raw data directory does not exist: {raw_root}")
        files.extend(
            path
            for path in sorted(raw_root.rglob("*"))
            if path.is_file() and path.name != "raw_manifest.json"
        )
    return files


def _write_run_log(
    path: Path,
    *,
    run_id: str,
    dataset_path: Path,
    n_rows: int,
    n_train: int,
    n_test: int,
    held_out_airfoils: list[str],
    test_metrics: dict[str, dict[str, float]],
) -> None:
    lines = [
        f"run_id={run_id}",
        f"dataset={dataset_path.as_posix()}",
        f"rows={n_rows}",
        "split=grouped_by_naca",
        f"train_rows={n_train}",
        f"test_rows={n_test}",
        f"held_out_airfoils={','.join(held_out_airfoils)}",
    ]
    for target in TARGET_COLUMNS:
        values = test_metrics[target]
        lines.append(
            f"{target}_test rmse={values['rmse']:.8f} "
            f"mae={values['mae']:.8f} r2={values['r2']:.8f}"
        )
    lines.append("status=completed")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
