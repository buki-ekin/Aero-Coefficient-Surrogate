"""Command line interface for AeroSurrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aero_surrogate.api import AeroSurrogate
from aero_surrogate.bundled import export_bundled_dashboard
from aero_surrogate.dashboard import export_html_dashboard
from aero_surrogate.data import load_dataset, save_dataset
from aero_surrogate.flow5 import import_flow5_directory, read_flow5_polar
from aero_surrogate.sklearn_surrogate import RandomForestSurrogate
from aero_surrogate.surrogate import save_model
from aero_surrogate.workflow import TrainingRunConfig, run_training_workflow


def main() -> None:
    parser = argparse.ArgumentParser(prog="aero-surrogate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    flow5_parser = subparsers.add_parser("import-flow5", help="Import flow5 polar files.")
    flow5_parser.add_argument("--output", default="data/processed/flow5_airfoils.csv")
    flow5_group = flow5_parser.add_mutually_exclusive_group(required=True)
    flow5_group.add_argument("--file")
    flow5_group.add_argument("--directory")
    flow5_parser.add_argument("--naca")
    flow5_parser.add_argument("--reynolds", type=float)
    flow5_parser.add_argument("--metadata")

    train_parser = subparsers.add_parser("train", help="Train the deployment model.")
    train_parser.add_argument("--data", default="data/processed/flow5_airfoils.csv")
    train_parser.add_argument("--output", default="models/flow5_random_forest.pkl")

    run_parser = subparsers.add_parser(
        "run",
        help="Run grouped validation and train the final deployment model.",
    )
    run_parser.add_argument("--data", default="data/processed/flow5_airfoils.csv")
    run_parser.add_argument("--raw-dir", default="data/raw")
    run_parser.add_argument("--run-root", default="runs")
    run_parser.add_argument("--test-fraction", type=float, default=0.2)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--deployment-model", default="models/flow5_random_forest.pkl")
    run_parser.add_argument("--cv-splits", type=int, default=5)
    run_parser.add_argument("--cv-repeats", type=int, default=3)
    run_parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip scientific PNG report generation.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict one airfoil state.")
    predict_parser.add_argument(
        "--model",
        default=None,
        help="Trusted pickle model path. Omit to use the bundled deployment model.",
    )
    predict_parser.add_argument("--naca", help="NACA 4-digit preset, for example 2412.")
    predict_parser.add_argument("--camber", type=float)
    predict_parser.add_argument("--camber-position", type=float)
    predict_parser.add_argument("--thickness", type=float)
    predict_parser.add_argument("--alpha-deg", type=float, required=True)
    predict_parser.add_argument("--reynolds", type=float, required=True)

    dashboard_parser = subparsers.add_parser("dashboard", help="Export the HTML predictor.")
    dashboard_parser.add_argument("--data", default="data/processed/flow5_airfoils.csv")
    dashboard_parser.add_argument("--run-dir", required=True)
    dashboard_parser.add_argument("--model", default="models/flow5_random_forest.pkl")
    dashboard_parser.add_argument("--output", default="output/dashboard/aerosurrogate_dashboard.html")

    summary_parser = subparsers.add_parser("summary", help="Print final project results.")
    summary_parser.add_argument("--data", default="data/processed/flow5_airfoils.csv")
    summary_parser.add_argument("--run-dir", required=True)
    summary_parser.add_argument("--model", default="models/flow5_random_forest.pkl")

    bundled_dashboard_parser = subparsers.add_parser(
        "export-bundled-dashboard",
        help="Copy the self-contained dashboard from an installed package.",
    )
    bundled_dashboard_parser.add_argument(
        "--output",
        default="aerosurrogate_dashboard.html",
    )

    args = parser.parse_args()

    if args.command == "import-flow5":
        _import_flow5(args)
    elif args.command == "train":
        _train(args)
    elif args.command == "run":
        _run(args)
    elif args.command == "predict":
        _predict(args)
    elif args.command == "dashboard":
        path = export_html_dashboard(
            dataset_path=args.data,
            run_dir=args.run_dir,
            output_path=args.output,
            model_path=args.model,
        )
        print(f"Saved dashboard to {path}")
    elif args.command == "summary":
        _summary(args)
    elif args.command == "export-bundled-dashboard":
        path = export_bundled_dashboard(args.output)
        print(f"Saved bundled dashboard to {path}")


def _import_flow5(args: argparse.Namespace) -> None:
    if args.file:
        if not args.naca or args.reynolds is None:
            raise SystemExit("--file requires --naca and --reynolds.")
        dataset = read_flow5_polar(args.file, args.naca, args.reynolds)
    else:
        if not args.metadata:
            raise SystemExit("--directory requires --metadata.")
        dataset = import_flow5_directory(args.directory, pd.read_csv(args.metadata))
    path = save_dataset(dataset, args.output)
    print(f"Saved {len(dataset)} flow5 rows to {path}")


def _train(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.data)
    model = RandomForestSurrogate().fit(dataset)
    path = save_model(model, args.output)
    print(f"Saved deployment model to {path}")


def _run(args: argparse.Namespace) -> None:
    result = run_training_workflow(
        TrainingRunConfig(
            dataset_path=args.data,
            raw_data_dir=args.raw_dir,
            test_fraction=args.test_fraction,
            random_seed=args.seed,
            run_root=args.run_root,
            run_id=args.run_id,
            deployment_model_path=args.deployment_model,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            generate_figures=not args.no_figures,
        )
    )
    print(f"Saved final run to {result['run_dir']}")
    print(f"Held-out airfoils: {', '.join(result['held_out_airfoils'])}")
    for target, values in result["test_metrics"].items():
        print(
            f"{target.upper()} test RMSE={values['rmse']:.5f}, "
            f"MAE={values['mae']:.5f}, R2={values['r2']:.5f}"
        )


def _predict(args: argparse.Namespace) -> None:
    surrogate = AeroSurrogate.load(args.model)
    if args.naca:
        prediction = surrogate.predict_naca(args.naca, args.alpha_deg, args.reynolds)
        print(f"NACA: {args.naca}")
    else:
        geometry = (args.camber, args.camber_position, args.thickness)
        if any(value is None for value in geometry):
            raise SystemExit(
                "Use --naca or provide --camber, --camber-position, and --thickness."
            )
        prediction = surrogate.predict_geometry(
            args.camber,
            args.camber_position,
            args.thickness,
            args.alpha_deg,
            args.reynolds,
        )
    print(f"alpha_deg: {args.alpha_deg:g}")
    print(f"reynolds: {args.reynolds:g}")
    for name, value in prediction.items():
        print(f"{name}: {value:.5f}")


def _summary(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.data)
    metrics_path = Path(args.run_dir) / "reports" / "metrics.json"
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    airfoils = sorted(dataset["naca"].unique())
    reynolds = sorted(float(value) for value in dataset["reynolds"].unique())
    print("AeroSurrogate final model")
    print("=" * 44)
    print("Data source : flow5 polar exports")
    print(f"Data rows   : {len(dataset):,}")
    print(f"NACA foils  : {len(airfoils)}")
    print(f"Re levels   : {', '.join(f'{value:,.0f}' for value in reynolds)}")
    print(f"Alpha range : {dataset['alpha_deg'].min():g} to {dataset['alpha_deg'].max():g} deg")
    print("Model       : Random Forest, multi-output")
    print(f"Model file  : {args.model}")
    print("Split       : grouped by NACA")
    print(f"Held out    : {', '.join(metrics['held_out_airfoils'])}")
    print("\nHeld-out flow5 validation")
    for target in ("cl", "cd", "cm"):
        values = metrics["test"][target]
        print(
            f"{target.upper():>2}  RMSE={values['rmse']:.6f}  "
            f"MAE={values['mae']:.6f}  R2={values['r2']:.6f}"
        )
    comparison_path = Path(args.run_dir) / "reports" / "model_comparison.json"
    if comparison_path.exists():
        with comparison_path.open(encoding="utf-8") as file:
            comparison = json.load(file)
        random_forest = comparison["models"]["random_forest"]
        print(
            "\nRepeated grouped validation "
            f"({comparison['n_repeats']} x {comparison['n_splits']} folds)"
        )
        for target in ("cl", "cd", "cm"):
            values = random_forest[target]
            print(
                f"{target.upper():>2}  mean RMSE={values['rmse_mean']:.6f}  "
                f"95% CI half-width={values['rmse_ci95']:.6f}"
            )


if __name__ == "__main__":
    main()
