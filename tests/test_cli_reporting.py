import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from aero_surrogate.cli import main
from aero_surrogate.data import save_dataset
from aero_surrogate.workflow import TrainingRunConfig, run_training_workflow


def report_dataset() -> pd.DataFrame:
    rows = []
    for code in ("0012", "1412", "2412", "4412", "6412"):
        camber = int(code[0]) / 100
        position = int(code[1]) / 10
        thickness = int(code[2:]) / 100
        for reynolds in (500_000.0, 1_000_000.0):
            for alpha in (-2.0, 0.0, 2.0):
                cl = 0.105 * alpha + 4.5 * camber
                rows.append(
                    {
                        "naca": f"NACA {code}",
                        "camber": camber,
                        "camber_position": position,
                        "thickness": thickness,
                        "alpha_deg": alpha,
                        "reynolds": reynolds,
                        "cl": cl,
                        "cd": 0.008 + 0.012 * cl**2,
                        "cm": -1.8 * camber - 0.0007 * alpha,
                    }
                )
    return pd.DataFrame(rows)


def run_cli(*arguments: str) -> str:
    stream = io.StringIO()
    with (
        patch.object(sys, "argv", ["aero-surrogate", *arguments]),
        redirect_stdout(stream),
    ):
        main()
    return stream.getvalue()


class ReportingTest(unittest.TestCase):
    def test_workflow_generates_all_scientific_figures(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = save_dataset(report_dataset(), root / "data.csv")
            result = run_training_workflow(
                TrainingRunConfig(
                    dataset_path=dataset_path.as_posix(),
                    run_root=(root / "runs").as_posix(),
                    run_id="figures",
                    n_estimators=5,
                    cv_splits=3,
                    cv_repeats=1,
                    generate_figures=True,
                )
            )
            figure_dir = result["run_dir"] / "reports" / "figures"
            expected = {
                "measured_vs_predicted.png",
                "residuals_vs_alpha.png",
                "rmse_by_airfoil.png",
                "grouped_cv_model_comparison.png",
                "data_coverage.png",
                "figure_manifest.json",
            }
            self.assertEqual(expected, {path.name for path in figure_dir.iterdir()})


class CommandLineTest(unittest.TestCase):
    def test_clean_install_commands_and_full_workflow(self):
        prediction = run_cli(
            "predict",
            "--naca",
            "2412",
            "--alpha-deg",
            "4",
            "--reynolds",
            "1000000",
        )
        self.assertIn("cl:", prediction)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = save_dataset(report_dataset(), root / "data.csv")
            raw_dir = root / "raw"
            raw_dir.mkdir()
            deployment_model = root / "model.pkl"
            run_root = root / "runs"

            trained = run_cli(
                "train",
                "--data",
                dataset_path.as_posix(),
                "--output",
                deployment_model.as_posix(),
            )
            self.assertIn("Saved deployment model", trained)

            run_output = run_cli(
                "run",
                "--data",
                dataset_path.as_posix(),
                "--raw-dir",
                raw_dir.as_posix(),
                "--run-root",
                run_root.as_posix(),
                "--run-id",
                "cli-run",
                "--deployment-model",
                deployment_model.as_posix(),
                "--cv-splits",
                "3",
                "--cv-repeats",
                "1",
                "--no-figures",
            )
            self.assertIn("Held-out airfoils", run_output)

            summary = run_cli(
                "summary",
                "--data",
                dataset_path.as_posix(),
                "--run-dir",
                (run_root / "cli-run").as_posix(),
                "--model",
                deployment_model.as_posix(),
            )
            self.assertIn("Repeated grouped validation", summary)

            dashboard_path = root / "dashboard.html"
            dashboard = run_cli(
                "dashboard",
                "--data",
                dataset_path.as_posix(),
                "--run-dir",
                (run_root / "cli-run").as_posix(),
                "--model",
                deployment_model.as_posix(),
                "--output",
                dashboard_path.as_posix(),
            )
            self.assertIn("Saved dashboard", dashboard)
            self.assertTrue(dashboard_path.exists())

            bundled_path = root / "bundled.html"
            bundled = run_cli(
                "export-bundled-dashboard",
                "--output",
                bundled_path.as_posix(),
            )
            self.assertIn("Saved bundled dashboard", bundled)
