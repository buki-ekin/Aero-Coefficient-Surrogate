import importlib.util
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from aero_surrogate.api import AeroSurrogate
from aero_surrogate.dashboard import _serialize_surrogate, export_html_dashboard
from aero_surrogate.data import save_dataset
from aero_surrogate.flow5 import naca4_parameters, read_flow5_polar
from aero_surrogate.metrics import evaluate_predictions, grouped_train_test_split
from aero_surrogate.sklearn_surrogate import RandomForestSurrogate
from aero_surrogate.workflow import TrainingRunConfig, run_training_workflow


def sample_dataset() -> pd.DataFrame:
    rows = []
    for code in ("0012", "1412", "2410", "2412", "4412", "6412"):
        geometry = naca4_parameters(code)
        for reynolds in (500_000.0, 1_000_000.0):
            for alpha in (-2.0, 0.0, 2.0, 4.0):
                camber = float(geometry["camber"])
                thickness = float(geometry["thickness"])
                cl = 0.105 * alpha + 4.5 * camber
                cd = 0.007 + 0.002 * (1_000_000 / reynolds) + 0.012 * cl**2
                cm = -1.8 * camber - 0.0007 * alpha + 0.01 * (thickness - 0.12)
                rows.append(
                    {
                        **geometry,
                        "alpha_deg": alpha,
                        "reynolds": reynolds,
                        "cl": cl,
                        "cd": cd,
                        "cm": cm,
                        "source_file": f"NACA_{code}_test.txt",
                    }
                )
    return pd.DataFrame(rows)


class Flow5ImportTest(unittest.TestCase):
    def test_naca4_parameters(self):
        parameters = naca4_parameters("NACA 2412")
        self.assertEqual(parameters["naca"], "NACA 2412")
        self.assertEqual(parameters["camber"], 0.02)
        self.assertEqual(parameters["camber_position"], 0.4)
        self.assertEqual(parameters["thickness"], 0.12)

    def test_read_flow5_polar_export(self):
        polar_text = """flow5 polar export
 alpha      CL        CD       CDp       Cm
 -2.0      0.100     0.0120   0.0010   -0.041
  0.0      0.310     0.0105   0.0011   -0.043
  2.0      0.520     0.0119   0.0012   -0.045
"""
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "naca2412.txt"
            path.write_text(polar_text, encoding="utf-8")
            dataset = read_flow5_polar(path, "NACA 2412", 1_000_000)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.loc[0, "naca"], "NACA 2412")
        self.assertEqual(dataset.loc[2, "cl"], 0.52)
        self.assertEqual(dataset.loc[2, "cm"], -0.045)


class ValidationTest(unittest.TestCase):
    def test_grouped_split_keeps_airfoils_separate(self):
        dataset = sample_dataset()
        train, test, held_out = grouped_train_test_split(dataset, random_seed=7)
        self.assertFalse(set(train["naca"]) & set(test["naca"]))
        self.assertEqual(set(test["naca"]), set(held_out))

    def test_grouped_split_is_deterministic(self):
        dataset = sample_dataset()
        first = grouped_train_test_split(dataset, random_seed=7)
        second = grouped_train_test_split(dataset, random_seed=7)
        self.assertTrue(first[0].equals(second[0]))
        self.assertTrue(first[1].equals(second[1]))
        self.assertEqual(first[2], second[2])

    def test_metrics_contain_all_targets(self):
        dataset = sample_dataset()
        model = RandomForestSurrogate(n_estimators=10, random_state=1).fit(dataset)
        metrics = evaluate_predictions(dataset, model.predict(dataset))
        self.assertEqual(set(metrics), {"cl", "cd", "cm"})
        self.assertIn("rmse", metrics["cl"])
        self.assertIn("r2", metrics["cm"])


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class RandomForestWorkflowTest(unittest.TestCase):
    def test_random_forest_predicts_three_coefficients(self):
        dataset = sample_dataset()
        model = RandomForestSurrogate(n_estimators=10, random_state=1).fit(dataset)
        prediction = model.predict(dataset.head(2))
        self.assertEqual(list(prediction.columns), ["cl", "cd", "cm"])
        self.assertEqual(len(prediction), 2)

    def test_public_api_accepts_naca_code(self):
        model = RandomForestSurrogate(n_estimators=10, random_state=1).fit(sample_dataset())
        prediction = AeroSurrogate(model).predict_naca("2412", 4.0, 1_000_000)
        self.assertEqual(set(prediction), {"cl", "cd", "cm"})

    def test_workflow_writes_final_run_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = save_dataset(sample_dataset(), root / "flow5.csv")
            result = run_training_workflow(
                TrainingRunConfig(
                    dataset_path=dataset_path.as_posix(),
                    raw_data_dir=None,
                    run_root=(root / "runs").as_posix(),
                    run_id="test_run",
                    n_estimators=10,
                    cv_splits=3,
                    cv_repeats=1,
                    generate_figures=False,
                )
            )
            run_dir = result["run_dir"]
            self.assertTrue((run_dir / "run.log").exists())
            self.assertTrue((run_dir / "outputs" / "validation_model.pkl").exists())
            self.assertTrue((run_dir / "outputs" / "deployment_model.pkl").exists())
            self.assertTrue((run_dir / "outputs" / "deployment_manifest.json").exists())
            self.assertTrue((run_dir / "reports" / "metrics.json").exists())
            self.assertTrue((run_dir / "reports" / "predictions.csv").exists())
            self.assertTrue((run_dir / "reports" / "cross_validation.csv").exists())
            self.assertTrue((run_dir / "reports" / "model_comparison.json").exists())

    def test_dashboard_embeds_deployment_model(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = save_dataset(sample_dataset(), root / "flow5.csv")
            result = run_training_workflow(
                TrainingRunConfig(
                    dataset_path=dataset_path.as_posix(),
                    raw_data_dir=None,
                    run_root=(root / "runs").as_posix(),
                    run_id="dashboard_test",
                    n_estimators=10,
                    cv_splits=3,
                    cv_repeats=1,
                    generate_figures=False,
                )
            )
            dashboard_path = export_html_dashboard(
                dataset_path,
                result["run_dir"],
                root / "dashboard.html",
            )
            html = dashboard_path.read_text(encoding="utf-8")
            self.assertIn('id="control-panel"', html)
            self.assertIn('id="coefficient-chart"', html)
            self.assertIn('"available":true', html)

    def test_serialized_forest_matches_python_prediction(self):
        model = RandomForestSurrogate(n_estimators=12, random_state=4).fit(sample_dataset())
        sample = {
            "camber": 0.02,
            "camber_position": 0.4,
            "thickness": 0.12,
            "alpha_deg": 4.0,
            "reynolds": 1_000_000.0,
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pkl"
            with path.open("wb") as file:
                pickle.dump(model, file)
            serialized = _serialize_surrogate(path)

        input_values = [sample[name] for name in serialized["features"]]
        totals = [0.0] * len(serialized["targets"])
        for tree in serialized["trees"]:
            node = 0
            while tree["feature"][node] >= 0:
                feature = tree["feature"][node]
                node = (
                    tree["left"][node]
                    if input_values[feature] <= tree["threshold"][node]
                    else tree["right"][node]
                )
            totals = [
                total + value for total, value in zip(totals, tree["value"][node])
            ]

        browser_values = [value / len(serialized["trees"]) for value in totals]
        python_values = model.predict(sample).iloc[0].tolist()
        for browser_value, python_value in zip(browser_values, python_values):
            self.assertAlmostEqual(browser_value, python_value, places=10)


if __name__ == "__main__":
    unittest.main()
