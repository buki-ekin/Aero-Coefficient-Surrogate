import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from aero_surrogate.api import AeroSurrogate
from aero_surrogate.bundled import export_bundled_dashboard, load_example_dataset
from aero_surrogate.data import validate_dataset
from aero_surrogate.sklearn_surrogate import (
    OutOfDomainWarning,
    RandomForestSurrogate,
)
from aero_surrogate.validation import (
    evaluate_model_systematics,
    repeated_grouped_folds,
)


def small_dataset() -> pd.DataFrame:
    rows = []
    for code in ("0012", "1412", "2412", "4412", "6412"):
        camber = int(code[0]) / 100
        camber_position = int(code[1]) / 10
        thickness = int(code[2:]) / 100
        for alpha in (-2.0, 2.0):
            cl = 0.1 * alpha + 4 * camber
            rows.append(
                {
                    "naca": f"NACA {code}",
                    "camber": camber,
                    "camber_position": camber_position,
                    "thickness": thickness,
                    "alpha_deg": alpha,
                    "reynolds": 1_000_000.0,
                    "cl": cl,
                    "cd": 0.01 + 0.01 * cl**2,
                    "cm": -1.5 * camber,
                }
            )
    return pd.DataFrame(rows)


class DatasetIntegrityTest(unittest.TestCase):
    def test_rejects_missing_numeric_value(self):
        dataset = small_dataset()
        dataset.loc[0, "cl"] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            validate_dataset(dataset)

    def test_rejects_duplicate_operating_point(self):
        dataset = pd.concat([small_dataset(), small_dataset().iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "duplicate operating points"):
            validate_dataset(dataset)

    def test_rejects_negative_drag(self):
        dataset = small_dataset()
        dataset.loc[0, "cd"] = -0.1
        with self.assertRaisesRegex(ValueError, "plausible range"):
            validate_dataset(dataset)


class DomainSafetyTest(unittest.TestCase):
    def test_prediction_warns_outside_training_domain(self):
        model = RandomForestSurrogate(n_estimators=5).fit(small_dataset())
        sample = {
            "camber": 0.02,
            "camber_position": 0.4,
            "thickness": 0.12,
            "alpha_deg": 20.0,
            "reynolds": 1_000_000.0,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.predict(sample)
        self.assertTrue(
            any(issubclass(item.category, OutOfDomainWarning) for item in caught)
        )
        self.assertIn("alpha_deg=20", model.domain_violations(sample)[0])

    def test_non_finite_prediction_input_is_rejected(self):
        model = RandomForestSurrogate(n_estimators=5).fit(small_dataset())
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            model.predict(
                {
                    "camber": 0.02,
                    "camber_position": 0.4,
                    "thickness": 0.12,
                    "alpha_deg": float("nan"),
                    "reynolds": 1_000_000.0,
                }
            )


class RepeatedValidationTest(unittest.TestCase):
    def test_repeated_folds_keep_groups_separate(self):
        folds = list(
            repeated_grouped_folds(
                small_dataset(),
                n_splits=3,
                n_repeats=2,
                random_seed=11,
            )
        )
        self.assertEqual(len(folds), 6)
        for fold in folds:
            self.assertFalse(set(fold.train["naca"]) & set(fold.test["naca"]))

    def test_model_comparison_contains_baselines_and_uncertainty(self):
        fold_metrics, summary = evaluate_model_systematics(
            small_dataset(),
            n_splits=3,
            n_repeats=1,
            n_estimators=5,
        )
        self.assertEqual(
            set(fold_metrics["model"]),
            {"random_forest", "linear_regression", "mean_baseline"},
        )
        self.assertIn("rmse_ci95", summary["models"]["random_forest"]["cl"])


class BundledDistributionTest(unittest.TestCase):
    def test_bundled_resources_support_clean_use(self):
        dataset = load_example_dataset()
        self.assertGreater(len(dataset), 1000)
        prediction = AeroSurrogate.load().predict_naca("2412", 4.0, 1_000_000)
        self.assertEqual(set(prediction), {"cl", "cd", "cm"})
        with TemporaryDirectory() as temp_dir:
            path = export_bundled_dashboard(Path(temp_dir) / "dashboard.html")
            self.assertGreater(path.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
