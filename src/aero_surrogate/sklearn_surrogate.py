"""Scikit-learn based surrogate models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aero_surrogate.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TrainingDomain,
    validate_dataset,
)


class OutOfDomainWarning(UserWarning):
    """Warning raised when a prediction leaves the observed training domain."""


@dataclass
class RandomForestSurrogate:
    """Multi-output Random Forest surrogate for flow5-generated polar data."""

    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    target_columns: tuple[str, ...] = tuple(TARGET_COLUMNS)
    n_estimators: int = 300
    random_state: int = 42
    min_samples_leaf: int = 2
    model: object | None = None
    training_domain: TrainingDomain | None = None

    def fit(self, dataset: pd.DataFrame) -> "RandomForestSurrogate":
        """Fit the Random Forest model to aerodynamic polar data."""

        try:
            from sklearn.ensemble import RandomForestRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RandomForestSurrogate requires scikit-learn. "
                "Install AeroSurrogate with its declared dependencies."
            ) from exc

        validated = validate_dataset(dataset)
        x = validated.loc[:, self.feature_columns].to_numpy(dtype=float)
        y = validated.loc[:, self.target_columns].to_numpy(dtype=float)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
        )
        self.model.fit(x, y)
        self.training_domain = TrainingDomain.from_dataset(validated)
        return self

    def predict(
        self,
        inputs: pd.DataFrame | dict[str, float],
        *,
        warn_outside_domain: bool = True,
    ) -> pd.DataFrame:
        """Predict coefficients and warn when inputs leave the training domain."""

        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        frame = self._coerce_inputs(inputs)
        if warn_outside_domain:
            self._warn_for_domain_violations(frame)
        predictions = self.model.predict(frame.loc[:, self.feature_columns].to_numpy(dtype=float))
        return pd.DataFrame(predictions, columns=self.target_columns, index=frame.index)

    def score_rmse(self, dataset: pd.DataFrame) -> dict[str, float]:
        """Return root mean squared error per target column."""

        prediction = self.predict(dataset)
        truth = dataset.loc[:, self.target_columns]
        errors = prediction.to_numpy() - truth.to_numpy()
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        return dict(zip(self.target_columns, rmse))

    def _coerce_inputs(self, inputs: pd.DataFrame | dict[str, float]) -> pd.DataFrame:
        if isinstance(inputs, pd.DataFrame):
            frame = inputs.copy()
        else:
            frame = pd.DataFrame([inputs])

        missing = sorted(set(self.feature_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Prediction inputs are missing required columns: {missing}")
        for column in self.feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = ~np.isfinite(
            frame.loc[:, self.feature_columns].to_numpy(dtype=float)
        )
        if invalid.any():
            raise ValueError("Prediction inputs must be finite numeric values.")
        return frame

    def domain_violations(
        self,
        inputs: pd.DataFrame | dict[str, float],
    ) -> list[str]:
        """Return unique training-domain violations without making a prediction."""

        if self.training_domain is None:
            return []
        frame = self._coerce_inputs(inputs)
        messages = {
            message
            for row in frame.loc[:, self.feature_columns].to_dict(orient="records")
            for message in self.training_domain.violations(row)
        }
        return sorted(messages)

    def _warn_for_domain_violations(self, frame: pd.DataFrame) -> None:
        messages = self.domain_violations(frame)
        if messages:
            warnings.warn(
                "Prediction is outside the observed training domain: "
                + "; ".join(messages)
                + ". Random Forest extrapolation is not scientifically supported.",
                OutOfDomainWarning,
                stacklevel=3,
            )
