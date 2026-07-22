"""Small public Python interface for the deployed AeroSurrogate model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aero_surrogate.flow5 import naca4_parameters
from aero_surrogate.surrogate import load_model


class AeroSurrogate:
    """Load a fitted model and query it with NACA or geometry inputs."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def load(
        cls,
        path: str | Path = "models/flow5_random_forest.pkl",
    ) -> "AeroSurrogate":
        """Load the final trained surrogate from disk."""

        return cls(load_model(path))

    def predict_naca(
        self,
        naca: str,
        alpha_deg: float,
        reynolds: float,
    ) -> dict[str, float]:
        """Predict CL, CD, and CM for a NACA 4-digit airfoil."""

        geometry = naca4_parameters(naca)
        return self.predict_geometry(
            camber=float(geometry["camber"]),
            camber_position=float(geometry["camber_position"]),
            thickness=float(geometry["thickness"]),
            alpha_deg=alpha_deg,
            reynolds=reynolds,
        )

    def predict_geometry(
        self,
        camber: float,
        camber_position: float,
        thickness: float,
        alpha_deg: float,
        reynolds: float,
    ) -> dict[str, float]:
        """Predict coefficients from normalized NACA-style geometry inputs."""

        prediction = self.model.predict(
            {
                "camber": float(camber),
                "camber_position": float(camber_position),
                "thickness": float(thickness),
                "alpha_deg": float(alpha_deg),
                "reynolds": float(reynolds),
            }
        ).iloc[0]
        return {name: float(value) for name, value in prediction.items()}
