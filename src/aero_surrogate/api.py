"""Small public Python interface for the deployed AeroSurrogate model."""

from __future__ import annotations

from importlib.resources import as_file, files
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
        path: str | Path | None = None,
    ) -> "AeroSurrogate":
        """Load a trusted model file or the model bundled with the package.

        Pickle model files can execute code during loading. Only pass files from
        a trusted source. When ``path`` is omitted, the checksum-controlled model
        shipped with AeroSurrogate is used.
        """

        if path is not None:
            return cls(load_model(path))

        resource = files("aero_surrogate.resources").joinpath(
            "flow5_random_forest.pkl"
        )
        if not resource.is_file():
            raise FileNotFoundError(
                "The bundled deployment model is unavailable. Reinstall the "
                "package or pass an explicit trusted model path."
            )
        with as_file(resource) as model_path:
            return cls(load_model(model_path))

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

    def domain_warnings(
        self,
        *,
        camber: float,
        camber_position: float,
        thickness: float,
        alpha_deg: float,
        reynolds: float,
    ) -> list[str]:
        """Return training-domain violations for a proposed prediction."""

        checker = getattr(self.model, "domain_violations", None)
        if checker is None:
            return []
        return checker(
            {
                "camber": camber,
                "camber_position": camber_position,
                "thickness": thickness,
                "alpha_deg": alpha_deg,
                "reynolds": reynolds,
            }
        )
