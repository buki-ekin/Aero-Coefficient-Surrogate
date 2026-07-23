"""Access to data and interface artifacts bundled in the wheel."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd


def load_example_dataset() -> pd.DataFrame:
    """Load the bundled model-ready flow5 example dataset."""

    resource = files("aero_surrogate.resources").joinpath("flow5_airfoils.csv")
    with resource.open("rb") as handle:
        return pd.read_csv(handle)


def export_bundled_dashboard(
    output_path: str | Path = "aerosurrogate_dashboard.html",
) -> Path:
    """Copy the self-contained dashboard from the installed package."""

    resource = files("aero_surrogate.resources").joinpath(
        "aerosurrogate_dashboard.html"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(resource.read_bytes())
    return destination
