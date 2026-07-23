"""AeroSurrogate public API."""

from aero_surrogate.api import AeroSurrogate
from aero_surrogate.bundled import export_bundled_dashboard, load_example_dataset
from aero_surrogate.flow5 import read_flow5_polar
from aero_surrogate.sklearn_surrogate import OutOfDomainWarning, RandomForestSurrogate
from aero_surrogate.surrogate import load_model, save_model
from aero_surrogate.workflow import TrainingRunConfig, run_training_workflow

__all__ = [
    "AeroSurrogate",
    "OutOfDomainWarning",
    "RandomForestSurrogate",
    "TrainingRunConfig",
    "export_bundled_dashboard",
    "load_model",
    "load_example_dataset",
    "read_flow5_polar",
    "run_training_workflow",
    "save_model",
]

__version__ = "1.1.0"
