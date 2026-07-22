"""Model serialization helpers."""

from __future__ import annotations

import pickle
from pathlib import Path


def save_model(model: object, output_path: str | Path) -> Path:
    """Serialize a fitted model with pickle."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)
    return path


def load_model(path: str | Path) -> object:
    """Load a previously saved surrogate model."""

    with Path(path).open("rb") as file:
        model = pickle.load(file)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded object is not a surrogate model.")
    return model
