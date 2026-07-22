"""Minimal example for querying the final flow5 Random Forest model."""

from aero_surrogate import AeroSurrogate

model = AeroSurrogate.load("models/flow5_random_forest.pkl")
prediction = model.predict_naca(
    naca="2412",
    alpha_deg=4.0,
    reynolds=1_000_000,
)

print(prediction)
