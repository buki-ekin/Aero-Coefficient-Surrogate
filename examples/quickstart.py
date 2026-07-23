"""Minimal clean-install example using the bundled deployment model."""

from aero_surrogate import AeroSurrogate

model = AeroSurrogate.load()
prediction = model.predict_naca(
    naca="2412",
    alpha_deg=4.0,
    reynolds=1_000_000,
)

print(prediction)
