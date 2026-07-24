# API Guide

## Default Bundled Model

```python
from aero_surrogate import AeroSurrogate

model = AeroSurrogate.load()
model.predict_naca("2412", alpha_deg=4, reynolds=1_000_000)
```

`AeroSurrogate.load()` uses the checksum-controlled deployment model bundled
with the package. Passing a path loads an external pickle and must only be done
for trusted files.

## Manual Geometry

```python
model.predict_geometry(
    camber=0.02,
    camber_position=0.4,
    thickness=0.12,
    alpha_deg=4,
    reynolds=1_000_000,
)
```

Geometry quantities are normalized by chord. For example, `camber=0.02` means
2% chord.

## Domain Checks

```python
model.domain_warnings(
    camber=0.02,
    camber_position=0.4,
    thickness=0.12,
    alpha_deg=20,
    reynolds=1_000_000,
)
```

Normal prediction calls emit `OutOfDomainWarning` when any feature leaves the
observed training range. The warning is a scientific guard, not a mathematical
guarantee for points inside the range.

## Bundled Data And Dashboard

```python
from aero_surrogate import export_bundled_dashboard, load_example_dataset

dataset = load_example_dataset()
dashboard_path = export_bundled_dashboard("dashboard.html")
```
