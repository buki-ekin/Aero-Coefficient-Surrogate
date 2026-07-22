# AeroSurrogate

AeroSurrogate predicts airfoil lift, drag, and moment coefficients with a
Random Forest model trained on flow5 polar data. The project uses flow5 for the
aerodynamic analyses and Python for data import, validation, model training,
reproducibility records, and the user interface.

## Workflow

```text
27 NACA airfoils
-> flow5 analyses at Re 500k, 1M, and 2M
-> 81 raw polar files
-> Python import and cleaning
-> 1,618 processed rows
-> grouped Random Forest validation
-> final model trained on all rows
-> Python API and HTML dashboard
```

flow5 already calculates `CL`, `CD`, and `CM`. Python does not replace these
values with formulas and does not fill unconverged points with synthetic data.

## Inputs And Outputs

Model inputs:

- maximum camber
- camber position
- thickness
- angle of attack
- Reynolds number

Model outputs:

- `CL`: lift coefficient
- `CD`: drag coefficient
- `CM`: pitching moment coefficient

## Data

The final dataset contains:

- 27 unique NACA 4-digit airfoils
- Reynolds numbers 500,000, 1,000,000, and 2,000,000
- requested angle-of-attack range from -6 to 14 degrees
- 1,618 converged flow5 operating points
- no missing values or duplicate operating points

Raw flow5 files are stored under `data/raw/`. The model-ready table is
`data/processed/flow5_airfoils.csv`. `data/raw/raw_manifest.json` records the
SHA-256 checksum of every final raw input and export.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The Conda environment can be created with:

```bash
conda env create -f environment.yml
conda activate aerosurrogate
```

## Reproduce The Dataset

The NACA geometry files, flow5 XML scripts, exported polar files, and logs are
included in `data/raw/`.

```bash
aero-surrogate import-flow5 \
  --directory data/raw/flow5_exports_multi_re \
  --metadata data/raw/flow5_exports_multi_re/metadata.csv \
  --output data/processed/flow5_airfoils.csv
```

## Run Validation And Training

```bash
aero-surrogate run \
  --data data/processed/flow5_airfoils.csv \
  --raw-dir data/raw \
  --run-id final_grouped_naca_seed42 \
  --deployment-model models/flow5_random_forest.pkl
```

Complete NACA profiles are held out during validation. The deployment model is
then trained separately on all 1,618 rows.

The held-out profiles are NACA 0021, 2418, 2424, 4415, and 4418. Results on
their 302 flow5 rows are:

| Target | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| CL | 0.03457 | 0.02306 | 0.99671 |
| CD | 0.00132 | 0.00107 | 0.95441 |
| CM | 0.00696 | 0.00486 | 0.96998 |

The final run folder contains:

```text
runs/final_grouped_naca_seed42/
├── config.json
├── manifest.json
├── meta.json
├── run.log
├── outputs/
│   ├── validation_model.pkl
│   ├── deployment_model.pkl
│   └── deployment_manifest.json
└── reports/
    ├── metrics.json
    ├── predictions.csv
    └── summary.json
```

## Make A Prediction

Command line:

```bash
aero-surrogate predict \
  --naca 2412 \
  --alpha-deg 4 \
  --reynolds 1000000
```

Python:

```python
from aero_surrogate import AeroSurrogate

model = AeroSurrogate.load()
result = model.predict_naca("2412", alpha_deg=4, reynolds=1_000_000)
print(result)
```

Manual geometry can be queried with `predict_geometry` using normalized camber,
camber position, and thickness values.

## Dashboard

Open `output/dashboard/aerosurrogate_dashboard.html`. The left panel contains
the geometry and operating-condition inputs. The right panel shows the current
`CL`, `CD`, and `CM` values and their curves over angle of attack.

The dashboard contains the fitted Random Forest, so it works locally without a
Python server.

## SCE Course Concepts

| Course concept | Project implementation |
| --- | --- |
| Data life cycle | raw flow5 files, processed CSV, model outputs |
| Metadata and provenance | source file, config, logs, checksums |
| FAIR and reuse | README, license, citation, standard formats |
| Research software engineering | package, CLI, API, tests, CI |
| Reproducibility | environment file and recorded run folder |
| Verification | importer, split, model, workflow, and dashboard tests |
| Validation | complete NACA profiles held out from training |
| Heterogeneous workflow | flow5 analysis and Python model pipeline |
| Scientific visualisation | interactive coefficient dashboard |

## Limitations

The model approximates flow5 outputs; it is not validated against wind-tunnel
measurements. Predictions should remain within the training domain. Reynolds
variation is sampled at three levels, and Random Forest predictions between
those levels are piecewise estimates.

## License And Citation

The software is released under the MIT License. Citation information is in
`CITATION.cff`.
