# Data

The final data source is flow5 v7.57.

```text
raw/flow5_alpha_sweep/
    27 NACA geometry files and three XML analysis scripts

raw/flow5_exports_multi_re/
    81 polar text files, three flow5 projects, three logs, and metadata.csv

raw/raw_manifest.json
    SHA-256 checksums for the final raw files

processed/flow5_airfoils.csv
    1,618 converged model-ready rows
```

## Processed Columns

Inputs:

- `camber`
- `camber_position`
- `thickness`
- `alpha_deg`
- `reynolds`

Outputs:

- `cl`
- `cd`
- `cm`

Provenance columns:

- `naca`
- `source_file`

## Rebuild

```bash
aero-surrogate import-flow5 \
  --directory data/raw/flow5_exports_multi_re \
  --metadata data/raw/flow5_exports_multi_re/metadata.csv \
  --output data/processed/flow5_airfoils.csv
```

The requested sweep contains 1,701 operating points. flow5 returned 1,618
converged rows. The missing 83 points are not filled or estimated.

The per-condition coverage heatmap is generated at:

```text
runs/final_grouped_cv_naca_seed42/reports/figures/data_coverage.png
```

## Integrity Checks

Before data can be saved or used for training, Aero Coefficient Surrogate checks:

- required columns
- finite numeric values
- non-empty NACA identifiers
- duplicate NACA/Reynolds/angle operating points
- broad physical plausibility ranges, including non-negative drag
