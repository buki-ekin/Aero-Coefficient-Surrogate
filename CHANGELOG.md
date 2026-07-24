# Changelog

## 1.2.0 - 2026-07-24

- Added a multi-stage, non-root Docker image for the bundled prediction
  package.
- Added container build and prediction checks to GitHub Actions.
- Added tag-triggered GitHub Container Registry publishing with SBOM and build
  provenance.
- Documented the container's reproducibility boundary and usage.
- Restored the declared Python 3.10 compatibility with a TOML parser fallback.

## 1.1.0 - 2026-07-23

- Added three-times repeated five-fold grouped validation.
- Added linear-regression and mean baselines on identical folds.
- Added confidence intervals and five reproducible scientific figures.
- Added strict dataset integrity checks and prediction-domain warnings.
- Bundled the deployment model, model-ready dataset, and HTML dashboard.
- Expanded the suite to 20 tests with 89% measured coverage.
- Added linting, clean-wheel installation, and multi-version CI.
- Added environment locking, FAIR documentation, CodeMeta, security guidance,
  and a Snakemake workflow for the Python stages.

## 1.0.0 - 2026-07-22

- Added the final 27-airfoil flow5 dataset at three Reynolds numbers.
- Added the final Random Forest surrogate.
- Added grouped validation with complete NACA profiles held out.
- Added run logs, checksums, deployment metadata, tests, and the HTML predictor.
