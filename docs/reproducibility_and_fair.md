# Reproducibility And FAIR Checklist

## Findable

- repository has a stable public URL
- package, run, data, and model names are consistent
- `CITATION.cff` and `codemeta.json` describe the software
- no DOI has been minted yet

## Accessible

- source, documentation, processed data, and final artifacts are stored in the
  repository
- a wheel can be built and installed without the source tree
- the wheel contains the model, example data, and dashboard
- a non-root Docker image provides an isolated prediction environment
- long-term archive access is not yet guaranteed

## Interoperable

- tabular data: CSV
- metadata and metrics: JSON
- geometry and solver configuration: DAT and XML
- human-readable raw exports and logs: TXT
- visual results: PNG
- interactive result: self-contained HTML

The pickle model is Python-specific and should be considered less
interoperable than the other products.

## Reusable

- MIT software license
- citation metadata
- defined model inputs, outputs, and limitations
- automated tests and clean-install CI
- exact final environment lock
- multi-stage container build checked in continuous integration
- raw and processed checksums
- public API, CLI, example, and dashboard

## Provenance

Each final run records:

- run configuration and random seed
- Git commit
- Python, operating system, and dependency versions
- checksums and sizes of all input files
- held-out airfoils and fold assignments
- fold-level and summary metrics
- prediction residuals
- validation and deployment model checksums
- runtime and plain-text log

## Remaining FAIR Work

1. Create a GitHub release for version 1.2.0.
2. Archive that release in Zenodo.
3. Add the resulting DOI to the README, citation, and CodeMeta records.
4. If practical, publish the final data as a separately citable dataset.
5. Consider a non-pickle portable model representation.
