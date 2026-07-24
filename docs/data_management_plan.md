# Data Management Plan

## Data Source

The project uses flow5 v7.57 polar exports for 27 NACA airfoils. No synthetic
rows are used in the final dataset.

## Storage

- NACA inputs, XML scripts, flow5 projects, logs, and polar files: `data/raw/`
- cleaned model table: `data/processed/flow5_airfoils.csv`
- trained models: `models/` and the final run folder
- metrics and provenance: `runs/final_grouped_naca_seed42/`

Raw files are not edited after export. Processed data can be rebuilt with the
documented import command.

## Metadata

Each processed row contains NACA code, normalized geometry, angle of attack,
Reynolds number, CL, CD, CM, and source filename. The analysis settings and
flow5 version are documented in the README and XML scripts.

## Integrity And Versions

SHA-256 manifests are stored for raw inputs, processed data, and the deployment
model. Git tracks source and documentation versions. Final releases use a
version tag. Dataset validation rejects incomplete, non-finite, duplicated, or
physically implausible model rows before training.

## Backup And Archive

Tracked code, data, documentation, and final artifacts are copied to the public
GitHub remote. The local repository is the working copy. This provides two
independent copies for the submitted material, but it is not a long-term
research archive.

For final publication, version 1.2.0 should be released on GitHub and archived
in Zenodo. The resulting DOI must then be added to `CITATION.cff`,
`codemeta.json`, and the README. No DOI is claimed before that deposit exists.

The Python delivery stage also has a versioned Docker build definition. The
container improves execution isolation but does not replace the raw-data
archive, Git history, manifests, or Zenodo deposit.

## Access And Reuse

CSV, JSON, TXT, XML, and DAT are used for portable data exchange. Software is
MIT licensed and citation information is provided in `CITATION.cff`.

## Retention

The final raw files, processed dataset, reproducible run, model, and dashboard
are kept together in the release. Temporary render files and interim datasets
are excluded. The intended minimum retention period is ten years after project
completion, subject to the continued availability of the repository or its
archival deposit.

## Responsibility

The project author is responsible for data integrity, versioning, release
creation, and responding to reuse questions. flow5 remains responsible for the
reference aerodynamic calculation; AeroSurrogate records rather than hides
that dependency.
