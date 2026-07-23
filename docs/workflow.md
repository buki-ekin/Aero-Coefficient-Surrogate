# Workflow Guide

## Boundary Between flow5 And Python

The flow5 stage is external because it requires the flow5 application. The
repository supplies NACA geometry files, three XML scripts, expected output
metadata, project files, logs, and exported polars.

After the exports exist, every Python stage is automated:

1. verify or regenerate raw checksums
2. import polar exports
3. validate the processed dataset
4. run the fixed grouped holdout
5. run repeated grouped validation and baselines
6. train the deployment model
7. generate scientific figures
8. export the self-contained dashboard

## Snakemake

```bash
snakemake -s workflow/Snakefile --cores 1
```

Snakemake provides dependency tracking, up-to-dateness checks, and
checkpoint-like restart behavior for the Python stages. It intentionally does
not pretend that flow5 can be executed portably on every platform.

## Manual Equivalent

The exact commands are recorded in the README. Every run is placed in a
separate directory so results are not silently mixed.
