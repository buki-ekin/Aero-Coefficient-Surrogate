# flow5 Analysis Record

## Settings

- flow5 version: 7.57
- analysis: fixed-speed foil polar
- Reynolds numbers: 500,000; 1,000,000; 2,000,000
- Mach number: 0
- Ncrit: 9
- angle of attack: -6 to 14 degrees, step 1 degree
- maximum XFoil iterations: 100
- foil panels: 200

The three scripts are:

```text
data/raw/flow5_alpha_sweep/aerosurrogate_flow5_re500000.xml
data/raw/flow5_alpha_sweep/aerosurrogate_flow5_re1000000.xml
data/raw/flow5_alpha_sweep/aerosurrogate_flow5_re2000000.xml
```

## Airfoils

```text
0006 0008 0009 0010 0012 0015 0018 0021 0024
1408 1410 1412
2408 2410 2412 2414 2415 2418 2421 2424
4412 4415 4418 4421 4424
6409 6412
```

The original request repeated NACA 0012 and 2415. Each profile is included once,
giving 27 unique airfoils.

## Outputs

flow5 produced 81 polar files. The complete requested grid would contain 1,701
rows. The processed dataset contains 1,618 converged rows; 83 unconverged points
remain absent.

## Import

```bash
aero-surrogate import-flow5 \
  --directory data/raw/flow5_exports_multi_re \
  --metadata data/raw/flow5_exports_multi_re/metadata.csv \
  --output data/processed/flow5_airfoils.csv
```
