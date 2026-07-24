# Container Guide

The Docker image provides an isolated, non-root Python environment containing
the AeroSurrogate package, deployment model, example dataset, and
self-contained dashboard.

The container covers the reproducible Python prediction and delivery stage. It
does not contain or automate the external flow5 GUI/reference-simulation stage.

## Build

```bash
docker build --tag aerosurrogate:1.2.0 .
```

The multi-stage build creates all Python wheels in a builder image and installs
them into the runtime image without contacting a package index. Only the
installed package and its runtime dependencies remain in the final image.

## Pull A Released Image

Version tags publish the tested image to GitHub Container Registry with an SBOM
and build-provenance attestation:

```bash
docker pull ghcr.io/buki-ekin/aerosurrogate:1.2.0
```

The registry artifact appears only after the corresponding `v1.2.0` Git tag has
been pushed and the release workflow has completed.

## Predict

```bash
docker run --rm aerosurrogate:1.2.0 \
  predict \
  --naca 2412 \
  --alpha-deg 4 \
  --reynolds 1000000
```

Replace `aerosurrogate:1.2.0` with
`ghcr.io/buki-ekin/aerosurrogate:1.2.0` to run the published image.

Expected coefficient values, allowing for display rounding:

```text
cl: 0.70149
cd: 0.00705
cm: -0.05329
```

## Export The Dashboard

Create a writable host directory and mount it into the container:

```bash
mkdir -p container-output
docker run --rm \
  --volume "$PWD/container-output:/output" \
  aerosurrogate:1.2.0 \
  export-bundled-dashboard \
  --output /output/aerosurrogate_dashboard.html
```

If the host directory does not permit writes by the container's non-root user,
export inside the container and copy the file with `docker cp` instead.

## Reproducibility Boundary

- Python version and operating-system family are defined by the base image.
- Python dependencies are resolved into wheels during the builder stage.
- The package, model, example data, and dashboard are embedded in the image.
- The process runs as an unprivileged user.
- GitHub Actions builds the image and runs a known prediction on every push and
  pull request.
- The base-image tag can receive upstream updates. The Git commit and release
  tag remain necessary provenance identifiers; a published image digest would
  be required for bit-identical container retrieval.
