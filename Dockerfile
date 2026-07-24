FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip wheel --wheel-dir /wheels .


FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="AeroSurrogate" \
      org.opencontainers.image.description="Reproducible flow5 airfoil surrogate prediction package" \
      org.opencontainers.image.source="https://github.com/buki-ekin/SCE-PROJECT" \
      org.opencontainers.image.version="1.2.0" \
      org.opencontainers.image.licenses="MIT"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN addgroup --system aerosurrogate \
    && adduser --system --ingroup aerosurrogate \
        --home /home/aerosurrogate aerosurrogate

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels aero-surrogate==1.2.0 \
    && rm -rf /wheels

USER aerosurrogate
WORKDIR /home/aerosurrogate

ENTRYPOINT ["aero-surrogate"]
CMD ["--help"]
