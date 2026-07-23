"""Import utilities for flow5 airfoil polar exports."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

FLOW5_TARGET_COLUMNS = {
    "alpha": "alpha_deg",
    "cl": "cl",
    "cd": "cd",
    "cm": "cm",
}


def naca4_parameters(code: str) -> dict[str, float | str]:
    """Convert a NACA 4-digit code into normalized geometry parameters."""

    match = re.search(r"(\d{4})", code)
    if not match:
        raise ValueError(f"Expected a NACA 4-digit code, got: {code}")

    digits = match.group(1)
    return {
        "naca": f"NACA {digits}",
        "camber": int(digits[0]) / 100.0,
        "camber_position": int(digits[1]) / 10.0,
        "thickness": int(digits[2:]) / 100.0,
    }


def read_flow5_polar(
    path: str | Path,
    naca_code: str,
    reynolds: float,
) -> pd.DataFrame:
    """Read one flow5/XFoil-style polar export and attach input metadata.

    flow5 polar exports are text files with a human-readable header followed by
    numeric columns. This parser accepts whitespace, comma, or semicolon
    separated rows as long as the numeric table contains alpha, CL, CD, and Cm.
    """

    source = Path(path)
    geometry = naca4_parameters(naca_code)
    rows: list[dict[str, float | str]] = []

    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        values = _numeric_values(line)
        if len(values) < 5:
            continue

        alpha_deg, cl, cd = values[:3]
        cm = values[4]
        rows.append(
            {
                **geometry,
                "alpha_deg": alpha_deg,
                "reynolds": float(reynolds),
                "cl": cl,
                "cd": cd,
                "cm": cm,
                "source_file": source.name,
            }
        )

    if not rows:
        raise ValueError(f"No polar rows found in {source}")

    return pd.DataFrame(rows)


def import_flow5_directory(
    input_dir: str | Path,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Import multiple flow5 polar exports using a metadata table.

    The metadata table must contain `file`, `naca`, and `reynolds` columns.
    """

    required = {"file", "naca", "reynolds"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"flow5 metadata is missing required columns: {missing}")

    base = Path(input_dir)
    frames = []
    for row in metadata.to_dict(orient="records"):
        frames.append(
            read_flow5_polar(
                base / str(row["file"]),
                naca_code=str(row["naca"]),
                reynolds=float(row["reynolds"]),
            )
        )

    return pd.concat(frames, ignore_index=True)


def _numeric_values(line: str) -> list[float]:
    parts = re.split(r"[\s,;]+", line.strip())
    values = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return []
    return values
