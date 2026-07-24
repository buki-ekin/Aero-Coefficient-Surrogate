"""Prepare flow5 input files for the wider AeroSurrogate alpha sweep."""

from __future__ import annotations

import math
from pathlib import Path


NACA_CODES = [
    "0006",
    "0008",
    "0009",
    "0010",
    "0012",
    "0015",
    "0018",
    "0021",
    "0024",
    "1408",
    "1410",
    "1412",
    "2408",
    "2410",
    "2412",
    "2414",
    "2415",
    "2418",
    "2421",
    "2424",
    "4412",
    "4415",
    "4418",
    "4421",
    "4424",
    "6409",
    "6412",
]

REYNOLDS_VALUES = (500_000, 1_000_000, 2_000_000)
ALPHA_MIN = -6
ALPHA_MAX = 14
ALPHA_STEP = 1
NCRIT = 9
MACH = 0.0
N_PANELS = 200
N_COORDINATE_POINTS = 120

ROOT = Path("data/raw/flow5_alpha_sweep")
FOIL_DIR = ROOT / "foils"
EXPORT_DIR = Path("data/raw/flow5_exports_multi_re")
METADATA_PATH = EXPORT_DIR / "metadata.csv"


def naca4_coordinates(code: str, n_points: int = 160) -> list[tuple[float, float]]:
    """Return closed NACA 4-digit coordinates in common .dat ordering."""

    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Expected a NACA 4-digit code, got: {code}")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = [math.pi * i / (n_points - 1) for i in range(n_points)]
    x_values = [(1.0 - math.cos(angle)) / 2.0 for angle in beta]

    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []

    for x in x_values:
        yt = 5.0 * t * (
            0.2969 * math.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

        if m == 0.0 or p == 0.0:
            yc = 0.0
            dyc_dx = 0.0
        elif x < p:
            yc = m / p**2 * (2.0 * p * x - x**2)
            dyc_dx = 2.0 * m / p**2 * (p - x)
        else:
            yc = m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2)
            dyc_dx = 2.0 * m / (1.0 - p) ** 2 * (p - x)

        theta = math.atan(dyc_dx)
        xu = x - yt * math.sin(theta)
        yu = yc + yt * math.cos(theta)
        xl = x + yt * math.sin(theta)
        yl = yc - yt * math.cos(theta)
        upper.append((xu, yu))
        lower.append((xl, yl))

    return list(reversed(upper)) + lower[1:]


def write_dat_file(code: str) -> Path:
    """Write one NACA .dat file for flow5."""

    path = FOIL_DIR / f"NACA_{code}.dat"
    coords = naca4_coordinates(code, n_points=N_COORDINATE_POINTS)
    lines = [f"NACA {code}"]
    lines.extend(f"{x:.8f} {y:.8f}" for x, y in coords)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_name(reynolds: int) -> str:
    """Return the stable flow5 run name for one Reynolds number."""

    return f"aerosurrogate_flow5_re{reynolds}"


def polar_filename(reynolds: int) -> str:
    """Return flow5's fixed-speed polar filename for one Reynolds number."""

    return f"T1-Re{reynolds / 1_000_000:.3f}-N{NCRIT:.1f}.txt"


def write_flow5_script(dat_files: list[Path], reynolds: int) -> Path:
    """Write one flow5 XML script for a fixed-Reynolds alpha sweep."""

    foil_entries = "\n".join(
        f"      <Foil_File_Name>{path.name}</Foil_File_Name>" for path in dat_files
    )
    name = run_name(reynolds)
    script_path = ROOT / f"{name}.xml"
    script = f"""<?xml version="1.0" encoding="UTF-8"?>
<xflscript version="1.0">
  <Metadata>
    <make_project_file>true</make_project_file>
    <project_file_name>{name}.fl5</project_file_name>
    <polar_text_output_format>txt</polar_text_output_format>
    <Directories>
      <output_dir>{EXPORT_DIR.as_posix()}</output_dir>
      <foil_files_dir>{FOIL_DIR.as_posix()}</foil_files_dir>
      <recursive_scan>false</recursive_scan>
    </Directories>
    <MultiThreading>
      <Allow_Multithreading>true</Allow_Multithreading>
      <max_threads>4</max_threads>
      <Thread_Priority>Normal</Thread_Priority>
    </MultiThreading>
  </Metadata>
  <foil_analysis>
    <Foil_Files>
{foil_entries}
    </Foil_Files>
    <Batch_Analysis_Data>
      <Polar_Type>FIXEDSPEEDPOLAR</Polar_Type>
      <Forced_Top_Transition>1.0</Forced_Top_Transition>
      <Forced_Bottom_Transition>1.0</Forced_Bottom_Transition>
      <Batch_Range>
        <Reynolds>{reynolds}</Reynolds>
        <Mach>{MACH}</Mach>
        <NCrit>{NCRIT}</NCrit>
      </Batch_Range>
    </Batch_Analysis_Data>
    <OpPoint_Range>
      <Alpha>0,{ALPHA_MAX},{ALPHA_STEP}</Alpha>
      <Alpha>0,{ALPHA_MIN},{ALPHA_STEP}</Alpha>
      <Spec_Alpha>true</Spec_Alpha>
      <From_Zero>true</From_Zero>
    </OpPoint_Range>
    <Options>
      <Max_XFoil_Iterations>100</Max_XFoil_Iterations>
      <Repanel_Foils>true</Repanel_Foils>
      <Foil_Panels>{N_PANELS}</Foil_Panels>
    </Options>
    <Output>
      <make_polars_text_file>true</make_polars_text_file>
      <make_polars_bin_file>false</make_polars_bin_file>
      <make_oppoints>false</make_oppoints>
    </Output>
  </foil_analysis>
</xflscript>
"""
    script_path.write_text(script, encoding="utf-8")
    return script_path


def write_expected_metadata() -> Path:
    """Write metadata for the polar files generated by this flow5 script."""

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["file,naca,reynolds"]
    for reynolds in REYNOLDS_VALUES:
        for code in NACA_CODES:
            naca = f"NACA {code}"
            relative_path = (
                f"{run_name(reynolds)}/Foil_polars/{naca}/"
                f"{polar_filename(reynolds)}"
            )
            lines.append(f'"{relative_path}","{naca}",{reynolds}')
    METADATA_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return METADATA_PATH


def main() -> None:
    FOIL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    dat_files = [write_dat_file(code) for code in NACA_CODES]
    script_paths = [
        write_flow5_script(dat_files, reynolds) for reynolds in REYNOLDS_VALUES
    ]
    metadata_path = write_expected_metadata()
    print(f"Wrote {len(dat_files)} foil files to {FOIL_DIR}")
    for script_path in script_paths:
        print(f"Wrote flow5 script to {script_path}")
    print(f"Wrote expected metadata to {metadata_path}")
    print(f"Expected flow5 output root: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
