"""Write checksums for the final flow5 inputs and exports."""

from pathlib import Path

from aero_surrogate.reproducibility import input_manifest, write_json


RAW_ROOT = Path("data/raw")
SOURCE_DIRS = [
    RAW_ROOT / "flow5_alpha_sweep",
    RAW_ROOT / "flow5_exports_multi_re",
]


def main() -> None:
    files = [
        path
        for directory in SOURCE_DIRS
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {".DS_Store", "raw_manifest.json"}
    ]
    output = write_json(RAW_ROOT / "raw_manifest.json", input_manifest(files))
    print(f"Wrote checksums for {len(files)} raw files to {output}")


if __name__ == "__main__":
    main()
