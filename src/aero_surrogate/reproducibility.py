"""Small helpers for recording reproducible computational experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    """Return the SHA-256 checksum for a file."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Create checksum metadata for important input files."""

    manifest = []
    for path in paths:
        file_path = Path(path)
        manifest.append(
            {
                "path": file_path.as_posix(),
                "sha256": sha256(file_path),
                "size_bytes": file_path.stat().st_size,
            }
        )
    return manifest


def environment_metadata(project_root: str | Path | None = None) -> dict[str, Any]:
    """Collect lightweight environment and version metadata."""

    root = Path(project_root) if project_root else Path.cwd()
    packages = {"aero-surrogate": _project_version(root)}
    for package in ("numpy", "pandas", "scikit-learn"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_commit(project_root),
    }


def _project_version(project_root: Path) -> str:
    pyproject_path = project_root / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as file:
            return str(tomllib.load(file)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        try:
            return version("aero-surrogate")
        except PackageNotFoundError:
            return "not-installed"


def git_commit(project_root: str | Path | None = None) -> str | None:
    """Return the current git commit hash when the project is in a repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root) if project_root else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> Path:
    """Write stable, readable JSON to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path
