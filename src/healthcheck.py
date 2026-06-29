#!/usr/bin/env python3
"""Project environment and data sanity checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "xarray",
    "scipy",
    "sklearn",
    "xgboost",
]

OPTIONAL_PACKAGES = [
    "matplotlib",
    "jupyter",
]

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "noaaSSTcoralData.nc"
PIPELINE_FILE = ROOT / "src" / "train.py"
NOTEBOOK_FILE = ROOT / "notebooks" / "exploration.ipynb"
PROJECT_VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def is_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> int:
    failed = False

    print("== Python Environment ==")
    print(f"Executable: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    if PROJECT_VENV_PYTHON.exists():
        using_project_venv = Path(sys.executable).resolve() == PROJECT_VENV_PYTHON.resolve()
        print(
            f"[{'OK' if using_project_venv else 'WARN'}] project virtualenv: "
            f"{'active' if using_project_venv else f'available at {PROJECT_VENV_PYTHON.relative_to(ROOT)}'}"
        )

    print("== Dependency Check ==")
    for module in REQUIRED_PACKAGES:
        installed = is_installed(module)
        print(f"[{'OK' if installed else 'MISSING'}] {module}")
        if not installed:
            failed = True

    print("\n== Optional Packages ==")
    for module in OPTIONAL_PACKAGES:
        installed = is_installed(module)
        print(f"[{'OK' if installed else 'MISSING'}] {module}")

    print("\n== Project Files ==")
    for path in [DATA_FILE, PIPELINE_FILE]:
        exists = path.exists()
        print(f"[{'OK' if exists else 'MISSING'}] {path.relative_to(ROOT)}")
        if not exists:
            failed = True

    notebook_exists = NOTEBOOK_FILE.exists()
    print(f"[{'OK' if notebook_exists else 'WARN'}] notebooks/exploration.ipynb")

    print("\nResult:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
