#!/usr/bin/env python3
"""Check Python imports needed by the phase-1 SignalMC-MED workflow."""

from __future__ import annotations

import importlib.util


MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "wfdb",
    "neurokit2",
    "torch",
    "pyPPG",
]


def main() -> int:
    missing = []
    for module in MODULES:
        ok = importlib.util.find_spec(module) is not None
        print(f"{'OK     ' if ok else 'MISSING'} {module}")
        if not ok:
            missing.append(module)
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

