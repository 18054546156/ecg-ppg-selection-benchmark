#!/usr/bin/env python3
"""Check whether required remote data/checkpoint files exist."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcmed-dir", default="/root/project/data/mc-med")
    parser.add_argument("--signalmcmed-dir", default="/root/project/data/signalmcmed")
    parser.add_argument("--checkpoint-dir", default="/root/project/checkpoints")
    args = parser.parse_args()

    required = [
        Path(args.mcmed_dir) / "visits.csv",
        Path(args.mcmed_dir) / "labs.csv",
        Path(args.mcmed_dir) / "pmh.csv",
        Path(args.mcmed_dir) / "split_chrono_train.csv",
        Path(args.mcmed_dir) / "split_chrono_val.csv",
        Path(args.mcmed_dir) / "split_chrono_test.csv",
        Path(args.mcmed_dir) / "waveforms",
        Path(args.signalmcmed_dir) / "signalmc-med_csns.pkl",
    ]
    optional_checkpoints = [
        Path(args.checkpoint_dir) / "papagei_s.pt",
        Path(args.checkpoint_dir) / "1_lead_ECGFounder.pth",
        Path(args.checkpoint_dir) / "D-BETA" / "checkpoints" / "pytorch_model.bin",
        Path(args.checkpoint_dir) / "csfm-base.pt",
    ]

    missing = [str(path) for path in required if not path.exists()]
    missing_optional = [str(path) for path in optional_checkpoints if not path.exists()]

    print("Required data:")
    for path in required:
        print(f"  {'OK     ' if path.exists() else 'MISSING'} {path}")
    print("Optional checkpoints:")
    for path in optional_checkpoints:
        print(f"  {'OK     ' if path.exists() else 'MISSING'} {path}")

    if missing:
        print("\nCannot run SignalMC-MED official scripts until required data is present.")
        return 2
    if missing_optional:
        print("\nData is present, but some model-specific feature extraction jobs will be skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

