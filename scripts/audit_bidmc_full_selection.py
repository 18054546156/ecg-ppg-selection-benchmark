#!/usr/bin/env python3
"""Audit a BIDMC full all-method selection benchmark result directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "selection_runs.csv",
    "selection_status.csv",
    "selection_summary.csv",
    "selection_summary.json",
    "selected_indices.json",
    "run_config.json",
    "audit_manifest.json",
    "reproduce_command.sh",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/bidmc_selection_full")
    parser.add_argument("--min-source-segments", type=int, default=2500)
    parser.add_argument("--min-train-fraction", type=float, default=0.50)
    parser.add_argument("--expected-test-fraction", type=float, default=0.30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    require(out_dir.exists(), f"missing result directory: {out_dir}")
    for name in REQUIRED_FILES:
        require((out_dir / name).exists(), f"missing required file: {out_dir / name}")

    summary = json.loads((out_dir / "selection_summary.json").read_text(encoding="utf-8"))
    metadata = summary.get("metadata", {})
    source_segments = int(metadata.get("source_segments", 0))
    train_segments = int(metadata.get("train_segments", 0))
    test_segments = int(metadata.get("test_segments", 0))
    require(source_segments >= args.min_source_segments, f"source_segments too small: {source_segments}")
    require(train_segments >= args.min_train_fraction * source_segments, f"train pool too small: {train_segments}/{source_segments}")
    require(test_segments > 0, "test split is empty")
    observed_test_fraction = test_segments / max(1, source_segments)
    require(
        abs(observed_test_fraction - args.expected_test_fraction) <= 0.08,
        f"test fraction unexpected: {observed_test_fraction:.3f}",
    )
    require(metadata.get("result_level") == "full_bidmc_train_pool_migration_benchmark", "result_level is not full benchmark")

    status_df = pd.read_csv(out_dir / "selection_status.csv")
    require("full_train_reference" in set(status_df["method"]), "missing full_train_reference")
    full_status = status_df.loc[status_df["method"] == "full_train_reference", "status"].iloc[0]
    require(full_status == "passed", f"full_train_reference did not pass: {full_status}")
    require("preselect" in set(status_df["method"]), "missing preselect task-mismatch row")
    preselect_status = status_df.loc[status_df["method"] == "preselect", "status"].iloc[0]
    require(preselect_status == "not_run_task_mismatch", f"preselect status unexpected: {preselect_status}")

    runs_df = pd.read_csv(out_dir / "selection_runs.csv")
    require(len(runs_df) > 0, "selection_runs.csv is empty")
    require((runs_df["test_n"] > 0).all(), "some runs have empty test_n")

    passed_methods = sorted(status_df.loc[status_df["status"] == "passed", "method"].unique())
    print("AUDIT PASS")
    print(f"source_segments={source_segments} train_segments={train_segments} test_segments={test_segments}")
    print("passed_methods=" + ",".join(passed_methods))
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
