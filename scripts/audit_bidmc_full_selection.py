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
    "formal_bidmc_table.csv",
    "formal_bidmc_table.md",
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
    selected_indices = json.loads((out_dir / "selected_indices.json").read_text(encoding="utf-8"))
    splits = metadata.get("splits", [])
    require(splits, "metadata.splits is empty")
    for split in splits:
        overlap = set(split.get("record_overlap", []))
        require(not overlap, f"train/test record leakage for split_seed={split.get('split_seed')}: {sorted(overlap)}")
        require(int(split.get("train_segments", 0)) > 0, f"empty train split for split_seed={split.get('split_seed')}")
        require(int(split.get("test_segments", 0)) > 0, f"empty test split for split_seed={split.get('split_seed')}")

    require("full_train_reference" in set(status_df["method"]), "missing full_train_reference")
    full_statuses = set(status_df.loc[status_df["method"] == "full_train_reference", "status"])
    require(full_statuses == {"passed"}, f"full_train_reference did not pass for every split: {full_statuses}")
    require("preselect" in set(status_df["method"]), "missing preselect task-mismatch row")
    preselect_status = status_df.loc[status_df["method"] == "preselect", "status"].iloc[0]
    require(preselect_status == "not_run_task_mismatch", f"preselect status unexpected: {preselect_status}")

    selection_status = status_df[
        ~status_df["method"].isin(["full_train_reference", "preselect"])
    ].copy()
    require(len(selection_status) > 0, "no selection method status rows")
    required_status_cols = {
        "selected_key",
        "requested_budget",
        "raw_selected_n",
        "selected_n",
        "budget_met",
        "train_n",
        "test_n",
        "fill_strategy",
        "note",
    }
    missing_status_cols = sorted(required_status_cols - set(selection_status.columns))
    require(not missing_status_cols, f"selection_status missing columns: {missing_status_cols}")
    passed = selection_status[selection_status["status"] == "passed"]
    require(len(passed) > 0, "no passed selector rows")
    for _, row in passed.iterrows():
        method = str(row["method"])
        key = str(row["selected_key"])
        require(key in selected_indices, f"missing selected indices for passed row: {key}")
        indices = selected_indices[key]
        requested = int(row["requested_budget"])
        selected_n = int(row["selected_n"])
        train_n = int(row["train_n"])
        require(selected_n == requested, f"{key} selected_n={selected_n} requested={requested}")
        require(len(indices) == selected_n, f"{key} selected_indices length mismatch")
        require(len(set(indices)) == len(indices), f"{key} has duplicate indices")
        require(all(isinstance(idx, int) for idx in indices), f"{key} contains non-int indices")
        require(all(0 <= idx < train_n for idx in indices), f"{key} has out-of-range train indices")
        budget_met = row["budget_met"]
        if isinstance(budget_met, str):
            budget_met = budget_met.lower() == "true"
        require(bool(budget_met), f"{key} did not meet requested budget")
        if method == "probcover_official":
            require(str(row["fill_strategy"]) in {"none", "k_center_unselected"}, f"{key} unexpected ProbCover fill strategy")
        else:
            require(int(row.get("filled_n", 0)) == 0, f"{key} unexpected non-ProbCover fill")

    failed = selection_status[selection_status["status"] == "failed"]
    for _, row in failed.iterrows():
        require(str(row.get("note", "")).strip(), f"failed row lacks note: {row.to_dict()}")

    runs_df = pd.read_csv(out_dir / "selection_runs.csv")
    require(len(runs_df) > 0, "selection_runs.csv is empty")
    require((runs_df["test_n"] > 0).all(), "some runs have empty test_n")
    if "requested_budget" in runs_df.columns:
        non_full_runs = runs_df[runs_df["method"] != "full_train_reference"]
        require(
            (non_full_runs["selected_n"].astype(int) == non_full_runs["requested_budget"].astype(int)).all(),
            "some passed runs did not meet requested_budget",
        )

    passed_methods = sorted(status_df.loc[status_df["status"] == "passed", "method"].unique())
    print("AUDIT PASS")
    print(f"source_segments={source_segments} train_segments={train_segments} test_segments={test_segments}")
    print("passed_methods=" + ",".join(passed_methods))
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
