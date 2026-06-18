#!/usr/bin/env python3
"""Run the full BIDMC all-method data-selection benchmark.

This entry point is intentionally separate from the smoke test. It always uses
the full record-level training pool after the 70/30 split and records the exact
configuration needed for later audit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from official_selection_adapters import select_probcover_official, select_typiclust_official
from run_bidmc_all_selection_smoke import (
    evaluate_selected,
    make_class_bins,
    select_cords_craig,
    select_cords_glister,
    select_cords_gradmatch,
    select_elfs_core_official,
    select_moderate_official,
    validate_indices,
)
from run_bidmc_selection_benchmark import (
    prepare_matrix,
    probcover_delta,
    select_k_center,
    select_random,
    split_by_record,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = [
    "random",
    "k_center",
    "typiclust_official",
    "probcover_official",
    "moderate_coreset_official_core",
    "elfs_official_core_proxy_score",
    "cords_gradmatch",
    "cords_glister",
    "cords_craig",
]


def parse_csv_floats(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return parsed


def parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return parsed


def parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(parsed) - set(DEFAULT_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown methods: {', '.join(unknown)}")
    return parsed


def build_feature_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ppg_cols = [col for col in df.columns if col.startswith("ppg_") and col != "ppg_channel"]
    ecg_cols = [col for col in df.columns if col.startswith("ecg_") and col not in {"ecg_channel", "ecg_hr_bpm"}]
    papagei_cols = [col for col in df.columns if col.startswith("papagei_")]
    feature_sets = {
        "ppg_handcrafted": ppg_cols,
        "ecg_ppg_handcrafted": ecg_cols + ppg_cols,
        "papagei": papagei_cols,
    }
    selection_sets = {
        "papagei": papagei_cols,
        "handcrafted": ppg_cols,
        "ecg_ppg_handcrafted": ecg_cols + ppg_cols,
    }
    return feature_sets, selection_sets


def summarize_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    metric_cols = ["mae", "pearson", "selected_n", "selected_records", "selected_record_fraction", "train_n", "test_n"]
    for keys, group in runs_df.groupby(["method", "budget_fraction", "feature_set"], dropna=False):
        row = dict(zip(["method", "budget_fraction", "feature_set"], keys))
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0)) if len(group) > 1 else 0.0
        row["runs"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feature_set", "budget_fraction", "mae_mean", "method"])


def make_selector(
    method: str,
    x_select: np.ndarray,
    labels: np.ndarray,
    train_n: int,
    budget: int,
    seed: int,
    args: argparse.Namespace,
):
    if method == "random":
        return select_random(train_n, budget, seed)
    if method == "k_center":
        return select_k_center(x_select, budget, seed)
    if method == "typiclust_official":
        return select_typiclust_official(
            x_select,
            budget,
            seed,
            min_cluster_size=args.typiclust_min_cluster_size,
            max_clusters=args.typiclust_max_clusters,
            k_nn=args.typiclust_k_nn,
            verbose=args.official_selector_verbose,
        )
    if method == "probcover_official":
        delta = probcover_delta(x_select, args.probcover_k_nn, args.probcover_quantile)
        return select_probcover_official(x_select, budget, delta, verbose=args.official_selector_verbose)[0]
    if method == "moderate_coreset_official_core":
        return select_moderate_official(x_select, labels, budget)
    if method == "elfs_official_core_proxy_score":
        return select_elfs_core_official(x_select, labels, budget)
    if method == "cords_gradmatch":
        return select_cords_gradmatch(x_select, labels, budget, seed, args.batch_size)
    if method == "cords_glister":
        return select_cords_glister(x_select, labels, budget, seed, args.batch_size)
    if method == "cords_craig":
        return select_cords_craig(x_select, labels, budget, seed, args.batch_size)
    raise ValueError(f"unknown method: {method}")


def write_audit_files(
    out_dir: Path,
    args: argparse.Namespace,
    metadata: dict,
    status_df: pd.DataFrame,
    command_argv: list[str],
) -> None:
    config = {
        "script": "scripts/run_bidmc_all_selection_benchmark.py",
        "argv": command_argv,
        "cwd": str(Path.cwd()),
        "environment": {
            "python": sys.executable,
            "pid": os.getpid(),
        },
        "parameters": {
            "features_csv": str(args.features_csv),
            "out_dir": str(args.out_dir),
            "selection_feature_set": args.selection_feature_set,
            "budget_fractions": args.budget_fractions,
            "seeds": args.seeds,
            "split_seed": args.split_seed,
            "test_fraction": args.test_fraction,
            "pca_dim": args.pca_dim,
            "methods": args.methods,
            "batch_size": args.batch_size,
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    audit = {
        "audit_status": "ready_for_review",
        "protocol_guardrails": [
            "record_level_split",
            "full_training_pool_after_split",
            "held_out_test_set_not_selected",
            "full_train_reference_reported",
            "preselect_recorded_as_task_mismatch_not_run",
        ],
        "metadata": metadata,
        "status_counts": status_df["status"].value_counts().to_dict() if not status_df.empty else {},
        "outputs": {
            "runs_csv": "selection_runs.csv",
            "status_csv": "selection_status.csv",
            "summary_csv": "selection_summary.csv",
            "summary_json": "selection_summary.json",
            "selected_indices_json": "selected_indices.json",
            "run_config_json": "run_config.json",
        },
    }
    (out_dir / "audit_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    reproduce = " ".join(command_argv)
    (out_dir / "reproduce_command.sh").write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{reproduce}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="results/bidmc/bidmc_segment_features.csv")
    parser.add_argument("--out-dir", default="results/bidmc_selection_full")
    parser.add_argument("--selection-feature-set", choices=["papagei", "handcrafted", "ecg_ppg_handcrafted"], default="papagei")
    parser.add_argument("--budget-fractions", type=parse_csv_floats, default=[0.10])
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2])
    parser.add_argument("--methods", type=parse_methods, default=list(DEFAULT_METHODS))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--typiclust-min-cluster-size", type=int, default=1)
    parser.add_argument("--typiclust-max-clusters", type=int, default=500)
    parser.add_argument("--typiclust-k-nn", type=int, default=10)
    parser.add_argument("--probcover-k-nn", type=int, default=10)
    parser.add_argument("--probcover-quantile", type=float, default=0.50)
    parser.add_argument("--official-selector-verbose", action="store_true")
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv).replace([np.inf, -np.inf], np.nan)
    feature_sets, selection_sets = build_feature_sets(df)
    selection_cols = selection_sets[args.selection_feature_set]
    if not selection_cols:
        raise RuntimeError(f"No columns found for selection feature set {args.selection_feature_set}")

    train_df, test_df = split_by_record(df, args.test_fraction, args.split_seed)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    required = train_df[["ecg_hr_bpm"] + selection_cols].replace([np.inf, -np.inf], np.nan)
    train_df = train_df[required.notna().all(axis=1)].reset_index(drop=True)

    if len(train_df) < 0.5 * len(df):
        raise RuntimeError(
            f"Training pool unexpectedly small: {len(train_df)} of {len(df)} segments. "
            "This script is for full-pool benchmarking, not smoke tests."
        )

    x_select = prepare_matrix(train_df, selection_cols, args.pca_dim, args.split_seed)
    labels = make_class_bins(train_df["ecg_hr_bpm"].to_numpy(dtype=float), num_bins=4)

    run_rows: list[dict] = []
    status_rows: list[dict] = []
    selected_indices: dict[str, list[int]] = {}
    for budget_fraction in args.budget_fractions:
        budget = max(1, int(math.ceil(float(budget_fraction) * len(train_df))))
        for seed in args.seeds:
            for method in args.methods:
                started = time.time()
                key = f"{method}|budget={budget_fraction:g}|seed={seed}"
                try:
                    selected = make_selector(method, x_select, labels, len(train_df), budget, seed, args)
                    selected = validate_indices(selected, len(train_df))
                    selected_indices[key] = selected.astype(int).tolist()
                    elapsed = time.time() - started
                    rows = evaluate_selected(method, train_df, test_df, selected, feature_sets)
                    for row in rows:
                        row["budget_fraction"] = float(budget_fraction)
                        row["seed"] = int(seed)
                    run_rows.extend(rows)
                    status_rows.append(
                        {
                            "method": method,
                            "budget_fraction": float(budget_fraction),
                            "seed": int(seed),
                            "status": "passed",
                            "elapsed_sec": elapsed,
                            "selected_n": int(len(selected)),
                            "note": "Full BIDMC train-pool benchmark.",
                        }
                    )
                except Exception as exc:
                    status_rows.append(
                        {
                            "method": method,
                            "budget_fraction": float(budget_fraction),
                            "seed": int(seed),
                            "status": "failed",
                            "elapsed_sec": time.time() - started,
                            "selected_n": 0,
                            "note": f"{type(exc).__name__}: {exc}",
                        }
                    )

    full_rows = evaluate_selected("full_train_reference", train_df, test_df, np.arange(len(train_df)), feature_sets)
    for row in full_rows:
        row["budget_fraction"] = 1.0
        row["seed"] = int(args.split_seed)
    run_rows.extend(full_rows)
    status_rows.append(
        {
            "method": "full_train_reference",
            "budget_fraction": 1.0,
            "seed": int(args.split_seed),
            "status": "passed",
            "elapsed_sec": 0.0,
            "selected_n": int(len(train_df)),
            "note": "Reference using the complete BIDMC training pool after record-level split.",
        }
    )
    status_rows.append(
        {
            "method": "preselect",
            "budget_fraction": np.nan,
            "seed": np.nan,
            "status": "not_run_task_mismatch",
            "elapsed_sec": 0.0,
            "selected_n": 0,
            "note": "Original PreSelect is text/LLM pretraining data selection, not ECG+PPG waveform/window selection.",
        }
    )

    runs_df = pd.DataFrame(run_rows)
    status_df = pd.DataFrame(status_rows)
    summary_df = summarize_runs(runs_df)

    runs_df.to_csv(out_dir / "selection_runs.csv", index=False)
    status_df.to_csv(out_dir / "selection_status.csv", index=False)
    summary_df.to_csv(out_dir / "selection_summary.csv", index=False)
    (out_dir / "selected_indices.json").write_text(json.dumps(selected_indices, indent=2), encoding="utf-8")

    metadata = {
        "dataset": "BIDMC PPG and Respiration Dataset",
        "source_segments": int(len(df)),
        "train_segments": int(len(train_df)),
        "test_segments": int(len(test_df)),
        "train_records": int(train_df["record"].nunique()),
        "test_records": int(test_df["record"].nunique()),
        "split": {
            "type": "record_level",
            "test_fraction": float(args.test_fraction),
            "split_seed": int(args.split_seed),
        },
        "selection_feature_set": args.selection_feature_set,
        "selection_representation_dim": int(x_select.shape[1]),
        "budget_fractions": args.budget_fractions,
        "seeds": args.seeds,
        "methods": args.methods + ["full_train_reference", "preselect_task_mismatch"],
        "result_level": "full_bidmc_train_pool_migration_benchmark",
        "label_proxy": "ecg_hr_bpm quantile bins for supervised selection adapters",
    }
    (out_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "status": status_df.astype(object).where(pd.notna(status_df), None).to_dict(orient="records"),
                "summary": summary_df.astype(object).where(pd.notna(summary_df), None).to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_audit_files(out_dir, args, metadata, status_df, [sys.executable, "scripts/run_bidmc_all_selection_benchmark.py", *sys.argv[1:]])

    print(status_df.to_string(index=False))
    if not summary_df.empty:
        display = summary_df[summary_df["feature_set"] == "ppg_handcrafted"][
            ["method", "budget_fraction", "mae_mean", "pearson_mean", "selected_record_fraction_mean", "runs"]
        ]
        print(display.to_string(index=False))
    print(f"Wrote {out_dir / 'selection_runs.csv'}")
    print(f"Wrote {out_dir / 'selection_status.csv'}")
    print(f"Wrote {out_dir / 'selection_summary.json'}")
    print(f"Wrote {out_dir / 'audit_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
