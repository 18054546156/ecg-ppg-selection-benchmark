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


def optional_csv_ints(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    return parse_csv_ints(value)


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


def fill_with_k_center(x_select: np.ndarray, selected: np.ndarray, budget: int, seed: int) -> np.ndarray:
    selected = np.unique(np.asarray(selected, dtype=int).reshape(-1))
    if len(selected) >= budget:
        return selected[:budget]
    n_samples = int(x_select.shape[0])
    if len(selected) == 0:
        return select_k_center(x_select, budget, seed)

    selected_list = selected.astype(int).tolist()
    selected_mask = np.zeros(n_samples, dtype=bool)
    selected_mask[selected] = True
    min_dist = np.min(np.sum((x_select[:, None, :] - x_select[selected][None, :, :]) ** 2, axis=2), axis=1)
    min_dist = np.nan_to_num(min_dist, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)
    min_dist[selected_mask] = -np.inf
    while len(selected_list) < budget and (~selected_mask).any():
        idx = int(np.argmax(min_dist))
        if selected_mask[idx]:
            break
        selected_list.append(idx)
        selected_mask[idx] = True
        dist = np.sum((x_select - x_select[idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, np.nan_to_num(dist, nan=np.inf, posinf=np.inf, neginf=np.inf))
        min_dist[selected_mask] = -np.inf
    return np.asarray(selected_list, dtype=int)


def validate_and_finalize_selection(
    method: str,
    raw_selected: np.ndarray,
    *,
    train_n: int,
    budget: int,
    seed: int,
    x_select: np.ndarray,
    probcover_fill_strategy: str,
) -> tuple[np.ndarray, dict]:
    raw = np.asarray(raw_selected, dtype=int).reshape(-1)
    if len(raw) and (np.any(raw < 0) or np.any(raw >= train_n)):
        raise RuntimeError(f"selector returned out-of-range indices for n={train_n}")
    unique = np.unique(raw)
    duplicate_n = int(len(raw) - len(unique))
    if len(unique) > budget:
        unique = unique[:budget]

    fill_strategy = "none"
    filled_n = 0
    final = unique
    if len(final) < budget:
        if method == "probcover_official" and probcover_fill_strategy == "k_center":
            final = fill_with_k_center(x_select, final, budget, seed)
            fill_strategy = "k_center_unselected"
            filled_n = int(len(final) - len(unique))
        else:
            raise RuntimeError(f"{method} selected {len(final)} of requested budget {budget}")

    final = validate_indices(final, train_n)
    if len(final) != budget:
        raise RuntimeError(f"{method} finalized {len(final)} of requested budget {budget}")
    info = {
        "requested_budget": int(budget),
        "raw_selected_n": int(len(unique)),
        "duplicate_removed_n": duplicate_n,
        "selected_n": int(len(final)),
        "budget_met": bool(len(final) == budget),
        "raw_effective_budget_fraction": float(len(unique) / budget),
        "final_effective_budget_fraction": float(len(final) / budget),
        "fill_strategy": fill_strategy,
        "filled_n": filled_n,
    }
    return final, info


def probcover_coverage(features: np.ndarray, selected: np.ndarray, delta: float) -> float:
    if len(selected) == 0:
        return 0.0
    graph = np.linalg.norm(features[selected, None, :] - features[None, :, :], axis=2) < float(delta)
    return float(graph.any(axis=0).mean())


def write_formal_table(out_dir: Path, summary_df: pd.DataFrame, status_df: pd.DataFrame, primary_feature_set: str) -> None:
    selection_status = status_df[status_df["method"].isin(DEFAULT_METHODS)].copy()
    status_rows: list[dict] = []
    if not selection_status.empty:
        for keys, group in selection_status.groupby(["method", "budget_fraction"], dropna=False):
            row = dict(zip(["method", "budget_fraction"], keys))
            row["attempts"] = int(len(group))
            row["passed_attempts"] = int((group["status"] == "passed").sum())
            row["failed_attempts"] = int((group["status"] == "failed").sum())
            for col in ["requested_budget", "raw_selected_n", "selected_n", "filled_n", "elapsed_sec"]:
                if col in group:
                    row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            failures = group.loc[group["status"] != "passed", "note"].dropna().astype(str).unique().tolist()
            row["failure_notes"] = " | ".join(failures[:3])
            status_rows.append(row)
    status_summary = pd.DataFrame(status_rows)

    perf = summary_df[summary_df["feature_set"] == primary_feature_set].copy() if not summary_df.empty else pd.DataFrame()
    keep_cols = [
        "method",
        "budget_fraction",
        "feature_set",
        "mae_mean",
        "mae_std",
        "pearson_mean",
        "pearson_std",
        "selected_record_fraction_mean",
        "selected_record_fraction_std",
        "runs",
    ]
    perf = perf[[col for col in keep_cols if col in perf.columns]]
    if status_summary.empty:
        formal = perf
    elif perf.empty:
        formal = status_summary
    else:
        formal = status_summary.merge(perf, on=["method", "budget_fraction"], how="left")
    if not formal.empty:
        formal = formal.sort_values(["budget_fraction", "mae_mean", "method"], na_position="last")
    formal.to_csv(out_dir / "formal_bidmc_table.csv", index=False)
    try:
        markdown = formal.to_markdown(index=False)
    except Exception:
        markdown = formal.to_csv(index=False)
    (out_dir / "formal_bidmc_table.md").write_text(
        f"# Formal BIDMC Table\n\nPrimary evaluation feature set: `{primary_feature_set}`\n\n{markdown}\n",
        encoding="utf-8",
    )


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
        return select_random(train_n, budget, seed), {}
    if method == "k_center":
        return select_k_center(x_select, budget, seed), {}
    if method == "typiclust_official":
        return select_typiclust_official(
            x_select,
            budget,
            seed,
            min_cluster_size=args.typiclust_min_cluster_size,
            max_clusters=args.typiclust_max_clusters,
            k_nn=args.typiclust_k_nn,
            verbose=args.official_selector_verbose,
        ), {}
    if method == "probcover_official":
        delta = probcover_delta(x_select, args.probcover_k_nn, args.probcover_quantile)
        selected, used_delta, raw_coverage = select_probcover_official(
            x_select, budget, delta, verbose=args.official_selector_verbose
        )
        return selected, {"probcover_delta": used_delta, "probcover_raw_coverage": raw_coverage}
    if method == "moderate_coreset_official_core":
        return select_moderate_official(x_select, labels, budget), {}
    if method == "elfs_official_core_proxy_score":
        return select_elfs_core_official(x_select, labels, budget), {}
    if method == "cords_gradmatch":
        return select_cords_gradmatch(x_select, labels, budget, seed, args.batch_size), {}
    if method == "cords_glister":
        return select_cords_glister(x_select, labels, budget, seed, args.batch_size), {}
    if method == "cords_craig":
        return select_cords_craig(x_select, labels, budget, seed, args.batch_size), {}
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
            "split_seeds": args.split_seeds,
            "split_seed": args.split_seed,
            "test_fraction": args.test_fraction,
            "pca_dim": args.pca_dim,
            "methods": args.methods,
            "batch_size": args.batch_size,
            "probcover_fill_strategy": args.probcover_fill_strategy,
            "primary_report_feature_set": args.primary_report_feature_set,
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
            "formal_bidmc_table_csv": "formal_bidmc_table.csv",
            "formal_bidmc_table_md": "formal_bidmc_table.md",
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
    parser.add_argument("--split-seeds", type=optional_csv_ints, default=None)
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
    parser.add_argument("--probcover-fill-strategy", choices=["k_center"], default="k_center")
    parser.add_argument("--primary-report-feature-set", choices=["ppg_handcrafted", "ecg_ppg_handcrafted", "papagei"], default="ppg_handcrafted")
    parser.add_argument("--official-selector-verbose", action="store_true")
    args = parser.parse_args()
    split_seeds = args.split_seeds if args.split_seeds is not None else [args.split_seed]

    features_csv = Path(args.features_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv).replace([np.inf, -np.inf], np.nan)
    feature_sets, selection_sets = build_feature_sets(df)
    selection_cols = selection_sets[args.selection_feature_set]
    if not selection_cols:
        raise RuntimeError(f"No columns found for selection feature set {args.selection_feature_set}")

    run_rows: list[dict] = []
    status_rows: list[dict] = []
    selected_indices: dict[str, list[int]] = {}
    split_metadata: list[dict] = []
    for split_seed in split_seeds:
        train_df, test_df = split_by_record(df, args.test_fraction, split_seed)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_records_before_filter = set(train_df["record"].astype(str).unique())
        test_records = set(test_df["record"].astype(str).unique())
        if train_records_before_filter & test_records:
            raise RuntimeError(f"record leakage in split_seed={split_seed}")

        required = train_df[["ecg_hr_bpm"] + selection_cols].replace([np.inf, -np.inf], np.nan)
        train_df = train_df[required.notna().all(axis=1)].reset_index(drop=True)

        if len(train_df) < 0.5 * len(df):
            raise RuntimeError(
                f"Training pool unexpectedly small for split_seed={split_seed}: {len(train_df)} of {len(df)} segments. "
                "This script is for full-pool benchmarking, not smoke tests."
            )

        x_select = prepare_matrix(train_df, selection_cols, args.pca_dim, split_seed)
        labels = make_class_bins(train_df["ecg_hr_bpm"].to_numpy(dtype=float), num_bins=4)
        train_records = set(train_df["record"].astype(str).unique())
        split_metadata.append(
            {
                "split_seed": int(split_seed),
                "train_segments": int(len(train_df)),
                "test_segments": int(len(test_df)),
                "train_records": int(train_df["record"].nunique()),
                "test_records": int(test_df["record"].nunique()),
                "train_record_ids": sorted(train_records),
                "test_record_ids": sorted(test_records),
                "record_overlap": sorted(train_records & test_records),
            }
        )

        for budget_fraction in args.budget_fractions:
            budget = max(1, int(math.ceil(float(budget_fraction) * len(train_df))))
            for seed in args.seeds:
                for method in args.methods:
                    started = time.time()
                    key = f"{method}|split={split_seed}|budget={budget_fraction:g}|seed={seed}"
                    print(
                        f"[select] split_seed={split_seed} budget_fraction={budget_fraction:g} "
                        f"seed={seed} method={method} requested_budget={budget}",
                        flush=True,
                    )
                    try:
                        raw_selected, selector_extra = make_selector(method, x_select, labels, len(train_df), budget, seed, args)
                        selected, selection_info = validate_and_finalize_selection(
                            method,
                            raw_selected,
                            train_n=len(train_df),
                            budget=budget,
                            seed=seed,
                            x_select=x_select,
                            probcover_fill_strategy=args.probcover_fill_strategy,
                        )
                        if method == "probcover_official" and "probcover_delta" in selector_extra:
                            selection_info["probcover_delta"] = float(selector_extra["probcover_delta"])
                            selection_info["probcover_raw_coverage"] = float(selector_extra["probcover_raw_coverage"])
                            selection_info["probcover_final_coverage"] = probcover_coverage(
                                x_select, selected, float(selector_extra["probcover_delta"])
                            )
                        selected_indices[key] = selected.astype(int).tolist()
                        elapsed = time.time() - started
                        rows = evaluate_selected(method, train_df, test_df, selected, feature_sets)
                        for row in rows:
                            row["budget_fraction"] = float(budget_fraction)
                            row["requested_budget"] = int(budget)
                            row["raw_selected_n"] = int(selection_info["raw_selected_n"])
                            row["fill_strategy"] = selection_info["fill_strategy"]
                            row["filled_n"] = int(selection_info["filled_n"])
                            row["split_seed"] = int(split_seed)
                            row["seed"] = int(seed)
                        run_rows.extend(rows)
                        status_rows.append(
                            {
                                "method": method,
                                "budget_fraction": float(budget_fraction),
                                "split_seed": int(split_seed),
                                "seed": int(seed),
                                "status": "passed",
                                "elapsed_sec": elapsed,
                                "train_n": int(len(train_df)),
                                "test_n": int(len(test_df)),
                                "selected_key": key,
                                "note": "Full BIDMC train-pool benchmark.",
                                **selection_info,
                                **selector_extra,
                            }
                        )
                        print(
                            f"[pass] {key} selected_n={len(selected)} elapsed_sec={elapsed:.2f}",
                            flush=True,
                        )
                    except Exception as exc:
                        status_rows.append(
                            {
                                "method": method,
                                "budget_fraction": float(budget_fraction),
                                "split_seed": int(split_seed),
                                "seed": int(seed),
                                "status": "failed",
                                "elapsed_sec": time.time() - started,
                                "train_n": int(len(train_df)),
                                "test_n": int(len(test_df)),
                                "selected_key": key,
                                "requested_budget": int(budget),
                                "raw_selected_n": 0,
                                "selected_n": 0,
                                "budget_met": False,
                                "raw_effective_budget_fraction": 0.0,
                                "final_effective_budget_fraction": 0.0,
                                "fill_strategy": "none",
                                "filled_n": 0,
                                "note": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"[fail] {key} {type(exc).__name__}: {exc}", flush=True)

        full_rows = evaluate_selected("full_train_reference", train_df, test_df, np.arange(len(train_df)), feature_sets)
        for row in full_rows:
            row["budget_fraction"] = 1.0
            row["requested_budget"] = int(len(train_df))
            row["raw_selected_n"] = int(len(train_df))
            row["fill_strategy"] = "none"
            row["filled_n"] = 0
            row["split_seed"] = int(split_seed)
            row["seed"] = int(split_seed)
        run_rows.extend(full_rows)
        status_rows.append(
            {
                "method": "full_train_reference",
                "budget_fraction": 1.0,
                "split_seed": int(split_seed),
                "seed": int(split_seed),
                "status": "passed",
                "elapsed_sec": 0.0,
                "train_n": int(len(train_df)),
                "test_n": int(len(test_df)),
                "selected_key": f"full_train_reference|split={split_seed}|budget=1|seed={split_seed}",
                "requested_budget": int(len(train_df)),
                "raw_selected_n": int(len(train_df)),
                "duplicate_removed_n": 0,
                "selected_n": int(len(train_df)),
                "budget_met": True,
                "raw_effective_budget_fraction": 1.0,
                "final_effective_budget_fraction": 1.0,
                "fill_strategy": "none",
                "filled_n": 0,
                "note": "Reference using the complete BIDMC training pool after record-level split.",
            }
        )
    status_rows.append(
        {
            "method": "preselect",
            "budget_fraction": np.nan,
            "split_seed": np.nan,
            "seed": np.nan,
            "status": "not_run_task_mismatch",
            "elapsed_sec": 0.0,
            "train_n": 0,
            "test_n": 0,
            "selected_key": "",
            "requested_budget": 0,
            "raw_selected_n": 0,
            "selected_n": 0,
            "budget_met": False,
            "fill_strategy": "none",
            "filled_n": 0,
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
    write_formal_table(out_dir, summary_df, status_df, args.primary_report_feature_set)

    train_segment_values = [item["train_segments"] for item in split_metadata]
    test_segment_values = [item["test_segments"] for item in split_metadata]
    train_record_values = [item["train_records"] for item in split_metadata]
    test_record_values = [item["test_records"] for item in split_metadata]

    metadata = {
        "dataset": "BIDMC PPG and Respiration Dataset",
        "source_segments": int(len(df)),
        "train_segments": int(min(train_segment_values)),
        "test_segments": int(min(test_segment_values)),
        "train_segments_min": int(min(train_segment_values)),
        "train_segments_max": int(max(train_segment_values)),
        "test_segments_min": int(min(test_segment_values)),
        "test_segments_max": int(max(test_segment_values)),
        "train_records": int(min(train_record_values)),
        "test_records": int(min(test_record_values)),
        "train_records_min": int(min(train_record_values)),
        "train_records_max": int(max(train_record_values)),
        "test_records_min": int(min(test_record_values)),
        "test_records_max": int(max(test_record_values)),
        "split": {
            "type": "record_level",
            "test_fraction": float(args.test_fraction),
            "split_seed": int(args.split_seed),
            "split_seeds": [int(item) for item in split_seeds],
        },
        "splits": split_metadata,
        "selection_feature_set": args.selection_feature_set,
        "selection_representation_dim": int(len(selection_cols) if args.pca_dim <= 0 else min(args.pca_dim, len(selection_cols))),
        "budget_fractions": args.budget_fractions,
        "seeds": args.seeds,
        "split_seeds": [int(item) for item in split_seeds],
        "primary_report_feature_set": args.primary_report_feature_set,
        "probcover_fill_strategy": args.probcover_fill_strategy,
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
        display = summary_df[summary_df["feature_set"] == args.primary_report_feature_set][
            ["method", "budget_fraction", "mae_mean", "pearson_mean", "selected_record_fraction_mean", "runs"]
        ]
        print(display.to_string(index=False))
    print(f"Wrote {out_dir / 'selection_runs.csv'}")
    print(f"Wrote {out_dir / 'selection_status.csv'}")
    print(f"Wrote {out_dir / 'formal_bidmc_table.csv'}")
    print(f"Wrote {out_dir / 'selection_summary.json'}")
    print(f"Wrote {out_dir / 'audit_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
