#!/usr/bin/env python3
"""Run BIDMC public-data selection baselines on saved ECG+PPG features.

This is a migration/smoke benchmark, not an original-paper reproduction. The
selection rules follow the official algorithm definitions where they can be
cleanly adapted to a fixed feature table:

- K-Center/CoreSet: farthest-first traversal in representation space.
- TypiClust: KMeans clusters, then most typical/dense point per cluster.
- ProbCover: greedy maximum coverage with a distance-radius graph.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from official_selection_adapters import select_probcover_official, select_typiclust_official


METRIC_FEATURE_SETS = {
    "ppg_handcrafted": "PPG hand-crafted features",
    "ecg_ppg_handcrafted": "ECG+PPG hand-crafted features",
    "papagei": "PaPaGei PPG embeddings",
}


def finite_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].replace([np.inf, -np.inf], np.nan).dropna()


def prepare_matrix(df: pd.DataFrame, columns: list[str], pca_dim: int, seed: int) -> np.ndarray:
    values = df[columns].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
    scaled = StandardScaler().fit_transform(values.to_numpy(dtype=np.float64))
    if pca_dim > 0 and scaled.shape[1] > pca_dim and scaled.shape[0] > pca_dim:
        scaled = PCA(n_components=pca_dim, random_state=seed).fit_transform(scaled)
    return scaled.astype(np.float32)


def split_by_record(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = np.array(sorted(df["record"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(records)
    n_test = max(1, int(math.ceil(len(records) * test_fraction)))
    test_records = set(records[:n_test])
    train_mask = ~df["record"].isin(test_records)
    return df[train_mask].copy(), df[~train_mask].copy()


def select_random(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def select_k_center(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    n = x.shape[0]
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    min_dist = np.sum((x - x[selected[0]]) ** 2, axis=1)
    min_dist[selected[0]] = -np.inf
    while len(selected) < k:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        dist = np.sum((x - x[idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected] = -np.inf
    return np.array(selected, dtype=int)


def typicality_scores(x: np.ndarray, k_nn: int) -> np.ndarray:
    if x.shape[0] <= 2:
        return np.ones(x.shape[0], dtype=np.float64)
    k_nn = max(1, min(k_nn, x.shape[0] - 1))
    distances, _ = NearestNeighbors(n_neighbors=k_nn + 1).fit(x).kneighbors(x)
    mean_distance = distances[:, 1:].mean(axis=1)
    return 1.0 / (mean_distance + 1e-5)


def select_typiclust(x: np.ndarray, k: int, seed: int, min_cluster_size: int, max_clusters: int, k_nn: int) -> np.ndarray:
    n = x.shape[0]
    if k >= n:
        return np.arange(n)
    n_clusters = max(1, min(k, max_clusters, n))
    if n_clusters <= 50:
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    else:
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, n_init=3, random_state=seed)
    labels = clusterer.fit_predict(x)
    cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    cluster_order = [
        int(cluster_id)
        for cluster_id, size in sorted(zip(cluster_ids, cluster_sizes), key=lambda item: (-item[1], item[0]))
        if size > min_cluster_size
    ]
    if not cluster_order:
        cluster_order = [int(cluster_id) for cluster_id, _ in sorted(zip(cluster_ids, cluster_sizes), key=lambda item: (-item[1], item[0]))]

    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    cluster_pointer = 0
    while len(selected) < k and available.any():
        cluster_id = cluster_order[cluster_pointer % len(cluster_order)]
        cluster_pointer += 1
        indices = np.flatnonzero((labels == cluster_id) & available)
        if len(indices) == 0:
            continue
        scores = typicality_scores(x[indices], min(k_nn, max(1, len(indices) // 2)))
        chosen = int(indices[int(np.argmax(scores))])
        selected.append(chosen)
        available[chosen] = False

    if len(selected) < k:
        remaining = np.flatnonzero(available)
        global_scores = typicality_scores(x[remaining], k_nn)
        fill = remaining[np.argsort(global_scores)[::-1][: k - len(selected)]]
        selected.extend(int(idx) for idx in fill)
    return np.array(selected, dtype=int)


def probcover_delta(x: np.ndarray, k_nn: int, quantile: float) -> float:
    if x.shape[0] <= 2:
        return 0.0
    n_neighbors = min(k_nn + 1, x.shape[0])
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(x)
    kth = distances[:, -1]
    return float(np.quantile(kth, quantile))


def select_probcover(x: np.ndarray, k: int, delta: float, seed: int) -> tuple[np.ndarray, float, float]:
    n = x.shape[0]
    if k >= n:
        return np.arange(n), delta, 1.0
    graph = pairwise_distances(x, metric="euclidean", n_jobs=1) < delta
    covered = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    selected: list[int] = []
    rng = np.random.default_rng(seed)
    for _ in range(k):
        gains = graph[:, ~covered].sum(axis=1).astype(np.float64)
        gains[~available] = -1.0
        max_gain = gains.max()
        if max_gain <= 0:
            candidates = np.flatnonzero(available)
            chosen = int(rng.choice(candidates))
        else:
            candidates = np.flatnonzero(gains == max_gain)
            chosen = int(rng.choice(candidates))
        selected.append(chosen)
        available[chosen] = False
        covered |= graph[chosen]
    return np.array(selected, dtype=int), delta, float(covered.mean())


def evaluate_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_positions: np.ndarray,
    feature_cols: list[str],
    label_col: str,
) -> dict[str, float]:
    selected_train = train_df.iloc[selected_positions]
    train_usable = finite_frame(selected_train, feature_cols + [label_col])
    test_usable = finite_frame(test_df, feature_cols + [label_col])
    if len(train_usable) < 10 or len(test_usable) < 10:
        return {"train_n": int(len(train_usable)), "test_n": int(len(test_usable)), "mae": float("nan"), "pearson": float("nan")}
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    x_train = train_usable[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_usable[label_col].to_numpy(dtype=np.float64)
    x_test = test_usable[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_usable[label_col].to_numpy(dtype=np.float64)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = float(np.mean(np.abs(y_test - pred)))
    corr = float(pearsonr(y_test, pred)[0]) if np.std(pred) > 1e-8 else float("nan")
    return {"train_n": int(len(train_usable)), "test_n": int(len(test_usable)), "mae": mae, "pearson": corr}


def summarize_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "pearson", "selected_n", "selected_records", "selected_record_fraction", "probcover_coverage"]
    rows = []
    group_cols = ["method", "budget_fraction", "feature_set"]
    for keys, group in runs_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0)) if len(group) > 1 else 0.0
        row["runs"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["feature_set", "budget_fraction", "mae_mean", "method"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="results/bidmc/bidmc_segment_features.csv")
    parser.add_argument("--out-dir", default="results/bidmc_selection")
    parser.add_argument("--selection-feature-set", choices=["papagei", "handcrafted", "ecg_ppg_handcrafted"], default="papagei")
    parser.add_argument("--budgets", default="0.10,0.25,0.50")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--typiclust-min-cluster-size", type=int, default=5)
    parser.add_argument("--typiclust-max-clusters", type=int, default=500)
    parser.add_argument("--typiclust-k-nn", type=int, default=20)
    parser.add_argument("--probcover-k-nn", type=int, default=10)
    parser.add_argument("--probcover-quantile", type=float, default=0.50)
    parser.add_argument("--selector-runtime", choices=["official", "local"], default="official")
    parser.add_argument("--official-selector-verbose", action="store_true")
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    df = df.replace([np.inf, -np.inf], np.nan)
    ppg_cols = [col for col in df.columns if col.startswith("ppg_") and col != "ppg_channel"]
    ecg_cols = [col for col in df.columns if col.startswith("ecg_") and col not in {"ecg_channel", "ecg_hr_bpm"}]
    papagei_cols = [col for col in df.columns if col.startswith("papagei_")]
    if args.selection_feature_set == "papagei" and not papagei_cols:
        raise RuntimeError("No PaPaGei columns found; rerun BIDMC pipeline with --use-papagei.")

    feature_sets = {
        "ppg_handcrafted": ppg_cols,
        "ecg_ppg_handcrafted": ecg_cols + ppg_cols,
        "papagei": papagei_cols,
    }
    selection_cols = {
        "papagei": papagei_cols,
        "handcrafted": ppg_cols,
        "ecg_ppg_handcrafted": ecg_cols + ppg_cols,
    }[args.selection_feature_set]

    train_df, test_df = split_by_record(df, args.test_fraction, args.split_seed)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    selection_required = train_df[["ecg_hr_bpm"] + selection_cols].replace([np.inf, -np.inf], np.nan)
    train_df = train_df[selection_required.notna().all(axis=1)].reset_index(drop=True)
    x_select = prepare_matrix(train_df, selection_cols, args.pca_dim, args.split_seed)

    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    methods = ["random", "k_center", "typiclust", "probcover"]
    run_rows = []

    for budget_fraction in budgets:
        k = max(1, int(math.ceil(budget_fraction * len(train_df))))
        for seed in seeds:
            selected_by_method: dict[str, tuple[np.ndarray, float | None, float | None]] = {}
            selected_by_method["random"] = (select_random(len(train_df), k, seed), None, None)
            selected_by_method["k_center"] = (select_k_center(x_select, k, seed), None, None)
            if args.selector_runtime == "official":
                typiclust_selected = select_typiclust_official(
                    x_select,
                    k,
                    seed,
                    min_cluster_size=args.typiclust_min_cluster_size,
                    max_clusters=args.typiclust_max_clusters,
                    k_nn=args.typiclust_k_nn,
                    verbose=args.official_selector_verbose,
                )
            else:
                typiclust_selected = select_typiclust(
                    x_select,
                    k,
                    seed,
                    args.typiclust_min_cluster_size,
                    args.typiclust_max_clusters,
                    args.typiclust_k_nn,
                )
            selected_by_method["typiclust"] = (typiclust_selected, None, None)
            delta = probcover_delta(x_select, args.probcover_k_nn, args.probcover_quantile)
            if args.selector_runtime == "official":
                selected, used_delta, coverage = select_probcover_official(
                    x_select,
                    k,
                    delta,
                    verbose=args.official_selector_verbose,
                )
            else:
                selected, used_delta, coverage = select_probcover(x_select, k, delta, seed)
            selected_by_method["probcover"] = (selected, used_delta, coverage)

            for method in methods:
                selected_positions, used_delta, coverage = selected_by_method[method]
                selected_records = int(train_df.iloc[selected_positions]["record"].nunique())
                for feature_set, cols in feature_sets.items():
                    metrics = evaluate_regression(train_df, test_df, selected_positions, cols, "ecg_hr_bpm")
                    run_rows.append(
                        {
                            "method": method,
                            "budget_fraction": budget_fraction,
                            "seed": seed,
                            "feature_set": feature_set,
                            "selected_n": int(len(selected_positions)),
                            "selected_records": selected_records,
                            "selected_record_fraction": float(selected_records / train_df["record"].nunique()),
                            "train_records": int(train_df["record"].nunique()),
                            "test_records": int(test_df["record"].nunique()),
                            "probcover_delta": used_delta,
                            "probcover_coverage": coverage if coverage is not None else float("nan"),
                            **metrics,
                        }
                    )

    full_positions = np.arange(len(train_df))
    for feature_set, cols in feature_sets.items():
        metrics = evaluate_regression(train_df, test_df, full_positions, cols, "ecg_hr_bpm")
        run_rows.append(
            {
                "method": "full_train",
                "budget_fraction": 1.0,
                "seed": args.split_seed,
                "feature_set": feature_set,
                "selected_n": int(len(full_positions)),
                "selected_records": int(train_df["record"].nunique()),
                "selected_record_fraction": 1.0,
                "train_records": int(train_df["record"].nunique()),
                "test_records": int(test_df["record"].nunique()),
                "probcover_delta": float("nan"),
                "probcover_coverage": float("nan"),
                **metrics,
            }
        )

    runs_df = pd.DataFrame(run_rows)
    summary_df = summarize_runs(runs_df)
    runs_csv = out_dir / "bidmc_selection_runs.csv"
    summary_csv = out_dir / "bidmc_selection_summary.csv"
    summary_json = out_dir / "bidmc_selection_summary.json"
    runs_df.to_csv(runs_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    metadata = {
        "dataset": "BIDMC PPG and Respiration Dataset",
        "input_features": str(features_csv),
        "selection_feature_set": args.selection_feature_set,
        "selection_representation_dim": int(x_select.shape[1]),
        "selector_runtime": args.selector_runtime,
        "train_segments": int(len(train_df)),
        "test_segments": int(len(test_df)),
        "train_records": int(train_df["record"].nunique()),
        "test_records": int(test_df["record"].nunique()),
        "budgets": budgets,
        "seeds": seeds,
        "methods": methods + ["full_train"],
        "feature_sets": METRIC_FEATURE_SETS,
        "outputs": {
            "runs_csv": str(runs_csv),
            "summary_csv": str(summary_csv),
        },
    }
    summary_json.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary": summary_df.astype(object).where(pd.notna(summary_df), None).to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_df = summary_df[summary_df["feature_set"] == "ppg_handcrafted"].copy()
    if not plot_df.empty:
        plt.figure(figsize=(7, 4))
        for method, group in plot_df.groupby("method"):
            group = group.sort_values("budget_fraction")
            plt.plot(group["budget_fraction"], group["mae_mean"], marker="o", label=method)
        plt.xlabel("Selected training fraction")
        plt.ylabel("Test MAE predicting ECG HR (bpm)")
        plt.title("BIDMC Selection Benchmark: PPG Features")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "bidmc_selection_ppg_mae.png", dpi=160)
        plt.close()

    display_cols = ["method", "budget_fraction", "feature_set", "mae_mean", "pearson_mean", "selected_record_fraction_mean"]
    print(summary_df[display_cols].to_string(index=False))
    print(f"Wrote {runs_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
