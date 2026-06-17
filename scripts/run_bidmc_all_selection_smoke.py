#!/usr/bin/env python3
"""Smoke-test all managed data-selection methods on BIDMC ECG+PPG features.

This is an integration test, not an original-paper reproduction. Supervised
gradient methods use ECG-derived HR quantile bins as a small classification
proxy so that their official selection strategies can execute on BIDMC.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from official_selection_adapters import select_probcover_official, select_typiclust_official
from run_bidmc_selection_benchmark import (
    evaluate_regression,
    prepare_matrix,
    probcover_delta,
    select_k_center,
    select_random,
    split_by_record,
)


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")


class ProxyNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, embedding_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU())
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs, last: bool = False, freeze: bool = False):
        del freeze
        embedding = self.encoder(inputs)
        logits = self.classifier(embedding)
        if last:
            return logits, embedding
        return logits

    def get_embedding_dim(self) -> int:
        return int(self.classifier.in_features)


def make_class_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    series = pd.Series(values)
    labels = pd.qcut(series, q=num_bins, labels=False, duplicates="drop")
    labels = labels.fillna(labels.median()).astype(int).to_numpy()
    labels -= labels.min()
    return labels.astype(np.int64)


def sample_train_subset(train_df: pd.DataFrame, max_train: int, seed: int) -> pd.DataFrame:
    if len(train_df) <= max_train:
        return train_df.reset_index(drop=True)
    labels = make_class_bins(train_df["ecg_hr_bpm"].to_numpy(dtype=float), num_bins=4)
    idx = np.arange(len(train_df))
    train_idx, _ = train_test_split(idx, train_size=max_train, random_state=seed, stratify=labels)
    return train_df.iloc[np.sort(train_idx)].reset_index(drop=True)


def validate_indices(indices: np.ndarray, n: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int).reshape(-1)
    if len(indices) == 0:
        raise RuntimeError("selector returned no indices")
    if np.any(indices < 0) or np.any(indices >= n):
        raise RuntimeError(f"selector returned out-of-range indices for n={n}")
    unique = np.unique(indices)
    if len(unique) != len(indices):
        indices = unique
    return indices


def select_moderate_official(features: np.ndarray, labels: np.ndarray, budget: int) -> np.ndarray:
    module_dir = str(ROOT / "third_party" / "2023_ICLR_Moderate-DS")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    selection = importlib.import_module("selection")
    distance = selection.get_distance(features, labels)
    args = SimpleNamespace(rate=float(budget / len(features)))
    drop_idx = set(int(i) for i in np.asarray(selection.get_prune_idx(args, distance)).reshape(-1))
    selected = np.array([i for i in range(len(features)) if i not in drop_idx], dtype=int)
    if len(selected) > budget:
        selected = selected[np.argsort(np.abs(distance[selected] - np.median(distance)))[:budget]]
    return selected


def select_elfs_core_official(features: np.ndarray, labels: np.ndarray, budget: int) -> np.ndarray:
    module_dir = str(ROOT / "third_party" / "elfs")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    coreset = importlib.import_module("core.data.Coreset")
    center = features.mean(axis=0, keepdims=True)
    proxy_score = -np.linalg.norm(features - center, axis=1)
    data_score = {
        "targets": torch.tensor(labels, dtype=torch.long),
        "proxy_representativeness": torch.tensor(proxy_score, dtype=torch.float32),
    }
    with torch.no_grad():
        selected = coreset.CoresetSelection.score_monotonic_selection(
            data_score,
            key="proxy_representativeness",
            ratio=float(budget / len(features)),
            descending=True,
            class_balanced=False,
        )
    return selected.detach().cpu().numpy().astype(int)


def install_asyncore_stub() -> None:
    if "asyncore" in sys.modules:
        return
    module = types.ModuleType("asyncore")
    module.read = object()
    sys.modules["asyncore"] = module


def install_sklearn_load_boston_stub() -> None:
    import sklearn.datasets as sklearn_datasets

    try:
        getattr(sklearn_datasets, "load_boston")
        return
    except Exception:
        pass

    def load_boston(return_X_y: bool = False, *args, **kwargs):
        del args, kwargs
        data = np.zeros((1, 13), dtype=np.float32)
        target = np.zeros((1,), dtype=np.float32)
        if return_X_y:
            return data, target
        return SimpleNamespace(data=data, target=target)

    sklearn_datasets.load_boston = load_boston


def install_sentence_transformers_stub() -> None:
    if "sentence_transformers" in sys.modules:
        return

    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def encode(self, texts, *args, **kwargs):
            del args, kwargs
            return np.zeros((len(texts), 1), dtype=np.float32)

    module.SentenceTransformer = SentenceTransformer
    module.util = types.SimpleNamespace()
    sys.modules["sentence_transformers"] = module


def install_transformers_vit_stub() -> None:
    class ViTFeatureExtractor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return {}

    class ViTModel(nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

        def forward(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(last_hidden_state=torch.zeros((1, 1, 1)))

    class BertConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

    class BertModel(nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

        def forward(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(pooler_output=torch.zeros((1, 1)))

    transformers = types.ModuleType("transformers")
    transformers.ViTFeatureExtractor = ViTFeatureExtractor
    transformers.ViTModel = ViTModel
    transformers.BertConfig = BertConfig
    transformers.BertModel = BertModel
    sys.modules["transformers"] = transformers


def install_submodlib_stub() -> None:
    if "submodlib" in sys.modules:
        return

    class _SubmodFunction:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def maximize(self, *args, **kwargs):
            del args, kwargs
            return []

    module = types.ModuleType("submodlib")
    module.FacilityLocationFunction = _SubmodFunction
    module.LogDeterminantFunction = _SubmodFunction
    module.GraphCutFunction = _SubmodFunction
    module.DisparityMinFunction = _SubmodFunction
    module.DisparitySumFunction = _SubmodFunction
    sys.modules["submodlib"] = module


def install_torchtext_stub() -> None:
    if "torchtext" in sys.modules:
        return
    module = types.ModuleType("torchtext")
    data_module = types.ModuleType("torchtext.data")
    module.data = data_module
    sys.modules["torchtext"] = module
    sys.modules["torchtext.data"] = data_module


def install_datasets_stub() -> None:
    if "datasets" in sys.modules and hasattr(sys.modules["datasets"], "load_dataset"):
        return
    module = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        del args, kwargs
        return {}

    module.load_dataset = load_dataset
    sys.modules["datasets"] = module


def install_tkinter_stub() -> None:
    if "tkinter" in sys.modules:
        return
    module = types.ModuleType("tkinter")
    module.TkVersion = 8.6
    module.TclVersion = 8.6
    module.Tk = object
    module.Tcl = object
    module.Frame = object
    module.Canvas = object
    module.PhotoImage = object
    module.StringVar = object
    module.IntVar = object
    module.BooleanVar = object
    module.DoubleVar = object
    module.mainloop = lambda *args, **kwargs: None
    for name in ["filedialog", "font", "messagebox", "simpledialog"]:
        submodule = types.ModuleType(f"tkinter.{name}")
        setattr(module, name, submodule)
        sys.modules[f"tkinter.{name}"] = submodule
    sys.modules["tkinter"] = module
    sys.modules["_tkinter"] = types.ModuleType("_tkinter")


def load_cords_strategy(module_name: str, class_name: str):
    install_asyncore_stub()
    install_sklearn_load_boston_stub()
    install_sentence_transformers_stub()
    install_transformers_vit_stub()
    install_submodlib_stub()
    install_torchtext_stub()
    install_datasets_stub()
    install_tkinter_stub()
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    cords_root = str(ROOT / "third_party" / "cords")
    if cords_root not in sys.path:
        sys.path.insert(0, cords_root)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_cords_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, ProxyNet, nn.Module]:
    idx = np.arange(len(features))
    train_idx, val_idx = train_test_split(idx, test_size=0.25, random_state=seed, stratify=labels)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = ProxyNet(features.shape[1], int(labels.max()) + 1)
    loss = nn.CrossEntropyLoss(reduction="none")
    return train_loader, val_loader, model, loss


def select_cords_gradmatch(features: np.ndarray, labels: np.ndarray, budget: int, seed: int, batch_size: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader, model, loss = make_cords_inputs(features, labels, seed, batch_size)
    cls = load_cords_strategy("cords.selectionstrategies.SL.gradmatchstrategy", "GradMatchStrategy")
    logger = logging.getLogger("cords.gradmatch")
    selector = cls(
        train_loader,
        val_loader,
        model,
        loss,
        eta=0.01,
        device="cpu",
        num_classes=int(labels.max()) + 1,
        linear_layer=False,
        selection_type="PerBatch",
        logger=logger,
        valid=False,
        v1=False,
    )
    idxs, _weights = selector.select(budget, model.state_dict())
    return np.asarray(idxs[:budget], dtype=int)


def select_cords_glister(features: np.ndarray, labels: np.ndarray, budget: int, seed: int, batch_size: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader, model, loss = make_cords_inputs(features, labels, seed, batch_size)
    cls = load_cords_strategy("cords.selectionstrategies.SL.glisterstrategy", "GLISTERStrategy")
    logger = logging.getLogger("cords.glister")
    selector = cls(
        train_loader,
        val_loader,
        model,
        loss,
        eta=0.01,
        device="cpu",
        num_classes=int(labels.max()) + 1,
        linear_layer=False,
        selection_type="Supervised",
        greedy="RGreedy",
        logger=logger,
        r=4,
    )
    idxs, _weights = selector.select(budget, model.state_dict())
    return np.asarray(idxs[:budget], dtype=int)


def select_cords_craig(features: np.ndarray, labels: np.ndarray, budget: int, seed: int, batch_size: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader, model, loss = make_cords_inputs(features, labels, seed, batch_size)
    cls = load_cords_strategy("cords.selectionstrategies.SL.craigstrategy", "CRAIGStrategy")
    logger = logging.getLogger("cords.craig")
    selector = cls(
        train_loader,
        val_loader,
        model,
        loss,
        device="cpu",
        num_classes=int(labels.max()) + 1,
        linear_layer=False,
        if_convex=True,
        selection_type="Supervised",
        logger=logger,
        optimizer="naive",
    )
    idxs, _weights = selector.select(budget, model.state_dict())
    return np.asarray(idxs[:budget], dtype=int)


def evaluate_selected(
    method: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected: np.ndarray,
    metric_feature_sets: dict[str, list[str]],
) -> list[dict]:
    rows = []
    selected = validate_indices(selected, len(train_df))
    for feature_set, cols in metric_feature_sets.items():
        metrics = evaluate_regression(train_df, test_df, selected, cols, "ecg_hr_bpm")
        rows.append(
            {
                "method": method,
                "status": "passed",
                "feature_set": feature_set,
                "selected_n": int(len(selected)),
                "selected_records": int(train_df.iloc[selected]["record"].nunique()),
                "selected_record_fraction": float(train_df.iloc[selected]["record"].nunique() / train_df["record"].nunique()),
                **metrics,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="results/bidmc/bidmc_segment_features.csv")
    parser.add_argument("--out-dir", default="results/bidmc_selection_all_smoke")
    parser.add_argument("--selection-feature-set", choices=["papagei", "handcrafted", "ecg_ppg_handcrafted"], default="papagei")
    parser.add_argument("--max-train", type=int, default=192)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv).replace([np.inf, -np.inf], np.nan)
    ppg_cols = [col for col in df.columns if col.startswith("ppg_") and col != "ppg_channel"]
    ecg_cols = [col for col in df.columns if col.startswith("ecg_") and col not in {"ecg_channel", "ecg_hr_bpm"}]
    papagei_cols = [col for col in df.columns if col.startswith("papagei_")]
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
    if not selection_cols:
        raise RuntimeError(f"No columns found for selection feature set {args.selection_feature_set}")

    train_df, test_df = split_by_record(df, args.test_fraction, args.split_seed)
    required = train_df[["ecg_hr_bpm"] + selection_cols].replace([np.inf, -np.inf], np.nan)
    train_df = train_df[required.notna().all(axis=1)].reset_index(drop=True)
    train_df = sample_train_subset(train_df, args.max_train, args.seed)
    test_df = test_df.reset_index(drop=True)
    x_select = prepare_matrix(train_df, selection_cols, args.pca_dim, args.seed)
    labels = make_class_bins(train_df["ecg_hr_bpm"].to_numpy(dtype=float), num_bins=4)
    budget = min(args.budget, len(train_df))

    selector_fns = {
        "random": lambda: select_random(len(train_df), budget, args.seed),
        "k_center": lambda: select_k_center(x_select, budget, args.seed),
        "typiclust_official": lambda: select_typiclust_official(
            x_select,
            budget,
            args.seed,
            min_cluster_size=1,
            max_clusters=100,
            k_nn=10,
        ),
        "probcover_official": lambda: select_probcover_official(
            x_select,
            budget,
            probcover_delta(x_select, 10, 0.50),
        )[0],
        "moderate_coreset_official_core": lambda: select_moderate_official(x_select, labels, budget),
        "elfs_official_core_proxy_score": lambda: select_elfs_core_official(x_select, labels, budget),
        "cords_gradmatch": lambda: select_cords_gradmatch(x_select, labels, budget, args.seed, args.batch_size),
        "cords_glister": lambda: select_cords_glister(x_select, labels, budget, args.seed, args.batch_size),
        "cords_craig": lambda: select_cords_craig(x_select, labels, budget, args.seed, args.batch_size),
    }

    run_rows: list[dict] = []
    status_rows: list[dict] = []
    for method, selector_fn in selector_fns.items():
        started = time.time()
        try:
            selected = selector_fn()
            selected = validate_indices(selected, len(train_df))
            elapsed = time.time() - started
            run_rows.extend(evaluate_selected(method, train_df, test_df, selected, feature_sets))
            status_rows.append(
                {
                    "method": method,
                    "status": "passed",
                    "elapsed_sec": elapsed,
                    "selected_n": int(len(selected)),
                    "note": "BIDMC smoke only; not an original-paper reproduction.",
                }
            )
        except Exception as exc:
            status_rows.append(
                {
                    "method": method,
                    "status": "failed",
                    "elapsed_sec": time.time() - started,
                    "selected_n": 0,
                    "note": f"{type(exc).__name__}: {exc}",
                }
            )

    status_rows.append(
        {
            "method": "preselect",
            "status": "not_run_task_mismatch",
            "elapsed_sec": 0.0,
            "selected_n": 0,
            "note": "Original PreSelect pipeline is text/LLM pretraining data selection, not ECG+PPG waveform/window selection.",
        }
    )

    full_rows = evaluate_selected("full_train_reference", train_df, test_df, np.arange(len(train_df)), feature_sets)
    run_rows.extend(full_rows)
    status_rows.append(
        {
            "method": "full_train_reference",
            "status": "passed",
            "elapsed_sec": 0.0,
            "selected_n": int(len(train_df)),
            "note": "Reference using all smoke-train windows.",
        }
    )

    runs_df = pd.DataFrame(run_rows)
    status_df = pd.DataFrame(status_rows)
    runs_path = out_dir / "selection_smoke_runs.csv"
    status_path = out_dir / "selection_smoke_status.csv"
    json_path = out_dir / "selection_smoke_summary.json"
    runs_df.to_csv(runs_path, index=False)
    status_df.to_csv(status_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "dataset": "BIDMC PPG and Respiration Dataset",
                    "features_csv": str(args.features_csv),
                    "selection_feature_set": args.selection_feature_set,
                    "train_windows": int(len(train_df)),
                    "test_windows": int(len(test_df)),
                    "budget": int(budget),
                    "label_proxy": "ecg_hr_bpm quantile bins for supervised selection smoke",
                    "result_level": "migration_smoke_not_original_reproduction",
                },
                "status": status_df.astype(object).where(pd.notna(status_df), None).to_dict(orient="records"),
                "runs": runs_df.astype(object).where(pd.notna(runs_df), None).to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(status_df.to_string(index=False))
    if not runs_df.empty:
        display = runs_df[runs_df["feature_set"] == "ppg_handcrafted"][
            ["method", "status", "selected_n", "mae", "pearson", "selected_record_fraction"]
        ].sort_values(["method"])
        print(display.to_string(index=False))
    print(f"Wrote {runs_path}")
    print(f"Wrote {status_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
