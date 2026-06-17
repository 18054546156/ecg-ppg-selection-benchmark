#!/usr/bin/env python3
"""Adapters that call upstream selection code on ECG+PPG feature matrices."""

from __future__ import annotations

import importlib
import io
import sys
import types
from contextlib import contextmanager
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[1]
TYPICLUST_ROOT = REPO_ROOT / "third_party" / "TypiClust" / "deep-al"


def _install_faiss_cpu_stub() -> None:
    """Provide the tiny FAISS surface TypiClust needs when faiss is unavailable."""
    try:
        importlib.import_module("faiss")
        return
    except ImportError:
        pass

    class IndexFlatL2:
        def __init__(self, dim: int):
            self.dim = dim
            self._features: np.ndarray | None = None

        def add(self, features: np.ndarray) -> None:
            features = np.asarray(features, dtype=np.float32)
            if features.ndim != 2 or features.shape[1] != self.dim:
                raise ValueError(f"expected [N, {self.dim}] features, got {features.shape}")
            self._features = features

        def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            if self._features is None:
                raise RuntimeError("IndexFlatL2.search called before add")
            queries = np.asarray(queries, dtype=np.float32)
            distances = pairwise_distances(queries, self._features, metric="sqeuclidean")
            k = min(k, self._features.shape[0])
            indices = np.argpartition(distances, kth=np.arange(k), axis=1)[:, :k]
            row_ids = np.arange(distances.shape[0])[:, None]
            ordered = np.argsort(distances[row_ids, indices], axis=1)
            indices = indices[row_ids, ordered]
            distances = distances[row_ids, indices]
            return distances.astype(np.float32), indices.astype(np.int64)

    faiss_stub = types.ModuleType("faiss")
    faiss_stub.IndexFlatL2 = IndexFlatL2
    faiss_stub.index_cpu_to_all_gpus = lambda index: index
    sys.modules["faiss"] = faiss_stub


def _load_typiclust_modules():
    if not TYPICLUST_ROOT.exists():
        raise FileNotFoundError(f"TypiClust checkout not found at {TYPICLUST_ROOT}")
    _install_faiss_cpu_stub()
    root = str(TYPICLUST_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    ds_utils = importlib.import_module("pycls.datasets.utils")
    typiclust_mod = importlib.import_module("pycls.al.typiclust")
    probcover_mod = importlib.import_module("pycls.al.prob_cover")
    return ds_utils, typiclust_mod, probcover_mod


def _cpu_get_nn(features: np.ndarray, num_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if num_neighbors <= 0 or len(features) <= 1:
        return np.zeros((len(features), 0), dtype=np.float32), np.zeros((len(features), 0), dtype=np.int64)
    n_neighbors = min(num_neighbors + 1, len(features))
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors).fit(features).kneighbors(features)
    return (distances[:, 1:] ** 2).astype(np.float32), indices[:, 1:].astype(np.int64)


def _construct_probcover_graph_cpu(self, batch_size: int = 500) -> pd.DataFrame:
    rel_features = np.asarray(self.rel_features, dtype=np.float32)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ds: list[np.ndarray] = []
    for start in range(0, len(rel_features), batch_size):
        cur = rel_features[start : start + batch_size]
        dist = pairwise_distances(cur, rel_features, metric="euclidean")
        x, y = np.nonzero(dist < self.delta)
        xs.append(x.astype(np.int64) + start)
        ys.append(y.astype(np.int64))
        ds.append(dist[x, y].astype(np.float32))
    if not xs:
        return pd.DataFrame({"x": [], "y": [], "d": []})
    return pd.DataFrame({"x": np.concatenate(xs), "y": np.concatenate(ys), "d": np.concatenate(ds)})


@contextmanager
def _maybe_silence(verbose: bool):
    if verbose:
        yield
    else:
        with redirect_stdout(io.StringIO()):
            yield


@contextmanager
def _patched_typiclust_runtime(features: np.ndarray):
    ds_utils, typiclust_mod, probcover_mod = _load_typiclust_modules()
    features = np.asarray(features, dtype=np.float32)

    old_ds_load = ds_utils.load_features
    old_typi_load = typiclust_mod.ds_utils.load_features
    old_prob_load = probcover_mod.ds_utils.load_features
    old_get_nn = typiclust_mod.get_nn
    old_prob_graph = probcover_mod.ProbCover.construct_graph

    def load_features(_ds_name: str, _seed: int = 1, train: bool = True, normalized: bool = True):
        del _ds_name, _seed, train, normalized
        return features

    ds_utils.load_features = load_features
    typiclust_mod.ds_utils.load_features = load_features
    probcover_mod.ds_utils.load_features = load_features
    typiclust_mod.get_nn = _cpu_get_nn
    probcover_mod.ProbCover.construct_graph = _construct_probcover_graph_cpu
    try:
        yield typiclust_mod, probcover_mod
    finally:
        ds_utils.load_features = old_ds_load
        typiclust_mod.ds_utils.load_features = old_typi_load
        probcover_mod.ds_utils.load_features = old_prob_load
        typiclust_mod.get_nn = old_get_nn
        probcover_mod.ProbCover.construct_graph = old_prob_graph


def select_typiclust_official(
    features: np.ndarray,
    budget: int,
    seed: int,
    *,
    min_cluster_size: int = 5,
    max_clusters: int = 500,
    k_nn: int = 20,
    verbose: bool = False,
) -> np.ndarray:
    """Run the upstream TypiClust class on a supplied feature matrix."""
    n_samples = int(features.shape[0])
    if budget >= n_samples:
        return np.arange(n_samples, dtype=int)
    if budget <= 0:
        return np.array([], dtype=int)

    cfg = {"DATASET": {"NAME": "ECG_PPG_FEATURES"}, "RNG_SEED": int(seed)}
    l_set = np.array([], dtype=int)
    u_set = np.arange(n_samples, dtype=int)
    np.random.seed(seed)

    with _patched_typiclust_runtime(features) as (typiclust_mod, _):
        cls = typiclust_mod.TypiClust
        old_attrs = (cls.MIN_CLUSTER_SIZE, cls.MAX_NUM_CLUSTERS, cls.K_NN)
        cls.MIN_CLUSTER_SIZE = int(min_cluster_size)
        cls.MAX_NUM_CLUSTERS = int(max_clusters)
        cls.K_NN = int(k_nn)
        try:
            with _maybe_silence(verbose):
                selector = cls(cfg, l_set, u_set, int(budget), is_scan=False)
                active_set, _remain_set = selector.select_samples()
        finally:
            cls.MIN_CLUSTER_SIZE, cls.MAX_NUM_CLUSTERS, cls.K_NN = old_attrs
    return np.asarray(active_set, dtype=int)


def select_probcover_official(
    features: np.ndarray,
    budget: int,
    delta: float,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Run the upstream ProbCover class on a supplied feature matrix."""
    n_samples = int(features.shape[0])
    if budget >= n_samples:
        return np.arange(n_samples, dtype=int), float(delta), 1.0
    if budget <= 0:
        return np.array([], dtype=int), float(delta), 0.0

    cfg = {"DATASET": {"NAME": "ECG_PPG_FEATURES"}, "RNG_SEED": 0}
    l_set = np.array([], dtype=int)
    u_set = np.arange(n_samples, dtype=int)

    with _patched_typiclust_runtime(features) as (_typiclust_mod, probcover_mod):
        with _maybe_silence(verbose):
            selector = probcover_mod.ProbCover(cfg, l_set, u_set, int(budget), float(delta))
            active_set, _remain_set = selector.select_samples()

    active_set = np.asarray(active_set, dtype=int)
    graph = pairwise_distances(np.asarray(features, dtype=np.float32), metric="euclidean") < float(delta)
    coverage = float(graph[active_set].any(axis=0).mean()) if len(active_set) else 0.0
    return active_set, float(delta), coverage
