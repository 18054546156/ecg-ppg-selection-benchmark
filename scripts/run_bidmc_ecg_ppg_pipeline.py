#!/usr/bin/env python3
"""Run a public ECG+PPG smoke pipeline on the BIDMC dataset.

The task is intentionally a sanity benchmark rather than a replacement for
SignalMC-MED: use PPG-derived features to predict ECG-derived heart rate with
record-grouped cross validation, while also saving synchronized ECG/PPG segment
quality features.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import torch
import wfdb
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def read_records(data_dir: Path) -> list[str]:
    records_file = data_dir / "RECORDS"
    if records_file.exists():
        return [
            line.strip()
            for line in records_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return sorted(path.stem for path in data_dir.glob("*.hea"))


def find_channel(sig_names: list[str], candidates: list[str]) -> int | None:
    normalized = [name.upper() for name in sig_names]
    for candidate in candidates:
        candidate = candidate.upper()
        for i, name in enumerate(normalized):
            if name == candidate:
                return i
    for candidate in candidates:
        candidate = candidate.upper()
        for i, name in enumerate(normalized):
            if candidate in name:
                return i
    return None


def resample_to(signal: np.ndarray, fs: float, target_fs: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if int(round(fs)) == target_fs:
        return signal
    target_len = int(round(signal.shape[0] * target_fs / fs))
    return scipy.signal.resample(signal, target_len)


def robust_zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(signal)
    if not finite.any():
        return np.zeros_like(signal)
    median = np.nanmedian(signal)
    iqr = np.nanpercentile(signal, 75) - np.nanpercentile(signal, 25)
    if not np.isfinite(iqr) or iqr < 1e-8:
        iqr = np.nanstd(signal)
    return (np.nan_to_num(signal, nan=median) - median) / (iqr + 1e-8)


def clean_ecg(signal: np.ndarray, fs: int) -> np.ndarray:
    try:
        return np.asarray(nk.ecg_clean(signal, sampling_rate=fs), dtype=np.float64)
    except Exception:
        return robust_zscore(signal)


def clean_ppg(signal: np.ndarray, fs: int) -> np.ndarray:
    try:
        return np.asarray(nk.ppg_clean(signal, sampling_rate=fs), dtype=np.float64)
    except Exception:
        return robust_zscore(signal)


def estimate_ecg_hr(signal: np.ndarray, fs: int) -> float:
    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.float64)
    except Exception:
        return float("nan")
    return peaks_to_hr(peaks, fs)


def estimate_ppg_hr(signal: np.ndarray, fs: int) -> float:
    try:
        _, info = nk.ppg_peaks(signal, sampling_rate=fs)
        peaks = np.asarray(info.get("PPG_Peaks", []), dtype=np.float64)
    except Exception:
        return float("nan")
    return peaks_to_hr(peaks, fs)


def peaks_to_hr(peaks: np.ndarray, fs: int) -> float:
    if peaks.size < 2:
        return float("nan")
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 0.25) & (ibi < 2.5)]
    if ibi.size == 0:
        return float("nan")
    return float(60.0 / np.median(ibi))


def quality(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    finite_fraction = float(np.isfinite(signal).mean())
    if finite_fraction < 0.95:
        return finite_fraction
    z = robust_zscore(signal)
    std = float(np.std(z))
    flat_fraction = float(np.mean(np.abs(np.diff(z)) < 1e-6)) if z.size > 1 else 1.0
    clipped_std_score = min(std, 3.0) / 3.0
    return max(0.0, min(1.0, finite_fraction * clipped_std_score * (1.0 - flat_fraction)))


def basic_features(signal: np.ndarray, prefix: str) -> dict[str, float]:
    signal = robust_zscore(signal)
    if signal.size == 0:
        return {f"{prefix}_{name}": float("nan") for name in BASIC_FEATURE_NAMES}
    diffs = np.diff(signal)
    features = {
        f"{prefix}_mean": float(np.mean(signal)),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_min": float(np.min(signal)),
        f"{prefix}_max": float(np.max(signal)),
        f"{prefix}_iqr": float(np.percentile(signal, 75) - np.percentile(signal, 25)),
        f"{prefix}_mad": float(np.median(np.abs(signal - np.median(signal)))),
        f"{prefix}_skew": float(scipy.stats.skew(signal, nan_policy="omit")),
        f"{prefix}_kurtosis": float(scipy.stats.kurtosis(signal, nan_policy="omit")),
        f"{prefix}_diff_std": float(np.std(diffs)) if diffs.size else 0.0,
        f"{prefix}_zero_cross": float(np.mean(np.diff(np.signbit(signal)) != 0)) if signal.size > 1 else 0.0,
    }
    return features


BASIC_FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "iqr",
    "mad",
    "skew",
    "kurtosis",
    "diff_std",
    "zero_cross",
]


def load_papagei(checkpoint: Path, device: str):
    sys.path.insert(0, str(ROOT / "third_party" / "papagei-foundation-model"))
    from linearprobing.utils import load_model_without_module_prefix
    from models.resnet import ResNet1DMoE

    model_config = {
        "base_filters": 32,
        "kernel_size": 3,
        "stride": 2,
        "groups": 1,
        "n_block": 18,
        "n_classes": 512,
        "n_experts": 3,
    }
    model = ResNet1DMoE(
        in_channels=1,
        base_filters=model_config["base_filters"],
        kernel_size=model_config["kernel_size"],
        stride=model_config["stride"],
        groups=model_config["groups"],
        n_block=model_config["n_block"],
        n_classes=model_config["n_classes"],
        n_experts=model_config["n_experts"],
    )
    model = load_model_without_module_prefix(model, str(checkpoint))
    model.to(device)
    model.eval()
    return model


def papagei_embeddings(
    model,
    ppg_segments: list[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    if not ppg_segments:
        return np.empty((0, 512), dtype=np.float32)
    tensor = torch.tensor(np.stack(ppg_segments).astype(np.float32)).unsqueeze(1)
    outputs = []
    with torch.inference_mode():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device)
            out = model(batch)[0].detach().cpu().numpy()
            outputs.append(out)
    return np.concatenate(outputs, axis=0)


def cross_val_regression(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> dict[str, float]:
    usable = df[feature_cols + [label_col, "record"]].replace([np.inf, -np.inf], np.nan).dropna()
    if usable["record"].nunique() < 2 or len(usable) < 10:
        return {"n": int(len(usable)), "mae": float("nan"), "pearson": float("nan")}
    groups = usable["record"].to_numpy()
    n_splits = min(5, usable["record"].nunique())
    splitter = GroupKFold(n_splits=n_splits)
    y_true_all = []
    y_pred_all = []
    X = usable[feature_cols].to_numpy(dtype=np.float64)
    y = usable[label_col].to_numpy(dtype=np.float64)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    corr = float(pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 1e-8 else float("nan")
    return {"n": int(len(y_true)), "mae": mae, "pearson": corr}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/project/data/bidmc")
    parser.add_argument("--out-dir", default="/root/project/results/bidmc")
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--target-fs", type=int, default=125)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--use-papagei", action="store_true")
    parser.add_argument("--papagei-checkpoint", default="/root/project/checkpoints/papagei_s.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_records(data_dir)
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise FileNotFoundError(f"No BIDMC records found in {data_dir}")

    segment_len = args.segment_seconds * args.target_fs
    rows: list[dict[str, float | int | str]] = []
    ppg_for_papagei: list[np.ndarray] = []
    papagei_row_indices: list[int] = []

    for record_i, record_name in enumerate(records, start=1):
        record_path = data_dir / record_name
        try:
            record = wfdb.rdrecord(str(record_path))
        except Exception as exc:
            print(f"[skip] {record_name}: {exc}")
            continue
        sig_names = list(record.sig_name)
        ecg_idx = find_channel(sig_names, ["II", "ECG", "V", "AVR"])
        ppg_idx = find_channel(sig_names, ["PLETH", "PPG"])
        if ecg_idx is None or ppg_idx is None:
            print(f"[skip] {record_name}: no ECG/PPG channels in {sig_names}")
            continue
        fs = float(record.fs)
        ecg = resample_to(record.p_signal[:, ecg_idx], fs, args.target_fs)
        ppg = resample_to(record.p_signal[:, ppg_idx], fs, args.target_fs)
        n = min(ecg.shape[0], ppg.shape[0])
        ecg = ecg[:n]
        ppg = ppg[:n]
        n_segments = n // segment_len
        print(f"[record] {record_i}/{len(records)} {record_name}: {n_segments} segments")
        for segment_i in range(n_segments):
            start = segment_i * segment_len
            end = start + segment_len
            ecg_raw = ecg[start:end]
            ppg_raw = ppg[start:end]
            ecg_clean = clean_ecg(ecg_raw, args.target_fs)
            ppg_clean = clean_ppg(ppg_raw, args.target_fs)

            ecg_hr = estimate_ecg_hr(ecg_clean, args.target_fs)
            ppg_hr = estimate_ppg_hr(ppg_clean, args.target_fs)
            row = {
                "record": record_name,
                "segment": segment_i,
                "start_sec": segment_i * args.segment_seconds,
                "duration_sec": args.segment_seconds,
                "fs": args.target_fs,
                "ecg_channel": sig_names[ecg_idx],
                "ppg_channel": sig_names[ppg_idx],
                "ecg_hr_bpm": ecg_hr,
                "ppg_hr_bpm": ppg_hr,
                "hr_abs_diff": abs(ecg_hr - ppg_hr)
                if np.isfinite(ecg_hr) and np.isfinite(ppg_hr)
                else float("nan"),
                "ecg_quality": quality(ecg_raw),
                "ppg_quality": quality(ppg_raw),
            }
            row.update(basic_features(ecg_clean, "ecg"))
            row.update(basic_features(ppg_clean, "ppg"))
            rows.append(row)
            if args.use_papagei:
                ppg_for_papagei.append(robust_zscore(ppg_clean).astype(np.float32))
                papagei_row_indices.append(len(rows) - 1)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No synchronized ECG+PPG segments were extracted.")

    if args.use_papagei:
        checkpoint = Path(args.papagei_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        model = load_papagei(checkpoint, device)
        emb = papagei_embeddings(model, ppg_for_papagei, device, args.batch_size)
        emb_cols = [f"papagei_{i}" for i in range(emb.shape[1])]
        emb_df = pd.DataFrame(np.nan, index=df.index, columns=emb_cols)
        for emb_i, row_i in enumerate(papagei_row_indices):
            emb_df.loc[row_i, emb_cols] = emb[emb_i]
        df = pd.concat([df, emb_df], axis=1)
    else:
        emb_cols = []

    feature_csv = out_dir / "bidmc_segment_features.csv"
    df.to_csv(feature_csv, index=False)

    ppg_cols = [col for col in df.columns if col.startswith("ppg_") and col != "ppg_channel"]
    ecg_cols = [col for col in df.columns if col.startswith("ecg_") and col not in {"ecg_channel", "ecg_hr_bpm"}]
    summary = {
        "dataset": "BIDMC PPG and Respiration Dataset",
        "records_seen": len(records),
        "records_used": int(df["record"].nunique()),
        "segments": int(len(df)),
        "segment_seconds": args.segment_seconds,
        "target_fs": args.target_fs,
        "finite_ecg_hr_segments": int(np.isfinite(df["ecg_hr_bpm"]).sum()),
        "finite_ppg_hr_segments": int(np.isfinite(df["ppg_hr_bpm"]).sum()),
        "median_hr_abs_diff": float(np.nanmedian(df["hr_abs_diff"])),
        "ppg_to_ecg_hr": cross_val_regression(df, ppg_cols, "ecg_hr_bpm"),
        "ecg_ppg_to_ecg_hr": cross_val_regression(df, ecg_cols + ppg_cols, "ecg_hr_bpm"),
    }
    if emb_cols:
        summary["papagei_to_ecg_hr"] = cross_val_regression(df, emb_cols, "ecg_hr_bpm")

    summary_json = out_dir / "bidmc_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_df = df[["ecg_hr_bpm", "ppg_hr_bpm", "record"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not plot_df.empty:
        plt.figure(figsize=(6, 6))
        plt.scatter(plot_df["ecg_hr_bpm"], plot_df["ppg_hr_bpm"], s=10, alpha=0.45)
        lo = float(min(plot_df["ecg_hr_bpm"].min(), plot_df["ppg_hr_bpm"].min()))
        hi = float(max(plot_df["ecg_hr_bpm"].max(), plot_df["ppg_hr_bpm"].max()))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        plt.xlabel("ECG-derived HR (bpm)")
        plt.ylabel("PPG-derived HR (bpm)")
        plt.title("BIDMC ECG/PPG HR Agreement")
        plt.tight_layout()
        plt.savefig(out_dir / "bidmc_hr_agreement.png", dpi=160)
        plt.close()

    print(json.dumps(summary, indent=2))
    print(f"Wrote {feature_csv}")
    print(f"Wrote {summary_json}")
    return 0 if math.isfinite(summary["ppg_to_ecg_hr"]["mae"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
