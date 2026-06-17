#!/usr/bin/env python3
"""Build SignalMC-MED CSN list from MC-MED when the official pkl is absent.

The SignalMC-MED paper defines the benchmark as visits with 10 minutes of
overlapping single-lead ECG and PPG. This script follows that criterion using
MC-MED's WFDB headers and writes `signalmc-med_csns.pkl`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pickle
from pathlib import Path

import pandas as pd
import wfdb


def header_start(header: wfdb.io.record.Record) -> dt.datetime:
    return dt.datetime.combine(header.base_date, header.base_time)


def has_overlap(mcmed_dir: Path, csn: int, seconds: int) -> bool:
    csn_str = str(csn)
    base = mcmed_dir / "waveforms" / csn_str[-3:] / csn_str
    ecg_path = base / "II" / f"{csn_str}_1"
    ppg_path = base / "Pleth" / f"{csn_str}_1"
    try:
        ecg = wfdb.rdheader(str(ecg_path))
        ppg = wfdb.rdheader(str(ppg_path))
    except Exception:
        return False
    if ecg.fs != 500 or ppg.fs != 125:
        return False

    ecg_start = header_start(ecg)
    ppg_start = header_start(ppg)
    ecg_end = ecg_start + dt.timedelta(seconds=ecg.sig_len / ecg.fs)
    ppg_end = ppg_start + dt.timedelta(seconds=ppg.sig_len / ppg.fs)
    overlap = (min(ecg_end, ppg_end) - max(ecg_start, ppg_start)).total_seconds()
    return overlap >= seconds


def load_split_csns(mcmed_dir: Path) -> list[int]:
    csns: list[int] = []
    for name in ["split_chrono_train.csv", "split_chrono_val.csv", "split_chrono_test.csv"]:
        path = mcmed_dir / name
        values = pd.read_csv(path, header=None)[0].dropna().astype(int).tolist()
        csns.extend(values)
    return csns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcmed-dir", default="/root/project/data/mc-med")
    parser.add_argument("--signalmcmed-dir", default="/root/project/data/signalmcmed")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--expected-count", type=int, default=22256)
    args = parser.parse_args()

    mcmed_dir = Path(args.mcmed_dir)
    signalmcmed_dir = Path(args.signalmcmed_dir)
    signalmcmed_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    csns = load_split_csns(mcmed_dir)
    for i, csn in enumerate(csns, start=1):
        if i % 1000 == 0:
            print(f"{i}/{len(csns)} checked; selected={len(selected)}")
        if has_overlap(mcmed_dir, csn, args.seconds):
            selected.append(csn)

    out = signalmcmed_dir / "signalmc-med_csns.pkl"
    with out.open("wb") as f:
        pickle.dump(selected, f)

    print(f"Wrote {len(selected)} CSNs to {out}")
    if len(selected) != args.expected_count:
        print(
            f"WARNING: selected count differs from paper target {args.expected_count}. "
            "Review MC-MED version, extraction status, and official SignalMC-MED updates."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

