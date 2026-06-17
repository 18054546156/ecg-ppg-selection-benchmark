#!/usr/bin/env python3
"""List or compare registered paper reproduction targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "manifests" / "reproduction_targets.json"
METRIC_KEYS = ["age_reg", "sex_cls", "ed_dispo_cls", "labs_reg", "icd10_cls"]


def load_target(target_id: str) -> dict:
    manifest = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    for target in manifest["targets"]:
        if target["id"] == target_id:
            return target
    known = ", ".join(target["id"] for target in manifest["targets"])
    raise SystemExit(f"Unknown target '{target_id}'. Known targets: {known}")


def row_key(row: dict) -> tuple[str, str]:
    return row["model"], row["modality"]


def metric_mean(value: float | list[float]) -> float:
    if isinstance(value, list):
        return float(value[0])
    return float(value)


def print_target(target: dict) -> None:
    print(f"{target['id']} :: {target['table']} :: {target['source_url']}")
    print("model\tmodality\tage_reg\tsex_cls\ted_dispo_cls\tlabs_reg\ticd10_cls")
    for row in target["rows"]:
        values = [f"{row[key][0]:.3f}+/-{row[key][1]:.3f}" for key in METRIC_KEYS]
        print("\t".join([row["model"], row["modality"], *values]))


def compare(target: dict, result_path: Path) -> int:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_rows = {row_key(row): row for row in result["rows"]}
    failures = 0
    for target_row in target["rows"]:
        key = row_key(target_row)
        if key not in result_rows:
            print(f"FAIL missing row: {key[0]} / {key[1]}")
            failures += len(METRIC_KEYS)
            continue
        result_row = result_rows[key]
        for metric_key in METRIC_KEYS:
            paper_mean, paper_std = target_row[metric_key]
            local_mean = metric_mean(result_row[metric_key])
            tolerance = max(0.005, paper_std)
            delta = abs(local_mean - paper_mean)
            status = "PASS" if delta <= tolerance else "FAIL"
            if status == "FAIL":
                failures += 1
            print(
                f"{status} {key[0]} / {key[1]} / {metric_key}: "
                f"local={local_mean:.3f} paper={paper_mean:.3f} tol={tolerance:.3f}"
            )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="signalmcmed_table2_test")
    parser.add_argument("--result", type=Path, help="Aggregated result JSON with rows matching the target manifest.")
    args = parser.parse_args()

    target = load_target(args.target)
    if not args.result:
        print_target(target)
        return 0
    return compare(target, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
