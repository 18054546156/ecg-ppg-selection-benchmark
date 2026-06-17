#!/usr/bin/env python3
"""Create runnable SignalMC-MED official scripts without editing upstream code."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "SignalMC-MED"


def quote(path: str) -> str:
    return path.rstrip("/")


def replace_paths(text: str, args: argparse.Namespace) -> str:
    line_replacements = {
        "mcmed_dir_path": f'mcmed_dir_path = "{quote(args.mcmed_dir)}"',
        "signalmcmed_dir_path": f'signalmcmed_dir_path = "{quote(args.signalmcmed_dir)}"',
    }
    for name, replacement in line_replacements.items():
        text = re.sub(
            rf"^{name} = \"INSERT-PATH-HERE\".*$",
            replacement,
            text,
            flags=re.MULTILINE,
        )

    replacements = {
        'model_path = "INSERT-PATH-HERE/papagei_s.pt"': (
            f'model_path = "{quote(args.checkpoint_dir)}/papagei_s.pt"'
        ),
        'model_path = "INSERT-PATH-HERE/1_lead_ECGFounder.pth"': (
            f'model_path = "{quote(args.checkpoint_dir)}/1_lead_ECGFounder.pth"'
        ),
        'model_path = "INSERT-PATH-HERE/D-BETA/sample.pt"': (
            f'model_path = "{quote(args.checkpoint_dir)}/D-BETA/checkpoints/pytorch_model.bin"'
        ),
        'model_config_path = "INSERT-PATH-HERE/D-BETA/configs/config.json"': (
            f'model_config_path = "{quote(args.repo_root)}/third_party/D-BETA/configs/config.json"'
        ),
        'csfm_checkpoint_path = "INSERT-PATH-HERE"': (
            f'csfm_checkpoint_path = "{quote(args.checkpoint_dir)}/csfm-base.pt"'
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))
        else:
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcmed-dir", default="/root/project/data/mc-med")
    parser.add_argument("--signalmcmed-dir", default="/root/project/data/signalmcmed")
    parser.add_argument("--checkpoint-dir", default="/root/project/checkpoints")
    parser.add_argument("--repo-root", default="/root/project")
    parser.add_argument("--out-dir", default="/root/project/runs/signalmcmed_official")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in UPSTREAM.rglob("*"):
        rel = path.relative_to(UPSTREAM)
        if ".git" in rel.parts:
            continue
        target = out_dir / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".py":
            text = path.read_text(encoding="utf-8")
            target.write_text(replace_paths(text, args), encoding="utf-8")
        else:
            shutil.copy2(path, target)

    third_party = Path(args.repo_root) / "third_party"
    links = {
        "papagei-foundation-model": third_party / "papagei-foundation-model",
        "ECGFounder": third_party / "ECGFounder",
        "D-BETA": third_party / "D-BETA",
        "xecg": third_party / "bench-xecg",
    }
    for name, src in links.items():
        if src.exists():
            link_or_copy(src, out_dir / name)

    print(f"Materialized official SignalMC-MED scripts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
