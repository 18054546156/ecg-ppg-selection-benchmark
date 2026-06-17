#!/usr/bin/env python3
"""Clone or update official upstream repositories and pin their commits."""

from __future__ import annotations

import json
import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "upstreams.json"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stdout.strip())
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkout-pins",
        action="store_true",
        help="Checkout commits already recorded in the manifest instead of updating pins.",
    )
    args = parser.parse_args()

    upstreams = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key, item in upstreams.items():
        path = ROOT / item["local_path"]
        url = item["url"]
        pinned_commit = item.get("commit")
        item.pop("clone_error", None)
        try:
            if path.exists() and (path / ".git").exists():
                print(f"[update] {key}: {path}")
                run(["git", "fetch", "--all", "--tags", "--prune"], cwd=path)
            else:
                print(f"[clone] {key}: {url} -> {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                run(["git", "clone", url, str(path)])

            if args.checkout_pins and pinned_commit:
                print(f"[checkout] {key}: {pinned_commit}")
                run(["git", "checkout", "--detach", pinned_commit], cwd=path)

            commit = run(["git", "rev-parse", "HEAD"], cwd=path)
            item["commit"] = commit
            print(f"[pin] {key}: {commit}")
        except Exception as exc:
            item["commit"] = None
            item["clone_error"] = str(exc)
            print(f"[error] {key}: {exc}")

    MANIFEST.write_text(
        json.dumps(upstreams, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
