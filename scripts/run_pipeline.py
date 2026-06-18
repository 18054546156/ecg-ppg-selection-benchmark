#!/usr/bin/env python3
"""Run a registered pipeline from manifests/pipelines.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINES_PATH = ROOT / "manifests" / "pipelines.json"
REMOTE_ROOT = os.environ.get("LAB_CONTAINER_PROJECT_ROOT", "/root/project")


def load_pipeline(pipeline_id: str) -> dict:
    manifest = json.loads(PIPELINES_PATH.read_text(encoding="utf-8"))
    for pipeline in manifest["pipelines"]:
        if pipeline["id"] == pipeline_id:
            return pipeline
    known = ", ".join(pipeline["id"] for pipeline in manifest["pipelines"])
    raise SystemExit(f"Unknown pipeline '{pipeline_id}'. Known pipelines: {known}")


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def remote_exec(command: str) -> None:
    run([sys.executable, "scripts/remote_exec.py", "--cwd", REMOTE_ROOT, command])


def run_bidmc(args: argparse.Namespace, pipeline: dict) -> None:
    if args.remote:
        if args.sync:
            run([sys.executable, "scripts/sync_to_remote.py", "--remote-root", REMOTE_ROOT])
        if args.download:
            remote_exec(f"bash {pipeline['download_script']}")
        extra_args = args.extra_args or pipeline.get("default_args", [])
        remote_exec(f"{REMOTE_ROOT}/.venv/bin/python " + pipeline["run_script"] + " " + " ".join(extra_args))
        return

    if args.download:
        run(["bash", pipeline["download_script"]])
    extra_args = args.extra_args or pipeline.get("default_args", [])
    run([sys.executable, pipeline["run_script"], *extra_args])


def run_python_pipeline(args: argparse.Namespace, pipeline: dict) -> None:
    if args.remote:
        if args.sync:
            run([sys.executable, "scripts/sync_to_remote.py", "--remote-root", REMOTE_ROOT])
        extra_args = args.extra_args or pipeline.get("default_args", [])
        remote_exec(f"{REMOTE_ROOT}/.venv/bin/python " + pipeline["run_script"] + " " + " ".join(extra_args))
        return

    extra_args = args.extra_args or pipeline.get("default_args", [])
    run([sys.executable, pipeline["run_script"], *extra_args])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline_id")
    parser.add_argument("--remote", action="store_true", help="Run through scripts/remote_exec.py on the lab container.")
    parser.add_argument("--sync", action="store_true", help="Sync local management code before remote execution.")
    parser.add_argument("--download", action="store_true", help="Run the registered download step before the pipeline.")
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args

    pipeline = load_pipeline(args.pipeline_id)
    if pipeline["id"] == "bidmc_ecg_ppg_open":
        run_bidmc(args, pipeline)
        return 0
    if pipeline["id"] in {"bidmc_selection_benchmark", "bidmc_selection_all_smoke", "bidmc_selection_full_benchmark"}:
        run_python_pipeline(args, pipeline)
        return 0

    raise SystemExit(
        f"Pipeline '{pipeline['id']}' is registered but not directly runnable by this launcher yet. "
        f"Use {pipeline.get('run_script')}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
