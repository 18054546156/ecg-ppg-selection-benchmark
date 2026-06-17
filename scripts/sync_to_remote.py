#!/usr/bin/env python3
"""Sync this management repository to the lab container.

Large raw data, checkpoints, and result directories are excluded by default.
Official upstream code can either be copied from local `third_party/` or cloned
on the remote from `manifests/upstreams.json`.
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import os
from pathlib import Path

import paramiko


REMOTE_ROOT = "/root/project"
ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_PATTERNS = [
    ".git/*",
    "data/*",
    "results/*",
    "__pycache__/*",
    "*/__pycache__/*",
    ".pytest_cache/*",
    "*/.pytest_cache/*",
    ".venv/*",
    "venv/*",
]


def excluded(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(fnmatch.fnmatch(rel, pattern) for pattern in EXCLUDE_PATTERNS)


def ensure_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = [part for part in path.split("/") if part]
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("LAB_CONTAINER_HOST"))
    parser.add_argument("--port", default=int(os.environ.get("LAB_CONTAINER_PORT", "21022")), type=int)
    parser.add_argument("--user", default=os.environ.get("LAB_CONTAINER_USER", "root"))
    parser.add_argument("--password-env", default="LAB_CONTAINER_PASSWORD")
    parser.add_argument("--include-third-party", action="store_true")
    args = parser.parse_args()

    if not args.host:
        parser.error("missing --host or LAB_CONTAINER_HOST")

    password = os.environ.get(args.password_env)
    if not password:
        password = getpass.getpass("SSH password: ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    sftp = client.open_sftp()
    ensure_dir(sftp, REMOTE_ROOT)

    copied = 0
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if excluded(rel):
            continue
        if rel.startswith("third_party/") and not args.include_third_party:
            continue
        remote_path = f"{REMOTE_ROOT}/{rel}"
        if path.is_dir():
            ensure_dir(sftp, remote_path)
        else:
            ensure_dir(sftp, str(Path(remote_path).parent).replace("\\", "/"))
            sftp.put(str(path), remote_path)
            copied += 1
            print(f"[copy] {rel}")

    sftp.close()
    client.close()
    print(f"Copied {copied} files to {REMOTE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
