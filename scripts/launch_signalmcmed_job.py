#!/usr/bin/env python3
"""Launch one materialized SignalMC-MED official script on the remote machine."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import paramiko


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", help="Script name inside /root/project/runs/signalmcmed_official")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host", default=os.environ.get("LAB_CONTAINER_HOST"))
    parser.add_argument("--port", default=int(os.environ.get("LAB_CONTAINER_PORT", "21022")), type=int)
    parser.add_argument("--user", default=os.environ.get("LAB_CONTAINER_USER", "root"))
    parser.add_argument("--password-env", default="LAB_CONTAINER_PASSWORD")
    parser.add_argument("--project-root", default="/root/project")
    parser.add_argument("--python", default="/root/project/.venv/bin/python")
    args = parser.parse_args()

    if not args.host:
        parser.error("missing --host or LAB_CONTAINER_HOST")

    password = os.environ.get(args.password_env)
    if not password:
        print(f"Missing password env: {args.password_env}", file=sys.stderr)
        return 2

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = args.script.replace("/", "_").replace(".py", "")
    log_path = f"{args.project_root}/results/logs/{timestamp}_{safe_name}.log"
    cmd = (
        f"mkdir -p {args.project_root}/results/logs && "
        f"cd {args.project_root}/runs/signalmcmed_official && "
        f"CUDA_VISIBLE_DEVICES={args.gpu} nohup {args.python} {args.script} "
        f"> {log_path} 2>&1 & echo $! && echo {log_path}"
    )

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
    stdin, stdout, stderr = client.exec_command(cmd)
    print(stdout.read().decode("utf-8", errors="replace").strip())
    err = stderr.read().decode("utf-8", errors="replace")
    client.close()
    if err:
        print(err, file=sys.stderr, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
