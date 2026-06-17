#!/usr/bin/env python3
"""Run a command on the lab container via SSH."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import sys

import paramiko


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run on the remote host",
    )
    parser.add_argument("--cwd", default="/root/project")
    parser.add_argument("--host", default=os.environ.get("LAB_CONTAINER_HOST"))
    parser.add_argument("--port", default=int(os.environ.get("LAB_CONTAINER_PORT", "21022")), type=int)
    parser.add_argument("--user", default=os.environ.get("LAB_CONTAINER_USER", "root"))
    parser.add_argument("--password-env", default="LAB_CONTAINER_PASSWORD")
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
    if not args.command:
        parser.error("missing remote command")
    command = f"cd {shlex.quote(args.cwd)} && {' '.join(shlex.quote(part) for part in args.command)}"
    stdin, stdout, stderr = client.exec_command(command)
    for line in iter(stdout.readline, ""):
        print(line, end="")
    err = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if err:
        print(err, file=sys.stderr, end="")
    client.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
