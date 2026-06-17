#!/usr/bin/env bash
set -euo pipefail

cd /root/project

python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r configs/requirements-signalmcmed-phase1.txt
python scripts/check_imports.py

