#!/usr/bin/env bash
set -euo pipefail

cd /root/project
mkdir -p checkpoints checkpoints/D-BETA/checkpoints

if [[ -d .venv ]]; then
  . .venv/bin/activate
fi

wget -c -O checkpoints/papagei_s.pt \
  "https://zenodo.org/records/13983110/files/papagei_s.pt?download=1"

python -m pip install -q "huggingface_hub[hf_xet]"

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path("/root/project/checkpoints")

ecgfounder = hf_hub_download(
    repo_id="PKUDigitalHealth/ECGFounder",
    filename="1_lead_ECGFounder.pth",
    local_dir=root,
    local_dir_use_symlinks=False,
)
print(f"Downloaded ECGFounder: {ecgfounder}")

token = os.environ.get("HF_TOKEN")
if token:
    dbeta = hf_hub_download(
        repo_id="Manhph2211/D-BETA",
        filename="pytorch_model.bin",
        token=token,
        local_dir=root / "D-BETA" / "checkpoints",
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded D-BETA: {dbeta}")
else:
    print("HF_TOKEN is not set; skipping gated D-BETA checkpoint.")
PY

