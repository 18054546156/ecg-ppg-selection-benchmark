# SignalMC-MED Official Runbook

This runbook keeps the first reproduction phase as close as possible to the
official repository.

## Upstream Code

Official SignalMC-MED source is pinned in `manifests/upstreams.json` and stored
under `third_party/SignalMC-MED`.

The upstream scripts contain `INSERT-PATH-HERE` placeholders. We do not edit the
upstream files directly. Instead:

```bash
python3 scripts/materialize_signalmcmed_official.py
```

creates runnable copies in `/root/project/runs/signalmcmed_official`.

Create the remote Python environment:

```bash
bash scripts/setup_remote_env.sh
```

## Required Data Layout

```text
/root/project/data/mc-med/
  visits.csv
  labs.csv
  pmh.csv
  split_chrono_train.csv
  split_chrono_val.csv
  split_chrono_test.csv
  waveforms/

/root/project/data/signalmcmed/
  signalmc-med_csns.pkl

/root/project/checkpoints/
  papagei_s.pt
  1_lead_ECGFounder.pth
  D-BETA/checkpoints/pytorch_model.bin
```

MC-MED is a PhysioNet dataset. It must be downloaded with valid PhysioNet
credentialed access and extracted before the official SignalMC-MED scripts can
run.

Download MC-MED when PhysioNet credentials are available:

```bash
export PHYSIONET_USER="..."
export PHYSIONET_PASSWORD="..."
bash scripts/download_mcmed_physionet.sh
```

If the official `signalmc-med_csns.pkl` is not available, generate it from
MC-MED WFDB headers using the paper's 10-minute ECG+PPG overlap criterion:

```bash
/root/project/.venv/bin/python scripts/build_signalmcmed_csns.py
```

Download public checkpoints:

```bash
bash scripts/download_public_checkpoints.sh
```

D-BETA is gated on Hugging Face. Set `HF_TOKEN` after accepting its model terms
to download that checkpoint.

## Phase-1 Official Jobs

Feature extraction:

```bash
cd /root/project/runs/signalmcmed_official
CUDA_VISIBLE_DEVICES=0 /root/project/.venv/bin/python extract-features_ecg-domain-features.py
CUDA_VISIBLE_DEVICES=0 /root/project/.venv/bin/python extract-features_ppg-domain-features-60sec.py
CUDA_VISIBLE_DEVICES=0 /root/project/.venv/bin/python extract-features_papagei.py
```

Evaluation:

```bash
cd /root/project/runs/signalmcmed_official
/root/project/.venv/bin/python evaluation_test_10min_ecg-domain-features.py
/root/project/.venv/bin/python evaluation_test_10min_ppg-domain-features-60sec.py
/root/project/.venv/bin/python evaluation_test_10min_papagei.py
/root/project/.venv/bin/python get-results_main-model-comp-tables_test.py
```

The official evaluation scripts run train proportions
`1.0, 0.75, 0.50, 0.25, 0.10, 0.05`; the paper comparison table script reports
the requested `1.0, 0.50, 0.25, 0.10` aggregate.
