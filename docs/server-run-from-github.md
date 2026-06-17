# Server Run From GitHub

Use this when the management repository has been pushed to GitHub and you want
to run it directly on the lab server.

## Fresh Clone

```bash
cd /root
rm -rf /root/project
git clone <GITHUB_REPO_URL> /root/project
cd /root/project
```

## Environment

```bash
bash scripts/setup_remote_env.sh
```

## Official Upstream Code

The GitHub repository only stores the management layer. It does not store
`third_party/`, raw data, checkpoints, or result files. Recreate official
upstreams from the pinned manifest:

```bash
cd /root/project
. /root/project/.venv/bin/activate
python scripts/clone_upstreams.py --checkout-pins
```

For the BIDMC selection smoke, the required upstreams are mainly TypiClust,
CORDS, ELFS, and Moderate Coreset. The full manifest also includes ECG/PPG
foundation model repositories for later phases.

## Public BIDMC Data

```bash
cd /root/project
bash scripts/download_bidmc_open.sh
```

## Run The Current Selection Benchmarks

Quick integration smoke:

```bash
cd /root/project
/root/project/.venv/bin/python scripts/run_pipeline.py bidmc_selection_all_smoke
```

Feature-level BIDMC benchmark:

```bash
cd /root/project
/root/project/.venv/bin/python scripts/run_pipeline.py bidmc_selection_benchmark
```

Open ECG+PPG BIDMC pipeline:

```bash
cd /root/project
/root/project/.venv/bin/python scripts/run_pipeline.py bidmc_ecg_ppg_open
```

## Outputs

```text
results/bidmc/
results/bidmc_selection_benchmark/
results/bidmc_selection_all_smoke/
```

## MC-MED Later

After PhysioNet access is approved, set your PhysioNet credentials only in the
shell environment, not in the repository:

```bash
export PHYSIONET_USER="..."
export PHYSIONET_PASSWORD="..."
bash scripts/download_mcmed_physionet.sh
```
