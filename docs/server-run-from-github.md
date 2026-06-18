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

Before an auditable rerun, remove old small-sample smoke outputs so they cannot
be confused with the full result:

```bash
cd /root/project
rm -rf results/bidmc_selection_all_smoke results/bidmc_selection_full
```

Quick integration smoke:

```bash
cd /root/project
/root/project/.venv/bin/python scripts/run_pipeline.py bidmc_selection_all_smoke
```

Feature-level BIDMC benchmark with four core methods:

```bash
cd /root/project
/root/project/.venv/bin/python scripts/run_pipeline.py bidmc_selection_benchmark
```

Auditable full train-pool all-method benchmark:

```bash
cd /root/project
/root/project/.venv/bin/python scripts/run_pipeline.py bidmc_selection_full_benchmark
/root/project/.venv/bin/python scripts/audit_bidmc_full_selection.py --out-dir results/bidmc_selection_full
```

Equivalent direct command:

```bash
cd /root/project
/root/project/.venv/bin/python scripts/run_bidmc_all_selection_benchmark.py \
  --features-csv results/bidmc/bidmc_segment_features.csv \
  --out-dir results/bidmc_selection_full \
  --selection-feature-set papagei \
  --budget-fractions 0.10 \
  --seeds 0,1,2 \
  --split-seed 42 \
  --test-fraction 0.30 \
  --pca-dim 16 \
  --batch-size 8
```

Open ECG+PPG BIDMC pipeline:

```bash
cd /root/project
/root/project/.venv/bin/python scripts/run_pipeline.py bidmc_ecg_ppg_open
```

## Outputs

```text
results/bidmc/
results/bidmc_selection/
results/bidmc_selection_full/
results/bidmc_selection_all_smoke/
```

The audit files for the full benchmark are:

```text
results/bidmc_selection_full/run_config.json
results/bidmc_selection_full/audit_manifest.json
results/bidmc_selection_full/reproduce_command.sh
results/bidmc_selection_full/selection_status.csv
results/bidmc_selection_full/selection_runs.csv
results/bidmc_selection_full/selection_summary.csv
results/bidmc_selection_full/selection_summary.json
results/bidmc_selection_full/selected_indices.json
```

Expected audit ending:

```text
AUDIT PASS
source_segments=2544 train_segments=<about 70%> test_segments=<about 30%>
```

## MC-MED Later

After PhysioNet access is approved, set your PhysioNet credentials only in the
shell environment, not in the repository:

```bash
export PHYSIONET_USER="..."
export PHYSIONET_PASSWORD="..."
bash scripts/download_mcmed_physionet.sh
```
