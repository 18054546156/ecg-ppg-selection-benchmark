# ECG+PPG Reproduction Workspace

This repository is an experiment-management layer for reproducing the original
open-source ECG/PPG foundation-model, signal-quality, and data-selection
baselines.

Principles:

- Use the original official repositories under `third_party/`.
- Pin every upstream repository by commit in `manifests/upstreams.json`.
- Keep local code as thin orchestration only: setup, launch, sync, and result
  bookkeeping.
- Record original paper metrics separately from migrated SignalMC-MED metrics.
- Treat public-data smoke tests separately from original-paper reproduction.

Remote execution helpers read the target from environment variables or CLI
flags:

- `LAB_CONTAINER_HOST`
- `LAB_CONTAINER_PORT`, default `21022`
- `LAB_CONTAINER_USER`, default `root`
- `LAB_CONTAINER_PASSWORD`
- Project directory: `/root/project`

## Layout

```text
configs/             Experiment configs and protocol definitions.
data/                Local data placeholders only. Raw data is not committed.
docs/                Reproduction notes and original metric ledger.
manifests/           Upstream repository pins and algorithm registry.
results/             Local result summaries and downloaded logs.
scripts/             Thin setup, sync, and launch helpers.
third_party/         Official upstream source repositories.
```

## Managed Pipelines

Registered datasets, pipelines, and paper targets live in:

```text
manifests/datasets.json
manifests/pipelines.json
manifests/reproduction_targets.json
```

List the SignalMC-MED paper target values:

```bash
python scripts/check_reproduction_target.py --target signalmcmed_table2_test
```

Run the public BIDMC ECG+PPG smoke pipeline through the launcher:

```bash
python scripts/run_pipeline.py bidmc_ecg_ppg_open --remote --sync --download
```

Run the BIDMC public data-selection benchmark:

```bash
python scripts/run_pipeline.py bidmc_selection_benchmark --remote --sync
```

## First Targets

1. Reproduce official SignalMC-MED baselines at 100%, 50%, 25%, and 10% train
   proportions with five runs.
2. Migrate data-selection baselines to the same SignalMC-MED protocol.
3. Run selected original-paper sanity checks when the original datasets are
   accessible.
