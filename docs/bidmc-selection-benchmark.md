# BIDMC Selection Benchmark

This is a public-data migration benchmark for data-selection algorithms. It is
not an original-paper reproduction target.

## Input

The benchmark consumes the BIDMC ECG+PPG feature table produced by:

```bash
python scripts/run_pipeline.py bidmc_ecg_ppg_open --remote --sync --download
```

Default remote input:

```text
/root/project/results/bidmc/bidmc_segment_features.csv
```

## Protocol

- Unit of selection: 10-second BIDMC ECG+PPG segment.
- Split: deterministic record-level train/test split with seed 42 and 30% held
  out records for test.
- Selection representation: PaPaGei segment embeddings, standardized and reduced
  to 32 dimensions with PCA for distance-based selection.
- Budgets: 10%, 25%, 50% of the train segments, plus a 100% full-train baseline.
- Repeats: five seeds for stochastic selection and KMeans initialization.
- Downstream task: predict ECG-derived heart rate on held-out records.
- Metrics: test MAE, test Pearson, selected record coverage.

## Selection Methods

- `random`: uniform random subset.
- `k_center`: farthest-first CoreSet/K-Center selection, adapted to batch mode.
- `typiclust`: by default calls the upstream `third_party/TypiClust`
  `TypiClust` class through `scripts/official_selection_adapters.py`.
- `probcover`: by default calls the upstream `third_party/TypiClust`
  `ProbCover` class through `scripts/official_selection_adapters.py`. The
  radius is chosen from the BIDMC representation scale using the median
  10-nearest-neighbor distance.
- `--selector-runtime local` keeps the earlier local adapted implementation
  available for debugging only.

Official algorithm sources are pinned under:

```text
third_party/active_learning_coreset
third_party/TypiClust
```

The TypiClust/ProbCover adapter replaces the upstream image feature loader with
the BIDMC ECG+PPG feature matrix and uses a CPU distance fallback when FAISS or
CUDA is not available. The selection classes and greedy selection logic still
come from the cloned upstream repository.

## Run

```bash
python scripts/run_pipeline.py bidmc_selection_benchmark --remote --sync
```

## Full All-Method Benchmark

Use this when the result will be audited or reported. It is not the smoke test:

```bash
python scripts/run_pipeline.py bidmc_selection_full_benchmark --remote --sync
```

Protocol:

- Uses all usable BIDMC segments after the deterministic record-level 70/30
  split. No `--max-train` subsampling is applied.
- Selects from the training pool only.
- Evaluates every selected subset on the complete held-out test split.
- Default budget is 10% of the full training pool.
- Default seeds are `0,1,2`.
- Reports `full_train_reference` and records `preselect` as
  `not_run_task_mismatch`.

The older `bidmc_selection_all_smoke` pipeline deliberately uses a tiny
training subset and exists only to check that upstream imports and selectors can
execute. Do not use smoke outputs for research claims.

## Outputs

```text
/root/project/results/bidmc_selection/bidmc_selection_runs.csv
/root/project/results/bidmc_selection/bidmc_selection_summary.csv
/root/project/results/bidmc_selection/bidmc_selection_summary.json
/root/project/results/bidmc_selection/bidmc_selection_ppg_mae.png
```

Local copies should be placed under:

```text
results/bidmc_selection/
```

Full all-method benchmark outputs:

```text
results/bidmc_selection_full/selection_runs.csv
results/bidmc_selection_full/selection_status.csv
results/bidmc_selection_full/selection_summary.csv
results/bidmc_selection_full/selection_summary.json
results/bidmc_selection_full/selected_indices.json
results/bidmc_selection_full/run_config.json
results/bidmc_selection_full/audit_manifest.json
results/bidmc_selection_full/reproduce_command.sh
```
