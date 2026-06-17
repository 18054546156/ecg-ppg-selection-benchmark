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
