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
- Formal split protocol: deterministic record-level 70/30 train/test splits
  with split seeds `42,43,44`. Test records are never used by selectors.
- Selection representation: ECG+PPG handcrafted segment features, standardized
  and reduced to 16 dimensions with PCA for distance-based selection.
- Budgets: 10%, 25%, 50% of the train segments, plus a 100% full-train baseline.
- Repeats: selector seeds `0,1,2` for every split/budget pair.
- Downstream task: predict ECG-derived heart rate on held-out records. The
  formal table uses the PPG-only downstream feature set as the primary report
  view; all downstream feature sets remain in `selection_runs.csv`.
- Metrics: test MAE, test Pearson, selected record coverage.

## Selection Methods

- `random`: uniform random subset.
- `k_center`: farthest-first CoreSet/K-Center selection, adapted to batch mode.
- `typiclust`: by default calls the upstream `third_party/TypiClust`
  `TypiClust` class through `scripts/official_selection_adapters.py`.
- `probcover`: by default calls the upstream `third_party/TypiClust`
  `ProbCover` class through `scripts/official_selection_adapters.py`. The
  radius is chosen from the BIDMC representation scale using the median
  10-nearest-neighbor distance. If the greedy cover rule returns fewer samples
  than the requested budget, the benchmark uses the fixed
  `k_center_unselected` fill policy and records both raw and final counts.
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
come from the cloned upstream repository. The TypiClust wrapper also guards
empty clusters and zero-neighbor singleton clusters; if a budget still cannot be
filled, the row is marked failed with the reason instead of reporting NaN.

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
- Default budgets are `10%,25%,50%` of the full training pool.
- Default split seeds are `42,43,44`; default selector seeds are `0,1,2`.
- Reports `full_train_reference` and records `preselect` as
  `not_run_task_mismatch`.
- Writes a formal report table to `formal_bidmc_table.csv` and
  `formal_bidmc_table.md`.

The older `bidmc_selection_all_smoke` pipeline deliberately uses a tiny
training subset and exists only to check that upstream imports and selectors can
execute. Do not use smoke outputs for research claims.

## Audit Rules

The formal benchmark must pass:

- Every passed selector row has `selected_n == requested_budget`.
- Selected indices are unique integer indices into the train pool.
- Selected indices are never test indices; record-level train/test overlap must
  be empty for every split seed.
- ProbCover under-selection is either absent or documented as
  `k_center_unselected` fill with `raw_selected_n`, `filled_n`, and final
  `selected_n`.
- TypiClust empty-cluster or zero-neighbor issues are handled before reporting;
  unresolved cases remain failed rows with a non-empty reason.

Run the audit:

```bash
python scripts/audit_bidmc_full_selection.py --out-dir results/bidmc_selection_full
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
results/bidmc_selection_full/formal_bidmc_table.csv
results/bidmc_selection_full/formal_bidmc_table.md
```
