# Reproduction Ledger

The ledger separates two kinds of results.

1. Original-paper reproduction: run the original repository on the original
   task, dataset, split, and metric whenever the data is accessible.
2. SignalMC-MED migration: use the same SignalMC-MED splits, train
   proportions, and metrics for every model or selection method.

## Phase 1

| Method | Source | Original metric target | Local status | Remote status |
| --- | --- | --- | --- | --- |
| BIDMC public ECG+PPG smoke pipeline | Public BIDMC WFDB data | Not an original-paper target; public pipeline sanity metrics only | passed | passed |
| SignalMC-MED official baselines | Official SignalMC-MED repo | Table 2 targets in `manifests/reproduction_targets.json`; Pearson for regression, AUROC for classification | scaffolded | pending MC-MED access |
| PaPaGei | Official PaPaGei repo | Paper reports average classification and regression gains on PPG tasks | scaffolded | pending data/checkpoints |
| ECG-FM | Official ECG-FM repo | Paper reports AF and LVEF AUROC targets | scaffolded | pending data/checkpoints |
| CoreSet/K-Center | Official CoreSet repo | Original CV active-learning benchmark | scaffolded | pending embeddings |
| TypiClust/ProbCover | Official TypiClust repo | Original low-budget AL benchmark | scaffolded | pending embeddings |
| BIDMC all selection-method integration smoke | Official/adapted upstream selectors | Not an original-paper target; checks adapters for Random, K-Center, TypiClust, ProbCover, Moderate, ELFS-core, GradMatch, GLISTER, CRAIG, and PreSelect status | passed | passed |
| vital_sqi | Official vital_sqi repo | SQI extraction and quality assignment | scaffolded | pending waveforms |

See `docs/reproduction-criteria.md` for the acceptance rule used to decide
whether a result is a paper reproduction or only a migrated/smoke result.

## SignalMC-MED Migration Protocol

| Field | Value |
| --- | --- |
| Train proportions | 100%, 50%, 25%, 10% |
| Runs | 5 |
| Classification metric | AUROC |
| Regression metric | Pearson |
| Selection budgets | 10%, 25%, 50% |
| Modalities | ECG, PPG, ECG+PPG |

## Data Access Notes

Raw clinical data and model checkpoints are intentionally not committed to git.
Put local data under `data/` and remote data under `/root/project/data/`.
