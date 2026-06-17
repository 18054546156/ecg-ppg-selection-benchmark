# Reproduction Criteria

## Three Result Levels

1. Smoke pipeline: a public-data engineering check. BIDMC belongs here. It
   proves the ECG+PPG path can download WFDB records, extract synchronized
   segments and features, load a checkpoint, and produce sanity metrics. It is
   not a paper reproduction result for SignalMC-MED.
2. Original-paper reproduction: run the original official repository on the
   original dataset, split, protocol, and metrics. This is the only result that
   should be described as reproducing the paper.
3. Migration experiment: run a method on our common ECG+PPG data-selection
   protocol. These results are useful for our project, but they must be labeled
   as migrated comparisons rather than original-paper reproduction.

## SignalMC-MED Paper Target

Primary source:

- Paper: https://arxiv.org/abs/2603.09940
- Official code: `third_party/SignalMC-MED`
- Target manifest: `manifests/reproduction_targets.json`

The SignalMC-MED paper evaluates 22,256 emergency department visits with
synchronized 10-minute single-lead ECG and PPG. Its benchmark has 20 downstream
tasks grouped as:

- Age regression
- Sex classification
- ED disposition classification
- Laboratory value regression, averaged across 8 lab tasks
- Prior ICD-10 diagnosis classification, averaged across 9 diagnosis groups

The paper Table 2 reports test-set results aggregated over train proportions
10%, 25%, 50%, and 100%. Regression tasks use Pearson correlation. Classification
tasks use AUROC. Downstream training and hyperparameter selection are repeated
5 times.

## Acceptance Rule

Strict reproduction means all of the following are true:

- MC-MED data version and SignalMC-MED CSN selection match the paper.
- Chronological train/validation/test split is unchanged, with no patient
  overlap.
- Official upstream code and pinned commit are used.
- Feature extraction, visit-level aggregation, downstream model type, validation
  hyperparameter selection, train proportions, and number of runs match the
  official scripts.
- The final aggregated Table 2 cells match the paper after rounding to 3
  decimals.

Practical acceptance for small numerical drift:

- A reported cell passes if `abs(local_mean - paper_mean) <= max(0.005,
  paper_std)`.
- Any systematic drift across many cells is a failure even if individual cells
  are near tolerance, because it usually means a data/split/checkpoint mismatch.

Use:

```bash
python scripts/check_reproduction_target.py --target signalmcmed_table2_test
```

When we have an aggregated local result JSON matching the target row schema:

```bash
python scripts/check_reproduction_target.py \
  --target signalmcmed_table2_test \
  --result results/signalmcmed/table2_aggregated.json
```

## BIDMC Status

BIDMC has no SignalMC-MED paper target. Its current pass condition is:

- all 53 BIDMC records are readable;
- synchronized ECG and PPG segments are extracted;
- ECG-derived and PPG-derived heart-rate estimates are finite for nearly all
  segments;
- grouped cross-validation emits finite MAE/Pearson metrics;
- result files are written under `results/bidmc/` locally and
  `/root/project/results/bidmc/` remotely.

Current BIDMC result is recorded under `results/bidmc/bidmc_summary.json`.
