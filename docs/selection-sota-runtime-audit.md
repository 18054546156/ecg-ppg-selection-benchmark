# Data Selection SOTA and Runtime Audit

Last checked: 2026-06-18.

This note separates three different questions that are easy to mix up:

1. whether a method is a current or strong data-selection baseline in its own
   paper setting;
2. whether its official code can be used without changing the core algorithm;
3. whether it is native to ECG+PPG waveform/window selection.

## Verdict

The current benchmark set is a good first SOTA/baseline panel, but it is not an
exhaustive list of all latest data-selection methods.

- Latest/near-latest methods currently represented:
  - ELFS, ICLR 2025: label-free coreset selection for vision.
  - PreSelect, ICML 2025: predictive text/LLM pretraining data selection.
  - DCoM, 2024: active learning method present in the TypiClust official repo,
    not yet adapted in our ECG+PPG runner.
- Strong modern baselines currently represented:
  - Moderate Coreset, ICLR 2023.
  - Coverage/low-budget active learning family: TypiClust and ProbCover.
  - CORDS methods: GradMatch, GLISTER, CRAIG.
  - CoreSet/K-Center and Random as required sanity baselines.
- ECG+PPG-native SOTA data selection:
  - No paper in this audit provides a direct ECG+PPG waveform-window benchmark
    comparing all these algorithms. For ECG+PPG, correctness is protocol
    alignment, not matching CIFAR/ImageNet/LLM numbers.

## Current Code Status

| Method | Original setting | Local code status | ECG+PPG status |
| --- | --- | --- | --- |
| Random | Generic sanity baseline | Local implementation | Ready |
| K-Center/CoreSet | Image active learning | Local feature-space implementation; official repo retained for provenance | Ready |
| TypiClust | Low-budget image active learning | Official class called through `scripts/official_selection_adapters.py` | Official core adapted |
| ProbCover | Low-budget image active learning | Official class called through `scripts/official_selection_adapters.py` | Official core adapted |
| DCoM | 2024 active learning for different learners | Present in `third_party/TypiClust`; not yet wired | Candidate |
| Moderate Coreset | Image data selection, robust/corrupted settings | Official `selection.py` core used | Official core adapted |
| ELFS | Label-free vision coreset selection | Official coreset score selector used; full pseudo-label/training-dynamics pipeline not yet ported | Partial official core |
| GLISTER | Supervised subset selection | CORDS official strategy used | Official workflow adapted |
| GradMatch | Supervised subset selection | CORDS official strategy used | Official workflow adapted |
| CRAIG | Gradient/submodular coreset | CORDS official strategy used; original repo retained | Official workflow adapted |
| DeepCore | Unified vision coreset library | Cloned, not yet wired into ECG runner | Candidate library |
| CCS | Coverage-centric coreset, high pruning | Cloned, not yet wired into ECG runner | Candidate |
| PreSelect | LLM/text pretraining corpus selection | Cloned; intentionally not run on waveform windows | Conceptual extension only |

## Runtime Estimate on the Current Lab Container

These are practical estimates for our ECG+PPG pipeline, not the original image
or LLM experiments.

| Scope | Methods | Expected time |
| --- | --- | --- |
| BIDMC smoke, 160 train windows, one seed | All currently wired methods except PreSelect | under 1 minute |
| BIDMC formal feature-level benchmark, 5 seeds x 4 budgets | Random, K-Center, TypiClust, ProbCover, Moderate-core, CORDS GradMatch/GLISTER/CRAIG, ELFS-core proxy | about 1-3 hours |
| BIDMC with real neural downstream training per selected subset | same selectors plus ECG/PPG model training | about 4-12 hours, depending on epochs |
| MC-MED feature extraction after access is ready | ECG+PPG encoders over all usable windows | several hours to 1 day |
| MC-MED formal benchmark, 5 seeds x 4 budgets, feature-level downstream models | current wired methods | about 0.5-2 days |
| MC-MED full neural retraining per selector/budget/seed | all runnable methods | several days to over 1 week |
| Full ELFS-style port | pseudo-label training, training dynamics, score generation, coreset retraining | multi-day, because original code uses 200-epoch clustering and 200-epoch/40k-iteration image training analogues |
| PreSelect original pipeline | text BPC, fastText filtering, LLM pretraining/eval | not applicable to ECG+PPG; original compute is LLM-scale |

## What Counts as a Correct ECG+PPG Baseline

A migrated method can enter the ECG+PPG baseline table only if:

- it keeps the original selection objective or official strategy class;
- all methods share the same ECG+PPG train/validation/test split;
- the same budget, seed list, representation, and downstream model are used;
- Random and full-train references are reported in the same table;
- the method is clearly labeled as `OFFICIAL_WORKFLOW_ADAPTED`,
  `OFFICIAL_CORE_ADAPTED`, `CANDIDATE`, or `CONCEPTUAL_EXTENSION`;
- results are reported as mean and standard deviation over seeds.

For BIDMC, use heart-rate regression metrics such as MAE and Pearson. For
MC-MED/SignalMC-MED-style tasks, use regression metrics for continuous labels
and AUROC/AUPRC or accuracy/F1 for classification labels.

## Source Notes

- ELFS README: 200-epoch cluster heads, 200-epoch training-dynamics collection,
  40k-iteration coreset training examples.
- PreSelect README: fastText scorer for text filtering, BPC calculation, and
  LLM pretraining/evaluation pipeline.
- Moderate Coreset README: CIFAR-100/Tiny ImageNet training, selection from a
  trained checkpoint, then coreset retraining.
- CORDS README: GLISTER, GradMatch, CRAIG, submodular selection, random
  strategies, mainly benchmarked on vision datasets.
- DeepCore README: unified coreset library covering many older and modern
  coreset methods on vision datasets.
- TypiClust README: official TypiClust, ProbCover, and DCoM implementation.
