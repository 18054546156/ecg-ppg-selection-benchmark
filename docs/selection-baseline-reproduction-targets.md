# Selection Baseline Reproduction Targets

This project separates original-paper reproduction from ECG+PPG migration. The
selection papers were not originally all compared on ECG+PPG, so ECG results
cannot be judged by matching their original image/text accuracy numbers.

## Original Paper Comparison Axes

| Method | Original comparison setting | Original metric style | What the paper establishes |
| --- | --- | --- | --- |
| CoreSet/K-Center | Batch active learning for image classification. | Test accuracy or error vs labeled budget. | Core-set selection can outperform common active-learning heuristics in CNN batch active learning. |
| TypiClust | Low-budget active learning on image datasets. | Classification accuracy vs annotation budget. | Typical examples are favored in the low-budget regime; with semi-supervised training, CIFAR-10 with 10 selected labels reaches 93.2% accuracy in the paper abstract. |
| ProbCover | Low-budget active learning through probability coverage. | Classification accuracy vs annotation budget. | The paper reports strong low-budget active-learning performance by maximizing coverage in representation space. |
| GLISTER | Efficient/robust supervised subset selection. | Test accuracy, validation likelihood, robustness under noise/imbalance, runtime. | GLISTER is compared against CRAIG and other baselines; the paper reports it can be comparable to full training with 50% data in some settings. |
| GradMatch | Efficient supervised data subset selection. | Test accuracy vs subset fraction and training time. | GradMatch matches full/validation gradients and is compared with GLISTER and CRAIG variants; paper discussion emphasizes faster convergence. |
| CRAIG | Data-efficient training via gradient coreset. | Objective quality, test accuracy, and speedup. | CRAIG approximates the full gradient with a weighted subset and reports up to multi-x speedups while preserving accuracy. |
| Moderate Coreset | CIFAR-100/Tiny-ImageNet and perturbed image settings. | Top-1 image classification accuracy vs selection ratio. | Moderate selects examples near the score median and outperforms prior baselines in many corrupted/noisy settings. |
| ELFS | Label-free image coreset selection. | Image classification accuracy after pruning/selection. | ELFS improves label-free coreset baselines; the abstract reports +5.3% on CIFAR-10 and +7.1% on CIFAR-100 at 90% pruning over the best baseline. |
| PreSelect | LLM pretraining corpus selection. | Downstream LM benchmark performance at fixed token/compute budget. | PreSelect is a text pipeline; the paper reports 30B selected tokens can beat a 300B-token vanilla baseline. It is not a native ECG+PPG selector. |

## ECG+PPG Acceptance Rule

An ECG+PPG baseline is correct only if all of these hold:

- The upstream selector or workflow is called from the cloned official code
  where technically possible.
- Dataset/model adapters are explicitly labeled as adapters, not original-paper
  reproduction.
- The selection unit is a synchronized ECG+PPG window or MC-MED visit segment
  with stable sample IDs and record/patient grouped splits.
- All methods use the same train/validation/test split, budget, random seeds,
  feature representation, downstream model, and metric.
- The selector only returns indices; downstream evaluation is performed by the
  common ECG+PPG protocol.

For BIDMC public smoke, the only pass condition is valid execution:

- every runnable selector returns the requested number of in-range unique
  indices;
- downstream HR regression produces finite MAE and Pearson values;
- selected record coverage is reported;
- PreSelect is recorded as task-mismatch rather than forced into a waveform
  experiment.

For formal MC-MED evaluation, use SignalMC-MED-style metrics:

| Task group | Metric |
| --- | --- |
| Age regression | Pearson correlation |
| Lab regression | Pearson correlation, averaged by lab task |
| Sex classification | AUROC |
| ED disposition classification | AUROC |
| ICD-10 group classification | AUROC, averaged by diagnosis group |

Train proportions should be 10%, 25%, 50%, and 100%, repeated for 5 seeds/runs
when compute allows. A method is a strong ECG+PPG baseline if it is consistently
better than random at the same budget, approaches the full-train reference as
budget increases, and does not rely on record leakage or a different downstream
model.

## Current BIDMC Smoke Result

The current all-method smoke run uses 160 BIDMC train windows, 16 selected
windows, PaPaGei features for selection, and ECG-derived HR quantile bins as the
classification proxy for supervised gradient selectors. It is not a final
benchmark.

| Method | Smoke status |
| --- | --- |
| Random | passed |
| K-Center | passed |
| TypiClust official wrapper | passed |
| ProbCover official wrapper | passed |
| Moderate Coreset official core | passed |
| ELFS official core with proxy score | passed |
| CORDS GradMatch | passed |
| CORDS GLISTER | passed |
| CORDS CRAIG | passed |
| PreSelect | not run: text/LLM task mismatch |

Current output paths:

```text
results/bidmc_selection_all_smoke/selection_smoke_runs.csv
results/bidmc_selection_all_smoke/selection_smoke_status.csv
results/bidmc_selection_all_smoke/selection_smoke_summary.json

/root/project/results/bidmc_selection_all_smoke/selection_smoke_runs.csv
/root/project/results/bidmc_selection_all_smoke/selection_smoke_status.csv
/root/project/results/bidmc_selection_all_smoke/selection_smoke_summary.json
```

## Source URLs

- CoreSet/K-Center: https://arxiv.org/abs/1708.00489
- TypiClust: https://proceedings.mlr.press/v162/hacohen22a.html
- ProbCover: https://openreview.net/forum?id=u6MpfQPx9ck
- GLISTER: https://arxiv.org/abs/2012.10630
- GradMatch: https://proceedings.mlr.press/v139/killamsetty21a.html
- CRAIG: https://arxiv.org/abs/1906.01827
- Moderate Coreset: https://openreview.net/forum?id=7D5EECbOaf9
- ELFS: https://openreview.net/forum?id=yklJpvB7Dq
- PreSelect: https://arxiv.org/abs/2503.00808
