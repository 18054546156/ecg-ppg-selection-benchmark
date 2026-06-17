# BIDMC Open ECG+PPG Pipeline

This is a public-data smoke benchmark for the ECG+PPG workflow. It does not
replace SignalMC-MED; it verifies that our remote code path can download open
WFDB data, read synchronized ECG/PPG, extract segment-level features, and run a
small downstream sanity task.

Dataset: BIDMC PPG and Respiration Dataset, open on PhysioNet.

Remote commands:

```bash
bash scripts/download_bidmc_open.sh
/root/project/.venv/bin/python scripts/run_bidmc_ecg_ppg_pipeline.py --use-papagei --device cuda
```

Outputs:

```text
/root/project/results/bidmc/bidmc_segment_features.csv
/root/project/results/bidmc/bidmc_summary.json
/root/project/results/bidmc/bidmc_hr_agreement.png
```

The sanity target is ECG-derived heart rate. We report grouped cross-validation
metrics for predicting ECG-derived HR from PPG hand-crafted features and,
optionally, PaPaGei embeddings.

Managed entry:

```bash
python scripts/run_pipeline.py bidmc_ecg_ppg_open --remote --sync --download
```

Current result:

```text
records_used: 53
segments: 2544
ppg_to_ecg_hr.mae: 3.459813694635973
ppg_to_ecg_hr.pearson: 0.843944372513789
```

This is a public-data smoke result. It should not be reported as a
SignalMC-MED paper reproduction.
