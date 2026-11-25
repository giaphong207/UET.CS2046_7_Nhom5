# S7 - Group-vector Cross-Validation

Scope: Train-only sensitivity analysis.

Models:
- B1 Logistic Regression
- B2 Random Forest

Methods:
- StratifiedKFold, 3 folds
- GroupKFold, 3 folds

Primary metric: Average Precision.

Main files:
- s7_model_comparison.csv
- s7_fold_audit.csv
- s7_metrics_recomputed.csv
- S7_QA_REPORT.json
- s7_run_config.json
- s7_environment.json
- sha256_manifest.json

The NPZ files contain out-of-fold predictions
on training records only.

These results are not holdout test results.

Notebook source must be preserved separately
for full experimental reproducibility.
