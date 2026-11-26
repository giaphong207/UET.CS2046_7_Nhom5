# S9.5 - Historical Result Aggregation

## 1. Purpose

Aggregate existing cross-validation and validation
results from team members B and C.

This task performs no model training, no new model
selection, no holdout-test evaluation and no FINAL
evaluation.

Primary metric: Average Precision (AP).

## 2. Verified input inventory

- Source files: 34
- B source files: 13
- C source files: 21
- CSV source files: 29
- Total historical CSV rows: 230

The input snapshot includes only registry-listed
source files and the original source registry.

Original archive hashes recorded in the registry
are provenance metadata. The complete upstream
branch archives are not reverified by this package.

## 3. Consolidated outputs

- Model configurations: 38
- Main CV records: 34
- C hyperparameter candidates: 80
- Validation configurations: 19
- Validation records: 32
- Feature selection CV records: 25
- Threshold sensitivity comparisons: 13
- Group-vector CV comparisons: 3

Configuration counts are not counts of independent
newly trained models.

## 4. Quality assurance

- Source integrity: 34/34
- Model registry provenance: 38/38
- Confusion matrix metric checks: 288/288
- Sensitivity checks: 12/12
- Script build checks: 8/8
- Final QA: 10/10
- Byte-identical reproduction: 27/27
- Source metric checks: 580/580
- Negative tests: 5/5

The build report records the state before full CLI
execution. The later aggregation QA report
documents successful full CLI reproduction.

## 5. Reproduction instructions

Tested environment:

- Python 3.13.18
- NumPy 2.1.3
- Pandas 2.3.3

Install required dependencies if necessary:

    python -m pip install -r requirements.txt

From the extracted S9_5 directory, run:

    python c_aggregate_results.py --input-root inputs/S9_5_inputs --out-dir reproduced_results

The output directory must be new or empty.

Do not use Python optimization mode (-O).

The aggregation process does not require raw
BRFSS data, GPU acceleration or Internet access.

Expected output: 26 CSV files and one JSON file.

## 6. File organization

- c_aggregate_results.py: standalone aggregation tool
- results/: consolidated historical results
- qa/: QA reports, audits and execution logs
- inputs/: original verified aggregation snapshot
- aggregation_build_report.json: build provenance
- s9_5_environment.json: environment information
- s9_5_run_config.json: aggregation scope and protocol
- sha256_manifest.json: file integrity manifest

Source row numbers are one-based data-row positions,
excluding the CSV header.

## 7. Limitations

- Reproduction verifies result aggregation, not
  model retraining or independent model performance.

- AP and ROC-AUC were checked against stored results,
  not recalculated from original prediction scores.

- Validation metrics are not holdout-test metrics.

- Historical code/lock version mismatch reported
  for B remains unresolved.

- Official A4/A5 results are not included.

- Matching reported metrics do not establish
  independent evidence or identical predictions.

- Group-vector CV does not establish statistical
  equivalence or eliminate every leakage risk.

- ZIP-input execution mode has not been independently
  tested by the current aggregation QA.

- Cross-platform byte-identical reproduction
  has not been established.

- Full upstream protocol compliance was not
  independently reverified in this task.

- No causal, future-risk or clinical diagnostic
  claims are supported by this aggregation.

## 8. Status

Technical aggregation QA: PASS

Team cross-review: PENDING

Official FINAL pipeline approval: PENDING

FINAL evaluation: NOT PERFORMED IN THIS TASK

## 9. Locked script SHA-256

8abc4bbdcec57113fd2e6ed667e6dcf123900b64b3a6ddfd7357f72af9dc5a2b
