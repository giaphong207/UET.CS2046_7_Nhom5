# S8 - Paired Bootstrap and Visualization

## Scope

SYNTHETIC ONLY.

No BRFSS data or real holdout test predictions
were used during this QA exercise.

## Source code

- c_bootstrap.py
- c_make_figures.py

## Required NPZ fields

- idx_test: unique integer row indices
- y_test: binary ground-truth labels
- proba_<model_name>: prediction scores

Both models must have identical row indices,
row order and ground-truth labels.

External index files require a separately
verified integration procedure.

## Bootstrap configuration

- Replicates: 1000 valid samples
- Seed: 42
- Confidence level: 95%
- CI method: percentile bootstrap
- Primary metric: Average Precision
- Secondary metric: ROC-AUC
- Delta direction: B minus A
- Single-class samples: skip and redraw

## Example commands (synthetic data only)

Run from the extracted S8 directory:

```bash
python c_bootstrap.py --file-a synthetic_inputs/synthetic_predictions_A.npz --model-a synthetic_A --file-b synthetic_inputs/synthetic_predictions_B.npz --model-b synthetic_B --out-dir demo/bootstrap

python c_make_figures.py --file-a synthetic_inputs/synthetic_predictions_A.npz --model-a synthetic_A --file-b synthetic_inputs/synthetic_predictions_B.npz --model-b synthetic_B --bootstrap-dir demo/bootstrap --out-dir demo/figures --scope "SYNTHETIC ONLY"
```

## QA results

- Bootstrap tests: 5/5
- Visualization tests: 4/4
- Advanced tests: 16/16
- Total: 25/25 PASS

## Limitations

Synthetic QA does not establish model
performance on BRFSS data.

Before FINAL integration, verify prediction
provenance, row alignment and group approval.

Bootstrap confidence intervals are conditional
on the fitted models and fixed predictions.

A confidence interval containing zero does
not establish statistical equivalence.

The row bootstrap does not account for
complex survey design or dependent records.

Preserve the original Kaggle notebook separately.
