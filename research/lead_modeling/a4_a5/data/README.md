# Original research data and V3 identity

Source: [Kaggle/alexteboul Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), processed from CDC BRFSS 2015. This directory retains the existing ignored CSV; datasets are outside the Python package and must not be committed or overwritten.

The primary input is `diabetes_binary_health_indicators_BRFSS2015.csv`: **253,680 original rows, 21 features and the supplied binary survey label**. Preserve every source row, label and feature value. Do not deduplicate exact-X vectors, majority-vote conflicting labels, merge other datasets, or substitute the balanced 50/50 file.

Verified full SHA-256:

```text
19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0
```

From the repository root, the read-only canonical verifier checks this hash, schema, full original-position coverage/disjointness, the approved archive and exact B v9 two-stage membership/order:

```bash
python shared_splits/verify_shared_split.py --csv research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv
```

The archive is at root `shared_splits/shared_split_indices.npz`; source indices refer to the original CSV row order. Missing or mismatched inputs fail without a regenerated fallback. Ratios/seed alone cannot prove aligned splits. See [../../../../shared_splits/README.md](../../../../shared_splits/README.md) for the exact contract.

Read-only comparison with the three-class CSV found identical feature order and zero mismatching feature cells. These verified bytes map three-class codes 0/1/2 to binary 0/0/1, with 213,703/4,631/35,346 rows. The three-class file is verification evidence only and does not enter training. Its SHA-256 is `a971809d0786d2d7d7f6070f83dfe8af9d860ea33a79e0e2701c8a49f78f9861`.

Describe the target as the supplied `Diabetes_binary=1` label; the numeric mapping does not support the historical combined prediabetes/diabetes wording. Creator-level clinical semantics remain unverified. Exact-X equality does not establish respondent identity. This survey analysis is academic research and does not establish a diagnostic system.
