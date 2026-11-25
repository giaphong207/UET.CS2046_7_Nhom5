# Verified B v9 original-row split

`shared_split_indices.npz` is the exact supplied archive, imported only after both the supplied verifier and canonical verifier passed against the original CSV. Do not regenerate or reorder it. A4 loads these original source positions; A5 verifies the archive, ordered arrays, original cases and labels before paired analysis.

| Partition | Rows | Label-1 rows | Label-0 rows |
|---|---:|---:|---:|
| Train | 177,576 | 24,742 | 152,834 |
| Validation | 38,052 | 5,302 | 32,750 |
| Test | 38,052 | 5,302 | 32,750 |

The complete arrays cover positions 0..253679 once, with zero overlap. The exact recipe is `train_test_split(test_size=0.30, stratify=y, random_state=42)` followed by `train_test_split` of the temporary rows with `test_size=0.50`, their corresponding labels and seed 42. Preserve returned array order. A different two-stage order may share ratios and seed while producing different cases.

| Input | Full SHA-256 |
|---|---|
| Original binary CSV | `19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0` |
| Exact NPZ archive | `26f95eee418ae5510ce7e2f27369f02cf3385104dadb0b327715ef924c11bae3` |
| Unchanged reference manifest | `b07edc2cda18063bbf7feada4c67ec6755c6be124db9df71f520ebac670ee186` |

`shared_split_manifest.reference.json` preserves the supplied manifest unchanged. `shared_split_manifest.json` adds verified archive/normalized-array hashes and read-only ZIP reinspection evidence; the original `.npy` hashes are retained. The actual v9 ZIP SHA and raw-array match are recorded there and in the [V3 amendment](../research/lead_modeling/a4_a5/config/PROTOCOL_AMENDMENTS_V3.json). Both archived pipelines/logs still identify LightGBM. Split compatibility does not turn their historical model scores into official XGBoost evidence.

From the repository root, with the documented editable environment or root `src` wrapper:

```bash
python shared_splits/verify_shared_split.py --csv research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv
diabetes-verify-split --csv research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv
```

The verifier checks files/full hashes/schema, archive membership, bounds, integer/unique positions, disjointness/full coverage, original `.npy` and normalized ordered-array hashes, class counts and exact recipe/order. It rechecks input hashes before returning. No model fits or performance evaluations occur. Missing/inconsistent inputs fail safely; generated fallback is unavailable.

Primary training folds are saved with each compatible A4 run, separately from this outer archive. The historical test-first sorted splitter is labeled `qa_legacy_test_first_sorted` and retained for synthetic/quick QA only. Exact-X group outer partitions are a separate sensitivity design with approximate row counts; they never replace the primary archive silently.
