# UET Diabetes A4/A5 V3

The canonical Python package is [src/diabetes_research](src/diabetes_research/).
Configs, notebooks, historical documents, ignored datasets and run directories
remain in [research/lead_modeling/a4_a5](research/lead_modeling/a4_a5/).

V3 keeps original rows/labels, AP ranking, three-fold stratified training CV,
XGBoost B3, validation F2 with the smallest exact-tie threshold, SHAP defaults
500/100, separate exact-X sensitivity and three same-run paired AP comparisons.
The exact B v9 split archive has been verified against the original CSV.

**Status: software implementation tested; real TRAINING-ONLY GroupSplit AP analysis complete.** Primary fold-local SelectFromModel
is implemented with weighted B3-configured XGBoost gain ranking and exact top-k budgets.
Official evaluation authorization remains PENDING; official A4 stops at that release gate. Historical B v9
LightGBM results remain historical. Official A4/A5 and real SHAP/LIME successes
are not established.

Descriptive primary-training CV mean AP: Stratified **0.4362740784**, Grouped
**0.4358097385**, Group minus Stratified **−0.0004643398**. Grouped folds have
zero exact-X group overlap. These results do not select the primary model or
establish held-out performance. See the [remediation report](research/lead_modeling/a4_a5/docs/FINAL_P2_REMEDIATION_REPORT.md)
for the sealed experiment identity and current checks. Verified numeric labels
map three-class 0/1/2 to binary 0/0/1; original CSV labels remain unchanged.

From the repository root, using Python 3.11 or later:

```bash
python -m pip install -e '.[dev]'
python -m pip check
python -m compileall -q src
PYTHONPATH=src python -m pytest research/lead_modeling/a4_a5/tests -q
python shared_splits/verify_shared_split.py --csv research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv
```

The [integration report](research/lead_modeling/a4_a5/docs/V3_INTEGRATION_REPORT.md)
records actual validation, changes and remaining limitations.

See [the V3 implementation guide](research/lead_modeling/a4_a5/README.md),
[methodology notes](research/lead_modeling/a4_a5/docs/METHODOLOGY_NOTES.md)
and [protocol amendments](research/lead_modeling/a4_a5/config/PROTOCOL_AMENDMENTS_V3.json)
for commands, scientific boundaries and approval status. Use an editable
installation from the full repository; run assets are deliberately kept outside
the Python package. No datasets, model weights or experiment outputs belong in a
commit.

Local clean copies and Kaggle require the full repository layout: root
`src/diabetes_research/`, component V3 config/amendment, scripts and notebooks,
`shared_splits/` (including the archive), and root packaging/dependency metadata.
For notebook discovery, set the working directory inside that full copy; the
first cell searches the current directory and its parents. Mount the original
CSV separately and adjust `DATA`. Component-only copies cannot resolve V3.
Actual Kaggle runtime reproduction remains pending.
