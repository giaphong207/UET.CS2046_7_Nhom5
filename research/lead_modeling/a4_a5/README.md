# Diabetes A4/A5 V3 — implementation and reproducibility guide

The single canonical package is at repository-root `src/diabetes_research/`. It was relocated with `git mv`; there is no importable package in the old component `src/` directory. Configs, notebooks, historical PDFs/Word documents, ignored CSVs, virtual environments and existing results remain in this research component.

**Status: software implementation tested; real TRAINING-ONLY GroupSplit AP analysis complete.** Official full-data A4/A5, E6–E9, and actual SHAP/LIME runtime remain pending. The primary fold-local SelectFromModel pipeline is implemented and software-tested with weighted B3-configured XGBoost gain ranking, exact budgets and deterministic ties. Official A4 stops at the separate evaluation-authorization gate. ANOVA is saved separately as secondary synthetic/quick QA, without CV or primary model selection.

## Current scientific choices

| Choice | V3 implementation |
|---|---|
| Population | All 253,680 original rows and labels; no feature deduplication or majority voting |
| Primary metric | Exact sklearn Average Precision for CV/model ranking |
| Primary outer split | Verified B v9 original-row archive, 70/30 then 50/50, stratified seed 42, returned order preserved |
| Primary CV | Shuffled StratifiedKFold, 3 folds within primary training, seed 42; saved original-row fold IDs |
| Baselines | Internally fitted B0 Dummy, B1 unweighted scaled LR, B2 unweighted RF, B3 actual unweighted XGBoost; no RF fallback |
| A1/A2 | A1 reuses B3 parameters and adds full-training class weighting; A2 uses the same fitted model/scores and a validation F2 threshold |
| Threshold | Positive-class F2, beta=2, ascending `np.linspace(0.01,0.99,199)`; exact ties choose the smallest threshold |
| Reference | Separate validation/test reference CSVs at 0.5; AP and ROC-AUC stay invariant when only the threshold changes |
| Feature reduction | Primary SelectFromModel: weighted B3-configured XGBoost gain, exact k=21/15/10/5, stable original-column ties, fold-local CV and full-training refits; ANOVA saved separately as QA |
| Final choice | Highest validation AP; exact ties follow B1, B2, B3, A2, selected primary A3 |
| Group sensitivity | Separate exact-X outer diagnostics and 3-fold Stratified-vs-StratifiedGroup CV on primary training only; fixed prespecified XGBoost |
| Bootstrap | Exactly B2−B1, B3−B1, FINAL−B3; one locked A4 run, original test labels/cases, 1,000 attempts, seed 42, marginal 95% percentile CIs |
| SHAP | Permutation, 500 explained rows and 100 training-background rows by default; deterministic source IDs, chunks of 25 and bounded prediction/mask batches |
| LIME | Distinct `num_samples=1000`, unchanged |

The supplied v9 ZIP is independently confirmed as LightGBM: its two pipelines instantiate `lgb.LGBMClassifier`, tuning uses `num_leaves`, and the Kaggle log says LightGBM. Historical scores remain LightGBM evidence. Historical test results have already been seen; a new run does not establish an untouched confirmatory test. C's baseline outputs stay separate QA until predictions, fit settings, exact split indices and full source hash match. A4 fits its own baselines.

The original binary CSV is described as the supplied `Diabetes_binary=1` survey label. Verified three-class 0/1/2 maps to binary 0/0/1 (213,703 / 4,631 / 35,346 rows); it does not support the old combined prediabetes/diabetes wording. Labels are preserved; creator-level clinical semantics remain unverified.

## Installation and read-only verification on Mac

Commands use the **repository root**, with Python 3.11 or later. The existing virtual environment can stay in the component.

```bash
cd /Users/humanormal/Projects/UET.CS2046_7_Nhom5
source research/lead_modeling/a4_a5/.venv-diabetes311/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
python -m compileall -q src
PYTHONPATH=src python -m pytest research/lead_modeling/a4_a5/tests -q
python shared_splits/verify_shared_split.py --csv research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv
```

For explanations, install `python -m pip install -e '.[explain]'` in the same environment before creating the paired A4/A5 runs. Packaging supports an editable installation from the full repository; configs and the split archive are repository assets outside the Python wheel. Root requirements files forward to the preserved component dependency definitions. No environment lock is invented; runs save exact library versions and A5 requires compatible model-library versions.

The canonical CSV SHA-256 is `19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0`; the exact archive SHA-256 is `26f95eee418ae5510ce7e2f27369f02cf3385104dadb0b327715ef924c11bae3`. The verifier checks full bytes, original row coverage, disjointness, original `.npy` hashes and exact two-stage membership/order. Missing or mismatched inputs fail without a generated fallback.

## Synthetic quick A4 → A5

Use new data/output names on repeat runs. This example only creates synthetic QA data, using the regression fixture's documented domain generator.

```bash
PYTHONPATH=src:research/lead_modeling/a4_a5/tests python - <<'PYQA'
from pathlib import Path
from test_protocol import toy_data
path = Path('/private/tmp/uet_diabetes_v3_synthetic.csv')
toy_data(n=400).to_csv(path, index=False)
print(path)
PYQA
python research/lead_modeling/a4_a5/scripts/run_a4.py --data /private/tmp/uet_diabetes_v3_synthetic.csv --output-dir research/lead_modeling/a4_a5/results/a4_v3_synthetic_001 --quick --demo
python research/lead_modeling/a4_a5/scripts/run_a5.py --data /private/tmp/uet_diabetes_v3_synthetic.csv --a4-dir research/lead_modeling/a4_a5/results/a4_v3_synthetic_001 --output-dir research/lead_modeling/a4_a5/results/a5_v3_synthetic_001 --allow-smoke --bootstraps 1000
```

Quick mode retains the old, explicitly labeled `qa_legacy_test_first_sorted` split on the seeded subset, 1 configuration/family and 2 CV folds. Its source-row mapping is verified. Synthetic full-row `--demo` can use 3-fold CV. Neither is a primary research run; A5 requires `--allow-smoke` and retains the QA status. Full original-population training cannot be released with `--demo`.

Primary selector evidence is saved in `feature_selection_cv.csv`, `feature_selection_folds.json`, `feature_selection_models.joblib`, `feature_rankings.json` and `feature_sets.json`. A2 validation AP is fixed in `a2_reference_lock.json` before selector evaluation. Each k fits fresh weighted B3-configured gain rankers and downstream models in every training fold; stable exact ties use original columns. Reserved validation chooses the smallest eligible k within 0.01 AP of A2; A2 is fallback and selected A3 remains separately reported. CV AP is conditional on the B3 settings selected using the same folds, not nested CV. `secondary_anova.json` stays outside primary selection and test predictions.

For synthetic three-fold integration, use `--demo --cv 3 --trials 10` without `--quick` on a new constructed CSV/output directory. The shared official split remains mandatory in official mode.

## Group sensitivity

The sealed real **TRAINING-ONLY SENSITIVITY ANALYSIS** completed on 25 November 2026: Stratified mean AP **0.4362740784**, Grouped mean AP **0.4358097385**, Group minus Stratified **−0.0004643398**. All six folds use only the canonical primary training partition (177,576 rows, 162,300 exact-X groups); grouped folds have zero exact-X overlap. These are descriptive single-preset, single-seed CV results, not held-out performance, significance/equivalence evidence or a primary model-selection input. Official A4/A5 and actual SHAP/LIME remain pending.

The 34 read-only artifacts remain at `results/real_training_only_group_cv_20261125T064729+0700_gmrgtd23/`; report `TRAINING_ONLY_SENSITIVITY_REPORT.md`, manifest SHA-256 `a6d2175121d9a5117f25da4d36ec7be720b01355353c0e6cfeb826af2297e644`. They are ignored experimental outputs, preserved locally and excluded from staging. The [remediation report](docs/FINAL_P2_REMEDIATION_REPORT.md) records integrity checks and historical source preservation. No real experiment was rerun during remediation.

The default command only constructs/verifies the separate outer group partition. It fits no model and computes no held-out score:

```bash
python research/lead_modeling/a4_a5/scripts/run_group_sensitivity.py --data research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv --output-dir research/lead_modeling/a4_a5/results/group_v3_diagnostics_001
```

For any separately requested future run, `--train-cv` performs the prespecified six fits on primary **training** rows: three StratifiedKFold and three StratifiedGroupKFold folds. The unweighted estimator is frozen independently of A4 search winners. YAML and the approved amendment must match the canonical sensitivity contract; wrapper and actual native settings are checked before fitting. Drift fails without overrides. Report each fold's rows, groups and class prevalence, mean/std AP (`ddof=0`) and `mean AP Group − mean AP Stratified`. The outer split remains diagnostics only. A held-out group experiment requires a separately prespecified design/run ID; this CLI has no held-out group evaluation option.

On the actual CSV, outer diagnostics found 177,254 / 38,006 / 38,420 rows, 159,535 / 34,186 / 34,187 exact-X groups, and zero group crossings. These are sensitivity design counts, not performance results. GroupShuffleSplit ratios target groups, so row ratios are approximate. Exact X equality never establishes a respondent identity, even when labels conflict.

## Official command templates after review

These are **blocked templates** pending separate official training/evaluation authorization. The selector is implemented; synthetic QA does not release official experiments. Current A4 stops before fitting/scoring at the authorization gate.

Mac, from repository root:

```bash
python research/lead_modeling/a4_a5/scripts/run_a4.py --data research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv --output-dir research/lead_modeling/a4_a5/results/a4_v3_reviewed_001 --trials 10 --cv 3 --shared-split-indices shared_splits/shared_split_indices.npz --shared-split-manifest shared_splits/shared_split_manifest.json
python research/lead_modeling/a4_a5/scripts/run_a5.py --data research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv --a4-dir research/lead_modeling/a4_a5/results/a4_v3_reviewed_001 --output-dir research/lead_modeling/a4_a5/results/a5_v3_reviewed_001 --bootstraps 1000 --shap --lime --shap-samples 500 --shap-background-samples 100
```

Kaggle, after placing a reviewed **full repository copy** at the following working path and mounting the original Kaggle dataset. This local integration branch has not been pushed; do not assume GitHub contains V3 yet. Adjust the mount path if needed; verification still requires the exact CSV bytes.

```bash
cd /kaggle/working/UET.CS2046_7_Nhom5
python -m pip install -e '.[explain]'
python shared_splits/verify_shared_split.py --csv /kaggle/input/diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv
python research/lead_modeling/a4_a5/scripts/run_a4.py --data /kaggle/input/diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv --output-dir /kaggle/working/a4_v3_reviewed_001 --trials 10 --cv 3 --shared-split-indices shared_splits/shared_split_indices.npz --shared-split-manifest shared_splits/shared_split_manifest.json
python research/lead_modeling/a4_a5/scripts/run_a5.py --data /kaggle/input/diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv --a4-dir /kaggle/working/a4_v3_reviewed_001 --output-dir /kaggle/working/a5_v3_reviewed_001 --bootstraps 1000 --shap --lime --shap-samples 500 --shap-background-samples 100
```

## Artifact contract and review

A4 manifest/registry schema is v3. Critical artifacts include protocol/amendment/exposure snapshots, effective settings, code/environment hashes, original source IDs, exact split metadata, training fold IDs, baseline CV traces, primary selector fold models/AP/rankings, F2 curves, reference metrics, feature sets, selection and fitted models. `selection_lock.json` binds pre-test artifacts before scoring. New source requires fresh A4/A5 run IDs; stale v2 or historical LightGBM outputs are rejected.

A5 checks byte hashes before loading trusted model pickles, then verifies source/environment, original cases/labels, folds, XGBoost internals, selectors, locked F2 decisions and reference metrics. It recomputes probabilities and uses exact model scores, preserving ties. It never calls fit or model/threshold selection routines. Hashes detect corruption/mixed runs; they do not authenticate an unknown pickle creator. Only load trusted model registries.

`paired_bootstrap_ap.csv` is a long-form table for exactly three core pairs. `paired_bootstrap_summary.json` contains observed point deltas, percentile CIs, sign conventions, bound comparison IDs, attempted/valid/excluded counts and self-comparison notices. `paired_bootstrap_exclusions.csv` logs single-class draws; attempts are not replaced. All pairs share one draw stream. CIs are conditional on fixed models and the test split, with marginal rather than simultaneous coverage. Extra A1/A3 bootstrap pairs remain PENDING separate approval.

SHAP CLI overrides are `--shap-samples`, `--shap-background-samples`, `--shap-chunk-size`, and `--shap-prediction-batch-size`; YAML provides defaults. An explicit Independent masker uses every selected background row, including configured counts above 100; actual counts cap at available rows. `shap_selected_source_rows.npz` and metadata save source IDs/settings before processing, and attribution metadata is completed only on success. LIME remains separate at 1,000 perturbations/case. Current tests of SHAP planning use a controlled stub; actual library success is unverified.

The notebooks retain their historical filenames, with V3 content, root resolution and empty outputs. Use a reviewed **full repository copy** containing root `src/diabetes_research/`, component V3 config/amendment, all scripts/notebooks, `shared_splits/` archive/manifest/verifier and root packaging/dependency metadata. A component-only mount is insufficient. In a local clean copy, start from its root, component or notebooks directory. On Kaggle, copy the full layout to a writable directory such as `/kaggle/working/UET.CS2046_7_Nhom5` and change the notebook working directory to that root before the first cell. Discovery searches cwd and parents and rejects incomplete layouts. Mount the original CSV separately and set `DATA` to its mount; keep new output paths writable. Editable installation uses that full repository root. Path checks are software-tested; actual Kaggle and SHAP/LIME runtime are pending. See [METHODOLOGY_NOTES.md](docs/METHODOLOGY_NOTES.md), [A5 report template](docs/A5_Results_Discussion_Template.md), and [PROTOCOL_AMENDMENTS_V3.json](config/PROTOCOL_AMENDMENTS_V3.json). The two pre-existing audit/proposal documents are preserved byte-for-byte as historical evidence; their old paths and proposed F1/selector wording do not override V3 instructions.
