> **25 November 2026 P2 remediation:** Real TRAINING-ONLY GroupSplit AP is complete (Stratified 0.4362740784; Grouped 0.4358097385; Group minus Stratified −0.0004643398; zero grouped-fold exact-X overlap), as descriptive training-CV evidence only. Frozen sensitivity drift guards and full-repository notebook guidance are updated. Official full-data A4/A5 and actual SHAP/LIME remain pending. See [FINAL_P2_REMEDIATION_REPORT.md](FINAL_P2_REMEDIATION_REPORT.md). Earlier snapshots below retain their original findings and source identities.

> **25 November 2026 continuation:** The primary fold-local SelectFromModel pipeline is now implemented with weighted B3-configured XGBoost gain ranking, exact budgets and deterministic ties. **174 tests pass** (118 existing + 56 added), and a fresh synthetic three-fold A4→A5/artifact-integrity roundtrip passes. Official evaluation authorization remains PENDING. See [PRIMARY_SELECTOR_IMPLEMENTATION_REPORT.md](PRIMARY_SELECTOR_IMPLEMENTATION_REPORT.md). The report below preserves the 24 November integration snapshot and its historical source hashes.

# A4/A5 V3 integration report

Date: 2026-11-24, Asia/Ho_Chi_Minh. Repository: `/Users/humanormal/Projects/UET.CS2046_7_Nhom5`.

V3 is implemented on **`integration/a4-a5-v3`** with one canonical package under root `src/diabetes_research/`. Final software validation passed **118 tests in 55.55 seconds**, editable installation, dependency/compile/notebook/entrypoint checks, a fresh synthetic CLI A4→A5 roundtrip and synthetic GroupSplit CV. The actual CSV/shared split and real exact-X outer diagnostics were verified read-only, without model fitting or performance evaluation.

**Primary fold-local SelectFromModel is PENDING** its approved estimator, importance definition, deterministic feature ties and fold artifacts. Official A4 fails before fitting/scoring while this gap remains. Official A4/A5, E6–E9 and actual SHAP/LIME successes are unestablished. No commit, push, merge, rebase, teammate branch change or official experiment was performed.

## Authorization and source identity

The user performed fetch, integration-branch creation and the staged `git mv` locally after the approval service failed. Subsequent implementation used only the integration branch and preserved that index state. Explicit V3 user instructions override historical draft F1/LightGBM wording; the amendment records actual implementation authorization, without inventing team signatures or an experiment release.

| Evidence | Full identity |
|---|---|
| Current HEAD and `origin/Hưng` | `19cf6d2a6256e3163443bc31360036a6c16ff5b5` |
| Prior A4/A5 evidence commit | `2b8036d2f22401ceb57472de8e9572d7b897f4af` |
| Historical B evidence commit | `3e72683caf5cefcd4c05becacd152b2cff7afb9b` |
| Initial attached user request SHA-256 | `ee8dbdede8ce87ba2187c91fd47939829e75a74af05d9611e09108e2731dfd3b` |
| Final aggregate source-files SHA-256 | `8cdb60b1844a0d2e36b3a454a1063f96816d70757dba52371608d560694caa3b` |
| V3 YAML SHA-256 | `69355ba7c8469f5fe1eeddb7719496a87ac1befae1ce025083b59ad89412d546` |
| V3 amendment SHA-256 | `e25940cb103f6819f1e9d88e1b3320d07b31ae7051a20dde1c18515356c6db7a` |
| Verified shared manifest SHA-256 | `34f958e9975a8df8af5bf021f4c230478bdf30fce08e3e85f3c063bb95c36996` |

The aggregate binds 27 module/script/notebook/protocol/dependency/shared-asset files, including untracked implementation files. Git HEAD alone does not identify these changes. The full per-file map is saved at `/private/tmp/uet-diabetes-v3-final-vfgjpemm/final_source_identity.json` and in the compatible run snapshots. `diabetes_research.provenance.source_identity()` recomputes it from the full repository. Final QA artifacts were checked against this exact map; documentation changes do not enter that execution-source map.

## Path migration and exact changes

| Before | After |
|---|---|
| `research/lead_modeling/a4_a5/src/diabetes_research/` | `src/diabetes_research/` |
| Component-relative package imports | Root source resolution and editable console entrypoints |
| Component config/notebook/data/results/document locations | Existing locations retained |
| V2 protocol | Preserved V2 history plus separate V3 YAML/amendment |

The six moved files are `__init__.py`, `core.py`, `modeling.py`, `analysis.py`, `artifact_validation.py` and `provenance.py`. The old component `src/` directory is absent. Historical notebook filenames remain, with V3 content, root resolution, deterministic cell IDs and empty outputs. Packaging supports an editable install from a full repository copy; protocol/shared assets deliberately remain outside the Python wheel.

Tracked paths with unstaged edits, including the staged source moves:

```text
src/diabetes_research/__init__.py
src/diabetes_research/analysis.py
src/diabetes_research/artifact_validation.py
src/diabetes_research/core.py
src/diabetes_research/modeling.py
src/diabetes_research/provenance.py
research/lead_modeling/a4_a5/README.md
research/lead_modeling/a4_a5/data/README.md
research/lead_modeling/a4_a5/docs/A5_Results_Discussion_Template.md
research/lead_modeling/a4_a5/docs/CoReach_A4_A5_tasks.md
research/lead_modeling/a4_a5/docs/GitHub_Commit_Guide.md
research/lead_modeling/a4_a5/docs/METHODOLOGY_NOTES.md
research/lead_modeling/a4_a5/docs/PR_DESCRIPTION.md
research/lead_modeling/a4_a5/notebooks/A4_Training_Protocol_v2.ipynb
research/lead_modeling/a4_a5/notebooks/A5_Evaluation_XAI_v2.ipynb
research/lead_modeling/a4_a5/scripts/run_a4.py
research/lead_modeling/a4_a5/scripts/run_a5.py
research/lead_modeling/a4_a5/tests/test_audit_regressions.py
research/lead_modeling/a4_a5/tests/test_protocol.py
```

New integration files, all left untracked:

```text
.gitignore
README.md
pyproject.toml
requirements.txt
requirements-dev.txt
requirements-explain.txt
src/diabetes_research/paths.py
src/diabetes_research/shared_splits.py
src/diabetes_research/group_sensitivity.py
shared_splits/README.md
shared_splits/shared_split_indices.npz
shared_splits/shared_split_manifest.json
shared_splits/shared_split_manifest.reference.json
shared_splits/verify_shared_split.py
research/lead_modeling/a4_a5/config/A2_protocol_v3.yaml
research/lead_modeling/a4_a5/config/PROTOCOL_AMENDMENTS_V3.json
research/lead_modeling/a4_a5/scripts/run_group_sensitivity.py
research/lead_modeling/a4_a5/tests/conftest.py
research/lead_modeling/a4_a5/tests/test_v3_integration.py
research/lead_modeling/a4_a5/docs/V3_INTEGRATION_REPORT.md
```

The two pre-existing untracked audit/proposal documents are excluded from that change list and remain unchanged/unstaged. The original dependency definitions, V2 YAML, A1/A2/A3 PDFs, A4/A5 historical PDF/Word files and historical audit/fix/QA reports match the pre-edit snapshots. No existing dataset or experiment output was overwritten. All new QA run outputs are in new temporary directories.

## Implemented scientific and artifact behavior

| Area | V3 behavior and status |
|---|---|
| Population | All 253,680 original rows and labels; no feature deduplication, voting or relabeling |
| Ranking/CV | AP remains primary; B1–B3 share saved shuffled stratified 3-fold primary-training folds, seed 42; actual search effort recorded |
| B3 | Actual unweighted XGBoost, with controlled A1 weighting and no silent RF fallback |
| Threshold | Validation positive-class F2, beta=2, exactly ascending `np.linspace(0.01,0.99,199)`, smallest exact-tie threshold; `score >= threshold` |
| Reference metrics | Separate validation/test 0.5 CSVs; F1 secondary; threshold-only changes preserve AP/ROC-AUC |
| Feature reduction | Primary fold-local SelectFromModel PENDING; ANOVA k=21/15/10/5 retained only as full-training secondary synthetic/quick QA, no A3 CV |
| Group sensitivity | Separate exact-X outer diagnostics and frozen-estimator stratified-vs-stratified-group training-only AP; no pipeline selection or group-heldout scoring |
| Bootstrap | Exactly B2−B1, B3−B1, FINAL−B3; same locked run/original labels/cases; shared stream, 1,000 attempts, seed 42, marginal 95% percentile CIs |
| SHAP | Default 500 explained/100 training-background rows; CLI/config overrides, deterministic source IDs, 25-row chunks and bounded prediction/mask batches |
| LIME | Distinct 1,000 perturbations per case, unchanged |

A4 records original-row folds, F2 curves, effective configuration, protocol/amendment/exposure snapshots, full source/environment identity and a pre-test selection lock. A5 verifies byte hashes before trusted pickle loading, then checks source/environment, original cases/labels, folds, actual XGBoost internals, secondary feature sets, locked decisions and 0.5 reference metrics. It recomputes probabilities with the saved models while invoking no fit, search or model/threshold selection routine. Schema v3 requires fresh compatible A4/A5 run IDs; historical outputs are preserved and cannot be silently imported.

Bootstrap outputs include bound comparison IDs, observed point deltas, explicit AP(left)−AP(right) signs, attempted/valid/excluded counts, shared draw identities, single-class exclusion logs without replacement attempts, and self-comparison notices. A FINAL=B3 self-comparison is a zero-difference identity check. CIs condition on fixed models/test rows, exclude training variability and have marginal rather than simultaneous coverage. Extra A1/A3 pairs remain disabled pending separate approval.

SHAP saves selected original source positions and requested/actual budgets before attribution, then marks metadata completed only after successful output. An explicit Independent masker sets `max_samples` to the selected background count; this avoids a silent 100-row cap on larger overrides. Tests cover 9/150/300-row requests and availability limits. These are controlled-stub tests, not real explainer success. Actual SHAP/LIME were not installed or run.

## Reference inputs and shared-split verification

All reference files exist under `/Users/humanormal/Downloads/diabetes_v3_integration/`, were inspected before production modification, and match their recorded hashes. The reference group module was adapted with tuple equality, input/partition/fold invariants, original source IDs, frozen XGBoost and run provenance; it was not copied as trusted production code.

| Reference input | Full SHA-256 |
|---|---|
| `TEAM_PROTOCOL_V3.md` | `8ddd7296f7acd825896fb33c5f77288acce772ce51c0aff75d530f2469373eca` |
| `CODEX_IMPLEMENTATION_PROMPT.md` | `d5105049873b47cfbe00466a2e35e31d7649efe17562e1e4b2223339af473ab0` |
| `shared_splits/verify_shared_split.py` | `0034149933bdef72cdbf833301b0b4415b5d390493afdc541cda8eb5be38e477` |
| Supplied manifest | `b07edc2cda18063bbf7feada4c67ec6755c6be124db9df71f520ebac670ee186` |
| Supplied NPZ | `26f95eee418ae5510ce7e2f27369f02cf3385104dadb0b327715ef924c11bae3` |
| Reference group module | `48ca83b9786ba18581215c468c5c12238cbe6cdccb7a0b69312c52d7b55da2f3` |
| Reference group tests | `1ce045d4b2dabc2804af4bea72ee1121e5dd15ca19eeaf442ae79cb7421efbbc` |

The unchanged NPZ/reference manifest are byte-identical to supplied inputs. The verified manifest adds evidence without rewriting the reference. Both supplied and canonical verifiers passed on the actual CSV. The final installed canonical verifier passed again, checking full bytes/schema, integer/unique original positions, bounds, disjointness/full coverage, original `.npy` and normalized ordered-array hashes, and exact two-stage recipe membership **and order**. Missing/mismatched inputs fail without regeneration.

| Primary partition | Rows | Label-1 count | Label-0 count |
|---|---:|---:|---:|
| Train | 177,576 | 24,742 | 152,834 |
| Validation | 38,052 | 5,302 | 32,750 |
| Test | 38,052 | 5,302 | 32,750 |

The recipe is 70/30 stratified splitting followed by 50/50 stratified splitting of temporary positions, both seed 42, preserving returned array order. Complete coverage is 0..253679 with zero overlap. Ordered int64 array hashes:

```text
train      5ea5ac1de34615247c9e124d4a2fddf258ead135cd24bbebcf590ecf515d1b80
validation 56cb4605bdf7b9bf478978472c1f2328430094e5deeeeba7a2eb29a3df887431
test       d9f140366835500bb3cf1b73776ea0a2d5d0851300209d4a1d5926ca3e97f08c
```

## ZIP, labels and historical exposure

The actual ZIP is `/Users/humanormal/Downloads/e1e5bfcd-388b-4461-aed1-560efe3cc37d_UET_CS2046_7_Nhom5-Th-nh_v9.zip`, SHA-256 `2bc3d1eb891ca0d8adc5bb32b87e1421511f93ab953e52ea10d86b29574522b9`. Read-only reinspection confirmed `lgb.LGBMClassifier`/`num_leaves` in both B pipelines and LightGBM in the saved Kaggle log. Its raw split arrays match the supplied archive and original `.npy` hashes. No archived code was executed and no historical models/scores were imported into A4. Full archived-file fingerprints are in the amendment and verified manifest.

Historical AP **0.4286/0.4331 remains LightGBM evidence**. Historical test results were already viewed; original viewing date and exact exposed cases are unknown. A new run cannot claim untouched confirmatory evaluation.

The unchanged binary CSV hash is `19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0`. Read-only aligned three-class verification found identical feature order, zero mismatching feature cells and the mapping 0/1/2→0/0/1 with 213,703/4,631/35,346 rows. Three-class CSV hash: `a971809d0786d2d7d7f6070f83dfe8af9d860ea33a79e0e2701c8a49f78f9861`. Final byte hashes of both datasets and the ZIP match. Training labels were not changed; the supplied binary survey-label wording is used because the creator clinical dictionary remains unverified.

## Group diagnostics and synthetic sensitivity

The final real-data diagnostics-only run is `group_sensitivity_20261124T132214206988Z_3737d19a86bb`, at `/private/tmp/uet-diabetes-v3-final-vfgjpemm/real_group_diagnostics_final`. It verified 227,908 exact-X groups, no group crossing any outer boundary, and the following approximate row partitions:

| Sensitivity outer partition | Rows | Groups | Label-1 count | Label-1 prevalence |
|---|---:|---:|---:|---:|
| Train | 177,254 | 159,535 | 24,742 | 0.1395850023 |
| Validation | 38,006 | 34,186 | 5,286 | 0.1390833026 |
| Test | 38,420 | 34,187 | 5,318 | 0.1384174909 |

These are design diagnostics, not performance results. GroupShuffleSplit targets groups; rows are approximate. Exact X excludes y and does not identify respondents. Primary training has 177,576 rows and 162,300 groups. The real run records `train_cv_performed=false`, no group-heldout evaluation and no performance metrics. Earlier diagnostics were preserved at their original paths.

Synthetic group CLI QA used 300 constructed rows/100 exact-X groups with a deliberately conflicting label within one group. The primary synthetic training partition had 210 rows/98 groups. Run `group_sensitivity_20261124T132137863368Z_fd2914a85ed5` completed six frozen-XGBoost fits: three ordinary stratified and three stratified-group folds. Saved fold IDs cover only primary training, each validation partition covers that training set once, and group folds have zero crossings. Fold rows/groups/prevalence, mean/std AP (`ddof=0`) and Group−Stratified delta are saved as synthetic sensitivity only. No group-heldout predictions/metrics exist.

## Validation commands and outcomes

Validation used the existing `research/lead_modeling/a4_a5/.venv-diabetes311/bin/python`: Python 3.11.15, macOS 15.1.1 x86_64. Exact installed versions include numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, xgboost 3.2.0, scipy 1.17.1, matplotlib 3.11.2, joblib 1.6.0, PyYAML 6.0.3, nbformat 5.11.1 and pytest 9.2.0. SHAP/LIME/Optuna are not installed. Original dependency definitions were retained; no dependency lock or research environment identity was invented.

Commands were executed from the repository root unless indicated:

```bash
PIP_DISABLE_PIP_VERSION_CHECK=1 research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m pip install --no-deps --no-build-isolation -e .
research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m pip check
research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m compileall -q src research/lead_modeling/a4_a5/scripts shared_splits/verify_shared_split.py
MPLCONFIGDIR=/private/tmp/uet-diabetes-v3-final-vfgjpemm/mpl-cache XDG_CACHE_HOME=/private/tmp/uet-diabetes-v3-final-vfgjpemm/xdg-cache PYTHONPATH=src research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m pytest research/lead_modeling/a4_a5/tests -q
research/lead_modeling/a4_a5/.venv-diabetes311/bin/diabetes-verify-split --csv research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv
git diff --check
git status --short --branch
git diff --cached --name-status
```

| Check | Actual outcome |
|---|---|
| Editable packaging, no dependency download | Installed `uet-diabetes-research==0.3.0`; import resolves to root `src/diabetes_research/__init__.py` |
| Dependency consistency | `No broken requirements found` |
| Compile | Exit 0 for source, wrappers and verifier |
| Final full pytest | **118 passed in 55.55s**; 70 baseline cases retained/updated, 48 V3 cases |
| Focused SHAP defaults/override cases | 4 passed, 44 deselected in 4.56s |
| Four installed console entrypoints | A4/A5/group/verifier `--help` exit 0; relocated imports resolve |
| Four wrappers/unrelated cwd | Covered by regression suite, all passing |
| Both notebooks | nbformat-valid; unique IDs; all code parses/compiles; no outputs or execution counts |
| Actual CSV/shared archive | Full exact verification passed, no fitting/scoring |
| Fresh synthetic A4→A5 and group CV | Exit 0; final source/linked run/artifact hashes verified |
| Real group diagnostics | Exit 0; no crossing or performance calculation |
| Final diff/preservation checks | Whitespace clean; protected documents/dataset hashes unchanged; staged index retains only six moves |

Validation history: the original baseline passed 70 tests in 76.31s. An initial V3 run passed 68 and failed two overlap-tamper cases because the new checksum/lock guard rejected corruption earlier; the harness was updated to rebind those hashes so it still exercises deeper semantic rejection. Expanded suites then passed 111 tests in 100.40s and 115 in 90.21s. Test dtype warnings were corrected. The final three SHAP background override cases bring the suite to 118 passing tests. No unresolved failure remains.

The final roundtrip ran installed console scripts from `/private/tmp`, using `/private/tmp/uet-diabetes-v3-final-vfgjpemm/synthetic.csv`, `--quick --demo` for A4, and `--allow-smoke --bootstraps 1000` for A5. Fresh output directories were `a4_quick_final` and `a5_quick_final`. A4 source rows=400, modeling rows=260, CV=2 and split counts=182/39/39, all explicitly smoke QA. The full synthetic fixture in the regression suite uses stratified 3-fold CV with ten requested configurations per family.

```text
A4 run ID: a4_20261124T132110098665Z_9f41f7b8bfe3
A5 run ID: a5_20261124T132200081387Z_d06f2b660900
Selection lock ID: 195942419b5a983a18d97515a9b7629c1eb0a873d743e17632b7cb2da031b84a
Status of both: SMOKE_NOT_RESEARCH_RESULT
```

All three synthetic bootstrap pairs recorded 1,000 attempted, 1,000 valid and zero excluded draws with one shared stream and unique bound IDs. Separate two-case tests exercised single-class exclusions and same-model zero-difference notices. Synthetic AP/CIs are not reported as research outcomes. Saved A4/A5 and both final group-run artifact hashes match their manifests and the final source digest.

Detailed local QA evidence is at `/private/tmp/uet-diabetes-v3-final-vfgjpemm/`: `pytest_final.log`, `canonical_verifier.log`, final CLI logs, `final_source_identity.json`, `environment.json`, `qa_validation_summary.json` and the four final run directories. Matplotlib caches were routed to that temporary directory; optional explainer libraries were not installed. Historical/earlier QA directories remain intact.

## Remaining scientific limits and next authorized work

- Primary SelectFromModel needs approval and coordinated implementation of the estimator, importance/tie rules, per-fold feature names/counts, serialization and A5 checks. Official feature reduction and full-protocol completion remain blocked.
- Official training and held-out evaluation require separate user authorization after that review/implementation. No results or experiment signoffs are inferred from software QA.
- Real SHAP/LIME need installed compatible dependencies and a separately authorized compatible run; configuration/stub checks do not prove runtime success.
- Real training-only group CV AP is unrun. Its frozen preset is in V3 YAML; a group-heldout experiment would require a separately prespecified design/run ID and is not implemented by this runner.
- C's baseline integration remains pending exact source/split/settings/prediction agreement; A4 internally fits B0–B2. A owns the three paired intervals/PR/ROC, B split/EDA checks, C baseline cross-checks; updated templates avoid duplicate main tables.
- Clinical dictionary and authoritative instructor/proposal compliance are not authenticated by these hashes. Numeric label mapping is verified; labels remain unchanged. Previous historical test exposure limits confirmatory claims.
- Artifact checksums bind consistency and detect corruption/mixed runs; they do not authenticate an unknown pickle creator. A5 requires trusted model artifacts, a full compatible repository and exact model-library/Python versions.

Copy-paste Mac/Kaggle official command templates are in [../README.md](../README.md). They are explicitly blocked by the pending primary selector; there is no flag to promote ANOVA silently. Kaggle instructions require a reviewed full repository copy because this local branch has not been published.

## Preservation and final Git state

| Protected pre-existing document | Unchanged SHA-256 |
|---|---|
| `CROSS_BRANCH_PROTOCOL_AUDIT.md` | `617af031428f865366eb62f940920de5a4d23162ed4cfa512cf292baaf2420a1` |
| `PROTOCOL_FREEZE_PROPOSAL.md` | `9075f4e9f8c5cd1acbf3dc6c29dfabf573c43ec04c6203a22ad75c7d982bfd1f` |

Both were inspected as historical inputs, remain untracked and are not staged. Original A2 PDF SHA-256 remains `8e20a47680bdcb95ac153455cdd24c102d08b2ed2a56c71c96e58f59420fc98c`; V2 YAML remains `5328711de3652ca2c4da4a66b2429dbebd10941ed47765ba36e9de2ac4889bf1`.

```text
Branch: integration/a4-a5-v3...origin/Hưng
HEAD: 19cf6d2a6256e3163443bc31360036a6c16ff5b5
Staged: six R100 source moves performed by the user
Unstaged: 19 tracked paths, including the six relocated sources
Untracked: 20 new integration files plus the two preserved user documents
Commit/push/merge/rebase: none
```

The staged source versions contain only the original relocation, so the final V3 edits are reviewed through the unstaged diff and new files together. No additional files were staged. Datasets, environments, credentials, caches and large run outputs are excluded from the proposed source changes.
