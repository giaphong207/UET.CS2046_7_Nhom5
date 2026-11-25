# Final P2 remediation report

Date: **25 November 2026**, Asia/Ho_Chi_Minh. Branch: **integration/a4-a5-v3**. HEAD remains `19cf6d2a6256e3163443bc31360036a6c16ff5b5`.

**READY FOR HUMAN REVIEW.** All three P2 findings are resolved. The full regression suite passes **295 tests**, including all 174 previous cases and 121 additions. This establishes implementation readiness. **Official full-data A4/A5 evaluation and actual SHAP/LIME runtime remain PENDING.** No official fitting, held-out performance access, real experiment rerun, staging, commit, push, merge, rebase or teammate-branch change occurred.

Audit inputs were read before editing: the complete `AUDIT_REPORT.md`, `STAGING_PLAN.md`, `staging_whitelist.txt` and `PR_DRAFT.md` under `/private/tmp/uet-final-audit-i8y1wiyc/`, the historical selector implementation report, V3 YAML/amendment and the affected implementation. Remediation evidence and synthetic outputs are under `/private/tmp/uet-p2-remediation-1pi3tcjb/`.

| Finding / subsystem | Disposition | Resolution |
|---|---|---|
| P2-01 — frozen sensitivity drift | PASS / RESOLVED | Independent complete declared/wrapper/native contract; YAML/amendment agreement; rejection before fitting/output writes |
| P2-02 — current completion wording | PASS / RESOLVED | Software-tested selector and completed real training-only sensitivity are distinguished from pending official evaluation |
| P2-03 — component-only notebook mounting | PASS / RESOLVED | Both notebooks require full repository assets; clean-copy path discovery and incomplete-copy rejection pass |
| Primary selector, F2, shared split, bootstrap | PASS | Scientific rules unchanged; existing regressions and fresh synthetic integration pass |
| Historical artifact/data/protocol/index integrity | PASS | Protected documents, 34 sealed artifacts, CSVs, split and approval records unchanged; index byte-identical |
| Official scientific release / actual Kaggle / explainers | NEEDS REVIEW | Separate authorization and runtime evidence still required; no completion claim |

## P2-01: exact frozen sensitivity contract

`src/diabetes_research/sensitivity_contract.py:14` defines constants independently of editable YAML/amendments. Validation at lines 34–76 checks the complete key sets and typed values, handles the preserved NaN missing-value setting, and rejects numeric/boolean substitutions. Errors identify the conflicting parameter; nothing is silently corrected.

The approved ten declared settings remain exactly:

```python
{
    'n_estimators': 160, 'max_depth': 4, 'learning_rate': 0.08,
    'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0,
    'reg_lambda': 1.0, 'weight_mode': 'none', 'n_jobs': 2, 'random_state': 42,
}
```

The wrapper additionally requires `importance_type='gain'`. The complete 40-key native XGBoost constructor configuration includes objective `binary:logistic`, evaluation metric `logloss`, tree method `hist`, `scale_pos_weight=1.0`, `enable_categorical=False`, missing NaN and all declared non-weight settings. These native settings remain unspecified (`None`), exactly as in the sealed run:

```text
base_score, booster, callbacks, colsample_bylevel, colsample_bynode, device,
early_stopping_rounds, feature_types, feature_weights, gamma, grow_policy,
interaction_constraints, max_bin, max_cat_threshold, max_cat_to_onehot,
max_delta_step, max_leaves, monotone_constraints, multi_strategy,
num_parallel_tree, reg_alpha, sampling_method, validate_parameters, verbosity
```

No new explicit hyperparameter or approval is introduced. All six saved native configurations in the sealed `fold_fit_audit.json` pass this contract, checked without fitting or predicting.

`provenance.py:169` now requires YAML and the **APPROVED** group-sensitivity amendment to agree with the independent constants. Coordinated edits to both records also fail if they depart from the contract. V3 YAML and amendment bytes are unchanged.

`group_sensitivity.py:35` adds `FrozenSensitivityXGB`, a guarding subclass of the existing `FoldAwareXGB`, retaining the same native XGBClassifier. `frozen_sensitivity_pipeline()` at line 49 validates protocol, wrapper and native defaults without fitting. `run()` at line 217 performs preflight before data access or output creation. Every cloned fold model repeats wrapper/native validation before native fit. `modeling.py:83` extracts the existing constructor into `_build_core()` without changing its arguments or weighting behavior; primary A4 science is unchanged.

The direct comparison helper at `group_sensitivity.py:166` rejects unguarded/drifted XGBoost wrappers, added preprocessing, changed fold count or seed. The frozen scientific pipeline still uses no fitted preprocessing. Existing non-XGBoost synthetic QA with fold-local LR/scaling remains supported; that helper grants no authorization for an alternative real-data experiment. The production runner accepts no estimator substitution or tuning option.

## P2-02: current status and scientific wording

Current summaries now state that primary SelectFromModel is implemented/software-tested and real **TRAINING-ONLY GroupSplit AP sensitivity** is complete. Official full-data A4/A5, E6–E9 and actual SHAP/LIME remain pending. Corrected locations include root `README.md:12`, component `README.md:5`/`:72`, `CoReach_A4_A5_tasks.md:3`/`:45`, `METHODOLOGY_NOTES.md:39`, `PR_DESCRIPTION.md:7` and the A4 wrapper docstring at `scripts/run_a4.py:4`.

The new V3 report preface identifies current status while preserving its previous dated snapshots verbatim below it. `PRIMARY_SELECTOR_IMPLEMENTATION_REPORT.md` remains the original implementation snapshot; its old source digest and then-unrun sensitivity statements describe that historical task. The amendment's dated research-status statement is also preserved as historical evidence. Current status is in this report and the updated summaries, not fabricated approval records.

Verified numeric mapping remains **three-class 0/1/2 → binary 0/0/1**, with row counts **213,703 / 4,631 / 35,346** and previously verified zero feature-cell differences. Both original CSV hashes remain unchanged. No labels are rewritten. Creator clinical dictionary authority remains unverified.

The immutable real experiment remains at:

`research/lead_modeling/a4_a5/results/real_training_only_group_cv_20261125T064729+0700_gmrgtd23/`

Report: `TRAINING_ONLY_SENSITIVITY_REPORT.md`. Results: `summary_metrics.csv`, `train_cv_fold_details.csv`, `train_cv_ap_sensitivity.json`. Manifest: `manifest_group_sensitivity.json`, SHA-256 `a6d2175121d9a5117f25da4d36ec7be720b01355353c0e6cfeb826af2297e644`.

**TRAINING-ONLY SENSITIVITY ANALYSIS — descriptive evidence from the already completed sealed run; no rerun:**

| CV method | Fold 1 AP | Fold 2 AP | Fold 3 AP | Mean AP | Fold std, ddof=0 |
|---|---:|---:|---:|---:|---:|
| StratifiedKFold | 0.4411514452 | 0.4347660732 | 0.4329047166 | 0.4362740784 | 0.0035315429 |
| StratifiedGroupKFold | 0.4342067844 | 0.4364334549 | 0.4367889763 | 0.4358097385 | 0.0011427147 |

Group minus Stratified mean AP: **−0.0004643398**. Canonical primary training contains 177,576 rows, 24,742 positives (prevalence 0.1393318917), 162,300 exact-X groups and 885 conflicting-label groups. All six fits/AP measurements use only canonical primary training; grouped folds have zero exact-X overlap. Group identity uses equality of all 21 X values, excluding y, and does not identify respondents. The sealed run did not construct an alternative outer partition. Existing outer diagnostics remain a separate design.

These single-seed, single-preset, row-weighted training-CV results establish neither significance, equivalence, noninferiority nor causal duplicate effects. Fold standard deviation is not a confidence interval or standard error. They are not held-out performance, do not alter primary model/k/threshold selection and do not certify research or Kaggle completion.

## P2-03: portable full-repository notebooks

Both notebooks' setup Markdown and first code cell require root `src/diabetes_research/`, component V3 YAML/amendment, all scripts and both notebooks, shared split archive/manifest/verifier, and root/component packaging/dependency metadata. The first cell searches cwd and parents and rejects incomplete layouts before dataset/manifest access. There are no hardcoded Mac paths in notebook setup.

For a local clean copy, start inside the root, component or notebooks directory. For Kaggle, place the reviewed full repository in a writable working directory, change notebook cwd into that copy before discovery, mount the original CSV separately and adjust `DATA`. Output paths must be writable; editable installation uses the repository root. A component-only mount cannot resolve V3. These setup instructions are shared by both READMEs and notebooks. Actual Kaggle runtime remains pending.

Both notebooks validate with nbformat, have unique cell IDs and empty outputs/execution counts, and every code cell compiles. Only the discovery prefix was executed; no experiment, dataset-reading, held-out or explanation notebook cell ran.

## Added regressions and executed validations

`tests/test_sensitivity_contract.py` adds **113 cases**: authorized YAML/amendment/wrapper/native/cloning; independent mutations of all ten declared settings in YAML, amendment and both; missing/extra settings; family/approval rejection; all eleven live wrapper settings; all 40 native settings; missing/extra native keys and boolean/numeric drift; rejection before data/output access; direct-helper bypass and design-change rejection. Native-fit sentinels prove rejection occurs before fitting. Existing artifacts are preserved in failure probes.

`tests/test_notebook_portability.py` adds **8 cases**: both notebooks from three clean-copy working directories and component-only rejection. All previous 174 cases and their test files remain unchanged by this remediation.

| Check | Actual result | Evidence under `/private/tmp/uet-p2-remediation-1pi3tcjb/` |
|---|---|---|
| Focused new tests | **121 passed in 4.97s** | `focused-pytest.log` |
| Full regression suite | **295 passed in 109.03s** | `full-pytest.log` |
| Python compileall: root source, wrappers, all tests, split verifier | Exit 0 | Command below |
| Existing and clean-candidate pip check | No broken requirements | `pip-check.log`, `clean-pip-check.log` |
| Installed dependency ranges / all ten module imports | PASS | `clean_package_checks.json` |
| Fresh candidate editable install without downloads | Exit 0; existing dependency distributions reused | `clean-editable-install.log` |
| All four installed CLI helps in both environments | 8 successful checks, unrelated cwd | `clean_package_checks.json` |
| Both notebooks: nbformat/IDs/empty outputs/compile/discovery | PASS; no experiment cells executed | `clean_package_checks.json` |
| Clean candidate migration/notebook regressions | **15 passed, 41 deselected in 44.74s** | `clean-candidate-tests.log` |
| Actual canonical CSV/shared split verifier | PASS: full hashes and exact recipe/order/coverage | `canonical-verifier.log` |
| New contract against six saved real native configs | PASS, no fit/prediction | `sealed_native_contract_check.json` |
| Synthetic 3-fold A4→A5, installed candidate CLIs | Both exit 0; 400 synthetic rows, 280/60/60, ten configurations per B1/B2/B3 | `synthetic-a4-final.log`, `synthetic-a5.log` |
| Synthetic artifact validation with fit forbidden | PASS: 30 A4/15 A5 hashes; 12 selector fold pipelines and 4 full-train refits; columns/settings/run match | `synthetic_integrity_validation.json` |
| Three synthetic bootstrap comparisons | B2−B1/B3−B1/FINAL−B3, each 1,000 attempted/valid, zero excluded; AP(left)−AP(right), identical cases/labels/shared draw stream | `synthetic_integrity_validation.json` |
| Protected docs / sealed 34 / CSVs / split / approval bytes / index | PASS; sealed write bits remain removed | `preservation_verification.json` |
| Staged and unstaged whitespace / safe candidate inventory | PASS; no unsafe staged file or credential-pattern match | `candidate_inventory.json`, final Git snapshots |

Validation history is retained rather than represented as success on every invocation: the first focused set passed 99 cases before the direct-helper/portability additions. A standalone synthetic CLI was started before fixture generation finished and failed on missing input before fitting; the completed fixture then passed A4→A5. An auxiliary integrity script initially referenced the wrong existing selection key, and an inventory assertion initially treated the allowed data README as a dataset. Those QA assertions were corrected; final validations pass. No production regression failed in the final suite.

Validation interpreter: existing `research/lead_modeling/a4_a5/.venv-diabetes311/bin/python`, retaining its virtual-environment path. Python 3.11.15; numpy 2.4.6; pandas 2.3.3; sklearn 1.9.1; XGBoost 3.2.0; scipy 1.17.1; joblib 1.6.0; matplotlib 3.11.2; PyYAML 6.0.3; nbformat 5.11.1; pytest 9.2.0. SHAP/LIME/Optuna remain uninstalled. The fresh temporary editable environment reuses these installed dependencies; it does not claim a new downloaded lock or actual Kaggle/Linux execution.

```bash
MPLCONFIGDIR=/private/tmp/uet-p2-remediation-1pi3tcjb/mpl-cache XDG_CACHE_HOME=/private/tmp/uet-p2-remediation-1pi3tcjb/xdg-cache research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m pytest -q
PYTHONPYCACHEPREFIX=/private/tmp/uet-p2-remediation-1pi3tcjb/pycache research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m compileall -q src research/lead_modeling/a4_a5/scripts research/lead_modeling/a4_a5/tests shared_splits/verify_shared_split.py
research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m pip check
research/lead_modeling/a4_a5/.venv-diabetes311/bin/python shared_splits/verify_shared_split.py --csv research/lead_modeling/a4_a5/data/diabetes_binary_health_indicators_BRFSS2015.csv
```

## Integrity and source identity

Protected files remain untracked, unchanged and excluded from the staging plan:

| File | Preserved SHA-256 |
|---|---|
| `CROSS_BRANCH_PROTOCOL_AUDIT.md` | `617af031428f865366eb62f940920de5a4d23162ed4cfa512cf292baaf2420a1` |
| `PROTOCOL_FREEZE_PROPOSAL.md` | `9075f4e9f8c5cd1acbf3dc6c29dfabf573c43ec04c6203a22ad75c7d982bfd1f` |

All 34 sealed file hashes and directory inventory match `/private/tmp/uet-real-train-group-gmrgtd23/sealed_artifact_inventory.json`; file/directory write bits remain removed. Neither labels, frequencies, split identities nor historical manifests were changed. Original CSV SHA remains `19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0`; shared archive remains `26f95eee418ae5510ce7e2f27369f02cf3385104dadb0b327715ef924c11bae3`.

Current execution source digest is **`993961640a66d4eafc7d5c3a34d6ef72c81cc295426e99091e0d8d40e69fa7a1`**, covering 28 execution/protocol/dependency/shared-asset files; map: `current_source_identity.json`. The new contract enters provenance automatically through the root-module hash inventory. Tests/documentation are outside that execution-source map.

The sealed real run retains historical digest **`4578d2f060298c5592da1815a82a9a7f7fc9d6f1c99b86f39c73cabd7bddc83a`**. Its exact execution source remains preserved in `/private/tmp/uet-final-audit-i8y1wiyc/source_candidate_snapshot.tar.gz`, SHA-256 `9bd52952311ca37491ee06c12cc523d910c5198f45ad6d341cf4c80d362da108`. Current code is deliberately a different source revision. Do not rebind or “repair” historical manifests to the new digest; historical artifacts are not fresh current-source A4/A5 outputs. Hashes detect corruption but do not authenticate an unknown pickle author.

## Changed files and human staging plan

Only these **16 paths** changed or were added during remediation; other Git changes predate this task:

```text
README.md
research/lead_modeling/a4_a5/README.md
research/lead_modeling/a4_a5/docs/CoReach_A4_A5_tasks.md
research/lead_modeling/a4_a5/docs/FINAL_P2_REMEDIATION_REPORT.md
research/lead_modeling/a4_a5/docs/METHODOLOGY_NOTES.md
research/lead_modeling/a4_a5/docs/PR_DESCRIPTION.md
research/lead_modeling/a4_a5/docs/V3_INTEGRATION_REPORT.md
research/lead_modeling/a4_a5/notebooks/A4_Training_Protocol_v2.ipynb
research/lead_modeling/a4_a5/notebooks/A5_Evaluation_XAI_v2.ipynb
research/lead_modeling/a4_a5/scripts/run_a4.py
research/lead_modeling/a4_a5/tests/test_notebook_portability.py
research/lead_modeling/a4_a5/tests/test_sensitivity_contract.py
src/diabetes_research/group_sensitivity.py
src/diabetes_research/modeling.py
src/diabetes_research/provenance.py
src/diabetes_research/sensitivity_contract.py
```

The reviewed human staging whitelist is `/private/tmp/uet-p2-remediation-1pi3tcjb/staging_whitelist.txt`: the earlier 41 coherent integration candidates plus the contract, two tests and this report (**45 paths**). It contains ten root modules, packaging/ignores/dependencies, shared split identity assets and component config/wrappers/notebooks/tests/current documentation. The shared NPZ is a required 786,199-byte split identity asset. The existing six staged moves contain older source bytes and must not be committed alone. No staging command was executed; the whitelist is a review plan, not publication approval.

Exclude both protected documents unless explicitly approved, all CSV/ZIP data, credentials, environments, caches, build/egg-info directories, all experimental results/models/probabilities/plots and all `/private/tmp` audit/synthetic/source archives. Do not force-add ignored outputs. No unsafe file is currently staged.

Draft PR title and description are in [PR_DESCRIPTION.md](PR_DESCRIPTION.md). Title: **Integrate root A4/A5 V3 package, fold-local selection and frozen sensitivity safeguards**. The description distinguishes preserved real training-only evidence, tested software, historical source and pending official evaluation.

## Remaining scientific gates and limitations

`official_run_release.status` remains **PENDING** and `official_execution_enabled` remains **false**. A passing selector/sensitivity contract supplies no permission for official A4 fitting, held-out scoring, A5 evaluation or real SHAP/LIME. Runtime search-budget overrides and effective source/environment need review at any separately authorized release. Extra bootstrap pairs, group-heldout evaluation and C baseline integration remain separately scoped work.

Native schema/default differences fail closed; supported dependency ranges are broader than this exact validated environment, so another environment needs verification. Implicit XGBoost `None` settings are preserved rather than invented; exact library versions remain necessary. Packaging/discovery checks cover reviewed full repository copies, not standalone wheels or a published remote V3 branch. No Kaggle reproduction claim is made.

Gain is model-specific importance with correlation/split-opportunity bias, not causal/clinical relevance. A3 CV is conditional on same-fold B3 tuning, not nested CV. Reserved-validation AP tolerance is not held-out noninferiority. Bootstrap intervals are marginal and conditional on fitted models, excluding training uncertainty. Historical test exposure prevents an untouched confirmatory-test claim. Authoritative proposal/creator clinical dictionary and C baseline agreement remain unverified.

## Final Git status

Branch remains `integration/a4-a5-v3`, tracking `origin/Hưng`. The index is byte-identical to the start: six R100 source moves. There are 19 tracked paths with unstaged changes (including the six moved files) and 28 individual untracked files: 26 integration candidates and the two protected documents. Nothing was staged or published during remediation. Exact final status follows:

```text
## integration/a4-a5-v3...origin/Hưng
 M research/lead_modeling/a4_a5/README.md
 M research/lead_modeling/a4_a5/data/README.md
 M research/lead_modeling/a4_a5/docs/A5_Results_Discussion_Template.md
 M research/lead_modeling/a4_a5/docs/CoReach_A4_A5_tasks.md
 M research/lead_modeling/a4_a5/docs/GitHub_Commit_Guide.md
 M research/lead_modeling/a4_a5/docs/METHODOLOGY_NOTES.md
 M research/lead_modeling/a4_a5/docs/PR_DESCRIPTION.md
 M research/lead_modeling/a4_a5/notebooks/A4_Training_Protocol_v2.ipynb
 M research/lead_modeling/a4_a5/notebooks/A5_Evaluation_XAI_v2.ipynb
 M research/lead_modeling/a4_a5/scripts/run_a4.py
 M research/lead_modeling/a4_a5/scripts/run_a5.py
 M research/lead_modeling/a4_a5/tests/test_audit_regressions.py
 M research/lead_modeling/a4_a5/tests/test_protocol.py
RM research/lead_modeling/a4_a5/src/diabetes_research/__init__.py -> src/diabetes_research/__init__.py
RM research/lead_modeling/a4_a5/src/diabetes_research/analysis.py -> src/diabetes_research/analysis.py
RM research/lead_modeling/a4_a5/src/diabetes_research/artifact_validation.py -> src/diabetes_research/artifact_validation.py
RM research/lead_modeling/a4_a5/src/diabetes_research/core.py -> src/diabetes_research/core.py
RM research/lead_modeling/a4_a5/src/diabetes_research/modeling.py -> src/diabetes_research/modeling.py
RM research/lead_modeling/a4_a5/src/diabetes_research/provenance.py -> src/diabetes_research/provenance.py
?? .gitignore
?? README.md
?? pyproject.toml
?? requirements-dev.txt
?? requirements-explain.txt
?? requirements.txt
?? research/lead_modeling/a4_a5/config/A2_protocol_v3.yaml
?? research/lead_modeling/a4_a5/config/PROTOCOL_AMENDMENTS_V3.json
?? research/lead_modeling/a4_a5/docs/CROSS_BRANCH_PROTOCOL_AUDIT.md
?? research/lead_modeling/a4_a5/docs/FINAL_P2_REMEDIATION_REPORT.md
?? research/lead_modeling/a4_a5/docs/PRIMARY_SELECTOR_IMPLEMENTATION_REPORT.md
?? research/lead_modeling/a4_a5/docs/PROTOCOL_FREEZE_PROPOSAL.md
?? research/lead_modeling/a4_a5/docs/V3_INTEGRATION_REPORT.md
?? research/lead_modeling/a4_a5/scripts/run_group_sensitivity.py
?? research/lead_modeling/a4_a5/tests/conftest.py
?? research/lead_modeling/a4_a5/tests/test_notebook_portability.py
?? research/lead_modeling/a4_a5/tests/test_primary_selector.py
?? research/lead_modeling/a4_a5/tests/test_sensitivity_contract.py
?? research/lead_modeling/a4_a5/tests/test_v3_integration.py
?? shared_splits/
?? src/diabetes_research/group_sensitivity.py
?? src/diabetes_research/paths.py
?? src/diabetes_research/sensitivity_contract.py
?? src/diabetes_research/shared_splits.py
```
