# Primary SelectFromModel implementation report

Date: **2026-11-25**, Asia/Ho_Chi_Minh. Branch: **integration/a4-a5-v3**. This continues the historical [V3 integration report](V3_INTEGRATION_REPORT.md).

**Implementation is ready and synthetic validation passes. Official evaluation permission remains PENDING.** No official full-dataset A4 training, official held-out test scoring, commit, push, merge or rebase was performed. All model scoring in this task used constructed synthetic data. No teammate branch was changed. The six existing staged source renames and all earlier worktree changes were preserved.

## Scientific decisions actually implemented

| Area | Implemented decision |
|---|---|
| Ranking estimator | Fresh `FoldAwareXGB` using the already selected B3 XGBoost configuration, with `weight_mode='balanced'` and explicit `importance_type='gain'` |
| Class weighting | Negative/positive ratio computed only from the labels supplied to each ranking or downstream fit; no reserved validation/test labels enter a fit |
| Selector | `ExactTopKSelectFromModel`, a sklearn `SelectFromModel` subclass; `threshold=-inf`, `max_features=k`, `prefit=False`, `importance_getter='feature_importances_'`, `norm_order=1` |
| Exact budgets | k=21,15,10,5, including zero-gain columns where necessary; wrong budgets or invalid importance vectors fail |
| Ranking ties | Stable descending sorting of unrounded native normalized gain; exact ties use original input-column position |
| Transform order | Selected columns retain original feature-column order, not ranking order |
| CV | Fresh selector/ranker/downstream fits in each of the same saved stratified three-fold primary-training folds; AP evaluated on that fold's validation rows |
| CV interpretation | Descriptive diagnostics conditional on B3 settings selected using the same training CV; not independent nested CV and not used to select k |
| Full-training refit | Fresh selector and weighted downstream XGBoost on the complete primary training partition for each k; no train+validation refit |
| A2 reference | Full-feature weighted A2 reserved-validation AP saved before selector evaluation in `a2_reference_lock.json` |
| k decision | Smallest k with reserved-validation AP loss <=0.01 relative to A2, retaining the existing 1e-12 arithmetic tolerance; improvements qualify |
| Fallback and final arm | Full-feature A2 remains fallback; selected A3 remains separately reported. FINAL retains highest-validation-AP order B1,B2,B3,A2,selected A3. k=21 retains all columns and A2 score parity |
| Threshold | Positive-class F2, beta=2, validation only; existing ascending `np.linspace(0.01,0.99,199)`, smallest threshold on exact ties, `probability >= threshold` |
| Preserved design | Actual unweighted XGBoost B3, baseline search spaces, verified shared original-row split, three-fold CV and exactly B2−B1/B3−B1/FINAL−B3 paired bootstrap comparisons |
| Secondary ANOVA | Separately named ANOVA_k* full-training QA in `secondary_anova.json`; no CV, primary/FINAL selection or held-out scoring. Only quick/demo runs fit this secondary experiment |
| Held-out boundary | Model/k/threshold decisions and model artifacts locked before synthetic test prediction; test labels/metrics never rank features or select budgets/thresholds |

The historical unapproved `PROTOCOL_FREEZE_PROPOSAL.md` section 5 recommended a fixed Random Forest impurity ranker. Existing V3 YAML and amendments left estimator/importance/tie/fold details pending. The latest explicit user request supplies B3-configured weighted XGBoost gain ranking and is recorded as implementation authorization in the amendment. The historical proposal is preserved unchanged. No fixed Random Forest ranker was implemented and no team signature or official release was inferred. Other previously superseded historical F1/split wording remains historical; this task preserves current V3 F2/shared-split decisions.

## Artifact and provenance contract

A4 adds `feature_selection_cv.csv`, `feature_selection_folds.json`, `feature_selection_models.joblib`, `feature_rankings.json`, `a2_reference_lock.json` and separately scoped `secondary_anova.json`. Fold records include original ordered fit/validation source IDs, source/dataset/split/CV hashes, gain vectors and ranked names, selected support/names/actual counts, AP and ranking/downstream settings/ratios. Full-training ranking records cover every k. Fitted ranking/downstream estimators record fit-row IDs, fit-label hashes and fit sizes. All new artifacts are critical hash bindings and part of the pre-test selection lock.

A5 verifies bytes and source/environment identity before loading trusted pickles. It checks fold completeness and exact frozen fold identities, original columns and selected support/counts, nested estimator configuration and gain settings, actual XGBoost internals/weights, fit-row/label evidence, full-training refit rankings, A2 reference, k=21 score parity and validation AP/F2 evidence. It recomputes training-CV AP from the serialized fold models without fitting. A5 metadata records the selected A3/k and final columns from the same locked A4 run. Source hashes include both notebooks and the protocol/amendment.

Manifest/registry schema remains V3, with the new artifacts required. Existing pending-selector V3 outputs lack the new contract and are rejected. Fresh compatible run IDs are required; older outputs are preserved. Hashes and internal consistency checks do not authenticate an unknown pickle creator or prove arbitrary external training history; only trusted locally produced registries are accepted.

Current aggregate source SHA-256: `4578d2f060298c5592da1815a82a9a7f7fc9d6f1c99b86f39c73cabd7bddc83a` across 27 execution/protocol/dependency/shared-asset files. Full map: `/private/tmp/uet-primary-selector-glmu3son/final_source_identity.json`. Current YAML SHA-256: `77cc4f865fd74ab8a3704d54b69a471c613dffff4efd912017a4f4eb159b758c`. Current amendment SHA-256: `92301e9ccb561cee06c7d94b3247295fe534a7c90133bb17ddb5c1763a7c6add`. Documentation is excluded from that execution-source map.

## Validation results

| Check | Result |
|---|---|
| Dependencies | `python -m pip check`: No broken requirements found; no dependency changes/downloads required |
| Compilation | `python -m compileall -q src research/lead_modeling/a4_a5/scripts shared_splits/verify_shared_split.py`: exit 0 |
| Focused primary tests | 36 passed, 19 deselected in 3.76s before the final test addition |
| Full pytest | **174 passed in 109.87s**: all 118 existing regression cases retained, 56 new primary-selector cases |
| Notebook validity | Both notebooks nbformat-valid, code compilation passes, empty outputs and execution counts; covered by full suite |
| Entry points | All four wrappers import from unrelated cwd and `--help` pass in the full suite |
| Synthetic CLI A4→A5 | Exit 0 from `/private/tmp`, 300 original constructed rows, 210/45/45 partitions, 3 folds, 10 actual configurations per B1/B2/B3 |
| Primary synthetic effort | 12 fold pipeline fits and 4 full-training pipeline refits; each includes a ranking and downstream fit, 32 XGBoost fits total for primary feature selection |
| Artifact integrity | A4 critical hashes, A5 output hashes, linked source/run identity, selector CV/full artifacts and model probabilities verified; `synthetic_integrity_validation.json` saved |
| Three bootstrap comparisons | Each recorded 1,000 attempted/1,000 valid/0 excluded synthetic draws, one shared stream; single-class exclusions also covered by regression tests |
| Protection/Git checks | Both untracked protocol documents byte-identical; staged index binary diff unchanged; `git diff --check` exit 0 |

The new tests exercise exact k with zero/tied importance, unrounded near ties, original-column transform order, selector/ranker/downstream fit isolation, per-fit ratios, validation eligibility boundaries and fallback, protocol setting drift, tampered fold models/source IDs/weights/scores/counts, full-training support/configuration, A2 reference tampering, ANOVA substitution/separation, checksum rejection before unpickle and A4/A5 run/feature mismatch. A fixed-partition synthetic test flips only test labels and proves unchanged feature rankings, k and validation thresholds while requiring a selection lock before test prediction. Existing A5 no-fit guards now also forbid SelectFromModel fitting.

Validation history: an initial collection overlapped intermediate code edits and failed import resolution before the new provenance function was added. The first expanded suite passed 172/174; a new QA spy incorrectly assumed ANOVA prediction inputs were DataFrames, and one appended notebook statement lacked a newline. Both were fixed; the final full suite passes. A first synthetic driver resolved the virtual-environment Python symlink to system Python, which lacked joblib; it failed before training. The driver was corrected to retain the virtual-environment executable, and the final CLI/integrity roundtrip passes. These were corrected QA/notebook issues, not dependency installation failures or unresolved scientific conflicts.

Environment: Python 3.11.15, numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, XGBoost 3.2.0, scipy 1.17.1, joblib 1.6.0, pytest 9.2.0. SHAP/LIME/Optuna remain uninstalled. Exact environment JSON and logs are under `/private/tmp/uet-primary-selector-glmu3son`.

Synthetic A4 run: `a4_20261124T204703977613Z_2d170351c8af`. Synthetic A5 run: `a5_20261124T204726545571Z_553c8d85b9a9`. These are UTC identifiers; local reporting date is 25 November. Both retain `SYNTHETIC_QA_ONLY`. Synthetic AP/CI values are deliberately not presented as research results.

Reproduction from the repository root, retaining the existing Python environment:

```bash
PIP_DISABLE_PIP_VERSION_CHECK=1 research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m pip check
research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m compileall -q src research/lead_modeling/a4_a5/scripts shared_splits/verify_shared_split.py
MPLCONFIGDIR=/private/tmp/uet-primary-selector-glmu3son/mpl-cache XDG_CACHE_HOME=/private/tmp/uet-primary-selector-glmu3son/xdg-cache research/lead_modeling/a4_a5/.venv-diabetes311/bin/python -m pytest -q
research/lead_modeling/a4_a5/.venv-diabetes311/bin/python research/lead_modeling/a4_a5/scripts/run_a4.py --data /private/tmp/uet-primary-selector-glmu3son/synthetic_3fold.csv --output-dir /private/tmp/NEW_a4_3fold --demo --cv 3 --trials 10
research/lead_modeling/a4_a5/.venv-diabetes311/bin/python research/lead_modeling/a4_a5/scripts/run_a5.py --data /private/tmp/uet-primary-selector-glmu3son/synthetic_3fold.csv --a4-dir /private/tmp/NEW_a4_3fold --output-dir /private/tmp/NEW_a5_3fold --allow-smoke --bootstraps 1000
```

Use new output directory names. No official dataset verifier, official training or official test scoring was invoked in this continuation.

## Changed files in this continuation

These 21 paths differ from the start-of-turn snapshots or were added here. This list excludes earlier changes merely still present in Git status.

```text
README.md
research/lead_modeling/a4_a5/README.md
research/lead_modeling/a4_a5/config/A2_protocol_v3.yaml
research/lead_modeling/a4_a5/config/PROTOCOL_AMENDMENTS_V3.json
research/lead_modeling/a4_a5/docs/A5_Results_Discussion_Template.md
research/lead_modeling/a4_a5/docs/CoReach_A4_A5_tasks.md
research/lead_modeling/a4_a5/docs/GitHub_Commit_Guide.md
research/lead_modeling/a4_a5/docs/METHODOLOGY_NOTES.md
research/lead_modeling/a4_a5/docs/PRIMARY_SELECTOR_IMPLEMENTATION_REPORT.md
research/lead_modeling/a4_a5/docs/PR_DESCRIPTION.md
research/lead_modeling/a4_a5/docs/V3_INTEGRATION_REPORT.md
research/lead_modeling/a4_a5/notebooks/A4_Training_Protocol_v2.ipynb
research/lead_modeling/a4_a5/notebooks/A5_Evaluation_XAI_v2.ipynb
research/lead_modeling/a4_a5/tests/test_audit_regressions.py
research/lead_modeling/a4_a5/tests/test_primary_selector.py
research/lead_modeling/a4_a5/tests/test_v3_integration.py
src/diabetes_research/analysis.py
src/diabetes_research/artifact_validation.py
src/diabetes_research/core.py
src/diabetes_research/modeling.py
src/diabetes_research/provenance.py
```

## Preserved documents and remaining limitations

| Preserved untracked document | SHA-256 |
|---|---|
| `CROSS_BRANCH_PROTOCOL_AUDIT.md` | `617af031428f865366eb62f940920de5a4d23162ed4cfa512cf292baaf2420a1` |
| `PROTOCOL_FREEZE_PROPOSAL.md` | `9075f4e9f8c5cd1acbf3dc6c29dfabf573c43ec04c6203a22ad75c7d982bfd1f` |

Both remain untracked and were neither rewritten nor deleted. Historical results, datasets, shared-split assets, existing virtual environments and the staged six-file rename patch were preserved. New QA run files are only under `/private/tmp/uet-primary-selector-glmu3son` and pytest temporary directories.

Official mode no longer fails because SelectFromModel is unimplemented: its validated configuration is ready, and its explicit failure now identifies **pending official evaluation authorization**. A sentinel-based preflight regression establishes this without official fitting/scoring. `official_run_release` stays PENDING and `official_execution_enabled` stays false. This implementation task supplies no permission for official full-dataset training or held-out evaluation.

Remaining limitations: primary performance and E6–E9 are unestablished; real SHAP/LIME and real training-only group CV remain unrun. Gain is a model-specific split importance, normalized by XGBoost, subject to correlation and split-opportunity bias; it is not clinical/causal relevance. A3 CV is conditional on same-fold B3 tuning, not nested CV. The validation 0.01 rule is not test noninferiority. Previous historical test exposure, missing authoritative proposal/creator clinical dictionary and fixed-model marginal bootstrap limits remain disclosed. No new experiment or approval is invented to resolve them.

## Final Git status

The branch remains `integration/a4-a5-v3`, tracking `origin/Hưng`. Worktree is dirty from existing integration work plus this continuation. The index remains exactly the six previously staged R100 source moves; nothing was staged here. No commit, push, merge or rebase was performed.

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
?? research/lead_modeling/a4_a5/docs/PRIMARY_SELECTOR_IMPLEMENTATION_REPORT.md
?? research/lead_modeling/a4_a5/docs/PROTOCOL_FREEZE_PROPOSAL.md
?? research/lead_modeling/a4_a5/docs/V3_INTEGRATION_REPORT.md
?? research/lead_modeling/a4_a5/scripts/run_group_sensitivity.py
?? research/lead_modeling/a4_a5/tests/conftest.py
?? research/lead_modeling/a4_a5/tests/test_primary_selector.py
?? research/lead_modeling/a4_a5/tests/test_v3_integration.py
?? shared_splits/
?? src/diabetes_research/group_sensitivity.py
?? src/diabetes_research/paths.py
?? src/diabetes_research/shared_splits.py
```
