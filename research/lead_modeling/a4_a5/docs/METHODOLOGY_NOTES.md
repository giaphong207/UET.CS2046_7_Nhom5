# A4/A5 implementation and research-method reconciliation

Updated 18 November 2026. This records the current implementation before official experiments; it contains no research results or new citations.

## Available authority and historical documents

The available research source is `config/A2_protocol_v2.yaml`. A1/A2/A3 PDFs in the repository's `docs/lead-modeling/` and the component A4/A5 PDF/Word documents are preserved. The instructor assignment and approved `De_cuong_Diabetes (1).pdf` are absent. Their requirements and instructor-level compliance remain unverified.

Earlier A2 describes permutation importance within CV or SelectFromModel, while A3 discusses fold-local ranking. Current A4 uses **univariate ANOVA `SelectKBest(f_classif)` fitted on train**, followed by validation comparison of feature counts. This note reconciles descriptions with the existing implementation; it does not authorize changing the research method. The available YAML requires training-only ranking but does not name the ranking algorithm. Confirm the intended method against the missing proposal before the official run if it imposes a different method. No permutation-importance run, model-based ranking, weighted search CV, fold-specific selected sets or ranking-stability analysis has been completed.

## Executed modeling procedure

The task is binary classification of the BRFSS 2015 survey label `Diabetes_binary`. Label 1 combines surveyed prediabetes/diabetes. It is not a future-incidence target or a clinical diagnosis. The normal input is the 253,680-row binary CSV with 21 input features; its official checksum and actual data quality have not been verified locally.

Split rows by stratified 70/15/15 with seed 42. Validation and test sizes are `round(0.15*N)`; train is the remainder. Keep these partitions and the original class distributions; identical feature vectors are reported without treating them as individual identities or changing the approved row-level split. Vector-group sensitivity splitting is optional and unimplemented.

| Stage | Fitting and selection actually performed |
|---|---|
| B0 | Prior DummyClassifier fit once on train. |
| B1 | Unweighted Logistic Regression (`liblinear`, L2) with StandardScaler inside the train CV pipeline. All numeric inputs are scaled; ordinal codes remain numeric, with no one-hot encoding. |
| B2 | Unweighted Random Forest searched on train. |
| B3 | Unweighted XGBoost searched on train. |
| B1–B3 search | Seeded RandomizedSearchCV, StratifiedKFold with shuffle and seed 42, 3 folds by default, Average Precision selection and refit on full train. No test data are supplied. |
| A1 | Reuse B3's selected hyperparameters, change only weighting to negative/positive count from full train, then fit full train. The wrapper computes the ratio from each fit's labels, but this stage has no separate weighted CV search. |
| A2 | Same fitted A1 and probabilities; choose maximum-F1 threshold on validation. |
| A3 | For k=21/15/10/5, fit ANOVA selector and weighted XGBoost on full train with B3's parameters. Compare on validation. A3 has no separate CV. |
| Feature-count choice | Select the smallest k for which `AP_A2_validation - AP_A3_k_validation <= 0.01` absolute (numerical tolerance 1e-12). An AP improvement also qualifies. Keep k=21 as fallback. |
| Final-model choice | Highest validation AP among B1, B2, B3, A2 and the selected A3. Exact ties follow that candidate order, preserving the existing behavior. |
| Threshold and test | Baseline B0–B3/A1 rows report 0.5; A2/selected A3 use their validation-F1 thresholds. FINAL always uses its own validation-F1 threshold, even when a baseline wins. F1 ties within 1e-12 favor the highest threshold. Lock all choices before test scoring; no train+validation refit. Only selected A3 k is scored on test. |

AP uses `sklearn.metrics.average_precision_score`, not trapezoidal PR area. ROC-AUC, precision, recall, specificity, F1, accuracy and confusion counts are secondary metrics. Threshold changes alone leave AP/ROC-AUC unchanged. Ordinal-code numeric treatment and ANOVA ranking impose modeling assumptions that should be disclosed in the eventual report.

B1 has ten meaningful C values from 0.001 to 30, with `class_weight=None`. B2/B3 finite spaces have 72/288 configurations in normal mode. Default requests evaluate 10 configurations per family; a request for 15 evaluates only 10 for B1 and records exhaustion. Search counts distinguish requested CLI budget, effective budget after quick overrides, actual configurations, CV fits and refit fits. The manifest's `trials_per_family` now aliases the actual per-family map. A finite space is never padded with repeated configurations.

## Analysis and provenance

A4 copies the available YAML verbatim to `config.yaml`; executable settings are recorded separately in `runtime_config.json`. Seeds, split/CV parameters, complete finite search spaces/counts, Python/library versions, dataset and source-file hashes, run ID, UTC timestamps, duration and artifact checksums are saved. File hashes cover untracked modules/scripts, requirements and protocol; Git HEAD is supplementary. The YAML itself remains unchanged by the fix task.

A5 requires manifest v2, verifies critical hashes before loading the trusted model registry, checks compatible source/environment and validates all partitions, source-row mapping, feature/model IDs, fitted feature order, selected names and frozen threshold relationships. It recomputes saved test scores (`atol=1e-8`, `rtol=1e-7`) and metrics. All A5 statistics use exact recomputed scores, preserving model-score ties. These checks detect corruption and mixed runs; an unsigned manifest is not proof of a trusted creator. Only load model pickles from trusted sources. Legacy runs without integrity metadata must be rerun; incompatible environments/source require restoration or a new A4 run.

A5 records its own configuration, versions, source identity, artifact hashes, timestamps and A4 run/manifest/checksum linkage. Test labels are used for locked evaluation, descriptive calibration/Brier/error/subgroup analysis and paired bootstrap; they never cause refitting, model selection or threshold changes.

Effective runtime metadata also records each fitted pipeline step's complete learner parameters, including fixed settings. A5 verifies these against the registry and the saved CV winner. It checks distinct valid CV configurations, fold AP values, mean/std summaries and actual trial counts. Fixed scientific choices in the available YAML are checked against the implementation before snapshotting; CV-fold/search-budget overrides remain explicit. This preserves the existing hardcoded architecture rather than treating arbitrary YAML changes as implemented methods.

The paired AP comparison is FINAL minus B3, with seed 42, **1,000 attempts**, identical sampled positions for both models and a 95% percentile interval. Single-class samples are excluded without replacement; attempted, valid and excluded counts and policy are explicit. Research A5 enforces 1,000 attempts; smaller positive counts remain available for clearly tagged QA runs and are recorded as overrides. The interval is conditional on fixed fitted models and the held-out test set, not the entire training process.

The baseline-comparison intervals requested by A1/A2 (nonlinear B2/B3 minus B1) remain unimplemented. They require a declared analysis plan before the official experiment; the present task preserves the existing FINAL-minus-B3 comparison. FINAL=B3 produces a valid but uninformative self-comparison. No research completion claim is made. Earlier PDF/Word commands assume the component working directory; use the nested paths in the current README. Their weighted-CV/fold-ranking descriptions are superseded for current-code reporting by the explicit procedure table above; the source PDFs/Word files have been preserved.

All observed Sex and Age groups remain in the subgroup CSV. Groups with n<50 or a single class retain counts, the frozen threshold, `status=withheld` and a reason, with unavailable metrics left blank. Even an empty subgroup table has headers. n≥50 is an implementation reporting rule, not proof of statistical power or clinical fairness; Age is a code 1–13.

SHAP and LIME remain optional. SHAP seeds both sample/background selection and the permutation explainer; saves attribution values, settings and source rows; and explains positive-class probability through the final full pipeline. Its 15-row summary describes a sample, not population-wide importance. LIME saves seed, perturbation budget, categorical settings, background-row mapping and TP/FP/FN case provenance. **Neither optional explanation was runtime-verified in the current environment.** Perturbations/masking can create implausible feature combinations; explanations describe prediction associations, not causal health effects.

Quick mode preserves the existing seeded `min(260,N-2)` subset, 1 configuration per family and 2 folds. `source_row_indices.npy` maps subset positions to original CSV positions, and A5 validates that map. `row_index` is a 0-based source-file position, not a respondent ID. Quick/demo A5 requires `--allow-smoke` and preserves the QA status; no synthetic/quick output is an official result. Stacking and Optuna remain optional, unimplemented extensions.

## Artifact-name correspondence

The original YAML artifact names include conceptual deliverables. Their current implementation names are:

| YAML deliverable | Actual file |
|---|---|
| `versions.txt` | A4 `versions.json`; A5 `versions_a5.json` |
| `pr_curve.png`, `roc_curve.png`, `confusion_matrix.png` | A5 `precision_recall_test.png`, `roc_test.png`, `confusion_matrix_final.png` |
| `bootstrap_ap_delta.csv` | A5 `paired_bootstrap_ap.csv` and `paired_bootstrap_summary.json` |
| `ablation.csv` | A4 table combining all validation rows and the already-selected test rows |
| `readme.md`, `ai_usage_log.md` | Component README and AI log template exist; a completed research report and factual AI usage log still require actual work/records. |

See README for run commands and the fix report for executed software verification. PR readiness is distinct from research completion: official experiments, interpretation, authoritative-document review and any adopted optional extensions remain future work.
