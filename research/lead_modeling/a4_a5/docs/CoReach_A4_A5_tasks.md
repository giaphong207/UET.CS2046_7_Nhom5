# CoReach task definitions — A4/A5 V3

This file defines tasks; it does not post to CoReach or mark them Done. Assign actual owners/contributors and only record completion with reviewable evidence. **Current status: primary SelectFromModel implemented/software-tested and real TRAINING-ONLY GroupSplit AP complete; official full-data A4/A5, E6–E9 and actual SHAP/LIME runtime remain pending.** Preserve that distinction in task statuses.

## A4.1 — Original population, shared split and source integrity

Verify the original binary CSV full hash/schema, all 253,680 rows and 21 features, original labels, repeated X and conflicting labels. Accept only the exact shared B v9 archive after full hash, disjointness/coverage and two-stage membership/order checks. Save original-row CV folds. Verified numeric mapping is three-class 0/1/2 → binary 0/0/1; no CSV labels were changed. This is read-only semantics evidence; clinical dictionary verification remains PENDING.

Evidence: canonical verifier, `dataset_audit.json`, `split_manifest.json`, `split_indices.npz`, `source_row_indices.npy`, `cv_fold_indices.npz` and source hashes. The read-only real CSV/split verification has passed; it does not establish official model training.

## A4.2 — B0–B3 with training-only AP ranking

Depends on A4.1 and separate official release. Fit internal Dummy/LR/RF/XGBoost baselines, with stratified 3-fold CV, seed 42, AP ranking and fold-local B1 preprocessing. B3 must contain actual XGBoost; historical B v9 LightGBM results stay independently labeled. Report actual distinct search effort. C's cross-checks only enter the main shared table after source, split, fit settings and predictions match.

Evidence: `cv_results.csv`, validation/reference CSVs, model registry and runtime/manifest. Official completion remains pending; synthetic fits validate software only.

## A4.3 — Controlled class weighting

Depends on A4.2. Reuse B3-selected parameters for A1; compute weighting from the rows used in each fit. Compare B3/A1 at 0.5, without claiming a separate weighted CV search. Save exact settings and ratios. No research tradeoff is established until authorized results exist.

## A4.4 — Validation F2 threshold

Depends on A4.3. Maximize positive-class F2, beta=2, on the ascending `np.linspace(0.01,0.99,199)` grid; exact ties select the smallest threshold. Test never selects the threshold. Preserve the 0.5 reference and prove A1/A2 share scores/AP/ROC-AUC.

Evidence: `threshold_search.json`, `selection.json`, `selection_lock.json`, validation/test reference CSVs and F2 regression tests. Software coverage is complete; official operating-point claims are pending.

## A4.5 — Primary reduction k=21/15/10/5 (implemented; official results pending)

The explicit user instruction authorizes B3-configured weighted XGBoost gain SelectFromModel. Exact top-k support includes zero importance; exact gain ties use original column order. Fresh ranker/downstream fits use only each training fold, with a class ratio from its own labels; refit every k on full primary training. The smallest eligible k uses reserved-validation AP loss <=0.01 against locked full-feature A2. Keep A2 fallback and separately report selected A3. CV AP is descriptive conditional on the B3 search winner and does not select k. ANOVA remains separately named QA, never a primary substitute.

Evidence: amendment, fit-isolation/tie/tamper regressions, `feature_selection_cv.csv`, `feature_selection_folds.json`, `feature_selection_models.joblib`, `feature_rankings.json`, `feature_sets.json`, `a2_reference_lock.json` and selection lock. Synthetic checks establish implementation readiness. Official A4 remains gated by separate experiment authorization.

## A4.6 — Lock decisions and release held-out evaluation

Depends on approved A4.1–A4.5 and explicit experiment authorization. Persist the model/k/threshold/comparison lock before test scoring. Use new A4/A5 run IDs and preserve prior outputs. Never adapt methods from held-out results; acknowledge already-seen historical LightGBM test scores.

Evidence: `selection_lock.json`, `test_exposure_log.json`, `test_predictions.csv`, metrics/reference CSVs, trusted model registry and complete hashes. Official held-out release is not authorized by this implementation task.

## A4.S — Exact-X GroupSplit sensitivity

Keep separate from primary tasks/results. Use exact full X tuples excluding y; preserve conflicting labels and do not infer respondent identity. Verify no group crosses outer partitions or StratifiedGroupKFold boundaries. Report actual rows/groups/prevalences because GroupShuffleSplit ratios target groups and approximate rows.

The default runner performs diagnostics only. An authorized `--train-cv` run uses the frozen XGBoost preset for three ordinary-stratified and three stratified-group folds on primary training only. Report each fold, mean/std AP (`ddof=0`) and delta Group−Stratified as exploratory sensitivity; do not select a pipeline. Group-heldout performance requires a separately prespecified design/run ID and is unavailable in this runner.

Real **TRAINING-ONLY SENSITIVITY ANALYSIS** is complete: Stratified mean AP **0.4362740784**, Grouped mean AP **0.4358097385**, Group minus Stratified **−0.0004643398**. Grouped folds have zero exact-X overlap. These are descriptive training-CV results, not official held-out performance or significance/equivalence evidence; the primary model is unchanged.

Evidence: 34 sealed read-only files in `../results/real_training_only_group_cv_20261125T064729+0700_gmrgtd23/`, including `TRAINING_ONLY_SENSITIVITY_REPORT.md`, original source/fold IDs, `train_cv_ap_sensitivity.json` and `train_cv_fold_details.csv`. Manifest SHA-256 `a6d2175121d9a5117f25da4d36ec7be720b01355353c0e6cfeb826af2297e644`. See [FINAL_P2_REMEDIATION_REPORT.md](FINAL_P2_REMEDIATION_REPORT.md). Outer diagnostics are separately scoped; the sealed CV run constructed no outer alternative partition.

## A5.1 — Locked metrics and PR/ROC curves

Depends on A4.6. Validate critical bytes before trusted pickle loading, then verify source/environment, case IDs, original labels, folds and locked decisions without fitting or selecting. A owns PR/ROC/error plots; coordinate a single shared baseline table with B/C rather than creating duplicates. Synthetic roundtrips establish software behavior only.

## A5.2 — Exactly three paired-bootstrap AP comparisons

Depends on A4.6. Compute B2−B1, B3−B1 and FINAL−B3 from the same locked run/cases/original labels with one draw stream, 1,000 attempts, seed 42 and marginal 95% percentile CIs. Save observed deltas, reliable bound IDs, signs, attempted/valid/excluded counts, single-class exclusions and self-comparison notices. Extra A1/A3 pairs remain PENDING separate approval.

Evidence: `paired_bootstrap_ap.csv`, `paired_bootstrap_summary.json`, `paired_bootstrap_exclusions.csv` and linked A4/A5/split/lock identities. Intervals are conditional on fixed models; a FINAL=B3 self-comparison does not show improvement.

## A5.3 — SHAP global/local

Depends on A4.6 and installed explanation dependencies. Explain the locked whole pipeline's label-1 probability with permutation SHAP. Default 500 explained/100 training-background rows, seed 42; save selected source IDs and requested/actual budgets, with bounded chunk/mask/prediction processing. Config/CLI overrides must be recorded.

Evidence: successful explanation metadata, source-row archive and completed plots. Tests with a controlled stub verify planning, not actual SHAP success; this task remains experimentally pending.

## A5.4 — LIME for available TP/FP/FN

Depends on A4.6 and installed LIME. Keep `num_samples=1000` perturbations per case, separate from SHAP budgets. Save case source IDs, settings and success/missing-case outcomes. Explain model behavior without causal claims. Actual LIME success remains unestablished.

## A5.5 — Results, discussion and subgroup limitations

Depends on authorized verified outputs. Fill [A5_Results_Discussion_Template.md](A5_Results_Discussion_Template.md) using linked manifests/CSVs/logs. Report Sex/Age sample counts, single-class/small-group withholding, exact-X sensitivity separately, all three comparison limits, numeric label evidence and prior test exposure. Do not report historical LightGBM AP as official XGBoost or synthetic QA as research performance. Publish/post only under a separately authorized workflow.
