# SP2 — V3 Results and Discussion template

**No official experiment results are available.** Fill `[ ... ]` only from a separately authorized, compatible A4/A5 run after artifact verification. Synthetic QA and historical LightGBM numbers do not populate this template. Primary fold-local SelectFromModel is implemented and tested on synthetic data; official A3 and full-protocol results require separate evaluation authorization.

## 3.1. Population, protocol and run identity

Use all 253,680 original rows and 21 features from the pinned binary CSV, preserving labels and repeated feature vectors. The verified B v9 archive contains 177,576 training, 38,052 validation and 38,052 test positions in the exact two-stage stratified recipe order. Primary model ranking uses Average Precision (AP), with shuffled stratified 3-fold CV on primary training only, seed 42. B3 must actually be XGBoost. B1 preprocessing fits within each training fold; A1 reuses B3 parameters with training-derived class weighting, without claiming a separate weighted CV search.

The operating threshold maximizes positive-class F2 on validation: beta=2, ascending `np.linspace(0.01,0.99,199)`, smallest threshold on an exact tie, and `probability >= threshold`. Preserve metrics at the 0.5 reference. F1 remains a secondary metric. The exact final validation AP tie order remains B1, B2, B3, A2, selected primary A3.

Record A4/A5 run IDs `[ ... ]`, code/source digest `[ ... ]`, dataset and split archive digests `[ ... ]`, protocol/amendment digests `[ ... ]`, environment `[ ... ]`, actual search effort `[ ... ]`, original-row CV archive `[ ... ]`, and pre-test `selection_lock.json` digest `[ ... ]`. Cite the verified class counts and cross-split exact-X diagnostics from that run. A5 consumes locked decisions and never refits or selects models or thresholds.

## 3.2. Shared metrics table and curves

Link the single coordinated B0–B3/A1/A2/approved-A3/FINAL table `[ ... ]` and its source CSVs; do not create a second competing baseline report table here. A4's internally fitted B0–B2 remain the same-run comparisons. C's baseline outputs may enter the shared table only after predictions, fit settings, exact splits and full source hash match; otherwise identify them as separate QA.

Include AP, ROC-AUC, precision, recall, specificity, F2, secondary F1 and operating thresholds; identify the validation/test 0.5 reference CSVs. Link the locked PR/ROC curves and confusion matrix `[ ... ]`. A1 and A2 share probabilities, so their AP/ROC-AUC must be identical when only the threshold changes. Investigate any disagreement before writing conclusions.

## 3.3. Ablation and feature-reduction status

- Class weighting: compare B3/A1 at 0.5 with the same algorithm and B3-selected parameters.
- F2 threshold: compare A1 at 0.5 with A2 at the validation-selected threshold, using identical scores. Describe the observed precision/recall tradeoff without selecting from test.
- Primary A3: weighted XGBoost gain ranker reuses selected B3 parameters and computes class ratios from each fit's training labels. sklearn SelectFromModel uses -inf threshold and exact k=21/15/10/5. Exact gain ties use original column position; transformed columns retain original order. Report per-fold AP, source identities, full ranking vectors, selected names and actual counts from the verified selector artifacts `[ ... ]`.
- Eligibility: cite locked A2 validation AP `[ ... ]`, every k validation AP/loss `[ ... ]`, eligible budgets `[ ... ]` and smallest selected k `[ ... ]`. Full-feature A2 is fallback. Report selected A3 separately from FINAL. Preserve k=21/A2 score parity and validation F2 thresholds. The 0.01 rule is a validation selection criterion, not test noninferiority.
- CV interpretation: settings were selected by B3 on the same training folds. A3 fold AP is descriptive conditional on those settings, not nested CV; it never selects k. Report fold subset variation and gain bias/correlation limits.
- Secondary ANOVA: full-training `SelectKBest(f_classif)` is saved separately in `secondary_anova.json` for synthetic/quick QA only, without CV or held-out scoring. It does not populate the primary table or FINAL.

## 3.4. Three prespecified paired AP comparisons

Use exactly the following family from the same locked A4 run, aligned original test labels and case indices. A owns these comparison summaries and PR/ROC curves; coordinate with B/C to avoid duplicated report tables.

| Comparison key | Sign convention | Bound comparison ID | Observed delta AP | Marginal 95% percentile CI | Attempted / valid / excluded | Self-comparison notice |
|---|---|---|---:|---|---|---|
| `B2_minus_B1` | AP(B2) − AP(B1) | [ ... ] | [ ... ] | [ ... ] | [ ... ] | [ ... ] |
| `B3_minus_B1` | AP(B3) − AP(B1) | [ ... ] | [ ... ] | [ ... ] | [ ... ] | [ ... ] |
| `FINAL_minus_B3` | AP(FINAL) − AP(B3) | [ ... ] | [ ... ] | [ ... ] | [ ... ] | [ ... ] |

All three share one draw stream: 1,000 attempts, seed 42, same sampled cases across models and pairs. Exclude and log single-class draws without replacement attempts; attempted=valid+excluded. Cite `paired_bootstrap_ap.csv`, `paired_bootstrap_summary.json`, `paired_bootstrap_exclusions.csv`, the draw-stream digest and linked run/split/lock identities `[ ... ]`. Report the observed point delta separately from the bootstrap mean. If FINAL resolves to B3, flag that self-comparison and its zero difference rather than claiming improvement.

Intervals are marginal, conditional on fixed fitted models and this test split; they do not include training variability or simultaneous family coverage. A CI containing zero provides no clear directional evidence under this calculation. Additional A1/A3 intervals require separate prespecification/approval. Historical B v9 LightGBM test results were already seen, so describe these comparisons without an untouched confirmatory-test claim.

## 3.5. Exact-X GroupSplit sensitivity

Keep this section separate from primary metrics. Exact equality of all X features excludes y and does not establish respondent identity, including when labels conflict. The separate outer GroupShuffleSplit design targets 70/15/15 by groups, with approximate row ratios. Cite its run/design IDs, actual partition rows, groups, class prevalence and zero cross-partition group intersections `[ ... ]`.

For an authorized training-CV sensitivity run, report the frozen XGBoost settings and primary-training-only source IDs/folds `[ ... ]`. Compare ordinary StratifiedKFold AP with StratifiedGroupKFold AP using three folds each. Record fold rows/groups/prevalence and verify zero group crossings in group folds; ordinary folds may have crossings.

| Training-CV sensitivity | Mean AP | Population std AP (`ddof=0`) |
|---|---:|---:|
| Ordinary stratified | [ ... ] | [ ... ] |
| Stratified group | [ ... ] | [ ... ] |

Report delta = mean AP Group − mean AP Stratified `[ ... ]`, with an exploratory/sensitivity label. Do not use it to select the primary pipeline. Outer diagnostics contain no held-out performance; any group-heldout evaluation needs a separately prespecified design and run ID.

## 3.6. Error and subgroup analysis

Record TP/TN/FP/FN and inspected error-case source IDs `[ ... ]`; describe observed false negatives/positives without changing the selected method. For Sex/Age report every subgroup's n, positive count, status and withheld reason. Metrics are withheld for n<50 or single-class groups; Age is category code 1–13, not years. These descriptive results do not establish clinical fairness.

## 3.7. SHAP and LIME

Only claim success from an actual compatible explainer run. Permutation SHAP explains label-1 probability for the whole locked final pipeline. Defaults are **500 explained rows and 100 training-background rows**, seed 42, deterministic subsampling and chunks of 25 with bounded prediction/mask processing. Record requested/actual counts, all CLI/config settings, `shap_selected_source_rows.npz`, source IDs and success metadata `[ ... ]`; describe the summary as a subsample explanation. Stub-based software tests do not establish real SHAP success.

LIME retains its distinct **1,000 perturbations per case** for available TP/FP/FN examples. Record settings, source IDs and completed files `[ ... ]`, including missing case types. Perturbations can create implausible health combinations. Neither method supports a causal or clinical interpretation.

## 3.8. Limitations and evidence-based conclusions

The source is a cross-sectional self-reported survey, with no onset time or respondent identifiers. Use the supplied `Diabetes_binary=1` label wording. Verified numeric alignment maps binary 1 to three-class code 2; the creator clinical dictionary remains unverified, so the old combined prediabetes/diabetes wording is unsupported. No labels were changed. Direct clinical or Vietnamese-population generalization is unsupported.

Keep historical AP 0.4286/0.4331 labeled LightGBM, with previous test exposure acknowledged. Official A4/A5, E6–E9 and real SHAP/LIME completion remain unestablished until separately evidenced. Write conclusions only after checking the coordinated metrics table, all three comparison intervals, ablations, sensitivity status and verified explanation outputs. Report uncertain or absent improvements without changing the metric after observing results.
