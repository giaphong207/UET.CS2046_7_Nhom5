# CODEX Research and Code Audit — UET CS2046 Diabetes A4/A5

Audit started: 17 November 2026. Report completed: 18 November 2026 (Asia/Ho_Chi_Minh).

Repository: `UET.CS2046_7_Nhom5`; branch: `feature/lead-modeling-a`; observed HEAD: `8c8ba648d38c1ac4adad827b502d2ebb30892b12`.

Scope: audit only. Existing implementation, configuration, notebooks and research documents were read without modification. Verification artifacts were generated under `/private/tmp/diabetes_codex_audit_2xr3w8_y/`, outside the repository. No official-dataset training, dependency installation, commit, push, merge, rebase or deletion was performed. This report is the only project file created by the audit.

## 1. Executive Summary

**Final decision: BLOCKED. Findings: 0 P0, 3 P1, 9 P2 — 12 findings total.**

The implementation is a functioning research scaffold, with sound train/validation/test boundaries in the normal A4 code path. Eight existing pytest tests passed; core imports, dependency consistency, syntax compilation, notebook structure and both CLI help commands passed. A small synthetic A4→A5 run also completed. These checks establish software behavior on controlled inputs, not performance or validity on the official Kaggle dataset.

The three high-priority problems are incomplete A5 artifact consistency checks, missing immutable experiment configuration/provenance, and a Logistic Regression search space that permits only four configurations while the protocol and default budget describe approximately 10–15 per family. These should be resolved before pushing this implementation as ready for review.

No confirmed P0 incident or normal-path preprocessing/model-selection leakage was found. In particular, the deliberately inconsistent artifacts used in audit probes were copies of synthetic outputs; they are evidence of missing A5 guards, not evidence that the original A4 run contaminated its test set.

The two highest-priority source documents and the official CSV were not found in the repository. Accordingly, instructor-level compliance, exact compliance with the approved proposal, actual dataset integrity, official performance, and completion of A4/A5 cannot be certified. Missing documents and unexecuted research are recorded as verification limitations, rather than counted as invented implementation defects.

### Source of truth and evidence boundaries

The repository was inventoried with `rg --files --hidden --no-ignore`, excluding Git internals, virtual environments and generated caches. No applicable `AGENTS.md` was found in the repository or its parent directories. All 29 original project files were fingerprinted before execution.

| Priority/source | Located evidence | Inspection and limitation |
|---|---|---|
| 1 — Instructor assignment | `AI_Midterm_Project (1).pdf`: not found | NOT VERIFIED. No grading, submission or mandatory-extension requirements were inferred from an unseen assignment. |
| 2 — Approved group proposal | `De_cuong_Diabetes (1).pdf`: not found | NOT VERIFIED. References to it in other reports do not constitute inspection of the proposal. |
| 3 — Revised protocol | `research/lead_modeling/a4_a5/config/A2_protocol_v2.yaml` | Read all 133 lines. Highest-priority research source actually available. |
| 4 — A1 | `docs/lead-modeling/A1.pdf` | All 5 pages extracted and read. Revised planning document dated 16 November; explicitly no official training results. |
| 4 — A2 | `docs/lead-modeling/A2.pdf` | All 5 pages extracted and read. Revised experiment plan, including tuning, threshold, feature reduction and bootstrap. |
| 4 — A3 | `docs/lead-modeling/A3.pdf` | All 7 pages extracted and read. Theory/methodology chapter; numerical examples are labeled hypothetical. |
| 5 — A4 documentation | `research/lead_modeling/a4_a5/docs/A4_Thuc_nghiem_Machine_Learning_SP2.pdf` and `.docx` | PDF: 2 pages; Word XML paragraphs read. Implementation document, not completed results. |
| 5 — A5 documentation | `research/lead_modeling/a4_a5/docs/A5_Phan_tich_SHAP_LIME_SP2.pdf` and `.docx` | PDF: 3 pages; Word XML paragraphs read. Results/explanation framework with placeholders. |
| 5 — Implementation/support | All source, scripts, tests, both notebooks, three requirements files, README, data README, `.gitignore`, and every Markdown file under component `docs/` | Read. Notebooks have no saved executions or outputs. |
| Dataset/results | Official CSV absent; repository `results/` contains only `.gitkeep` | No official schema measurements, checksum, prevalence, duplicate counts, metrics, model artifact or research run could be inspected. |

PDF inspection here was text extraction, not a repeat of the historical visual rendering check claimed in `docs/QA_REPORT.md`. Cited primary papers were not retrieved; their numerical claims and bibliographic details were not independently authenticated.

### Location notation

To keep the traceability tables readable, `P` means `research/lead_modeling/a4_a5/`. `C` = `P/src/diabetes_research/core.py`; `M` = `P/src/diabetes_research/modeling.py`; `A` = `P/src/diabetes_research/analysis.py`; `Y` = `P/config/A2_protocol_v2.yaml`. References such as `M:137` are actual source line numbers. PDF citations use page and section numbers, not invented source lines.

## 2. Requirements Traceability Matrix

**Status meaning:** PASS = inspected implementation and, where feasible, software verification support the requirement; FAIL = a concrete mismatch was identified; PARTIAL = only part is implemented/evidenced; NOT VERIFIED = required evidence was unavailable. A code-level PASS never certifies official-dataset results.

| ID | Expected behavior | Implementation location | Evidence | Status |
|---|---|---|---|---|
| SRC-01 | Verify official instructor requirements | Source inventory; secondary references in A1–A3 | Assignment PDF absent. | NOT VERIFIED |
| SRC-02 | Verify approved group proposal and revised extension scope | Source inventory; Y:1; README:14 | Proposal absent; available revised documents and user request treat Stacking/Optuna as extensions. | NOT VERIFIED |
| D-01 | Use the full official binary CSV, 253,680 rows and 21 features | M:105, M:107; C:43 | Normal run checks filename and row count, loader checks exact feature set. Official file absent. Filename/size and a locally recorded hash do not authenticate Kaggle provenance by themselves. | NOT VERIFIED |
| D-02 | Target `Diabetes_binary`, labels 0/1, both classes | C:14, C:36, C:51 | Strict label check; invalid target rejected in controlled probe. | PASS |
| D-03 | Explain positive label as surveyed prediabetes/diabetes, without clinical/future-incidence claims | README:125; package `__init__.py:3`; A1 p.2 §1; A3 p.2 §2.1 | Available documentation consistently makes the correct distinction; original dataset dictionary unavailable. | PASS |
| D-04 | Validate exact feature schema and reject added index columns | C:40, C:43; tests:59 | Missing feature rejected; strict loader requires the expected 21 names. | PASS |
| D-05 | Reject missing and nonfinite values, with safe handling guidance | C:47, C:49 | NaN and infinity rejected; missing-data error directs imputation into CV pipeline. No global imputation exists. | PASS |
| D-06 | Validate binary, ordinal and count feature domains | C:55, C:58 | Binary checks pass; fractional ordinal/day values and negative BMI are accepted. F-05. | PARTIAL |
| D-07 | Record row count, prevalence, missingness and contradictory duplicate vectors | C:66; M:111, M:130 | Synthetic audit correctly counted duplicates and same-X/different-y vectors; official measurements absent. | PARTIAL |
| D-08 | Measure identical feature vectors across splits without calling them duplicate individuals | C:74; M:121; Y:102 | Global duplicate statistics only. Controlled constant-X fixture overlaps across all three splits; official overlap unknown. F-06. | PARTIAL |
| S-01 | Stratified 70/15/15, seed 42, disjoint exhaustive indices | C:80; M:121 | Synthetic 200 rows: 140/30/30; size-only synthetic labels of length 253,680: 177,576/38,052/38,052. Repeated split identical. | PASS |
| S-02 | Save split indices tied to source data | M:122, M:129 | Normal/demo run saves indices and source hash. Quick mode loses original-row mapping. F-12. | PARTIAL |
| S-03 | Preserve original class distribution in validation/test | C:85, C:87 | Stratification implemented; synthetic 20% positive prevalence preserved. Actual official prevalence and per-split counts unverified. | NOT VERIFIED |
| S-04 | Fit preprocessing after split, inside relevant CV folds | M:75, M:137, M:140 | LR scaler is part of the searched pipeline; no fitted preprocessing before split. | PASS |
| S-05 | Keep test out of model/hyperparameter/k/threshold selection | M:153, M:175, M:184, M:198, M:202 | Selection uses training CV and validation; test scoring follows saved selection. No A5 model fitting. | PASS |
| B-01 | B0 DummyClassifier | M:133 | `strategy='prior'`, trained on train only, evaluated on same held-out rows. | PASS |
| B-02 | B1 Logistic Regression | M:75, M:85 | Scaled `liblinear` LR; synthetic fitting and serialization succeeded. | PASS |
| B-03 | B2 Random Forest | M:77, M:86 | Seeded RF; synthetic fitting and serialization succeeded. | PASS |
| B-04 | B3 Gradient Boosting/XGBoost | M:26, M:78, M:90 | XGBoost wrapper; synthetic fitting and probability outputs succeeded. | PASS |
| B-05 | Consistent splits, scoring and comparison procedure | M:132, M:137, M:202 | Common AP scorer and partitions. Actual search-budget inconsistency is tracked separately under H-02. | PASS |
| H-01 | Stratified 3-fold CV on train; primary score AP | M:132, M:138, M:236 | Correct default; CV is passed only `Xt,yt`. Lower-cost audit demo deliberately used 2 folds. | PASS |
| H-02 | Approximately 10–15 search configurations per family, honestly recorded | Y:34; M:85, M:137, M:217 | LR space has 4 configurations; request for 10 runs only 4. Manifest stores requested count as `trials_per_family`. F-03. | FAIL |
| H-03 | Record search results, seed and reproducible choices | M:142, M:147, M:212 | Sampled params and mean/std CV AP saved; full search-space/configuration snapshot, timing and source identity missing. F-02. | PARTIAL |
| H-04 | Accurately identify optimization backend | M:18; README:14 | `RandomizedSearchCV` implemented. No Optuna search or Stacking code. Optional scope preserved. | PASS |
| I-01 | Compute class weights from the labels supplied to training | M:45, M:49, M:52, M:150 | Balanced XGB ratio uses only fit-time y; existing ratio test passed. Actual weighted A1 fits full train after unweighted B3 search. | PASS |
| I-02 | Never resample validation/test; training-fold-only oversampling if present | M:121; source review | No oversampling implemented. No validation/test resampling. | PASS |
| I-03 | Evaluate recall with precision and specificity | C:115, C:116, C:117; discussion template:28 | All metrics saved for B3/A1 at 0.5, then A1/A2 under their respective thresholds. Research tradeoffs await official outputs. | PASS |
| T-01 | Use 0.5 reference and validation-only maximum-F1 alternative | C:96; M:153, M:157, M:205 | Direct tie fixture and 40 brute-force threshold comparisons passed. | PASS |
| T-02 | Freeze selected threshold before test evaluation | M:188, M:198, M:199 | Saved selection precedes test-scoring block; no train+validation refit. | PASS |
| T-03 | AP/ROC-AUC remain unchanged when only threshold changes | C:112; M:151 | Hand check passed; synthetic A1/A2 probability vectors identical. | PASS |
| F-REQ-01 | Evaluate all 21/15/10/5 feature sets | M:162 | All four trained and compared on validation. Test scores generated for the selected k only. | PASS |
| F-REQ-02 | Fit feature ranking on permitted training data | M:163, M:165 | `SelectKBest(f_classif)` fit on `Xt,yt`; no validation/test-based ranking. | PASS |
| F-REQ-03 | Refit selector within a fold if selector participates in CV comparison | M:137, M:162 | Current A3 selector does not participate in CV; only its train-fit validation comparison runs. Pipeline would support fold-local fitting if used in CV. Do not describe that CV as already executed. F-07. | PASS |
| F-REQ-04 | Select smallest k with validation AP loss ≤0.01 absolute units | M:175, M:178, M:181 | Correct signed loss comparison; improvements also qualify. Full k=21 model matched A1 in software probe. | PASS |
| F-REQ-05 | Preserve selected names and distinguish eligible reduction from final model | M:189, M:192, M:212 | Names saved. `selected_feature_k` describes the A3 candidate; final may be a full-feature baseline. | PASS |
| E-01 | Correct confusion matrix, precision, recall, specificity, F1 and accuracy | C:107; A:70 | Hand-computed four-row fixture verified counts and all five rates; all saved synthetic test rows recomputed consistently. | PASS |
| E-02 | Correct AP and ROC-AUC from scores | C:112, C:113 | Hand fixture verified AP=5/6 and ROC-AUC=0.75; software example only. | PASS |
| E-03 | Correct PR/ROC curves and prevalence reference | A:51 | Synthetic PR/ROC files generated; scikit-learn curve functions used, AP labeled separately from trapezoidal PR area. | PASS |
| E-04 | Paired bootstrap uses identical indices and FINAL-minus-B3 direction | A:19, A:34, A:39, A:42 | Compared all retained replicate indices/differences against independent seeded resampling. | PASS |
| E-05 | Intended 1,000 resampling attempts, 95% percentile interval | A:19, A:33, A:43; A:245 | 1,000-attempt software checks passed; 2.5%/97.5% quantiles independently matched. | PASS |
| E-06 | Explicitly handle and report degenerate bootstrap samples | A:35, A:40, A:45 | Single-class samples skipped; all-invalid case raises. Valid count saved, but skip policy/count and attempted count are not self-contained in summary. F-10. | PARTIAL |
| E-07 | Interpret bootstrap conditionally on fixed fitted models | A:22, A:46, A:48; README:124 | Correct scope and zero-inclusion interpretation; no full-training uncertainty claim. | PASS |
| E-08 | FP/FN examples are selected consistently and traceable | A:107, A:118, A:124 | Highest-score FP and lowest-score FN chosen; row index, truth and score saved. Clinical/feature-profile interpretation remains manual. | PASS |
| E-09 | Age/Sex evaluation includes subgroup counts and limitations | A:94, A:100, A:103 | Counts saved only for retained groups; excluded groups vanish and empty CSV is unreadable. F-09. | PARTIAL |
| E-10 | Calibration/Brier are descriptive held-out analyses | A:83, A:91 | Quantile-bin calibration and Brier generated on synthetic run; no test-fitted calibration adjustment. | PASS |
| X-01 | SHAP/LIME explain the actual selected pipeline with correct feature order | A:128, A:136, A:152, A:163, A:205 | Code passes selected full pipeline, restoring original columns before scoring. Runtime explanations unavailable because dependencies are absent. | NOT VERIFIED |
| X-02 | Use train reference data, valid observed cases and explicit model output | A:131, A:132, A:146, A:159, A:167 | SHAP explains positive probability; LIME classification wrapper receives selected test cases. Masked/perturbed combinations need interpretation limits. | PARTIAL |
| X-03 | Distinguish global/local explanations and control cost/randomness | A:128, A:138, A:143, A:152, A:168 | Small SHAP sample/background and bounded LIME perturbations; incomplete SHAP seed/row metadata. F-11. | PARTIAL |
| X-04 | Report missing LIME/SHAP dependencies honestly | A:129, A:153; requirements-explain.txt | Both absent. LIME guard raises documented RuntimeError; SHAP raises ModuleNotFoundError. Neither marked runtime PASS. | NOT VERIFIED |
| X-05 | Avoid causal or clinical interpretation | A:149; README:12; discussion template:49 | Consistent limitations in explanatory documentation. | PASS |
| R-01 | Record dataset SHA-256 | C:23; M:129; A:184 | Synthetic hash saved; A5 rejected altered hash. Official dataset hash unavailable. | PASS |
| R-02 | Save immutable experiment configuration snapshot | Y:114; M:214 | No `config.yaml` snapshot or YAML read/copy; effective settings only partially reflected in manifest. F-02. | FAIL |
| R-03 | Record Python/library versions for each execution stage | C:132; M:131; A:227 | A4 saves versions; A5 does not record its current environment. Broad constraints are not a lockfile. F-02. | PARTIAL |
| R-04 | Serialize and reload the final pipeline consistently | M:212; A:202 | Clean synthetic registry reload and probability round trip passed; README documents trusted pickle sources. | PASS |
| R-05 | Verify artifact model/threshold/prediction/split consistency in A5 | A:192, A:200, A:203, A:207 | Five basic guards passed; three inconsistent synthetic artifact cases accepted. F-01. | FAIL |
| R-06 | Preserve experiment identifiers and prevent mixed output runs | M:102, M:222; A:177 | Nonempty output directories refused; directory name is implicit run ID. No immutable artifact/source-code identity in manifests. F-02. | PARTIAL |
| R-07 | Validate CLI required arguments and basic bounds | M:240, M:241; A:250 | Five invalid-argument cases returned exit 2 with specific messages. Overrides can still depart from the stated research defaults. | PASS |
| R-08 | README/notebook entry points match actual repository layout | README:19; A4 notebook:20,31,80; A5 notebook:31,58 | Documented root has no requirements/scripts/src/data; component paths differ. F-04. | FAIL |
| R-09 | Both notebooks are structurally valid and compatible with Python syntax | Both notebooks | nbformat validation passed; 9 code cells compiled; no full notebook execution due missing CSV/artifacts. | PARTIAL |
| R-10 | Dependencies work on this macOS Intel/Python 3.11 environment | requirements files; executed import/pip/pytest checks | Actual x86_64 environment passed core checks. Optional XAI and other allowed version combinations unverified. | PARTIAL |
| G-01 | Keep venvs, local datasets, secrets, caches and large outputs out of commits | `.gitignore:2,5,7,12,13`; Git inventory | Component ignore probes passed; no staged files or high-confidence secret matches in inspected text. Root-level ignore protection absent. | PARTIAL |
| G-02 | Do not present smoke/synthetic outputs as research | M:196, M:214; A:181; README:3 | Correct QA tags and default A5 refusal verified. No synthetic outputs placed in project results. | PASS |
| DEL-01 | Provide official results, ablation discussion and honest AI log | Y:119, Y:125, Y:126; documents/templates | Templates only; no completed official deliverables. Exact instructor acceptance criteria unavailable. | NOT VERIFIED |

## 3. Findings by Severity

| Severity | Count | Findings |
|---|---:|---|
| P0 — Critical | 0 | No confirmed incident identified within inspected evidence. |
| P1 — High | 3 | F-01–F-03 |
| P2 — Medium/Low | 9 | F-04–F-12 |

### P0

None confirmed. Missing official evidence is not proof of invalid results, and accepted inconsistent QA artifacts are not evidence of contamination in an original A4 run.

### P1 — High

#### F-01 — A5 does not bind predictions, threshold and partitions to the loaded model registry

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/analysis.py:192`, `:200`, `:202`, `:203`, `:207`, `:208`; `research/lead_modeling/a4_a5/src/diabetes_research/modeling.py:212`.

A5 checks source SHA, input column order, test-row indices, y_true and final model ID. It then trusts `prob_FINAL`/baseline probabilities from CSV and the threshold from `selection.json`. It does not compare the registry threshold, verify the registry feature-name contract, compare cached probabilities with registry predictions, or verify saved partitions are disjoint and exhaustive.

Three separate copies of a valid synthetic A4 run were tested: replacing FINAL probabilities with a constant inconsistent with the loaded model; changing only the registry threshold; replacing the saved train indices with test indices. All three were accepted by A5 and analysis completed. Five existing guards independently rejected incorrect hash, input feature order, test-row order, labels and final ID.

Impact: internally inconsistent run artifacts can produce A5 figures/statistics that do not correspond to the saved model or locked decision rule. If XAI is subsequently enabled, it can explain a different scoring function from the CSV used in the analysis. This latter consequence is inferred from the code path; XAI runtime was not exercised.

Required fix: validate partition contents and the complete feature/model/threshold contract, and bind the stored score columns to the corresponding fitted models using checksums and/or deterministic recomputation. Verification must keep the model and decision rule frozen. Add targeted failure checks for the three demonstrated cases.

#### F-02 — Experiment configuration and provenance are not captured immutably

Locations: `research/lead_modeling/a4_a5/config/A2_protocol_v2.yaml:114`; `research/lead_modeling/a4_a5/src/diabetes_research/modeling.py:99`, `:131`, `:147`, `:214`; `research/lead_modeling/a4_a5/src/diabetes_research/analysis.py:227`; `research/lead_modeling/a4_a5/src/diabetes_research/core.py:132`.

The implementation names `A2_protocol_v2` in its manifest, but neither entry point reads or snapshots that YAML. The synthetic run produced no `config.yaml`. CLI overrides, full search spaces, effective per-family trial counts, elapsed times, source-code identity and artifact hashes are not captured as a complete run specification. A5 also omits its own Python/library versions, so it cannot establish that analysis used the A4 environment.

Existing seed, source SHA, split indices, A4 versions, CV parameter rows and model registry provide useful partial provenance. They do not substitute for the required configuration snapshot. The inspected implementation files are untracked and therefore are not identified by the recorded Git HEAD alone.

Required fix: archive the approved protocol and effective run settings, identify the executed source revision/hash, record stage-specific environment and timing, and bind A5 to immutable A4 artifacts. A hardcoded implementation can remain hardcoded if that decision is documented; the immediate defect is missing evidence of the actual configuration, not a requirement to refactor the system into a YAML-driven framework.

#### F-03 — Logistic Regression cannot meet the stated search budget, and manifest counts can overstate it

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/modeling.py:85`, `:137`, `:143`, `:217`; `research/lead_modeling/a4_a5/config/A2_protocol_v2.yaml:34`; `research/lead_modeling/a4_a5/README.md:62`.

B1 searches only four C values with class_weight fixed to None. Search-space cardinalities are B1=4, B2=72 and B3=288. An actual B1 search requested with `n_iter=10` completed four configurations and warned:

```text
The total space of parameters 4 is smaller than n_iter=10. Running 4 iterations. For exhaustive searches, use GridSearchCV.
```

The manifest still records the requested scalar as `trials_per_family`. The CV CSV reveals the actual count, but the manifest and documentation do not. This contradicts the proposed approximate 10–15 configurations per family and weakens claims about comparable tuning effort.

Required fix: choose and document a suitable B1 budget/search space before the official experiment, or explicitly approve and report the smaller LR search as a limitation. Always store actual family-specific configuration and fit counts. Do not add irrelevant parameters merely to inflate the count.

### P2 — Medium/Low

#### F-04 — README and notebook working-directory assumptions do not match the component layout

Locations: `research/lead_modeling/a4_a5/README.md:19`, `:24`, `:48`, `:142`; `research/lead_modeling/a4_a5/notebooks/A4_Training_Protocol_v2.ipynb:20`, `:31`, `:55`, `:80`; `research/lead_modeling/a4_a5/notebooks/A5_Evaluation_XAI_v2.ipynb:31`, `:58`; `research/lead_modeling/a4_a5/docs/GitHub_Commit_Guide.md:6`, `:12`.

README changes to the repository root but installs root `requirements.txt` and invokes root `scripts/`. The actual files are nested under `research/lead_modeling/a4_a5/`. Both notebooks use `Path.cwd()` for component-relative data/scripts; A4 explicitly instructs opening from the repository root. Inventory confirmed the referenced root requirements, scripts, src, data and config paths are absent.

Required fix: state the component working directory consistently or resolve paths to the actual component. Keep Git staging examples scoped to the nested files. The nested `.gitignore` does not protect a venv or results folder created at the root by the current instructions.

#### F-05 — Domain checks accept negative BMI and fractional encoded/count values

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/core.py:49`, `:55`, `:58`, `:61`.

Controlled CSV probes accepted BMI=-5, Age=1.5, Education=2.5, Income=3.5, GenHlth=2.5, MentHlth=0.5 and PhysHlth=10.5. Bounds checks do not enforce integer codes/counts, and BMI has no domain check. Invalid binary values, labels, NaN and infinity were correctly rejected.

Required fix: validate integer membership/count semantics and a BMI domain justified by the actual dataset dictionary. Do not invent a clinical exclusion cutoff or delete official rows without documenting the decision. No claim is made that the official CSV actually contains these defects.

#### F-06 — Saved data audit omits per-split class counts and cross-split identical-vector counts

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/core.py:66`, `:74`, `:76`; `research/lead_modeling/a4_a5/src/diabetes_research/modeling.py:121`, `:130`, `:221`.

`dataset_audit.json` reports global duplicate/contradictory-vector counts; the manifest reports only partition sizes. Neither records positive/negative counts per split or intersections of feature-vector groups across splits.

A controlled 40-row constant-X fixture yielded 39 duplicate rows, one disagreeing-label vector and one shared vector for each pair of splits. That is expected under row-level stratification and does not establish duplicate individuals. The actual official overlap remains unknown.

Required fix: report per-split distributions and exact vector-overlap diagnostics, keeping the primary approved row split unless a separately declared sensitivity experiment is authorized. Group-aware splitting remains optional under Y:102.

#### F-07 — Research documents overstate or differ from the executed feature-selection/weighted-CV procedure

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/modeling.py:78`, `:140`, `:150`, `:163`, `:165`, `:189`; `research/lead_modeling/a4_a5/README.md:10`, `:56`, `:57`; `docs/lead-modeling/A2.pdf` p.3 §7; `docs/lead-modeling/A3.pdf` p.4 §2.6.3; A4 PDF p.1 §4 and p.2 §6.

A2 describes permutation importance in CV or SelectFromModel; current A4 uses an ANOVA filter, `SelectKBest(f_classif)`. A4 documentation names this filter, but A2 was not reconciled. The actual run cross-validates unweighted B3, then fits weighted A1 and all A3 selectors once on full train. It does not run weighted A1/A3 CV or save fold-specific selected sets/ranking stability.

The current train-fit/validation-evaluate procedure has no demonstrated feature-selection leakage. Having a selector inside a Pipeline does not establish that it was cross-validated. The higher-priority YAML permits training-only ranking but does not specify this ranking algorithm; therefore this is an academic consistency issue, not a confirmed violation of an unseen approved proposal.

Required fix: align A1–A5 descriptions with the actual algorithm and evaluation scope, record the filter choice and numerical treatment of ordinal codes, and distinguish univariate ranking from model-based importance. Any methodological change requires a documented decision before the official run.

#### F-08 — Final-model tie comment disagrees with the implemented rule

Location: `research/lead_modeling/a4_a5/src/diabetes_research/modeling.py:182`, `:184`, `:187`.

The comment says ties favor B3. The key `(validation_AP, -candidates.index(k))` favors the earliest candidate, B1, then B2, then B3. An equal-score fixture returned B1. This affects exact AP ties; it does not alter ordinary unequal-score selection.

Required fix: agree on the rule before training, make code/documentation agree, and record the final-model tie rule in selection metadata.

#### F-09 — Subgroup filtering silently removes groups and can create an unreadable CSV

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/analysis.py:94`, `:100`, `:104`, `:221`.

Groups with fewer than 50 observations or only one class disappear rather than retaining n, positive_n and a reason for withholding metrics. When all groups are excluded, `pd.DataFrame(out)` has no columns; the synthetic run produced a one-byte `subgroups_test.csv`, and `pd.read_csv` raised `EmptyDataError`.

Required fix: preserve a stable output schema and report counts/status for every observed age/sex group, using undefined values where a metric cannot be estimated. The n≥50 rule should be disclosed; it is not evidence of adequate power or clinical fairness.

#### F-10 — Bootstrap summaries do not fully disclose attempts, exclusions and seed

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/analysis.py:33`, `:35`, `:44`, `:230`.

The mathematical paired bootstrap and percentile interval passed software checks. The loop makes 1,000 attempts rather than retaining exactly 1,000 valid samples. In the synthetic A4→A5 run, 998 were retained. A two-row synthetic stress case retained 494 and skipped 506.

The summary records `n_valid_bootstrap` but not attempted count, skipped count, exclusion rule, seed, confidence level or named interval method as structured fields. Attempt count is separately in the A5 manifest; missing replicate numbers indirectly reveal exclusions.

Required fix: make bootstrap metadata self-contained and distinguish attempted/retained samples. If the group intends exactly 1,000 retained replicates, document that policy before changing the loop. There is no confirmed incorrect-AP or incorrect-CI finding here.

#### F-11 — Optional XAI reproducibility/evidence is incomplete

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/analysis.py:130`, `:137`, `:146`, `:152`, `:162`, `:168`; `research/lead_modeling/a4_a5/docs/QA_REPORT.md:5`, `:9`, `:10`.

SHAP row/background selection uses a seeded generator, but the permutation explainer is not given an explicit library seed. Metadata records counts and output units, not selected/background row indices, attribution values or the exact settings needed to identify the waterfall case. LIME has an explicit seed, but its perturbation budget and explanation provenance are not fully recorded in dedicated metadata.

SHAP and LIME are absent from the inspected environment. The historical QA report claims a successful SHAP run with different library versions, but no referenced artifacts are present to verify it. This does not prove the historical run failed. Current SHAP/LIME runtime functionality and stochastic stability remain NOT VERIFIED.

Required fix when the optional extension is pursued: control and record the explainer's randomness, row mappings, model identity, settings and outputs; install/validate optional dependencies in an authorized follow-up; repeat a small explanation run for stability. The 15-row SHAP summary must be described as a sampled global summary, not population-wide importance.

#### F-12 — A4 quick outputs cannot be consumed by A5, even with `--allow-smoke`

Locations: `research/lead_modeling/a4_a5/src/diabetes_research/modeling.py:115`, `:117`, `:122`; `research/lead_modeling/a4_a5/src/diabetes_research/analysis.py:189`, `:191`, `:248`.

Quick mode resets subset row positions and saves split indices without the source-to-subset map. On the 200-row synthetic CSV it retained 198 rows. A5 verified the original source hash, then rejected the row count with `Smoke A4 uses a subset: A5 needs exact original-to-subset mapping; rerun full A4`.

Required fix: either persist and validate the subset mapping for an end-to-end quick QA path, or document that `--allow-smoke` supports full-row demo outputs only. This limitation affects software validation, not official full-data model selection.

## 4. Test Execution Summary

All research-related numerical examples below use controlled synthetic inputs. None are dataset performance results.

### Environment and command context

Commands were run from `/Users/humanormal/Projects/UET.CS2046_7_Nhom5/research/lead_modeling/a4_a5` using `.venv-diabetes311/bin/python`. Execution used `PYTHONDONTWRITEBYTECODE=1`; pytest/CLI/probes also used `PYTHONPATH=src`, a temporary `MPLCONFIGDIR`, and a temporary `XDG_CACHE_HOME`. Pytest cache was disabled and its temporary files were directed outside the repository.

| Item | Observed value |
|---|---|
| Python | 3.11.15 |
| Host | macOS 15.1.1, x86_64 |
| numpy / pandas | 2.4.6 / 2.3.3 |
| scikit-learn / scipy | 1.9.1 / 1.17.1 |
| xgboost / matplotlib | 3.2.0 / 3.11.2 |
| joblib / PyYAML | 1.6.0 / 6.0.3 |
| nbformat / pytest | 5.11.1 / 9.2.0 |
| shap / lime / optuna | Not installed |

### Core checks — actual commands/calls and output

Dependency check:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-diabetes311/bin/python -m pip check
```

```text
No broken requirements found.
```

An inline Python import check imported numpy, pandas, sklearn, scipy, xgboost, matplotlib, joblib, yaml, nbformat and pytest successfully. The first import/pip call warned about unwritable default cache locations; Matplotlib used a temporary cache and pip disabled caching. These were cache warnings, not import/dependency failures. Subsequent checks explicitly used writable temporary caches.

Syntax and notebook checks were executed as inline Python, using these exact operations on all matching project files:

```python
compile(p.read_text(encoding='utf-8'), str(p), 'exec')
notebook = nbformat.read(p, as_version=4)
nbformat.validate(notebook)
compile(c.source, f'{p}:code_cell_{i+1}', 'exec')
```

```text
COMPILE PASS: 4 package modules, 2 scripts, 1 test module
NOTEBOOK VALID notebooks/A4_Training_Protocol_v2.ipynb code_cells 5 outputs 0 executed 0
NOTEBOOK VALID notebooks/A5_Evaluation_XAI_v2.ipynb code_cells 4 outputs 0 executed 0
```

The first line consolidates seven individual `COMPILE PASS <path>` messages. No notebook cells that train models were executed.

Existing pytest suite, with the actual arguments:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/diabetes_codex_audit_2xr3w8_y/mpl XDG_CACHE_HOME=/private/tmp/diabetes_codex_audit_2xr3w8_y/cache PYTHONPATH=src .venv-diabetes311/bin/python -m pytest tests -q -p no:cacheprovider --basetemp=/private/tmp/diabetes_codex_audit_2xr3w8_y/pytest
```

```text
........                                                                 [100%]
8 passed in 50.58s
```

Both commands below exited 0 and displayed their documented flags:

```bash
.venv-diabetes311/bin/python scripts/run_a4.py --help
.venv-diabetes311/bin/python scripts/run_a5.py --help
```

Five parser-error commands were executed with the same interpreter and temporary-cache environment:

```bash
.venv-diabetes311/bin/python scripts/run_a4.py
.venv-diabetes311/bin/python scripts/run_a4.py --data /private/tmp/nonexistent_diabetes_audit.csv --trials 0
.venv-diabetes311/bin/python scripts/run_a4.py --data /private/tmp/nonexistent_diabetes_audit.csv --cv 1
.venv-diabetes311/bin/python scripts/run_a5.py
.venv-diabetes311/bin/python scripts/run_a5.py --data /private/tmp/nonexistent_diabetes_audit.csv --a4-dir /private/tmp/nonexistent_a4_audit --bootstraps 0
```

```text
missing_required_a4 exit 2 run_a4.py: error: the following arguments are required: --data
zero_trials exit 2 run_a4.py: error: --trials must be 1..100
invalid_cv exit 2 run_a4.py: error: --cv must be >=2
missing_required_a5 exit 2 run_a5.py: error: the following arguments are required: --data, --a4-dir
zero_bootstraps exit 2 run_a5.py: error: --bootstraps must be positive
```

### Controlled numerical/domain probes

An inline harness was executed with `.venv-diabetes311/bin/python - /private/tmp/diabetes_codex_audit_2xr3w8_y`. It used the existing `toy_data()` helper, `load_dataset`, `stratified_indices`, `metrics_at_threshold`, `optimal_f1_threshold`, `bootstrap_paired_ap`, `subgroup_metrics` and `ParameterGrid`. No test/source file was added or edited.

```text
SCHEMA negative_BMI ACCEPTED
SCHEMA fractional_Age ACCEPTED
SCHEMA fractional_Education ACCEPTED
SCHEMA fractional_Income ACCEPTED
SCHEMA fractional_GenHlth ACCEPTED
SCHEMA fractional_MentHlth ACCEPTED
SCHEMA fractional_PhysHlth ACCEPTED
SCHEMA missing REJECTED ValueError
SCHEMA infinity REJECTED ValueError
SCHEMA invalid_binary REJECTED ValueError
SCHEMA invalid_target REJECTED ValueError
SEARCH_SPACE_CARDINALITIES B1=4 B2=72 B3=288
SPLIT_SYNTHETIC_200 train=140/28 positive validation=30/6 positive test=30/6 positive
SPLIT_SIZE_ONLY_253680_SYNTHETIC_LABELS train=177576 validation=38052 test=38052
DUPLICATE_SYNTHETIC_CROSS_SPLIT_UNIQUE_VECTORS 1 1 1
METRIC_HAND_CHECK_AND_THRESHOLD_BRUTE_FORCE_40 PASS
BOOTSTRAP_MANUAL_REFERENCE_1000 PASS valid_replicates 996
BOOTSTRAP_DEGENERATE_SYNTHETIC attempted 1000 valid 494 skipped 506
BOOTSTRAP_IDENTICAL_PREDICTIONS_1000 PASS
SUBGROUP_SMALL_SYNTHETIC output_rows 0 output_columns []
LIME DEPENDENCY_BLOCK RuntimeError LIME chưa cài: pip install -r requirements-explain.txt
SHAP DEPENDENCY_BLOCK ModuleNotFoundError No module named 'shap'
```

This block normalizes dictionary output and shortens four rejection messages; the acceptance/rejection outcomes and counts are the actual observations. The 253,680-length probe constructed labels only to verify split rounding; it did not load or train on a 253,680-row dataset.

The hand-check fixture was y=[0,0,1,1], p=[0.1,0.6,0.4,0.9], threshold=0.5: all confusion counts equal 1, precision/recall/specificity/F1/accuracy equal 0.5, AP=5/6 and ROC-AUC=0.75. Changing threshold preserved AP/ROC-AUC. The explicit F1 tie fixture y=[1,0,0,1], p=[0.9,0.8,0.7,0.6] selected 0.9, the highest winning threshold.

The paired-bootstrap reference independently generated the same seeded sample positions, checked every retained replicate's FINAL-minus-baseline AP difference, and checked the summary quantiles against the generated differences. This goes beyond the existing identical-predictions test.

### Synthetic integration and artifact guards

Executed via an inline Python harness using `.venv-diabetes311/bin/python -u - /private/tmp/diabetes_codex_audit_2xr3w8_y`. Its main calls were:

```python
df = toy_data(200)
df.to_csv(csv, index=False)
manifest = modeling.run(csv, root/'a4_demo', trials=1, cv_folds=2, demo=True)
summary = analysis.run(csv, root/'a4_demo', root/'a5_demo', bootstraps=1000, allow_smoke=True)
quick = modeling.run(csv, root/'a4_quick', trials=1, cv_folds=2, quick=True, demo=True)
```

Additional calls compared reloaded registry probabilities and every stored test metric with fresh calculations, exercised the actual B1 `n_iter=10` search, and tested independent copies of synthetic artifacts. Fault-injection runs used 10 bootstrap attempts to keep costs low; the main clean A5 run used 1,000.

Selected actual stdout:

```text
SYNTHETIC SOFTWARE VALIDATION ONLY; NO BRFSS RESEARCH PERFORMANCE CLAIMS
A4_SYNTHETIC PASS SYNTHETIC_QA_ONLY split_counts {'train': 140, 'validation': 30, 'test': 30}
SERIALIZATION_FEATURE_ORDER_AND_ALL_SAVED_TEST_METRICS PASS
A1_A2_SCORE_IDENTITY_AND_K21_EQUIVALENCE PASS
LR_REQUESTED_10 actual_trials 4
A5_DEFAULT_QA_REFUSAL PASS A4 run has status SYNTHETIC_QA_ONLY. A5 refuses to treat smoke as research.
A5_SYNTHETIC PASS valid_bootstraps 998
EMPTY_SUBGROUP_CSV confirmed EmptyDataError
ARTIFACT_GUARD wrong_hash PASS CSV SHA256 khác A4. Không dùng dữ liệu thay đổi hoặc stale source.
ARTIFACT_GUARD wrong_features PASS Feature schema differs from A4
ARTIFACT_GUARD wrong_test_order PASS Test row indices do not align with A4 test split
ARTIFACT_GUARD wrong_test_labels PASS Test ground truth differs from A4 predictions
ARTIFACT_GUARD wrong_final_id PASS Model registry and selection file disagree
FAULT_INJECTION prediction_model_mismatch ACCEPTED_BY_A5
FAULT_INJECTION registry_selection_threshold_mismatch ACCEPTED_BY_A5
FAULT_INJECTION train_test_split_overlap ACCEPTED_BY_A5
A4_QUICK_SYNTHETIC PASS SMOKE_NOT_RESEARCH_RESULT source_rows 200 manifest_rows 198
A5_QUICK_CHAIN confirmed failure Smoke A4 uses a subset: A5 needs exact original-to-subset mapping; rerun full A4
SYNTHETIC_PROBE_DURATION_SECONDS 12.81
```

The three `ACCEPTED_BY_A5` outcomes are guard failures, even though the diagnostic harness itself exited successfully. They are not presented as passing requirements.

The clean synthetic A4 run created 12 artifacts; clean A5 created 11, including four descriptive plot files, paired bootstrap outputs, errors and subgroup outputs. A4 `config.yaml` and `ablation.csv` were absent; A5 had no current-environment snapshot. No synthetic performance metrics or conclusions are used to fill research templates.

### Repository/document checks

Executed read-only Git commands included `git branch --show-current`, `git status --porcelain=v1 --untracked-files=all`, `git ls-files`, `git rev-parse HEAD`, `git diff --cached --name-only`, and `git check-ignore -v` for component venv, CSV, model-result, `.env` and cache paths. Initial status was 29 untracked project files, no staged changes, and no tracked project files in this checkout.

Ignore checks matched `.venv-*/`, `/data/*.csv`, `/results/*`, `.env`, `.env.*` and `__pycache__/` at the component level. Text-file scanning found no high-confidence token/private-key patterns, candidate secret files or project files larger than 5 MB. Values were suppressed by design. This is a bounded repository inspection, not a claim of comprehensive secret scanning of every environment/binary/Git object.

All five available PDFs were read using `pdftotext -layout <path> -`; both DOCX files were read using ZIP/XML extraction. No missing source was treated as inspected.

Original source/document hash comparison before report creation returned:

```text
ORIGINAL_PROJECT_FILE_HASH_CHANGES []
```

### Checks deliberately not performed

No official-dataset fitting or full notebook execution; no dependency installation or alternate-version compatibility matrix; no runtime SHAP/LIME explanations; no external primary-paper/proposal authentication; no historical run reproduction from absent artifacts. These remain NOT VERIFIED rather than PASS.

## 5. Academic Consistency Assessment

The available revised A1–A3 documents, A4/A5 templates and README agree on the binary survey task, AP as the primary score, 70/15/15 split, seed 42, B0–B3 baselines, controlled weighting/threshold ablations, and avoiding clinical, future-incidence or causal claims. They consistently label the work as planning/implementation without official experimental results. No fabricated research conclusion was identified in the inspected documents.

| Topic | Assessment | Required reconciliation |
|---|---|---|
| Stacking/Optuna scope | User request and available revised documents treat them as extensions. Code uses RandomizedSearchCV, with neither extension implemented. YAML entries sit under `optional_extension`. | Preserve this distinction. Verify original allocation/proposal when supplied; do not silently mark extension work complete. |
| Tuning budget | A1/A2/A3/A4 describe roughly 10–15 configurations per family; B1 can run only 4. | F-03: revise budget/search space and record actual effort. |
| Weighted CV | FoldAwareXGB supports fold-local weighting; actual searched B3 is unweighted and weighted A1 fits once on train. | F-07: describe the executed controlled ablation accurately. |
| Feature method | A2 p.3 §7 specifies permutation importance or SelectFromModel; A4 code/documentation use ANOVA SelectKBest. | F-07: reconcile the predeclared method without an unapproved change. |
| Feature stability/CV | A3 p.4 §2.6.3 discusses selected sets per fold; current outputs contain train-selected sets only. | Distinguish a capability or proposed check from an executed fold-wise experiment. |
| Ordinal representation | LR standardizes all columns and treats Age/Education/Income/GenHlth as numeric codes. Trees split these codes numerically; no one-hot encoding is present. A3 acknowledges this modeling assumption. | Record the actual representation and equal-spacing/linear-effect assumption for LR; do not describe the scaler as an implemented encoder. |
| Reduced-feature claim | `selected_feature_k` identifies an eligible A3 candidate; final model is independently selected by validation AP and may still use 21 inputs. | Report the actual final model/feature set separately from the eligible reduction result. |
| Baseline RQ confidence intervals | A1 p.4 RQ1 and A2 p.4 §9 request uncertainty for nonlinear-vs-LR comparisons. A5 hardcodes FINAL-minus-B3 only at A:214. | Complete the planned comparative analysis under a frozen analysis plan, or document its omission. FINAL=B3 gives a correct but uninformative self-comparison. |
| Ablation artifacts | Validation has all k levels; test outputs contain the selected k only. No `ablation.csv` is generated despite Y:119. | Create a traceable curated ablation deliverable from available outputs; never invent ungenerated test metrics for unselected k. |
| Historical QA | QA_REPORT cites sklearn 1.8, XGB 3.1.3 and SHAP 0.50, unlike the current environment; referenced artifacts are absent. | Treat historical claims as historical/unverified. Current evidence is in this audit. |
| AI usage / bibliography | AI log is a template only. A3 cites the LIME paper as KDD; A5 cites a NAACL version without the same link. Primary papers were not retrieved. | Preserve actual interaction records and harmonize citations to the exact versions consulted; do not infer that one venue is wrong without inspecting the source. |

Because the assignment and group proposal are missing, this assessment cannot certify rubric compliance or prove that every secondary document accurately reflects the approved methodology.

## 6. Data Leakage Assessment

| Investigation | Evidence and conclusion |
|---|---|
| 1. Train/validation/test contamination | C:80 constructs disjoint exhaustive row partitions; clean synthetic checks passed. Official data/run unverified. A5 accepts externally inconsistent partitions (F-01), which is separate from normal A4 behavior. |
| 2. Global preprocessing before split | None found. Schema checks/hash/global descriptive audit at M:105–112 do not fit a predictive transform. LR scaling is inside the searched pipeline. |
| 3. Feature-selection leakage | A3 selectors fit `Xt,yt` only at M:165. Validation selects k; test does not. A3 is not currently cross-validated, so no executed selector-CV result is being certified. |
| 4. Class-weight computation | M:49 uses only y supplied to fit; balanced A1 gets train y at M:150. No validation/test labels used in a weighting fit. |
| 5. Hyperparameter selection using test | Search receives `Xt,yt` at M:140, with AP and seeded stratified CV. No test eval_set or test early stopping. |
| 6. Threshold tuning using test | Threshold helper is called on validation y/scores at M:153 and M:170. Final threshold saved before test scoring. |
| 7. Misleading baseline comparisons | Common data/metrics are present. Budget discrepancy is confirmed (F-03). A1/A2 same score vector is correct; FINAL can be the same baseline with a validation-tuned threshold. |
| 8. Duplicate feature-vector handling | No automatic deletion. Global same-X/different-y audit is correct on software fixture. Cross-split vector counts missing (F-06); official overlap unknown; no respondent ID exists to prove repeated individuals. |
| 9. Serialization/loading | Clean joblib round trip passed. Registry mismatch safeguards are incomplete (F-01). README:99 correctly restricts pickle loading to trusted artifacts; no security incident identified. |
| 10. Feature ordering | A5 checks CSV columns against A4 manifest at A:187. SHAP/LIME wrappers reconstruct columns; original pipelines are serialized. Registry/manifest contract is not fully checked (F-01). |
| 11. Randomness/reproducibility | Split, search, estimators, bootstrap and LIME are seeded. A3 f_classif is deterministic. SHAP row selection seeded, explainer permutation randomness not explicitly controlled (F-11); immutable run provenance incomplete (F-02). |
| 12. Categorical/ordinal handling | No resampling interpolation is used in training. Numeric ordinal representation is a modeling assumption, not demonstrated leakage. Fractional-code acceptance is a validation defect (F-05). |
| 13. Missing-value assumptions | Loader rejects missing values before any fitting and directs fold-local imputation if data require it. No hidden global filling found. Official missingness unavailable. |
| 14. Final model use on held-out data | A4 fits train only, locks model/k/threshold from validation, then scores test. A5 fits no model and performs descriptive calibration/analysis. Test arrays are allocated earlier, but selection does not read their scores/labels. |

The code-level conclusion is **no demonstrated leakage in the normal A4 selection/fitting path**, with an independently confirmed A5 artifact validation weakness. The historical/full research pipeline cannot be certified without its actual data and artifacts.

AP/ROC-AUC measure score ordering across thresholds. Changing the locked classification rule can change confusion counts, precision, recall, specificity, F1 and accuracy, while leaving AP/ROC-AUC unchanged. The saved A1/A2 construction respects this distinction.

Bootstrap intervals describe row-resampling uncertainty conditional on the already-fitted models. They do not capture retraining, alternative splits/seeds, hyperparameter/feature selection, survey-design uncertainty or population/clinical generalization. The current core wording correctly limits that claim.

## 7. Missing Deliverables

Before A4/A5 can be considered complete:

1. Accessible official assignment and approved proposal, with a traceability review against their actual requirements.
2. The official binary CSV downloaded from its source into local ignored storage, with verified data dictionary, actual 253,680-row/21-feature schema, source SHA, permitted value domains, global and per-split class counts, and duplicate/contradictory-vector diagnostics.
3. Resolution of the three P1 findings and an agreed, snapshotted experiment specification including actual family-specific tuning budgets, ranking method, numerical feature representation and tie rules.
4. An approved official A4 experiment producing real audit, split, CV, selection, metric, prediction and model artifacts. Existing QA outputs cannot satisfy this item.
5. Real A5 plots/statistics/error/subgroup/calibration artifacts bound to that frozen A4 run, with bootstrap exclusions and analysis environment documented.
6. A curated ablation table/discussion. YAML's named `ablation.csv` is currently absent. Some artifact names differ between YAML and README/code (`versions.txt` versus `versions.json`, `pr_curve.png` versus `precision_recall_test.png`, and `bootstrap_ap_delta.csv` versus `paired_bootstrap_ap.csv`); establish an explicit mapping instead of treating equivalent formats as separate scientific failures.
7. Completed A4/A5 Results/Discussion with all placeholders resolved from official evidence, plus runtime/feature-reduction tradeoffs and the planned baseline uncertainty comparisons or documented omissions.
8. A truthful completed AI usage log and run README/manifest linkage. The existing `AI_Usage_Log_template.md` is not a filled log; instructor-level log details still require the missing assignment.
9. If optional XAI is claimed: installed/tested SHAP and LIME, final-pipeline explanations, reference/case mappings and reproducibility metadata. If these extensions are deferred, document that status. Stacking/Optuna likewise remain optional and unimplemented unless the team explicitly retains their original allocation.
10. A reviewable PR containing scoped code/documents and curated legitimate summaries, while keeping local data, environments, row-level dumps, pickles and bulky intermediates out of routine commits.

Absence of these research deliverables is consistent with the current README's unfinished status. The audit did not attempt to fill them with synthetic results.

## 8. Prioritized Fix Plan

| Gate | Required actions | Evidence expected |
|---|---|---|
| Before committing / GitHub push as review-ready implementation | Fix F-01 artifact consistency; fix F-02 configuration/provenance; resolve F-03 LR budget and manifest counts. Correct F-04 paths and F-08 tie documentation so the published workflow can be followed. | Targeted guard tests reject the demonstrated inconsistent artifacts; clean synthetic round trip still succeeds; immutable configuration and actual budget recorded; documented component paths exist. |
| Before opening the PR | Address F-05/F-09 output correctness, reconcile methodology text (F-07), document missing authoritative sources and research results, and clarify F-10/F-12 QA/statistical limitations. Attach this audit and scope optional features honestly. | Passing relevant software checks and a coherent README/report; no synthetic result presented as official; carefully scoped staged diff. |
| Before the official experiment | Inspect the missing sources and genuine CSV; establish all domain checks/distribution/duplicate diagnostics (F-05/F-06); freeze search/ranking/tie/threshold/analysis policy and package environment; decide on baseline uncertainty comparisons and optional XAI. | Approved effective protocol snapshot and authenticated local source identity, complete provenance, sufficient runtime budget, explicit authorization for the expensive run. |
| Before declaring A4/A5 complete | Perform the approved official run and locked A5 analysis, verify artifact coherence, produce ablation/results/discussion/AI log, and validate optional XAI only if claimed. | Actual official outputs traceable to data/code/configuration; limitations and negative results reported; teammate review of evidence. |

F-11 is optional-extension work and does not require adding SHAP/LIME or Stacking/Optuna to the core scope. F-12 can be handled by documenting the supported QA path if an end-to-end quick path is not required. Do not silently switch the feature-ranking method, primary split, search backend or statistical comparison to resolve documentation mismatches.

Keep `/data/*.csv`, venvs, `.env` files, caches and uncurated `/results/*` ignored within the component. The current `.gitignore` correctly protects these component paths; no root `.gitignore` exists. Curated real report assets should follow the team's documented publishing/storage choice, with a scoped review rather than forcing large intermediate files into Git.

No fixes, staging or pushes were performed as part of this audit.

## 9. Final Decision

**BLOCKED** for presenting this branch as PR-ready A4/A5 implementation.

Passing software checks support the basic scaffold, but they do not resolve the three confirmed high-priority issues. A5 can analyze inconsistent artifacts; the required experiment snapshot is missing; and the actual LR tuning budget contradicts its recorded/default per-family budget. Those issues need concrete fixes or an explicitly reviewed research limitation before this readiness decision can change.

Official results and completion remain separately NOT VERIFIED because the genuine CSV, assignment, proposal and research run artifacts are absent. The audit did not require or execute unauthorized full training. SHAP, LIME, Optuna and Stacking are not certified as implemented runtime features.

The only repository addition made by this task is `research/lead_modeling/a4_a5/docs/CODEX_AUDIT_REPORT.md`; the 29 pre-existing project files retained their recorded hashes.

Audited core/protocol fingerprints:

| File | SHA-256 |
|---|---|
| `P/config/A2_protocol_v2.yaml` | `5328711de3652ca2c4da4a66b2429dbebd10941ed47765ba36e9de2ac4889bf1` |
| `P/src/diabetes_research/core.py` | `d727cea6b825c280adce0ca25c908820bca181760f0718942d01bbb5f66480c3` |
| `P/src/diabetes_research/modeling.py` | `d8c552bd30dfcf0f0f119461356fcb5d9ca0b5df22c62055611fca7da01b2df1` |
| `P/src/diabetes_research/analysis.py` | `13b04f8f8d1d1aaf1ea04768c9772ecf3bf988af227fc0fbc596e7b4376f3e9a` |

