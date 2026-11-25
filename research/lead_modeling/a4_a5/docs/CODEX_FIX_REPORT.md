# A4/A5 Audit Fix and Verification Report

Date: 18 November 2026 (Asia/Ho_Chi_Minh). Scope: `research/lead_modeling/a4_a5/` only.

## 1. Executive summary

**Final decision: READY FOR PR WITH DOCUMENTED LIMITATIONS. Research completion: NOT COMPLETE.**

All three P1 findings are resolved in the current implementation and regression-tested. Eight P2 findings are resolved; F-11's evidence reporting and optional explanation provenance are addressed, with SHAP/LIME execution explicitly **NOT VERIFIED**. Thus 3/3 P1 are resolved, 9/9 P2 are addressed, and 8/9 P2 have completed core validation. The final suite has **70 passing tests: eight existing and 62 new test cases, zero failures**. Core compilation, imports/dependencies, notebook structure/path resolution, CLI help/error behavior, clean synthetic full-row and quick A4→A5 integration, and artifact fault injection all pass.

At task start, the working tree already contained changes newer than the audit: manifest v2, `provenance.py`, `artifact_validation.py`, ten LR C values, domain/diagnostic/subgroup/bootstrap/quick fixes and methodology notes. These updates were inspected and preserved; the audit's original fingerprints are not the session's starting fingerprints. This task supplied the missing regression suite, strengthened fixed-parameter/CV/protocol consistency, corrected both notebooks and completed documentation. Before/after below refers to the audited behavior versus the verified final working tree, not a claim that every fix was first written during this session.

No official training, dependency installation, staging, commit, push, merge or rebase was performed. No credentials were read/modified. Synthetic CSVs, pickles, plots and analysis outputs are under `/private/tmp/diabetes_codex_fix_5f0p88t8/`, not project results. Files outside the component and the available protocol are unchanged.

## 2. Finding disposition

| Finding | Audit before | Verified after | Status |
|---|---|---|---|
| F-01 (P1) | A5 accepted modified FINAL scores, threshold mismatch and overlapping partitions | Critical hashes plus independent split, registry, feature, selection, threshold, probability and metric checks; score recomputation before statistics | RESOLVED |
| F-02 (P1) | No complete immutable protocol/runtime/source/artifact provenance; no A5 environment | Verbatim protocol snapshot; requested/effective parameters, complete learner settings, seeds, split/CV/search effort, source/environment identity, execution metadata and linked artifact hashes | RESOLVED |
| F-03 (P1) | Four LR configurations despite requested ten; manifest overstated effort | Ten meaningful C values, unweighted baseline; requested/effective/actual counts, CV/refit counts and finite-space exhaustion | RESOLVED |
| F-04 (P2) | Repository-root examples and notebooks resolved nonexistent component paths | README/Git guide use nested component; notebooks resolve from repository/component/notebooks directories | RESOLVED |
| F-05 (P2) | Negative BMI and fractional ordinal/day codes accepted | Positive BMI without clinical cutoffs; integral codes/day counts within their documented domains | RESOLVED |
| F-06 (P2) | No per-split class/vector-overlap diagnostics | Per-split rows, positives, negatives, prevalence; exact shared-vector and affected-row counts | RESOLVED |
| F-07 (P2) | Documents implied different ranking algorithms/weighted CV/fold stability | Current-code methodology explicitly states ANOVA train-fit selection and validation choice; historical PDFs/Word preserved and reconciled in notes | RESOLVED for implementation/documentation; missing proposal remains unverified |
| F-08 (P2) | Tie comment promised B3 while code favored B1 | Preserve existing exact-AP candidate-order priority and record/document it | RESOLVED |
| F-09 (P2) | Withheld groups disappeared; empty CSV had no schema | All observed groups retain counts, frozen threshold, status and reason; even empty output has headers | RESOLVED |
| F-10 (P2) | Attempt/exclusion/seed/interval metadata incomplete | Explicit attempts, valid/excluded counts, no-replacement policy, seed, confidence, interval method and paired comparison | RESOLVED |
| F-11 (P2) | Optional explainer reproducibility and current runtime evidence incomplete | Seed/settings, model/run/source-row mappings and attribution/local-weight artifacts; historical claims labeled unverified; dependency errors tested | ADDRESSED WITH LIMITATION: real SHAP/LIME runtime NOT VERIFIED |
| F-12 (P2) | Quick A4 could not feed A5 | Exact subset-to-source mapping persisted and verified; quick A4→A5 succeeds only with explicit QA flag | RESOLVED |

## 3. Files changed during this session

Modified:

- `src/diabetes_research/modeling.py`: full fitted pipeline/learner configuration and fixed XGBoost settings in runtime metadata.
- `src/diabetes_research/provenance.py`: fixed scientific protocol drift and effective-methodology validation, including valid CV fold counts.
- `src/diabetes_research/artifact_validation.py`: verify distinct in-space CV trials, fold/summary AP, winner parameters and fitted runtime/registry parameters.
- Both notebooks: resolve the nested component from all three supported working directories.
- `README.md`: notebook paths, full runtime metadata and remaining analysis limits.
- `docs/METHODOLOGY_NOTES.md`: actual execution scope, protocol/CV consistency and outstanding baseline intervals.

Added:

- `tests/test_audit_regressions.py`: 62 collected regression cases.
- `docs/PR_DESCRIPTION.md`: local review description; no remote PR was created.
- `docs/CODEX_FIX_REPORT.md`: this report.

Existing core/analysis fixes and historical audit/QA/PDF/Word documents were inspected, retained and validated. The available YAML and all requirements files were preserved. A1–A3 and unrelated teammates' files were not edited.

## 4. Significant implementation details

### Artifact integrity and frozen selection

A5 requires compatible manifest v2/source/library provenance and checks critical artifact bytes before `joblib.load`. Model pickles must still come from trusted creators: checksums in an unsigned manifest detect corruption/mixing, not authenticate a creator.

All three partitions must contain valid unique in-bounds integer positions, be pairwise disjoint, exhaust the modeling dataset, match the recorded sizes and reproduce seed-42 stratification. Quick maps must match the recorded seeded subset. Prediction row indices are original CSV positions, not respondent IDs; labels must match those rows.

Registry/manifest/selection model IDs, final ID, thresholds, input feature order, fitted classes/names, selected names/order/count, pipeline/learner definitions, weighting, hyperparameters and selection metadata are reconciled. Selection remains based on saved validation evidence. Saved test scores are compared with freshly computed serialized-model probabilities using `atol=1e-8`, `rtol=1e-7`. A5 analyzes exact recomputed scores, so tolerated CSV noise cannot alter score ties/AP. Test metrics are independently recomputed with the locked thresholds. No model is refitted and no threshold/configuration is selected from test labels.

Twenty-five fault tests deliberately refresh artifact hashes after mutation, exercising semantic checks beyond byte checks. The separate corrupt-pickle test proves byte validation occurs before loading. Invalid artifacts fail before an A5 output directory/statistics are created.

### Reproducible configuration and honest search effort

A4 saves the available YAML verbatim, records CLI requests and effective settings/overrides separately, and hashes actual source files including untracked modules/scripts plus protocol/dependencies. Git HEAD is supplementary. Runtime metadata includes full finite search spaces, actual fitted pipeline parameters, seeds, split rounding/index space, CV/scoring/search settings, best parameters, requested/effective/actual configurations, CV/refit fits and durations. Manifests include dataset/artifact/source hashes, unique run IDs and UTC start/end/duration. Source/data mutation during execution prevents completion metadata.

A5 validates compatible Python major/minor and exact model-library versions/source hashes, records its own environment/configuration/source/artifact identity, and links its outputs to the A4 run and manifest hash. Legacy artifacts without integrity metadata must be rerun. Restoring compatible source/environment is required for existing v2 artifacts; broad requirements constraints are not a tested version matrix or lockfile.

The protocol remains hardcoded where previously hardcoded. Fixed scientific YAML choices are checked against implemented behavior so editing the plan cannot silently change execution. Existing CV/budget CLI overrides remain disclosed. CV traces must show distinct valid configurations, valid fold AP, consistent mean/std and the recorded winner. Fitted searched parameters and complete learner settings must agree with runtime/registry metadata.

LR varies only inverse L2 regularization strength: `C=[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30]`, with `class_weight=None` and existing `liblinear`/scaled baseline. Normal spaces are B1=10, B2=72, B3=288. Requests of 3/10/15 were actually exercised for LR: 3/10/10 distinct configurations, respectively. A request exceeding a finite space is capped honestly; no repeated configurations pad effort. Default synthetic A4 exercised ten configurations and three folds for all three families.

### Dataset, descriptive analysis and optional explanations

Binary features remain 0/1; ordinal ranges are GenHlth 1–5, Age 1–13, Education 1–6, Income 1–8; MentHlth/PhysHlth are integer days 0–30. BMI must be positive, with no invented clinical upper/lower exclusion cutoff; fractional positive BMI is accepted. Official data dictionary/actual domains remain to be authenticated with the missing genuine CSV. Boundary regressions check accepted endpoints and reject fractional encoded/count fields.

Dataset diagnostics retain original row splitting and report exact equal-X vectors, including contradictory labels globally and pairwise shared vectors/affected rows across splits. Identical vectors are not evidence of identical people. No automatic deduplication or group-split sensitivity experiment was introduced.

Subgroups retain every observed Sex/Age group and a fixed CSV schema. Metrics are withheld for n<50 or single-class groups; counts/status/reasons remain. n≥50 is an implementation reporting rule, not proof of power or fairness.

Bootstrap uses 1,000 attempts, seed 42, paired row positions, FINAL-minus-B3 AP and 95% percentile intervals. Single-class attempts are excluded without replacement. Summary records all required metadata; official-mode A5 rejects alternative attempt counts. QA overrides are explicit. An independent two-row stress check verifies retained replicate IDs/deltas and quantiles; this is software evidence, not a BRFSS result.

Optional SHAP/LIME remain optional and were not installed or executed. Current code records explainer seeds, settings, source-row/model/run identities and actual explanation values/weights when run. The SHAP summary is a sampled 15-row summary, not population importance. Tests exercise missing-dependency errors only and do not count as runtime explanation verification.

## 5. Regression tests

`tests/test_audit_regressions.py` adds 62 collected cases:

- Clean full-row synthetic A4→A5, ten configurations/three folds, 1,000 bootstrap attempts; prohibit A5 fit/threshold selection and verify locked artifacts unchanged.
- 25 semantic fault cases: final/baseline/NaN probabilities, threshold mismatches, overlap/incomplete splits, input/fitted/selected feature ordering, model IDs, labels/order/source mapping, seeds/counts/source/environment/method/learner configuration, duplicate CV trials/wrong winner and invalid test metrics.
- Checksum-before-unpickle and valid floating-point tolerance with exact recomputed scores.
- Quick 400→260 mapping/integration and default synthetic/quick refusal (the separate CLI harness also checks 200→198 mapping).
- LR cardinality/effort/exhaustion for requests of 3/10/15.
- Eleven invalid-domain cases, legitimate endpoints/positive fractional BMI, split class/vector diagnostics and exact/near AP tie cases.
- Four subgroup cases including empty/single-class output, independently paired bootstrap metadata/resampling, source-file identity, four fixed-protocol drift cases and official bootstrap-policy enforcement.
- Two optional-dependency error cases and both notebooks' path resolution from repository/component/notebooks directories.

The eight original tests are unchanged. No official performance assertion, fabricated research result or synthetic figure was inserted into research templates.

## 6. Executed verification commands and actual outputs

Working directory unless specified: `/Users/humanormal/Projects/UET.CS2046_7_Nhom5/research/lead_modeling/a4_a5`.

Common QA environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/private/tmp/diabetes_codex_fix_5f0p88t8/mpl
export XDG_CACHE_HOME=/private/tmp/diabetes_codex_fix_5f0p88t8/cache
export PYTHONPATH=src
```

### Dependency consistency and imports

```bash
PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 .venv-diabetes311/bin/python -m pip check
```

```text
No broken requirements found.
```

Inline `.venv-diabetes311/bin/python -` imported `numpy,pandas,sklearn,scipy,xgboost,matplotlib,joblib,yaml,nbformat,pytest` through `importlib.import_module` and printed `versions()`:

```text
CORE_IMPORTS PASS (10 modules)
python 3.11.15; macOS-15.1.1-x86_64-i386-64bit; x86_64
numpy 2.4.6; pandas 2.3.3; scikit-learn 1.9.1; scipy 1.17.1
xgboost 3.2.0; matplotlib 3.11.2; joblib 1.6.0; PyYAML 6.0.3
nbformat 5.11.1; pytest 9.2.0
shap / lime / optuna: not installed
```

Versions above are the actual printed JSON fields, rendered compactly. No installation was performed.

### Compile and notebook checks

Inline `.venv-diabetes311/bin/python -` executed:

```python
for folder in ('src', 'scripts', 'tests'):
    for p in Path(folder).rglob('*.py'):
        compile(p.read_text(), str(p), 'exec')
        print('COMPILE PASS', p)
for p in Path('notebooks').glob('*.ipynb'):
    nb = nbformat.read(p, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == 'code']
    for i, c in enumerate(code):
        compile(c.source, f'{p}:cell_{i}', 'exec')
```

```text
COMPILE PASS src/diabetes_research/analysis.py
COMPILE PASS src/diabetes_research/__init__.py
COMPILE PASS src/diabetes_research/core.py
COMPILE PASS src/diabetes_research/provenance.py
COMPILE PASS src/diabetes_research/modeling.py
COMPILE PASS src/diabetes_research/artifact_validation.py
COMPILE PASS scripts/run_a5.py
COMPILE PASS scripts/run_a4.py
COMPILE PASS tests/test_protocol.py
COMPILE PASS tests/test_audit_regressions.py
NOTEBOOK PASS notebooks/A5_Evaluation_XAI_v2.ipynb code_cells 4 executed 0 outputs 0
NOTEBOOK PASS notebooks/A4_Training_Protocol_v2.ipynb code_cells 5 executed 0 outputs 0
```

Notebook model/data cells were not executed. Directory-resolution code alone was executed by two regressions from all three supported directories. The two optional-dependency tests added after this compile check also passed in the final suite.

### Pytest: all three runs

Common QA environment applied to each command:

```bash
.venv-diabetes311/bin/python -m pytest tests/test_protocol.py -q -p no:cacheprovider --basetemp=/private/tmp/diabetes_codex_fix_5f0p88t8/pytest-baseline
.venv-diabetes311/bin/python -m pytest tests -q -p no:cacheprovider --basetemp=/private/tmp/diabetes_codex_fix_5f0p88t8/pytest-final
.venv-diabetes311/bin/python -m pytest tests -q -p no:cacheprovider --basetemp=/private/tmp/diabetes_codex_fix_5f0p88t8/pytest-complete
```

Actual summary outputs, respectively:

```text
8 passed in 35.22s
68 passed in 51.92s
70 passed in 52.55s
```

The last run followed addition of two explicit optional-dependency error tests. No test failure occurred and no assertions were weakened.

### CLI checks and synthetic CLI integration

Executed temporary harness:

```bash
PYTHONPATH=src:tests .venv-diabetes311/bin/python -u /private/tmp/diabetes_codex_fix_5f0p88t8/verify_cli.py
```

It ran these actual subprocess commands using the same `.venv-diabetes311/bin/python` interpreter:

```bash
python scripts/run_a4.py --help
python scripts/run_a5.py --help
python scripts/run_a4.py
python scripts/run_a4.py --data /private/tmp/diabetes_codex_fix_5f0p88t8/missing.csv --trials 0
python scripts/run_a4.py --data /private/tmp/diabetes_codex_fix_5f0p88t8/missing.csv --cv 1
python scripts/run_a5.py
python scripts/run_a5.py --data /private/tmp/diabetes_codex_fix_5f0p88t8/missing.csv --a4-dir /private/tmp/diabetes_codex_fix_5f0p88t8/missing_a4 --bootstraps 0
python scripts/run_a4.py --data /private/tmp/diabetes_codex_fix_5f0p88t8/cli_synthetic.csv --output-dir /private/tmp/diabetes_codex_fix_5f0p88t8/cli_a4 --quick --demo
python scripts/run_a5.py --data /private/tmp/diabetes_codex_fix_5f0p88t8/cli_synthetic.csv --a4-dir /private/tmp/diabetes_codex_fix_5f0p88t8/cli_a4 --output-dir /private/tmp/diabetes_codex_fix_5f0p88t8/cli_a5 --allow-smoke --bootstraps 1000
```

Actual harness output:

```text
A4_HELP PASS exit=0
A5_HELP PASS exit=0
A4_MISSING_REQUIRED PASS exit=2
A4_ZERO_TRIALS PASS exit=2
A4_INVALID_CV PASS exit=2
A5_MISSING_REQUIRED PASS exit=2
A5_ZERO_BOOTSTRAPS PASS exit=2
A4_SYNTHETIC_QUICK PASS exit=0
A5_SYNTHETIC_QUICK PASS exit=0
SYNTHETIC_CLI_CHAIN PASS; original model bytes preserved; QA only
```

Actual parser-error final lines:

```text
run_a4.py: error: the following arguments are required: --data
run_a4.py: error: --trials must be 1..100
run_a4.py: error: --cv must be >=2
run_a5.py: error: the following arguments are required: --data, --a4-dir
run_a5.py: error: --bootstraps must be positive
```

Help output began with the documented `--data`/`--a4-dir` usage. Full stdout/stderr/command arrays for all nine subprocesses are retained in temporary `cli_checks.json`. Synthetic CLI A4 retains 198 of 200 source rows; A5 links to the matching A4 run and preserves model bytes. Synthetic AP/F1/CI printouts are deliberately not used as research findings.

## 7. Protocol/documentation consistency and limitations

Read the complete audit, available protocol, A1/A2/A3 PDF text and relevant A4/A5 PDF/Word and Markdown documentation. PDF inspection was text extraction; no visual rendering or external citation authentication was claimed. The official instructor assignment, approved group proposal and genuine CSV are absent from this workspace. Their requirements, data dictionary/source identity and instructor-level compliance are not certified.

The current implementation/documentation agree on BRFSS 2015 binary survey classification; stratified 70/15/15 and seed 42; Dummy/LR/RF/XGBoost; train-only weighting; reference threshold 0.5 and validation-max-F1 threshold; 21/15/10/5 ANOVA feature sets; AP primary; train-only stratified 3-fold search CV; locked test; paired 95% percentile bootstrap with 1,000 attempts. B1–B3 undergo search CV; A1/A3 fit full train using B3's parameters and select threshold/k/model on validation. No weighted/feature CV or fold-ranking stability experiment is claimed. Ordinal codes remain numeric; LR scales all inputs. Final exact AP ties preserve existing B1/B2/B3/A2/selected-A3 priority.

The original YAML remains byte-identical and the original A1–A5 PDFs/Word are preserved. Current method notes explicitly reconcile earlier A2 permutation/SelectFromModel suggestions and A3/A4 fold-ranking wording without silently changing the implementation. Artifact-name correspondence and the fact that only selected A3 k is scored on test are documented; `ablation.csv` contains actual validation/test rows, not invented test evaluations of unselected k.

Remaining limits:

1. No official experiment/results or research-completion claim. Actual dataset/domain/provenance and authoritative assignment/proposal review are outstanding.
2. SHAP/LIME runtime/stability remain unverified. Their dependencies are optional and were not installed. Historical QA claims are historical and unverified for current code.
3. Baseline AP intervals requested in A1/A2 (B2/B3 minus B1) remain unimplemented; current analysis preserves FINAL minus B3. A final=B3 self-comparison is correct but uninformative. Declare any added analysis plan before the official run.
4. Available YAML/notes do not prove approval of a ranking algorithm by an unseen proposal. Confirm any higher-authority conflict before official experiments.
5. Exact environment/source compatibility was tested only on the recorded macOS Intel/Python 3.11 environment. Cross-version/architecture reproduction, locked dependency resolution and legacy artifacts were not certified.
6. Stacking/Optuna/group-vector sensitivity remain optional and unimplemented. Completed report/AI log require real experiment/interaction records; no histories, citations or results were fabricated.

## 8. Git/scope verification

From repository root, `git diff --stat`, `git diff --check`, `git diff --cached --stat` and `git ls-files` all returned empty output and exit 0. `git status --short` returned:

```text
?? docs/
?? research/
```

This checkout has no tracked project files: ordinary Git diff cannot show untracked component changes. No staging was done to manufacture a diff. A task-start snapshot under `/private/tmp/diabetes_codex_fix_5f0p88t8/before/` provides a separate review comparison of actual session edits.

SHA256 comparison against task-start inventory confirmed every modified existing file is inside the component; the available protocol and outside-component files are unchanged. No generated synthetic output was added under project data/results.

Snapshot diff results are recorded below after final report creation. The no-index diff includes only this session's modifications/additions; it does not erase or attribute pre-existing post-audit work.

## 9. Final readiness recommendation

**READY FOR PR WITH DOCUMENTED LIMITATIONS** for review of the Lead Modeling software. The local [PR description](PR_DESCRIPTION.md) and passing regression evidence make the scoped changes concrete and reviewable. No remote publication was performed.

**NOT RESEARCH-COMPLETE.** Official authenticated BRFSS experiments, authoritative-document reconciliation and reviewed results/discussion/actual AI records are still required. Optional XAI can only be marked verified after executing it.

### Actual task-start snapshot diff

Commands (snapshot paths outside repository):

```bash
git diff --no-index --stat /private/tmp/diabetes_codex_fix_5f0p88t8/before/research/lead_modeling/a4_a5 /private/tmp/diabetes_codex_fix_5f0p88t8/after/research/lead_modeling/a4_a5
git diff --no-index --check /private/tmp/diabetes_codex_fix_5f0p88t8/before/research/lead_modeling/a4_a5 /private/tmp/diabetes_codex_fix_5f0p88t8/after/research/lead_modeling/a4_a5
```

Actual `--stat` output:

```text
 .../research/lead_modeling/a4_a5/README.md         |   6 +
 .../lead_modeling/a4_a5/docs/CODEX_FIX_REPORT.md}  | 302 ++++++++++++++
 .../lead_modeling/a4_a5/docs/METHODOLOGY_NOTES.md  |   4 +
 .../lead_modeling/a4_a5/docs/PR_DESCRIPTION.md}    |   9 +
 .../a4_a5/notebooks/A4_Training_Protocol_v2.ipynb  |  15 +-
 .../a4_a5/notebooks/A5_Evaluation_XAI_v2.ipynb     |  15 +-
 .../src/diabetes_research/artifact_validation.py   |  50 ++-
 .../a4_a5/src/diabetes_research/modeling.py        |  20 +
 .../a4_a5/src/diabetes_research/provenance.py      |  39 +-
 .../a4_a5/tests/test_audit_regressions.py}         | 434 +++++++++++++++++++++
 10 files changed, 887 insertions(+), 7 deletions(-)
```

Both no-index commands exit 1 because no-index implies a differences exit code. Snapshot `--check` produced no whitespace diagnostics (empty stdout/stderr). Ordinary repository `git diff --check`: empty output, exit 0.
