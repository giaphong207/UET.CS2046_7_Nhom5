"""Audit regression and synthetic integration checks; never official BRFSS findings."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from diabetes_research import analysis, modeling
from diabetes_research.artifact_validation import validate_a4_artifacts
from diabetes_research.core import (
    FEATURE_COUNTS, SEED, TARGET, dataset_audit, load_dataset, select_final_model,
    sha256_file, split_diagnostics, stratified_indices, validate_split_indices, write_json,
)
from diabetes_research.provenance import (
    A4_CRITICAL_ARTIFACTS, COMPONENT_ROOT, artifact_hashes, read_json, source_identity,
    validate_protocol,
)
from test_protocol import toy_data


@pytest.fixture(scope='session')
def clean_a4(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_a4')
    data = root / 'synthetic.csv'
    toy_data().to_csv(data, index=False)
    output = root / 'a4'
    manifest = modeling.run(data, output, trials=10, cv_folds=3, demo=True)
    return data, output, manifest


def refresh_hashes(output):
    """Bypass byte guard deliberately to test independent semantic consistency checks."""
    manifest = read_json(output / 'manifest.json')
    manifest['artifact_hashes'] = artifact_hashes(output, A4_CRITICAL_ARTIFACTS)
    write_json(output / 'manifest.json', manifest)
    return manifest


def test_clean_a4_a5_roundtrip_preserves_locked_model(clean_a4, tmp_path, monkeypatch):
    data, a4, manifest = clean_a4
    before = artifact_hashes(a4)
    selection = read_json(a4 / 'selection.json')

    def forbidden(*args, **kwargs):
        raise AssertionError('A5 must not refit or select a new threshold')

    for cls in (modeling.FoldAwareXGB, modeling.LogisticRegression,
                modeling.RandomForestClassifier, modeling.SelectKBest, modeling.StandardScaler):
        monkeypatch.setattr(cls, 'fit', forbidden)
    monkeypatch.setattr(modeling, 'optimal_f1_threshold', forbidden)
    out = tmp_path / 'a5'
    summary = analysis.run(data, a4, out, allow_smoke=True)
    saved = read_json(out / 'manifest_a5.json')
    assert saved['source_a4_run_id'] == manifest['run_id']
    assert saved['source_manifest_sha256'] == sha256_file(a4 / 'manifest.json')
    assert saved['locked_threshold'] == selection['final_threshold']
    assert saved['status'] == 'SYNTHETIC_QA_ONLY'
    assert saved['artifact_hashes'] == artifact_hashes(out)
    assert summary['n_attempted_bootstrap'] == 1000
    assert summary['n_valid_bootstrap'] + summary['n_excluded_bootstrap'] == 1000
    assert before == artifact_hashes(a4)
    assert list(pd.read_csv(out / 'subgroups_test.csv')) == analysis.SUBGROUP_COLUMNS
    assert read_json(out / 'versions_a5.json')['python']
    verification = read_json(out / 'artifact_verification.json')
    assert verification['registry_metadata_verified'] and verification['test_metrics_verified']
    runtime = read_json(a4 / 'runtime_config.json')
    assert (a4 / 'config.yaml').read_bytes() == (COMPONENT_ROOT / 'config/A2_protocol_v2.yaml').read_bytes()
    assert runtime['effective']['search_spaces'] == modeling.random_spaces()
    assert manifest['actual_trials_per_family'] == {'B1': 10, 'B2': 10, 'B3': 10}
    for stage in (manifest, saved):
        assert stage['duration_seconds'] > 0
        assert stage['completed_at_utc'] >= stage['started_at_utc']
    registry = joblib.load(a4 / 'fitted_models.joblib')
    assert registry['models']['A1'] is registry['models']['A2']
    ablation = pd.read_csv(a4 / 'ablation.csv')
    assert set(ablation.loc[ablation.split == 'validation', 'model']) >= {f'A3_k{k}' for k in FEATURE_COUNTS}
    assert set(ablation.loc[(ablation.split == 'test') & ablation.model.str.startswith('A3'), 'model']) == {
        f"A3_k{selection['selected_feature_k']}"}


FAULTS = [
    ('final_probabilities', 'Saved test probabilities'),
    ('baseline_probabilities', 'Saved test probabilities'),
    ('nan_probabilities', 'finite in'),
    ('registry_threshold', 'selection threshold mismatch'),
    ('overlapping_splits', 'Split indices overlap'),
    ('incomplete_splits', 'not complete'),
    ('wrong_feature_order', 'feature ordering'),
    ('wrong_fitted_feature_order', 'configuration|fitted feature ordering'),
    ('wrong_selected_feature_order', 'Selected feature names/order'),
    ('wrong_registry_model_id', 'final model ID'),
    ('wrong_model_ids', 'Model registry IDs'),
    ('wrong_prediction_model_ids', 'columns/model IDs'),
    ('wrong_selection_threshold', 'locked validation threshold'),
    ('wrong_test_labels', 'ground truth'),
    ('wrong_test_order', 'row indices'),
    ('wrong_source_mapping', 'Source-row mapping'),
    ('wrong_runtime_seed', 'random seeds'),
    ('wrong_actual_count', 'search budget/count'),
    ('wrong_source_identity', 'source-code identity'),
    ('wrong_environment', 'model environment'),
    ('wrong_runtime_method', 'modeling methodology'),
    ('duplicate_cv_trials', 'duplicated or outside'),
    ('wrong_cv_winner', 'best parameters'),
    ('wrong_test_metric', 'test metrics differ'),
    ('wrong_runtime_model_params', 'model configuration'),
]


def inject_fault(output, fault):
    if fault in ('final_probabilities', 'baseline_probabilities', 'nan_probabilities',
                 'wrong_test_labels', 'wrong_test_order', 'wrong_prediction_model_ids'):
        path = output / 'test_predictions.csv'
        pred = pd.read_csv(path)
        if fault == 'final_probabilities': pred['prob_FINAL'] = 0.123456
        elif fault == 'baseline_probabilities': pred['prob_B3'] = 0.987654
        elif fault == 'nan_probabilities': pred.loc[0, 'prob_FINAL'] = np.nan
        elif fault == 'wrong_test_labels': pred.loc[0, 'y_true'] = 1 - pred.loc[0, 'y_true']
        elif fault == 'wrong_test_order': pred = pred.iloc[::-1]
        else: pred = pred.rename(columns={'prob_B1': 'prob_ALIEN'})
        pred.to_csv(path, index=False)
    elif fault in ('overlapping_splits', 'incomplete_splits'):
        path = output / 'split_indices.npz'
        with np.load(path) as archive:
            splits = {key: archive[key] for key in archive.files}
        if fault == 'overlapping_splits': splits['train'][0] = splits['test'][0]
        else: splits['train'] = splits['train'][1:]
        np.savez_compressed(path, **splits)
    elif fault in ('registry_threshold', 'wrong_feature_order', 'wrong_registry_model_id',
                   'wrong_model_ids', 'wrong_fitted_feature_order'):
        path = output / 'fitted_models.joblib'
        registry = joblib.load(path)
        if fault == 'registry_threshold': registry['final_threshold'] = (registry['final_threshold'] + 0.37) % 1
        elif fault == 'wrong_feature_order': registry['feature_names'] = registry['feature_names'][::-1]
        elif fault == 'wrong_registry_model_id': registry['final_model_id'] = 'ALIEN'
        elif fault == 'wrong_model_ids': registry['model_ids'][0] = 'ALIEN'
        else:
            scaler = registry['models']['B1'].named_steps['scaler']
            scaler.feature_names_in_ = scaler.feature_names_in_[::-1]
        joblib.dump(registry, path)
    elif fault == 'wrong_selected_feature_order':
        path = output / 'feature_sets.json'
        sets = read_json(path)
        sets['A3_k10'] = sets['A3_k10'][::-1]
        write_json(path, sets)
    elif fault == 'wrong_selection_threshold':
        path = output / 'selection.json'
        selection = read_json(path)
        selection['final_threshold'] = (selection['final_threshold'] + 0.37) % 1
        write_json(path, selection)
    elif fault == 'wrong_source_mapping':
        path = output / 'source_row_indices.npy'
        np.save(path, np.load(path)[::-1])
    elif fault in ('wrong_runtime_seed', 'wrong_actual_count', 'wrong_runtime_method', 'wrong_runtime_model_params'):
        path = output / 'runtime_config.json'
        runtime = read_json(path)
        if fault == 'wrong_runtime_seed': runtime['seeds']['split'] = 123
        elif fault == 'wrong_actual_count': runtime['search_counts']['B1']['actual_configurations'] += 1
        elif fault == 'wrong_runtime_method': runtime['effective']['feature_selection']['method'] = 'unapproved'
        else: runtime['effective']['fitted_model_configuration']['B1']['model']['parameters']['C'] = 999
        write_json(path, runtime)
    elif fault == 'wrong_source_identity':
        path = output / 'source_identity.json'
        identity = read_json(path)
        identity['file_hashes']['src/diabetes_research/modeling.py'] = '0' * 64
        write_json(path, identity)
    elif fault == 'wrong_environment':
        path = output / 'versions.json'
        environment = read_json(path)
        environment['scikit-learn'] = '0.0'
        write_json(path, environment)
    elif fault in ('duplicate_cv_trials', 'wrong_cv_winner'):
        path = output / 'cv_results.csv'
        trace = pd.read_csv(path)
        if fault == 'duplicate_cv_trials': trace.loc[1, 'params'] = trace.loc[0, 'params']
        else:
            rows = trace.loc[trace.model == 'B1']
            winner = rows.loc[rows.rank_cv_ap == 1].index[0]
            best = json.loads(trace.loc[winner, 'params'])
            wrong = next(json.loads(value) for value in rows.params if json.loads(value) != best)
            runtime_path = output / 'runtime_config.json'
            runtime = read_json(runtime_path)
            runtime['search_counts']['B1']['best_params'] = wrong
            write_json(runtime_path, runtime)
        trace.to_csv(path, index=False)
    elif fault == 'wrong_test_metric':
        path = output / 'test_metrics.csv'
        metrics = pd.read_csv(path)
        metrics.loc[0, 'tp'] += 1
        metrics.to_csv(path, index=False)
    else:
        raise AssertionError(f'Unknown fault {fault}')


@pytest.mark.parametrize('fault,reason', FAULTS, ids=[item[0] for item in FAULTS])
def test_a5_rejects_inconsistent_artifacts_before_statistics(clean_a4, tmp_path, fault, reason):
    data, clean, _ = clean_a4
    broken = tmp_path / 'broken_a4'
    shutil.copytree(clean, broken)
    inject_fault(broken, fault)
    refresh_hashes(broken)
    output = tmp_path / 'must_not_exist'
    with pytest.raises(ValueError, match=reason):
        analysis.run(data, broken, output, bootstraps=10, allow_smoke=True)
    assert not output.exists(), 'Invalid input must fail before any statistics/output'


def test_checksum_guard_precedes_pickle_loading(clean_a4, tmp_path, monkeypatch):
    data, clean, _ = clean_a4
    broken = tmp_path / 'broken'
    shutil.copytree(clean, broken)
    (broken / 'fitted_models.joblib').write_bytes(b'corrupt model')
    monkeypatch.setattr(joblib, 'load', lambda *args: pytest.fail('Checksum must precede unpickle'))
    with pytest.raises(ValueError, match='checksum mismatch: fitted_models'):
        validate_a4_artifacts(data, broken, read_json(broken / 'manifest.json'))


def test_probability_tolerance_uses_exact_model_scores(clean_a4, tmp_path):
    data, clean, _ = clean_a4
    output = tmp_path / 'noisy'
    shutil.copytree(clean, output)
    path = output / 'test_predictions.csv'
    pred = pd.read_csv(path)
    pred['prob_FINAL'] += 1e-10
    pred.to_csv(path, index=False)
    manifest = refresh_hashes(output)
    verified = validate_a4_artifacts(data, output, manifest)
    fresh = verified['final_model'].predict_proba(verified['Xtest'])[:, 1]
    np.testing.assert_array_equal(verified['predictions'].prob_FINAL, fresh)


def test_quick_a4_a5_source_row_mapping(tmp_path):
    data = tmp_path / 'synthetic.csv'
    toy_data(n=400).to_csv(data, index=False)
    a4, a5 = tmp_path / 'quick_a4', tmp_path / 'quick_a5'
    manifest = modeling.run(data, a4, trials=12, cv_folds=3, quick=True, demo=True)
    assert manifest['source_rows'] == 400 and manifest['total_rows'] == 260
    assert manifest['requested_trials_per_family'] == dict.fromkeys(('B1', 'B2', 'B3'), 12)
    assert manifest['actual_trials_per_family'] == dict.fromkeys(('B1', 'B2', 'B3'), 1)
    assert manifest['cv_folds'] == 2
    with pytest.raises(ValueError, match='refuses to treat smoke'):
        analysis.run(data, a4, a5)
    analysis.run(data, a4, a5, allow_smoke=True)
    assert read_json(a5 / 'manifest_a5.json')['status'] == 'SMOKE_NOT_RESEARCH_RESULT'
    with np.load(a4 / 'split_indices.npz') as splits:
        expected = np.load(a4 / 'source_row_indices.npy')[splits['test']]
    np.testing.assert_array_equal(pd.read_csv(a4 / 'test_predictions.csv').row_index, expected)


def test_default_a5_refuses_synthetic(clean_a4, tmp_path):
    data, a4, _ = clean_a4
    with pytest.raises(ValueError, match='refuses to treat smoke'):
        analysis.run(data, a4, tmp_path / 'a5')


@pytest.mark.parametrize('requested,expected,capped', [(3, 3, False), (10, 10, False), (15, 10, True)])
def test_lr_search_cardinality_and_actual_effort(requested, expected, capped):
    space = modeling.random_spaces()['B1']
    assert len(ParameterGrid(space)) == 10
    assert space['model__class_weight'] == [None]
    df = toy_data(n=100)
    model, rows, counts = modeling.search_model_family(
        modeling.initial_models()['B1'], space, df.drop(columns=TARGET), df[TARGET], requested,
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED))
    assert counts['requested_configurations'] == requested
    assert counts['actual_configurations'] == len(rows) == expected
    assert len({row['params'] for row in rows}) == expected
    assert counts['cv_fits'] == expected * 3 and counts['refit_fits'] == 1
    assert counts['budget_capped_to_finite_space'] is capped
    assert counts['space_exhausted'] is (expected == 10)
    assert model.named_steps['model'].class_weight is None


@pytest.mark.parametrize('feature,value', [
    ('BMI', -5), ('BMI', 0), ('Age', 1.5), ('Education', 2.5), ('Income', 3.5),
    ('GenHlth', 2.5), ('MentHlth', 0.5), ('PhysHlth', 10.5),
    ('Age', 14), ('PhysHlth', 31), ('HighBP', 0.5),
])
def test_invalid_dataset_domains(tmp_path, feature, value):
    df = toy_data().astype(float)
    df.loc[0, feature] = value
    path = tmp_path / 'invalid.csv'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=feature):
        load_dataset(path)


def test_legitimate_domain_edges_without_clinical_bmi_cutoffs(tmp_path):
    df = toy_data().astype(float)
    for feature, low, high in [('Age', 1, 13), ('Education', 1, 6), ('Income', 1, 8),
                               ('GenHlth', 1, 5), ('MentHlth', 0, 30), ('PhysHlth', 0, 30)]:
        df.loc[0, feature], df.loc[1, feature] = low, high
    df.loc[0, 'BMI'], df.loc[1, 'BMI'] = 0.1, 100.5
    path = tmp_path / 'valid.csv'
    df.to_csv(path, index=False)
    assert load_dataset(path)[0].loc[1, 'BMI'] == 100.5


def test_split_vector_diagnostics_and_class_counts():
    df = toy_data(n=40)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X.iloc[:] = X.iloc[0].to_numpy()
    splits = stratified_indices(y)
    audit = dataset_audit(X, y)
    diagnostics = split_diagnostics(X, y, splits)
    assert audit['duplicate_feature_vectors'] == 39
    assert audit['vectors_with_disagreeing_labels'] == 1
    for name, indices in splits.items():
        counts = diagnostics['class_counts'][name]
        assert counts['positive_count'] == int(y.iloc[indices].sum())
        assert counts['negative_count'] + counts['positive_count'] == counts['rows']
        assert counts['positive_fraction'] == y.iloc[indices].mean()
    for pair, overlap in diagnostics['identical_vectors_across_splits'].items():
        left, right = pair.split('__')
        assert overlap == dict(shared_unique_vectors=1, left_rows_with_shared_vector=len(splits[left]),
                               right_rows_with_shared_vector=len(splits[right]))


def test_final_model_exact_tie_uses_existing_candidate_order():
    candidates = ['B1', 'B2', 'B3', 'A2', 'A3_k5']
    assert select_final_model(candidates, dict.fromkeys(candidates, 0.3)) == 'B1'
    ap = dict.fromkeys(candidates, 0.3)
    ap['B3'] = 0.3 + 1e-14
    assert select_final_model(candidates, ap) == 'B3'


@pytest.mark.parametrize('case', ['small', 'single_class', 'empty', 'valid'])
def test_subgroups_stable_schema_and_withholding(tmp_path, case):
    n = 60 if case in ('single_class', 'valid') else (0 if case == 'empty' else 10)
    X = pd.DataFrame({'Sex': np.zeros(n), 'Age': np.ones(n)})
    y = np.zeros(n, dtype=int) if case == 'single_class' else np.arange(n) % 2
    frame = analysis.subgroup_metrics(X, y, np.full(n, 0.4), 0.4)
    path = tmp_path / 'subgroups.csv'
    frame.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert list(loaded) == analysis.SUBGROUP_COLUMNS
    assert len(loaded) == (0 if case == 'empty' else 2)
    if n:
        assert (loaded.n == n).all() and (loaded.positive_n == y.sum()).all()
        assert (loaded.threshold == 0.4).all()
        if case == 'valid': assert (loaded.status == 'evaluated').all()
        else:
            assert (loaded.status == 'withheld').all()
            assert loaded.average_precision.isna().all()
            assert loaded.withholding_reason.notna().all()


def test_paired_bootstrap_metadata_and_independent_resampling():
    y, final, baseline = np.array([0, 1]), np.array([0.1, 0.9]), np.array([0.9, 0.1])
    actual, summary = analysis.bootstrap_paired_ap(y, final, baseline)
    rng = np.random.default_rng(SEED)
    expected = []
    for attempt in range(1000):
        indices = rng.integers(0, 2, size=2)
        if len(np.unique(y[indices])) == 2:
            expected.append((attempt, average_precision_score(y[indices], final[indices]) -
                             average_precision_score(y[indices], baseline[indices])))
    np.testing.assert_array_equal(actual.replicate, [row[0] for row in expected])
    np.testing.assert_allclose(actual.delta_ap, [row[1] for row in expected])
    low, high = np.quantile(actual.delta_ap, [0.025, 0.975])
    assert summary['ci95_low'] == low and summary['ci95_high'] == high
    assert summary['n_attempted_bootstrap'] == 1000
    assert summary['n_valid_bootstrap'] == len(expected)
    assert summary['n_excluded_bootstrap'] == 1000 - len(expected) > 0
    assert summary['random_seed'] == 42 and summary['confidence_level'] == 0.95
    assert summary['paired_same_indices'] and 'percentile' in summary['interval_method']
    assert 'do not replace' in summary['exclusion_policy']


def test_source_identity_covers_untracked_code_not_only_git():
    identity = source_identity()
    for name in ('modeling', 'analysis', 'core', 'provenance', 'artifact_validation'):
        key = f'src/diabetes_research/{name}.py'
        assert identity['file_hashes'][key] == sha256_file(COMPONENT_ROOT / key)
    assert len(identity['source_files_sha256']) == 64


@pytest.mark.parametrize('section,key,value', [
    ('split', 'seed', 12), ('threshold', 'reference', 0.3),
    ('uncertainty', 'n_resamples', 200), ('metrics', 'primary', 'accuracy'),
])
def test_protocol_drift_is_not_silently_snapshotted(section, key, value):
    protocol = yaml.safe_load((COMPONENT_ROOT / 'config/A2_protocol_v2.yaml').read_text())
    validate_protocol(protocol)
    protocol[section][key] = value
    with pytest.raises(ValueError, match=section):
        validate_protocol(protocol)


def test_research_bootstrap_override_is_rejected_before_data_access(tmp_path):
    a4 = tmp_path / 'a4'
    a4.mkdir()
    write_json(a4 / 'manifest.json', {'status': 'RESEARCH_TEST_COMPLETE'})
    with pytest.raises(ValueError, match='1000 bootstrap attempts'):
        analysis.run(tmp_path / 'absent.csv', a4, tmp_path / 'a5', bootstraps=20)


@pytest.mark.parametrize('explanation', ['shap', 'lime'])
def test_optional_explanation_missing_dependency_is_explicit(tmp_path, monkeypatch, explanation):
    # Exercise dependency handling only; this is not SHAP/LIME runtime verification.
    import builtins
    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.split('.')[0] == explanation:
            raise ImportError(f'Test-controlled missing {explanation}')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing)
    args = (None, pd.DataFrame(), pd.DataFrame())
    with pytest.raises(RuntimeError, match=f'{explanation.upper()} is optional'):
        if explanation == 'shap': analysis.plot_shap(*args, tmp_path)
        else: analysis.plot_lime(*args, {}, tmp_path)


@pytest.mark.parametrize('name', ['A4_Training_Protocol_v2.ipynb', 'A5_Evaluation_XAI_v2.ipynb'])
def test_notebook_component_resolution_from_supported_working_directories(name, monkeypatch):
    notebook = json.loads((COMPONENT_ROOT / 'notebooks' / name).read_text())
    first_code = next(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    # Execute only directory resolution, never dataset/model cells.
    resolution = first_code.split('DATA')[0]
    for cwd in (COMPONENT_ROOT, COMPONENT_ROOT / 'notebooks', COMPONENT_ROOT.parents[2]):
        monkeypatch.chdir(cwd)
        scope = {}
        exec(compile(resolution, name, 'exec'), scope)
        assert scope['ROOT'] == COMPONENT_ROOT
