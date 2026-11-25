"""Independent drift probes; no original-data fitting or historical output writes."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import clone
from xgboost import XGBClassifier

from diabetes_research import group_sensitivity, modeling, provenance
from diabetes_research.paths import AMENDMENTS_PATH, PROTOCOL_PATH
from diabetes_research.sensitivity_contract import validate_sensitivity_native


APPROVED = {
    'n_estimators': 160, 'max_depth': 4, 'learning_rate': 0.08,
    'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0,
    'reg_lambda': 1.0, 'weight_mode': 'none', 'n_jobs': 2, 'random_state': 42,
}
DECLARED_DRIFT = [
    ('weight_mode', 'balanced'), ('n_estimators', 999), ('max_depth', 30),
    ('learning_rate', 0.5), ('min_child_weight', 8), ('subsample', 0.8),
    ('colsample_bytree', 0.8), ('reg_lambda', 9.0), ('n_jobs', 1), ('random_state', 7),
]
NATIVE_DRIFT = [
    *[(key, value) for key, value in DECLARED_DRIFT if key != 'weight_mode'],
    ('objective', 'binary:logitraw'), ('eval_metric', 'aucpr'),
    ('tree_method', 'approx'), ('importance_type', 'weight'), ('scale_pos_weight', 2.0),
    ('enable_categorical', True), ('missing', 0.0),
    ('base_score', 0.1), ('booster', 'dart'), ('callbacks', []),
    ('colsample_bylevel', 0.8), ('colsample_bynode', 0.8), ('device', 'cuda'),
    ('early_stopping_rounds', 3), ('feature_types', ['q']), ('feature_weights', [0.5]),
    ('gamma', 3.0), ('grow_policy', 'lossguide'), ('interaction_constraints', '[[0,1]]'),
    ('max_bin', 128), ('max_cat_threshold', 8), ('max_cat_to_onehot', 8),
    ('max_delta_step', 1), ('max_leaves', 9), ('monotone_constraints', '(1)'),
    ('multi_strategy', 'multi_output_tree'), ('num_parallel_tree', 3), ('reg_alpha', 2.0),
    ('sampling_method', 'gradient_based'), ('validate_parameters', True), ('verbosity', 3),
]


@pytest.fixture
def contract_inputs(tmp_path, monkeypatch):
    protocol = yaml.safe_load(PROTOCOL_PATH.read_bytes())
    amendments = json.loads(AMENDMENTS_PATH.read_bytes())
    amendment_path = tmp_path / 'amendments.json'
    amendment_path.write_text(json.dumps(amendments))
    monkeypatch.setattr(provenance, 'AMENDMENTS_PATH', amendment_path)
    return protocol, amendments, amendment_path


def test_authorized_yaml_amendment_wrapper_native_and_clone_pass(contract_inputs):
    protocol, amendments, _ = contract_inputs
    assert protocol['sensitivity']['frozen_estimator']['parameters'] == APPROVED
    assert amendments['decisions']['group_sensitivity']['frozen_estimator']['parameters'] == APPROVED
    provenance.validate_protocol(protocol)
    pipeline = group_sensitivity.frozen_sensitivity_pipeline(protocol)
    for estimator in (pipeline[-1], clone(pipeline)[-1]):
        assert estimator.get_params(deep=False) == {**APPROVED, 'importance_type': 'gain'}
        core = estimator._build_core(1.0)
        validate_sensitivity_native(core.get_params(deep=False))
        assert len(core.get_params()) == 40
        assert core.objective == 'binary:logistic'
        assert core.eval_metric == 'logloss'
        assert not hasattr(core, '_Booster')


@pytest.mark.parametrize('source', ['yaml', 'amendment', 'both'])
@pytest.mark.parametrize('parameter,value', DECLARED_DRIFT)
def test_declared_drift_fails_against_independent_contract(contract_inputs, source, parameter, value):
    protocol, amendments, path = contract_inputs
    if source in ('yaml', 'both'):
        protocol['sensitivity']['frozen_estimator']['parameters'][parameter] = value
    if source in ('amendment', 'both'):
        amendments['decisions']['group_sensitivity']['frozen_estimator']['parameters'][parameter] = value
        path.write_text(json.dumps(amendments))
    before = path.read_bytes()
    with pytest.raises(ValueError, match=parameter):
        group_sensitivity.frozen_sensitivity_pipeline(protocol)
    assert path.read_bytes() == before


@pytest.mark.parametrize('source', ['yaml', 'amendment'])
@pytest.mark.parametrize('fault', ['missing', 'extra', 'family', 'missing_estimator'])
def test_declared_missing_extra_or_substitution_fails(contract_inputs, source, fault):
    protocol, amendments, path = contract_inputs
    container = (protocol['sensitivity'] if source == 'yaml'
                 else amendments['decisions']['group_sensitivity'])
    if fault == 'missing':
        del container['frozen_estimator']['parameters']['subsample']
        match = 'subsample'
    elif fault == 'extra':
        container['frozen_estimator']['parameters']['objective'] = 'binary:logitraw'
        match = 'objective'
    elif fault == 'family':
        container['frozen_estimator']['family'] = 'RandomForestClassifier'
        match = 'family'
    else:
        del container['frozen_estimator']
        match = 'frozen_estimator'
    path.write_text(json.dumps(amendments))
    with pytest.raises(ValueError, match=match):
        provenance.validate_protocol(protocol)


def test_sensitivity_approval_cannot_be_missing(contract_inputs):
    protocol, amendments, path = contract_inputs
    amendments['decisions']['group_sensitivity']['status'] = 'PENDING'
    path.write_text(json.dumps(amendments))
    with pytest.raises(ValueError, match='group_sensitivity.status'):
        provenance.validate_protocol(protocol)


@pytest.mark.parametrize('parameter,value', [*DECLARED_DRIFT, ('importance_type', 'weight')])
def test_live_wrapper_drift_rejected_before_native_fit(parameter, value, monkeypatch):
    estimator = group_sensitivity.FrozenSensitivityXGB(**APPROVED)
    estimator.set_params(**{parameter: value})
    monkeypatch.setattr(XGBClassifier, 'fit', lambda *a, **k: pytest.fail('Forbidden native fit'))
    with pytest.raises(ValueError, match=parameter):
        estimator.fit(pd.DataFrame({'x': [0, 1]}), np.array([0, 1]))


@pytest.mark.parametrize('parameter,value', NATIVE_DRIFT)
def test_actual_native_drift_rejected_before_each_fit(parameter, value, monkeypatch):
    estimator = group_sensitivity.FrozenSensitivityXGB(**APPROVED)
    native_builder = modeling.FoldAwareXGB._build_core
    def changed_core(self, ratio):
        core = native_builder(self, ratio)
        core.set_params(**{parameter: value})
        return core
    monkeypatch.setattr(modeling.FoldAwareXGB, '_build_core', changed_core)
    monkeypatch.setattr(XGBClassifier, 'fit', lambda *a, **k: pytest.fail('Forbidden native fit'))
    with pytest.raises(ValueError, match=parameter):
        estimator.fit(pd.DataFrame({'x': [0, 1]}), np.array([0, 1]))


@pytest.mark.parametrize('fault,parameter', [
    ('missing', 'reg_alpha'), ('extra', 'unknown_option'),
    ('bool', 'scale_pos_weight'), ('bool', 'min_child_weight'), ('bool', 'enable_categorical'),
])
def test_native_missing_extra_and_boolean_numeric_drift(fault, parameter):
    parameters = modeling.FoldAwareXGB(**APPROVED)._build_core(1.0).get_params(deep=False)
    if fault == 'missing':
        del parameters[parameter]
    elif fault == 'extra':
        parameters[parameter] = 42
    else:
        parameters[parameter] = 0 if parameter == 'enable_categorical' else True
    with pytest.raises(ValueError, match=parameter):
        validate_sensitivity_native(parameters)


@pytest.mark.parametrize('fault', ['yaml', 'amendment', 'native'])
def test_runner_rejects_drift_before_data_access_and_output_mutation(
        contract_inputs, tmp_path, monkeypatch, fault):
    protocol, amendments, path = contract_inputs
    if fault == 'yaml':
        protocol['sensitivity']['frozen_estimator']['parameters']['weight_mode'] = 'balanced'
        match = 'weight_mode'
    elif fault == 'amendment':
        amendments['decisions']['group_sensitivity']['frozen_estimator']['parameters']['max_depth'] = 30
        path.write_text(json.dumps(amendments))
        match = 'max_depth'
    else:
        native_builder = modeling.FoldAwareXGB._build_core
        def changed_core(self, ratio):
            return native_builder(self, ratio).set_params(eval_metric='aucpr')
        monkeypatch.setattr(modeling.FoldAwareXGB, '_build_core', changed_core)
        match = 'eval_metric'
    config = tmp_path / 'protocol.yaml'
    config.write_text(yaml.safe_dump(protocol))
    monkeypatch.setattr(group_sensitivity, 'PROTOCOL_PATH', config)
    monkeypatch.setattr(group_sensitivity, 'load_dataset', lambda *a: pytest.fail('Forbidden data access'))
    monkeypatch.setattr(XGBClassifier, 'fit', lambda *a, **k: pytest.fail('Forbidden native fit'))
    output = tmp_path / 'new_output'
    with pytest.raises(ValueError, match=match):
        group_sensitivity.run(tmp_path / 'absent.csv', output, train_cv=True)
    assert not output.exists()
    output.mkdir()
    marker = output / 'existing_artifact.txt'
    marker.write_bytes(b'preserve exactly')
    with pytest.raises(ValueError, match=match):
        group_sensitivity.run(tmp_path / 'absent.csv', output, train_cv=True)
    assert list(output.iterdir()) == [marker]
    assert marker.read_bytes() == b'preserve exactly'


@pytest.mark.parametrize('parameter,value', DECLARED_DRIFT)
def test_direct_comparison_cannot_bypass_frozen_wrapper(parameter, value):
    estimator = modeling.FoldAwareXGB(**APPROVED).set_params(**{parameter: value})
    with pytest.raises(ValueError, match=parameter):
        group_sensitivity.compare_train_only_cv_ap(None, None, estimator)


@pytest.mark.parametrize('fault', ['unguarded', 'preprocessing', 'folds', 'seed'])
def test_direct_xgboost_comparison_requires_guarded_prespecified_design(fault):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    model = group_sensitivity.FrozenSensitivityXGB(**APPROVED)
    pipeline = Pipeline([('model', model)])
    kwargs = {}
    if fault == 'unguarded':
        pipeline = Pipeline([('model', modeling.FoldAwareXGB(**APPROVED))])
    elif fault == 'preprocessing':
        pipeline = Pipeline([('scale', StandardScaler()), ('model', model)])
    elif fault == 'folds':
        kwargs['n_folds'] = 2
    else:
        kwargs['seed'] = 7
    with pytest.raises(ValueError, match='sensitivity'):
        group_sensitivity.compare_train_only_cv_ap(None, None, pipeline, **kwargs)
