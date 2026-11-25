"""Frozen training-only sensitivity preset, independent of editable protocol files.

These settings describe the approved, sealed unweighted run. Native ``None``
values preserve its XGBoost defaults; they do not authorize new explicit values.
An incompatible native parameter schema fails closed pending human review.
"""
from __future__ import annotations

import math
from types import MappingProxyType


DECLARED_PARAMETERS = MappingProxyType({
    'n_estimators': 160, 'max_depth': 4, 'learning_rate': 0.08,
    'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0,
    'reg_lambda': 1.0, 'weight_mode': 'none', 'n_jobs': 2, 'random_state': 42,
})
WRAPPER_PARAMETERS = MappingProxyType({**DECLARED_PARAMETERS, 'importance_type': 'gain'})
NATIVE_PARAMETERS = MappingProxyType({
    **{key: value for key, value in WRAPPER_PARAMETERS.items() if key != 'weight_mode'},
    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
    'scale_pos_weight': 1.0, 'enable_categorical': False, 'missing': math.nan,
    **dict.fromkeys((
        'base_score', 'booster', 'callbacks', 'colsample_bylevel', 'colsample_bynode',
        'device', 'early_stopping_rounds', 'feature_types', 'feature_weights',
        'gamma', 'grow_policy', 'interaction_constraints', 'max_bin',
        'max_cat_threshold', 'max_cat_to_onehot', 'max_delta_step', 'max_leaves',
        'monotone_constraints', 'multi_strategy', 'num_parallel_tree', 'reg_alpha',
        'sampling_method', 'validate_parameters', 'verbosity',
    )),
})


def validate_parameters(actual, expected, location):
    """Check the complete key set and typed values, naming every first conflict."""
    if not isinstance(actual, dict):
        raise ValueError(f'{location}: parameters must be a mapping')
    for key in expected:
        if key not in actual:
            raise ValueError(f'{location}.{key}: missing frozen parameter')
    extra = sorted(set(actual) - set(expected))
    if extra:
        raise ValueError(f'{location}.{extra[0]}: unauthorized extra parameter')
    for key, value in expected.items():
        candidate = actual[key]
        if isinstance(value, float) and math.isnan(value):
            matches = type(candidate) is float and math.isnan(candidate)
        else:
            matches = type(candidate) is type(value) and candidate == value
        if not matches:
            raise ValueError(f'{location}.{key}: expected {value!r}, got {candidate!r}')


def validate_frozen_sensitivity(protocol, amendments):
    """Require YAML and the approved amendment to match an independent contract."""
    decision = amendments.get('decisions', {}).get('group_sensitivity', {})
    if decision.get('status') != 'APPROVED':
        raise ValueError('amendment.group_sensitivity.status: expected APPROVED')
    for location, estimator in (
        ('protocol.sensitivity.frozen_estimator', protocol.get('sensitivity', {}).get('frozen_estimator')),
        ('amendment.group_sensitivity.frozen_estimator', decision.get('frozen_estimator')),
    ):
        if not isinstance(estimator, dict):
            raise ValueError(f'{location}: frozen estimator must be a mapping')
        # Validate nested parameters separately so errors identify the scientific setting.
        validate_parameters({key: value for key, value in estimator.items() if key != 'parameters'},
                            {'family': 'FoldAwareXGB'}, location)
        validate_parameters(estimator.get('parameters'), DECLARED_PARAMETERS, location + '.parameters')


def validate_sensitivity_wrapper(parameters):
    validate_parameters(parameters, WRAPPER_PARAMETERS, 'sensitivity.wrapper')


def validate_sensitivity_native(parameters):
    validate_parameters(parameters, NATIVE_PARAMETERS, 'sensitivity.native')
