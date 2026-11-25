"""Read-only validation of locked A4 artifacts before A5 creates any statistics."""
from __future__ import annotations

from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .core import (
    F2_GRID, FEATURE_COUNTS, FINAL_MODEL_TIE_RULE, SEED, SPLIT_NAMES,
    feature_selection_policy, f2_threshold_curve, load_dataset, select_feature_budget, threshold_policy,
    metrics_at_threshold, sha256_file, stratified_indices,
    validate_split_indices,
)
from .modeling import (ExactTopKSelectFromModel, FoldAwareXGB, XGBClassifier,
                       model_configuration, primary_ranking_record, random_spaces)
from .provenance import MANIFEST_VERSION, read_json, verify_a4_provenance
from .shared_splits import LEGACY_QA_MODE, SHARED_MODE, array_sha256, verify_shared_split

PROBABILITY_ATOL = 1e-8
PROBABILITY_RTOL = 1e-7
CORE_MODEL_IDS = ['B0', 'B1', 'B2', 'B3', 'A1', 'A2', *[f'A3_k{k}' for k in FEATURE_COUNTS]]
A4_STATUSES = {'RESEARCH_TEST_COMPLETE', 'SYNTHETIC_QA_ONLY', 'SMOKE_NOT_RESEARCH_RESULT'}


def _threshold(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{label} must be a numeric probability threshold')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be a numeric probability threshold') from exc
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f'{label} must be finite and in [0, 1]')
    return result


def _source_mapping(a4, manifest, X, y):
    mapping = np.load(a4 / 'source_row_indices.npy', allow_pickle=False)
    if (mapping.ndim != 1 or mapping.dtype.kind not in 'iu' or not len(mapping)
            or (mapping < 0).any() or (mapping >= len(X)).any()
            or len(np.unique(mapping)) != len(mapping)):
        raise ValueError('Source-row mapping must contain unique in-bounds integer positions')
    if len(X) != manifest.get('source_rows') or len(mapping) != manifest.get('total_rows'):
        raise ValueError('Source/modeling row counts differ from A4')
    if manifest['status'] == 'SMOKE_NOT_RESEARCH_RESULT':
        expected, _ = train_test_split(
            np.arange(len(y)), train_size=min(260, len(y) - 2), stratify=y, random_state=SEED)
    else:
        expected = np.arange(len(X))
    if not np.array_equal(mapping, expected):
        raise ValueError('Source-row mapping differs from the recorded A4 sampling protocol')
    return mapping


def _selection(a4, manifest, features):
    selection = read_json(a4 / 'selection.json')
    k = selection.get('selected_feature_k')
    if isinstance(k, bool) or not isinstance(k, int) or k not in FEATURE_COUNTS:
        raise ValueError('Invalid selected feature count')
    candidates = ['B1', 'B2', 'B3', 'A2', f'A3_k{k}']
    if (selection.get('run_id') != manifest['run_id']
            or selection.get('candidate_order') != candidates
            or selection.get('final_model_tie_rule') != FINAL_MODEL_TIE_RULE):
        raise ValueError('Selection run/candidate-order/tie-rule metadata differs from A4')
    if selection.get('input_feature_names') != features:
        raise ValueError('Selection input feature ordering differs from A4')
    ap_candidates=selection.get('ap_validation_candidates', {})
    locked_id=selection.get('final_model')
    if (not isinstance(ap_candidates,dict) or set(ap_candidates)!=set(candidates)
            or locked_id not in candidates
            or any(not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1
                   for v in ap_candidates.values())):
        raise ValueError('Invalid locked validation model decision')
    locked_ap=ap_candidates[locked_id]
    if (any(value>locked_ap for value in ap_candidates.values())
            or any(ap_candidates[key]==locked_ap for key in candidates[:candidates.index(locked_id)])):
        raise ValueError('Final model ID differs from locked validation selection')
    validation = pd.read_csv(a4 / 'validation_metrics.csv', float_precision='round_trip')
    if list(validation.get('model', [])) != CORE_MODEL_IDS or not (validation['split'] == 'validation').all():
        raise ValueError('Validation metric model IDs/split differ from A4')
    saved_ap = dict(zip(validation.model, validation.average_precision))
    feature_ap = {f'A3_k{value}': saved_ap[f'A3_k{value}'] for value in FEATURE_COUNTS}
    reference = saved_ap['A2']
    expected_k,eligible=select_feature_budget(reference,{value:feature_ap[f'A3_k{value}'] for value in FEATURE_COUNTS})
    if (selection.get('reduction_tolerance_ap') != 0.01
            or selection.get('weighted_reference_ap') != reference
            or selection.get('ap_validation_feature_counts') != feature_ap
            or selection.get('eligible_feature_counts') != eligible
            or k != expected_k
            or selection.get('selected_reduction_model') != (f'A3_k{k}' if eligible else 'A2')
            or selection.get('separately_reported_a3_model') != f'A3_k{k}'
            or selection.get('full_feature_fallback') != 'A2'
            or selection['ap_validation_candidates'] != {key: saved_ap[key] for key in candidates}):
        raise ValueError('Feature-count/final selection differs from saved validation metrics')
    threshold = _threshold(selection.get('final_threshold'), 'Final threshold')
    if (selection.get('threshold_policy') != threshold_policy()
            or selection.get('feature_selection') != feature_selection_policy()
            or selection.get('primary_feature_selection_status') != 'IMPLEMENTED'
            or selection.get('feature_selection_cv_performed') is not True):
        raise ValueError('Selection threshold/feature methodology differs from the recorded QA protocol')
    f2_thresholds = selection.get('validation_f2_thresholds', {})
    if set(f2_thresholds) != set(CORE_MODEL_IDS):
        raise ValueError('Validation threshold model IDs are inconsistent')
    f2_thresholds = {key: _threshold(value, f'{key} threshold') for key, value in f2_thresholds.items()}
    if threshold != f2_thresholds[selection['final_model']]:
        raise ValueError('Final threshold differs from locked validation threshold')
    trace = read_json(a4 / 'threshold_search.json')
    if (trace.get('run_id') != manifest['run_id'] or trace.get('policy') != threshold_policy()
            or set(trace.get('models', {})) != set(CORE_MODEL_IDS)):
        raise ValueError('F2 threshold-search provenance differs from A4')
    # Verify the already-locked validation curve; A5 never invokes threshold selection.
    for key, saved in trace['models'].items():
        values = np.asarray(saved.get('f2_values'), dtype=float)
        winner = saved.get('winner_grid_index')
        if (values.shape != F2_GRID.shape or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()
                or isinstance(winner, bool) or not isinstance(winner, int)
                or not 0 <= winner < len(F2_GRID)):
            raise ValueError('Invalid F2 validation threshold curve')
        maximum = float(values.max())
        if (values[winner] != maximum or (values[:winner] == maximum).any()
                or saved.get('max_f2') != maximum
                or saved.get('selected_threshold') != float(F2_GRID[winner])
                or f2_thresholds[key] != float(F2_GRID[winner])):
            raise ValueError('F2 grid/smallest exact-tie decision differs from locked validation threshold')
    reference = pd.read_csv(a4 / 'validation_reference_metrics.csv', float_precision='round_trip')
    if (list(reference.get('model', [])) != CORE_MODEL_IDS
            or not (reference['split'] == 'validation').all() or not (reference.threshold == 0.5).all()
            or not np.allclose(reference.average_precision,validation.average_precision,atol=1e-12,rtol=0)):
        raise ValueError('Validation 0.5 reference/AP invariance differs from A4')
    test_ids = ['B0', 'B1', 'B2', 'B3', 'A1', 'A2', f'A3_k{k}']
    test_thresholds = {
        key: f2_thresholds[key] if key in ('A2', f'A3_k{k}') else 0.5 for key in test_ids}
    if selection.get('test_thresholds') != test_thresholds:
        raise ValueError('Test decision thresholds differ from locked selection')
    return selection, threshold, test_ids


def _verify_primary_pipeline(pipeline, k, features, source_ids, ytrain, b3_params):
    steps=pipeline.named_steps
    selector=steps.get('selection')
    if (list(steps) != ['selection','model'] or not isinstance(selector,SelectFromModel)
            or not isinstance(selector,ExactTopKSelectFromModel) or selector.k != k
            or selector.max_features != k or selector.prefit or selector.threshold != -np.inf
            or selector.importance_getter != 'feature_importances_' or selector.norm_order != 1
            or getattr(selector,'_sklearn_output_config',{}).get('transform') != 'pandas'):
        raise ValueError('Primary selector differs from exact top-k SelectFromModel settings')
    expected={**b3_params,'weight_mode':'balanced','importance_type':'gain'}
    if (list(selector.feature_names_in_) != features
            or not isinstance(selector.estimator,FoldAwareXGB) or selector.estimator.get_params() != expected):
        raise ValueError('Primary ranking settings/input columns differ from selected B3 gain configuration')
    ranker=selector.estimator_
    downstream=steps['model']
    ratio=float((np.asarray(ytrain)==0).sum()/(np.asarray(ytrain)==1).sum())
    for role,estimator in [('ranker',ranker),('downstream',downstream)]:
        if (not isinstance(estimator,FoldAwareXGB) or estimator.get_params() != expected
                or not isinstance(getattr(estimator,'core_',None),XGBClassifier)):
            raise ValueError(f'Primary {role} ranking settings differ from selected B3')
        actual=estimator.core_.get_params()
        if (any(actual.get(key) != value for key,value in expected.items() if key != 'weight_mode')
                or actual.get('tree_method') != 'hist' or actual.get('eval_metric') != 'logloss'
                or actual.get('scale_pos_weight') != ratio or estimator.weight_ratio_ != ratio):
            raise ValueError(f'Primary {role} gain/class weighting differs from allowed fit labels')
        if (not np.array_equal(getattr(estimator,'fit_source_row_indices_',None),source_ids)
                or getattr(estimator,'n_fit_samples_',None) != len(source_ids)
                or getattr(estimator,'fit_label_sha256_',None) != array_sha256(np.asarray(ytrain,dtype=int))):
            raise ValueError(f'Primary {role} fit identities differ from the allowed training fold')
    if not np.array_equal(selector.fit_source_row_indices_,source_ids):
        raise ValueError('Primary selector fit identities differ from the allowed training fold')
    if list(ranker.feature_names_in_) != features or ranker.n_features_in_ != len(features):
        raise ValueError('Primary ranker fitted column order differs from original features')
    selected=list(np.asarray(features)[selector.get_support()])
    if (len(selected) != k or downstream.n_features_in_ != k
            or list(downstream.feature_names_in_) != selected
            or ranker.core_.get_booster().feature_names != features
            or downstream.core_.get_booster().feature_names != selected):
        raise ValueError('Primary selected columns/actual feature count differ from fitted models')
    return selected


def _registry(a4, manifest, selection, features, ytrain, train_ids):
    # Checksum verification has already run. Only load artifacts from trusted creators.
    fit = joblib.load(a4 / 'fitted_models.joblib')
    if not isinstance(fit, dict) or fit.get('schema_version') != MANIFEST_VERSION:
        raise ValueError('Unsupported fitted-model registry schema')
    if (fit.get('run_id') != manifest['run_id']
            or fit.get('source_files_sha256') != manifest['source_files_sha256']):
        raise ValueError('Model registry provenance differs from A4')
    if fit.get('final_model_id') != selection['final_model']:
        raise ValueError('Model registry and selection file disagree on final model ID')
    if _threshold(fit.get('final_threshold'), 'Registry threshold') != selection['final_threshold']:
        raise ValueError('Model registry and selection threshold mismatch')
    if fit.get('feature_names') != features:
        raise ValueError('Model registry feature ordering differs from A4')
    if fit.get('selection') != selection:
        raise ValueError('Model registry and selection metadata disagree')
    models = fit.get('models')
    if (not isinstance(models, dict) or list(models) != CORE_MODEL_IDS
            or fit.get('model_ids') != CORE_MODEL_IDS or manifest.get('model_ids') != CORE_MODEL_IDS):
        raise ValueError('Model registry IDs differ from A4 core model IDs')
    runtime = read_json(a4 / 'runtime_config.json')
    if runtime['effective'].get('fitted_model_configuration') != model_configuration(models):
        raise ValueError('Fitted model configuration differs from effective A4 runtime metadata')
    for key in ('B1', 'B2', 'B3'):
        params = models[key].get_params()
        best = runtime['search_counts'][key].get('best_params', {})
        if not best or any(params.get(field) != value for field, value in best.items()):
            raise ValueError(f'{key} fitted hyperparameters differ from recorded best search parameters')

    actual_features = {}
    ratio = float((np.asarray(ytrain) == 0).sum() / (np.asarray(ytrain) == 1).sum())
    b3_params = None
    for key, model in models.items():
        if list(np.asarray(getattr(model, 'classes_', []))) != [0, 1]:
            raise ValueError(f'{key} model must have fitted binary classes [0, 1]')
        names = getattr(model, 'feature_names_in_', None)
        if (key != 'B0' and names is None) or (names is not None and list(names) != features):
            raise ValueError(f'{key} fitted feature ordering differs from A4')
        if key == 'B0':
            if not isinstance(model, DummyClassifier) or model.strategy != 'prior':
                raise ValueError('B0 must be the fitted prior DummyClassifier')
            actual_features[key] = features
            continue
        if not isinstance(model, Pipeline):
            raise ValueError(f'{key} must be a complete fitted pipeline')
        steps = model.named_steps
        estimator = steps.get('model')
        if key == 'B1':
            if (list(steps) != ['scaler', 'model'] or not isinstance(steps['scaler'], StandardScaler)
                    or not isinstance(estimator, LogisticRegression) or estimator.class_weight is not None):
                raise ValueError('B1 must be the scaled unweighted Logistic Regression baseline')
        elif key == 'B2':
            if (list(steps) != ['model'] or not isinstance(estimator, RandomForestClassifier)
                    or estimator.class_weight is not None):
                raise ValueError('B2 must be the fitted Random Forest baseline')
        else:
            if not isinstance(estimator, FoldAwareXGB):
                raise ValueError(f'{key} must contain the fitted XGBoost wrapper')
            if not isinstance(getattr(estimator, 'core_', None), XGBClassifier):
                raise ValueError(f'{key} must actually contain XGBClassifier; no LightGBM or RF fallback')
            core_params = estimator.core_.get_params()
            if (core_params.get('tree_method') != 'hist' or core_params.get('eval_metric') != 'logloss'
                    or core_params.get('scale_pos_weight') != estimator.weight_ratio_
                    or any(core_params.get(field) != value for field,value in estimator.get_params().items()
                           if field != 'weight_mode')):
                raise ValueError(f'{key} actual XGBoost parameters differ from its wrapper')
            params = estimator.get_params()
            if key == 'B3':
                if params['weight_mode'] != 'none' or estimator.weight_ratio_ != 1.0:
                    raise ValueError('B3 must remain the unweighted XGBoost baseline')
                b3_params = params
            else:
                if params != {**b3_params, 'weight_mode': 'balanced'} or not np.isclose(estimator.weight_ratio_, ratio):
                    raise ValueError(f'{key} weighting/hyperparameters differ from the controlled B3 ablation')
        if key.startswith('A3_k'):
            k = int(key.removeprefix('A3_k'))
            actual_features[key]=_verify_primary_pipeline(model,k,features,train_ids,ytrain,b3_params)
        else:
            if key != 'B1' and list(steps) != ['model']:
                raise ValueError(f'{key} pipeline steps differ from A4')
            actual_features[key] = features
    saved_features = read_json(a4 / 'feature_sets.json')
    expected_sets = {key: value for key, value in actual_features.items() if key.startswith('A3_k')}
    if saved_features != expected_sets or fit.get('model_feature_sets') != actual_features:
        raise ValueError('Selected feature names/order differ from the fitted selectors')
    final_features = actual_features[selection['final_model']]
    if (selection.get('final_feature_names') != final_features
            or selection.get('final_feature_count') != len(final_features)):
        raise ValueError('Final selected feature metadata differs from the actual model')
    return fit


def _verify_primary_evidence(a4, manifest, fit, Xtrain, ytrain, Xvalidation, yvalidation, train_ids, validation_ids, expected_folds):
    """Check serialized fold fits and recompute training-CV diagnostics without fitting."""
    folds=read_json(a4/'feature_selection_folds.json')
    context=dict(run_id=manifest['run_id'],source_files_sha256=manifest['source_files_sha256'],
                 dataset_sha256=manifest['dataset_sha256'],split_archive_sha256=manifest['split_archive_sha256'],
                 cv_fold_archive_sha256=sha256_file(a4/'cv_fold_indices.npz'),policy=feature_selection_policy())
    fold_registry=joblib.load(a4/'feature_selection_models.joblib')
    if (not isinstance(fold_registry,dict) or any(folds.get(key) != value or fold_registry.get(key) != value
                                               for key,value in context.items())):
        raise ValueError('Primary selector fold/source identities differ from A4')
    n_folds=manifest['cv_folds']
    expected=[(k,fold) for k in FEATURE_COUNTS for fold in range(n_folds)]
    records=folds.get('records',[])
    models=fold_registry.get('models',{})
    if ([(r.get('requested_k'),r.get('fold')) for r in records] != expected
            or list(models) != [f'A3_k{k}_fold_{fold}' for k,fold in expected]
            or folds.get('successful_folds_per_k') != {str(k):n_folds for k in FEATURE_COUNTS}
            or manifest.get('primary_selector_cv_fits') != len(expected)
            or manifest.get('primary_selector_refit_fits') != len(FEATURE_COUNTS)):
        raise ValueError('Primary selector fold coverage/feature counts differ from A4')
    table=pd.read_csv(a4/'feature_selection_cv.csv',float_precision='round_trip')
    expected_rows=[]
    positions={int(source_id):pos for pos,source_id in enumerate(train_ids)}
    b3_params=fit['models']['B3'].named_steps['model'].get_params()
    for record,(k,fold) in zip(records,expected):
        tr=expected_folds[f'fold_{fold}_train']; va=expected_folds[f'fold_{fold}_validation']
        model=models[f'A3_k{k}_fold_{fold}']
        tr_pos=[positions[int(i)] for i in tr]; va_pos=[positions[int(i)] for i in va]
        _verify_primary_pipeline(model,k,list(Xtrain.columns),tr,ytrain.iloc[tr_pos],b3_params)
        ranking=primary_ranking_record(model)
        ap=float(average_precision_score(ytrain.iloc[va_pos],model.predict_proba(Xtrain.iloc[va_pos])[:,1]))
        if (any(record.get(key) != value for key,value in ranking.items())
                or record.get('model') != f'A3_k{k}' or record.get('scope') != 'primary_training_cv'
                or record.get('train_source_row_indices') != tr.tolist()
                or record.get('validation_source_row_indices') != va.tolist()
                or record.get('validation_source_indices_sha256') != array_sha256(va)
                or record.get('ap') != ap):
            raise ValueError('Primary selector fold rankings/selected columns/AP/fit identities differ from fitted model')
        expected_rows.append(dict(model=f'A3_k{k}',fold=fold,requested_k=k,actual_feature_count=k,
                                  average_precision=ap,train_rows=len(tr),validation_rows=len(va),
                                  ranker_weight_ratio=ranking['ranker_weight_ratio'],
                                  downstream_weight_ratio=ranking['downstream_weight_ratio']))
    if table.to_dict(orient='records') != expected_rows:
        raise ValueError('Primary selector CV AP/feature counts differ from verified fold records')
    rankings=read_json(a4/'feature_rankings.json')
    expected_rankings={f'A3_k{k}':dict(scope='full_primary_training_refit',**primary_ranking_record(fit['models'][f'A3_k{k}']))
                       for k in FEATURE_COUNTS}
    if (rankings.get('run_id') != manifest['run_id']
            or rankings.get('source_files_sha256') != manifest['source_files_sha256']
            or rankings.get('policy') != feature_selection_policy() or rankings.get('models') != expected_rankings):
        raise ValueError('Primary full-training rankings/selected columns differ from fitted selectors')
    reference=read_json(a4/'a2_reference_lock.json')
    a2_ap=float(average_precision_score(yvalidation,fit['models']['A2'].predict_proba(Xvalidation)[:,1]))
    if reference != dict(run_id=manifest['run_id'],source_files_sha256=manifest['source_files_sha256'],
                         model='A2',average_precision=a2_ap,train_source_indices_sha256=array_sha256(train_ids),
                         validation_source_indices_sha256=array_sha256(validation_ids),locked_before_primary_selector=True):
        raise ValueError('Locked full-feature A2 reference differs from reserved-validation model scores')
    validation=pd.read_csv(a4/'validation_metrics.csv',float_precision='round_trip')
    trace=read_json(a4/'threshold_search.json')
    for key,model in fit['models'].items():
        p=model.predict_proba(Xvalidation)[:,1]
        if (float(validation.loc[validation.model==key,'average_precision'].iloc[0]) != float(average_precision_score(yvalidation,p))
                or trace['models'][key]['f2_values'] != f2_threshold_curve(yvalidation,p).tolist()):
            raise ValueError('Reserved-validation AP/F2 evidence differs from fitted A4 models')
    # Full-budget transform must preserve every original column and A2 scores.
    if not np.array_equal(fit['models']['A3_k21'].predict_proba(Xvalidation),fit['models']['A2'].predict_proba(Xvalidation)):
        raise ValueError('Full-feature A3 k=21 does not preserve A2 score parity')
    secondary=read_json(a4/'secondary_anova.json')
    if (secondary.get('run_id') != manifest['run_id'] or secondary.get('scope') != 'SECONDARY_QA_ONLY'
            or secondary.get('method') != 'SelectKBest(f_classif)' or secondary.get('training_cv_performed') is not False
            or secondary.get('used_for_primary_selection') is not False
            or set(secondary.get('models',{})) != (set() if manifest['status']=='RESEARCH_TEST_COMPLETE'
                                                   else {f'ANOVA_k{k}' for k in FEATURE_COUNTS})):
        raise ValueError('Secondary ANOVA experiment is not separated from the primary selector')


def _recompute_predictions(pred, models, selection, test_ids, Xtest):
    expected_columns = ['row_index', 'y_true', *[f'prob_{key}' for key in test_ids], 'prob_FINAL']
    if list(pred.columns) != expected_columns:
        raise ValueError('Prediction artifact columns/model IDs differ from A4')
    recomputed, differences = {}, {}
    for label in [*test_ids, 'FINAL']:
        key = selection['final_model'] if label == 'FINAL' else label
        if key not in recomputed:
            probabilities = np.asarray(models[key].predict_proba(Xtest), dtype=float)
            if (probabilities.shape != (len(Xtest), 2) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()):
                raise ValueError(f'{key} serialized model returned invalid probabilities')
            recomputed[key] = probabilities[:, 1]
        fresh = recomputed[key]
        column = f'prob_{label}'
        try:
            saved = pred[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Saved probabilities are not numeric: {column}') from exc
        if not np.isfinite(saved).all() or (saved < 0).any() or (saved > 1).any():
            raise ValueError(f'Saved probabilities must be finite in [0, 1]: {column}')
        if not np.allclose(saved, fresh, atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL):
            raise ValueError(f'Saved test probabilities differ from serialized model: {column}')
        differences[column] = float(np.max(np.abs(saved - fresh)))
        # Analyze exact model scores, preventing tolerated CSV noise from altering ties/AP.
        pred[column] = fresh
    return pred, differences


def _verify_test_metrics(a4, pred, selection, test_ids):
    recorded = pd.read_csv(a4 / 'test_metrics.csv', float_precision='round_trip')
    if list(recorded.get('model', [])) != [*test_ids, 'FINAL'] or not (recorded['split'] == 'test').all():
        raise ValueError('Saved test metric model IDs/split differ from A4')
    for _, row in recorded.iterrows():
        key = row['model']
        threshold = selection['final_threshold'] if key == 'FINAL' else selection['test_thresholds'][key]
        expected = metrics_at_threshold(pred.y_true, pred[f'prob_{key}'], threshold)
        for field, value in expected.items():
            if field not in row or not np.isfinite(row[field]):
                raise ValueError(f'Saved test metric is missing/invalid: {key}/{field}')
            if field in ('tn', 'fp', 'fn', 'tp', 'threshold'):
                agrees = row[field] == value
            else:
                agrees = np.isclose(row[field], value, atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL)
            if not agrees:
                raise ValueError(f'Saved test metrics differ from locked model: {key}/{field}')
    reference = pd.read_csv(a4 / 'test_reference_metrics.csv', float_precision='round_trip')
    if list(reference.get('model', [])) != [*test_ids, 'FINAL'] or not (reference['split'] == 'test').all():
        raise ValueError('Test reference model IDs/split differ from A4')
    for _, row in reference.iterrows():
        expected = metrics_at_threshold(pred.y_true, pred[f"prob_{row['model']}"], 0.5)
        for field,value in expected.items():
            if field not in row or not np.isclose(row[field],value,atol=PROBABILITY_ATOL,rtol=PROBABILITY_RTOL):
                raise ValueError('Saved test 0.5 reference metrics differ from the locked model')


def _verify_cv_trace(trace, runtime):
    """Validate actual distinct trials, fold scores and their saved winning parameters."""
    folds = runtime['effective']['cross_validation']['folds']
    for key, counts in runtime['search_counts'].items():
        rows = trace.loc[trace.model == key]
        if not np.array_equal(rows.trial.to_numpy(), np.arange(len(rows))):
            raise ValueError(f'CV trial indices are inconsistent: {key}')
        try:
            parameters = [json.loads(value) for value in rows.params]
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid CV parameter trace: {key}') from exc
        space = list(ParameterGrid(runtime['effective']['search_spaces'][key]))
        encoded = [json.dumps(value, sort_keys=True) for value in parameters]
        if len(set(encoded)) != len(encoded) or any(value not in space for value in parameters):
            raise ValueError(f'CV configurations are duplicated or outside the search space: {key}')
        columns = [f'fold_{fold}_ap' for fold in range(folds)]
        if any(column not in rows for column in columns):
            raise ValueError(f'Missing CV fold scores: {key}')
        scores = rows[columns].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError(f'Invalid CV fold scores: {key}')
        means, deviations = scores.mean(axis=1), scores.std(axis=1)
        if (not np.allclose(rows.mean_cv_ap, means, atol=1e-12, rtol=0)
                or not np.allclose(rows.std_cv_ap, deviations, atol=1e-12, rtol=0)):
            raise ValueError(f'CV summary scores disagree with fold scores: {key}')
        # Rank is diagnostic only; near-equal scores may legitimately have distinct ranks.
        if not np.isfinite(rows.rank_cv_ap).all() or (rows.rank_cv_ap < 1).any():
            raise ValueError(f'Invalid CV ranks: {key}')
        best_rows = rows.loc[rows.rank_cv_ap == 1]
        if best_rows.empty:
            raise ValueError(f'CV trace has no winning configuration: {key}')
        winner = int(best_rows.iloc[0].trial)
        if means[winner] < means.max() - 1e-12 or parameters[winner] != counts.get('best_params'):
            raise ValueError(f'CV winner differs from recorded best parameters: {key}')


def validate_a4_artifacts(data_path, a4_dir, manifest):
    """Return verified data/models/fresh scores; never fit or change a selection."""
    a4, data_path = Path(a4_dir), Path(data_path)
    if manifest.get('status') not in A4_STATUSES:
        raise ValueError('Unknown A4 research/QA status')
    provenance = verify_a4_provenance(a4, manifest)
    quick = manifest['status'] == 'SMOKE_NOT_RESEARCH_RESULT'
    if provenance['runtime']['effective']['search_spaces'] != random_spaces(quick):
        raise ValueError('A4 search spaces differ from this component')
    cv_trace = pd.read_csv(a4 / 'cv_results.csv')
    actual_counts = cv_trace.groupby('model', sort=False).size().to_dict()
    if actual_counts != manifest['actual_trials_per_family']:
        raise ValueError('Saved CV trace disagrees with actual search counts')
    _verify_cv_trace(cv_trace, provenance['runtime'])
    if sha256_file(data_path) != manifest['dataset_sha256']:
        raise ValueError('CSV SHA256 khác A4. Không dùng dữ liệu thay đổi hoặc stale source.')
    X, y = load_dataset(data_path)
    if sha256_file(data_path) != manifest['dataset_sha256']:
        raise ValueError('CSV changed while validating A5 input')
    features = list(X.columns)
    if features != manifest.get('features'):
        raise ValueError('Feature schema/order differs from A4')
    mapping = _source_mapping(a4, manifest, X, y)
    X = X.iloc[mapping].reset_index(drop=True)
    y = y.iloc[mapping].reset_index(drop=True)
    effective_split = provenance['runtime']['effective']['split']
    if effective_split.get('modeling_rows') != len(X) or effective_split.get('source_rows') != manifest['source_rows']:
        raise ValueError('Runtime source/modeling dimensions differ from A4')
    with np.load(a4 / 'split_indices.npz', allow_pickle=False) as archive:
        splits = validate_split_indices(
            {name: archive[name] for name in archive.files}, len(X), manifest.get('split_counts'))
    split_record = read_json(a4 / 'split_manifest.json')
    if (split_record.get('mode') != effective_split.get('mode')
            or split_record.get('archive_sha256') != manifest['split_archive_sha256']
            or split_record.get('normalized_array_sha256') != {
                key:array_sha256(ix) for key,ix in splits.items()}
            or effective_split.get('normalized_array_sha256') != split_record['normalized_array_sha256']):
        raise ValueError('Stored split manifest/archive/source positions differ from A4')
    if effective_split['mode'] == SHARED_MODE:
        expected_splits, verified_split = verify_shared_split(
            data_path, a4 / 'split_indices.npz', a4 / 'shared_split_manifest.json')
        if (verified_split['input_manifest_sha256'] != manifest.get('shared_split_manifest_sha256')
                or split_record.get('verified_B_v9_recipe_and_order') is not True):
            raise ValueError('Canonical split manifest verification differs from A4')
    elif effective_split['mode'] == LEGACY_QA_MODE and manifest['status'] != 'RESEARCH_TEST_COMPLETE':
        expected_splits = stratified_indices(y, seed=SEED)
    else:
        raise ValueError('Unapproved split mode; canonical archive cannot have a generated fallback')
    if any(not np.array_equal(splits[name], expected_splits[name]) for name in SPLIT_NAMES):
        raise ValueError('Saved split indices differ from the exact recorded A4 recipe/order')
    cv = provenance['runtime']['effective']['cross_validation']
    train_ids = mapping[splits['train']]
    expected_folds = {}
    for fold,(tr,va) in enumerate(StratifiedKFold(n_splits=cv['folds'],shuffle=True,random_state=SEED).split(
            X.iloc[splits['train']],y.iloc[splits['train']])):
        expected_folds[f'fold_{fold}_train'] = train_ids[tr]
        expected_folds[f'fold_{fold}_validation'] = train_ids[va]
    with np.load(a4 / 'cv_fold_indices.npz', allow_pickle=False) as archive:
        if (set(archive.files) != set(expected_folds)
                or any(not np.array_equal(archive[key],value) for key,value in expected_folds.items())):
            raise ValueError('Saved primary-training CV folds/source positions differ from A4')
    class_counts = {}
    for name, ix in splits.items():
        positive = int(y.iloc[ix].sum())
        class_counts[name] = dict(rows=len(ix), negative_count=len(ix)-positive,
                                  positive_count=positive, positive_fraction=positive/len(ix))
        if not 0 < positive < len(ix):
            raise ValueError(f'{name} split must contain both classes')
    audit = read_json(a4 / 'dataset_audit.json')
    if (manifest.get('split_class_counts') != class_counts
            or audit.get('split_diagnostics', {}).get('class_counts') != class_counts):
        raise ValueError('Saved per-split class diagnostics differ from source data')
    selection, threshold, test_ids = _selection(a4, manifest, features)
    pred = pd.read_csv(a4 / 'test_predictions.csv')
    if 'row_index' not in pred or not np.array_equal(pred.row_index.to_numpy(), mapping[splits['test']]):
        raise ValueError('Test row indices do not align with the A4 source-row mapping')
    if 'y_true' not in pred or not np.array_equal(pred.y_true.to_numpy(), y.iloc[splits['test']].to_numpy()):
        raise ValueError('Test ground truth differs from A4 source labels')
    Xtrain = X.iloc[splits['train']].reset_index(drop=True)
    Xtest = X.iloc[splits['test']].reset_index(drop=True)
    fit = _registry(a4, manifest, selection, features, y.iloc[splits['train']], train_ids)
    _verify_primary_evidence(a4,manifest,fit,Xtrain,y.iloc[splits['train']],
                             X.iloc[splits['validation']],y.iloc[splits['validation']],
                             train_ids,mapping[splits['validation']],expected_folds)
    pred, differences = _recompute_predictions(pred, fit['models'], selection, test_ids, Xtest)
    _verify_test_metrics(a4, pred, selection, test_ids)
    return dict(
        provenance=provenance, selection=selection, predictions=pred,
        final_model=fit['models'][fit['final_model_id']], final_threshold=threshold,
        Xtrain=Xtrain, Xtest=Xtest, ytest=y.iloc[splits['test']].to_numpy(),
        source_train_indices=mapping[splits['train']], source_test_indices=mapping[splits['test']],
        verification=dict(
            run_id=manifest['run_id'], split_indices_verified=True, source_mapping_verified=True,
            registry_metadata_verified=True, selected_features_verified=True,
            primary_selector_folds_verified=True, primary_rankings_verified=True,
            primary_feature_selection_status='IMPLEMENTED',
            probability_atol=PROBABILITY_ATOL, probability_rtol=PROBABILITY_RTOL,
            max_absolute_probability_differences=differences,
            analysis_scores='recomputed from serialized fitted models; no refit',
            test_metrics_verified=True))

