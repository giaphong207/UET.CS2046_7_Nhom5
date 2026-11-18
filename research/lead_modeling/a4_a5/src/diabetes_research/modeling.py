"""A4 reproducible baseline + imbalance + threshold + feature ablation.

No test labels/scores are used for model, hyperparameter, feature-count or threshold
selection. Test runs once after the validation decisions have been locked.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .core import (TARGET, SEED, FEATURE_COUNTS, FINAL_MODEL_TIE_RULE, dataset_audit, load_dataset,
                   metrics_at_threshold, optimal_f1_threshold,
                   select_final_model, sha256_file, split_diagnostics,
                   stratified_indices, versions, write_json)
from .provenance import (A4_CRITICAL_ARTIFACTS, MANIFEST_VERSION, artifact_hashes,
                         finish_execution, snapshot_protocol, source_identity, start_execution)

class FoldAwareXGB(ClassifierMixin, BaseEstimator):
    """Calculate `scale_pos_weight` on labels passed to *each* fit, not all CV labels.

    Useful for a fair comparison between weighted and unweighted XGBoost.
    """
    def __init__(self, n_estimators=160, max_depth=4, learning_rate=0.08,
                 min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, weight_mode='none', n_jobs=2, random_state=SEED):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.learning_rate=learning_rate
        self.min_child_weight=min_child_weight
        self.subsample=subsample
        self.colsample_bytree=colsample_bytree
        self.reg_lambda=reg_lambda
        self.weight_mode=weight_mode
        self.n_jobs=n_jobs
        self.random_state=random_state

    def fit(self, X, y):
        y=np.asarray(y).astype(int)
        if set(np.unique(y))!={0,1}:
            raise ValueError('XGBoost train fold must include both classes')
        ratio=(y==0).sum()/(y==1).sum()
        if self.weight_mode not in ('balanced','none'):
            raise ValueError('weight_mode must be balanced or none')
        self.weight_ratio_ = float(ratio if self.weight_mode=='balanced' else 1.0)
        self.core_=XGBClassifier(n_estimators=self.n_estimators,
            max_depth=self.max_depth,learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,reg_lambda=self.reg_lambda,
            scale_pos_weight=self.weight_ratio_,tree_method='hist',
            eval_metric='logloss',random_state=self.random_state,
            n_jobs=self.n_jobs)
        self.core_.fit(X,y)
        self.classes_=self.core_.classes_
        self.n_features_in_=X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, 'feature_names_in_'):
            del self.feature_names_in_
        return self

    def predict_proba(self,X):
        return self.core_.predict_proba(X)

    def predict(self,X):
        return self.core_.predict(X)


def initial_models(quick=False):
    n=45 if quick else 220
    return {
        'B1':Pipeline([('scaler',StandardScaler()),
                       ('model',LogisticRegression(solver='liblinear',max_iter=1500,random_state=SEED))]),
        'B2':Pipeline([('model',RandomForestClassifier(n_estimators=(35 if quick else 180),n_jobs=2,random_state=SEED))]),
        'B3':Pipeline([('model',FoldAwareXGB(n_estimators=n,n_jobs=2,weight_mode='none'))]),
    }


def random_spaces(quick=False):
    sizes=[35,50] if quick else [100,160,220,300]
    return {
        # Vary inverse L2 regularization meaningfully; keep the unweighted baseline.
        'B1':{'model__C':[0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0],
              'model__class_weight':[None]},
        'B2':{'model__n_estimators':sizes,
              'model__max_depth':[None,6,12],
              'model__min_samples_leaf':[1,3,6],
              'model__max_features':['sqrt',0.8]},
        'B3':{'model__n_estimators':sizes,
              'model__max_depth':[3,4,6],
              'model__learning_rate':[0.03,0.06,0.1],
              'model__subsample':[0.8,1.0],
              'model__colsample_bytree':[0.8,1.0],
              'model__min_child_weight':[1,3]},
    }


def search_model_family(estimator, space, X, y, trials, cv):
    """Evaluate a finite, seeded AP search and report requested and real effort."""
    cardinality = len(ParameterGrid(space))
    n_iter = min(trials, cardinality)
    search = RandomizedSearchCV(
        estimator, space, n_iter=n_iter, cv=cv, scoring='average_precision',
        random_state=SEED, refit=True, n_jobs=1, error_score='raise', return_train_score=False)
    started = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - started
    actual = len(search.cv_results_['params'])
    counts = dict(requested_configurations=int(trials), actual_configurations=actual,
                  finite_space_size=cardinality, space_exhausted=(actual == cardinality),
                  budget_capped_to_finite_space=(trials > cardinality),
                  cv_fits=actual * cv.n_splits, refit_fits=1, duration_seconds=elapsed,
                  best_params=search.best_params_)
    rows = []
    for i, params in enumerate(search.cv_results_['params']):
        row = dict(trial=i, mean_cv_ap=float(search.cv_results_['mean_test_score'][i]),
                   std_cv_ap=float(search.cv_results_['std_test_score'][i]),
                   mean_fit_time=float(search.cv_results_['mean_fit_time'][i]),
                   rank_cv_ap=int(search.cv_results_['rank_test_score'][i]),
                   params=json.dumps(params, sort_keys=True))
        for fold in range(cv.n_splits):
            row[f'fold_{fold}_ap'] = float(search.cv_results_[f'split{fold}_test_score'][i])
        rows.append(row)
    return search.best_estimator_, rows, counts


def model_configuration(models):
    """Record fixed and searched learner parameters without serializing estimator objects."""
    result = {}
    for name, pipeline in models.items():
        if isinstance(pipeline, Pipeline):
            steps = {}
            for step_name, estimator in pipeline.named_steps.items():
                params = estimator.get_params(deep=False)
                # SelectKBest's score function is the only callable parameter here.
                params = {key: (value.__name__ if callable(value) else value)
                          for key, value in params.items()}
                steps[step_name] = dict(type=type(estimator).__name__, parameters=params)
            result[name] = steps
        else:
            result[name] = dict(type=type(pipeline).__name__, parameters=pipeline.get_params(deep=False))
    return result


def run(data_path,output_dir, trials=10, cv_folds=3, quick=False, demo=False):
    execution, started = start_execution('a4')
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)) or not 1 <= trials <= 100:
        raise ValueError('trials must be an integer in 1..100')
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, (int, np.integer)) or cv_folds < 2:
        raise ValueError('cv_folds must be an integer >=2')
    requested_trials, requested_cv = int(trials), int(cv_folds)
    data_path=Path(data_path).expanduser().resolve()
    out=Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory already populated: {out}. Use a new --output-dir.')
    if not data_path.is_file():
        load_dataset(data_path)  # Preserve the helpful missing-dataset error.
    source_hash=sha256_file(data_path)
    X,y=load_dataset(data_path)
    if sha256_file(data_path) != source_hash:
        raise ValueError('CSV changed while loading; use an immutable dataset file')
    official_file='diabetes_binary_health_indicators_BRFSS2015.csv'
    is_official_candidate=(data_path.name==official_file and len(X)==253680)
    if not is_official_candidate and not (quick or demo):
        raise ValueError('Chế độ nghiên cứu yêu cầu file ' + official_file +
                         ' với 253680 mẫu. Dùng --demo cho dữ liệu tổng hợp QA.')
    audit=dataset_audit(X,y)
    source_rows=len(X)
    source_indices=np.arange(source_rows)
    if quick:
        # Save the exact map; local split positions never become respondent IDs.
        source_indices,_=train_test_split(source_indices,train_size=min(260,len(y)-2),
                                         stratify=y,random_state=SEED)
        X=X.iloc[source_indices].reset_index(drop=True)
        y=y.iloc[source_indices].reset_index(drop=True)
        trials=min(trials,1)
        cv_folds=2
    status=('SMOKE_NOT_RESEARCH_RESULT' if quick else
            ('SYNTHETIC_QA_ONLY' if demo else 'RESEARCH_TEST_COMPLETE'))
    splits=stratified_indices(y)
    diagnostics=split_diagnostics(X,y,splits)
    audit['modeling_rows']=len(X)
    audit['split_diagnostics']=diagnostics
    Xt=X.iloc[splits['train']]
    yt=y.iloc[splits['train']]
    Xv=X.iloc[splits['validation']]
    yv=y.iloc[splits['validation']]
    Xtest=X.iloc[splits['test']]
    ytest=y.iloc[splits['test']]
    if cv_folds > int(yt.value_counts().min()):
        raise ValueError('cv_folds exceeds the minority-class count in training')
    if any(len(np.unique(y.iloc[ix])) != 2 for ix in splits.values()):
        raise ValueError('Each train/validation/test split must contain both classes')

    out.mkdir(parents=True,exist_ok=True)
    protocol,protocol_hash=snapshot_protocol(out)
    identity=source_identity()
    np.save(out/'source_row_indices.npy',source_indices,allow_pickle=False)
    np.savez_compressed(out/'split_indices.npz',**splits)
    (out/'dataset_sha256.txt').write_text(source_hash+'\n',encoding='ascii')
    write_json(out/'dataset_audit.json',audit)
    write_json(out/'versions.json',versions())
    write_json(out/'source_identity.json',identity)

    spaces=random_spaces(quick)
    cv=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=SEED)
    models={'B0':DummyClassifier(strategy='prior')}
    models['B0'].fit(Xt,yt)
    trace,search_counts=[],{}
    for name,est in initial_models(quick).items():
        model,rows,counts=search_model_family(est,spaces[name],Xt,yt,int(trials),cv)
        models[name]=model
        search_counts[name]=counts
        trace.extend({'model':name,**row} for row in rows)
    pd.DataFrame(trace).to_csv(out/'cv_results.csv',index=False)
    xgb_params=models['B3'].named_steps['model'].get_params()
    weighted=FoldAwareXGB(**{**xgb_params,'weight_mode':'balanced'})
    # Controlled weighting ablation: reuse unweighted B3 hyperparameters, fit on full train.
    models['A1']=Pipeline([('model',weighted)]).fit(Xt,yt)
    models['A2']=models['A1']
    val_prob={key:mdl.predict_proba(Xv)[:,1] for key,mdl in models.items()}
    chosen_thresholds={key:optimal_f1_threshold(yv,p) for key,p in val_prob.items()}
    result_rows=[]
    for key,p in val_prob.items():
        threshold=chosen_thresholds[key] if key=='A2' else 0.5
        result_rows.append({'model':key,'split':'validation',
                            **metrics_at_threshold(yv,p,threshold)})
    # A3 uses a train-fit filter and validation selection; no A3 CV is executed here.
    for k in FEATURE_COUNTS:
        m=Pipeline([('selection',SelectKBest(f_classif,k=k)),
                    ('model',FoldAwareXGB(**{**xgb_params,'weight_mode':'balanced'}))])
        m.fit(Xt,yt)
        key=f'A3_k{k}'
        models[key]=m
        p=m.predict_proba(Xv)[:,1]
        chosen_thresholds[key]=optimal_f1_threshold(yv,p)
        result_rows.append({'model':key,'split':'validation',
                            **metrics_at_threshold(yv,p,chosen_thresholds[key])})
    validation=pd.DataFrame(result_rows)
    validation.to_csv(out/'validation_metrics.csv',index=False)
    base_ap=float(validation.loc[validation.model=='A2','average_precision'].iloc[0])
    feature_ap={f'A3_k{k}':float(validation.loc[
        validation.model==f'A3_k{k}','average_precision'].iloc[0]) for k in FEATURE_COUNTS}
    eligible=[k for k in sorted(FEATURE_COUNTS) if base_ap-feature_ap[f'A3_k{k}'] <= 0.01+1e-12]
    selected_k=min(eligible) if eligible else 21
    # Exact AP ties follow this existing candidate order (B1 before B2 before B3).
    candidates=['B1','B2','B3','A2',f'A3_k{selected_k}']
    val_AP={key:float(validation.loc[validation.model==key,'average_precision'].iloc[0])
            for key in candidates}
    final_id=select_final_model(candidates,val_AP)
    final_threshold=chosen_thresholds[final_id]
    input_features=list(X.columns)
    feature_sets={key:list(Xt.columns[mdl.named_steps['selection'].get_support()])
                  for key,mdl in models.items() if key.startswith('A3_k')}
    model_feature_sets={key:feature_sets.get(key,input_features) for key in models}
    write_json(out/'feature_sets.json',feature_sets)
    test_model_ids=['B0','B1','B2','B3','A1','A2',f'A3_k{selected_k}']
    test_thresholds={key:(chosen_thresholds[key] if key in ['A2',f'A3_k{selected_k}'] else 0.5)
                     for key in test_model_ids}
    selection=dict(
        run_id=execution['run_id'], final_model=final_id, final_threshold=float(final_threshold),
        input_feature_names=input_features, final_feature_names=model_feature_sets[final_id],
        final_feature_count=len(model_feature_sets[final_id]),
        selected_feature_k=int(selected_k), reduction_tolerance_ap=0.01,
        eligible_feature_counts=eligible, ap_validation_feature_counts=feature_ap,
        ap_validation_candidates=val_AP, weighted_reference_ap=base_ap,
        candidate_order=candidates, final_model_tie_rule=FINAL_MODEL_TIE_RULE,
        validation_f1_thresholds=chosen_thresholds, test_thresholds=test_thresholds,
        threshold_selection='max F1 on validation; ties highest threshold',
        interpretation=('SMOKE NOT RESEARCH' if quick else ('SYNTHETIC DEMO' if demo else 'LOCKED TEST PROTOCOL')),
        note='Do not change choices based on test scores.')
    write_json(out/'selection.json',selection)

    # All model/k/threshold decisions above are now locked. No test-based refit/selection.
    rows=[]
    preds=pd.DataFrame({'row_index':source_indices[splits['test']],'y_true':ytest.to_numpy()})
    for key in test_model_ids:
        prob=models[key].predict_proba(Xtest)[:,1]
        preds[f'prob_{key}']=prob
        rows.append({'model':key,'split':'test',
                     **metrics_at_threshold(ytest,prob,test_thresholds[key])})
    preds['prob_FINAL']=models[final_id].predict_proba(Xtest)[:,1]
    rows.append({'model':'FINAL','split':'test',
                 **metrics_at_threshold(ytest,preds['prob_FINAL'],final_threshold)})
    test_metrics=pd.DataFrame(rows)
    test_metrics.to_csv(out/'test_metrics.csv',index=False)
    preds.to_csv(out/'test_predictions.csv',index=False)
    ablation=pd.concat([validation,test_metrics],ignore_index=True)
    ablation['selected_feature_count']=[
        len(model_feature_sets[final_id if key=='FINAL' else key]) for key in ablation.model]
    ablation.to_csv(out/'ablation.csv',index=False)
    joblib.dump(dict(
        schema_version=MANIFEST_VERSION, run_id=execution['run_id'],
        source_files_sha256=identity['source_files_sha256'],
        models=models, model_ids=list(models), final_model_id=final_id,
        final_threshold=final_threshold, feature_names=input_features,
        model_feature_sets=model_feature_sets, selection=selection),
        out/'fitted_models.joblib',compress=3)

    overrides={}
    planned_cv=protocol['cross_validation']['folds']
    if cv_folds != planned_cv:
        overrides['cv_folds']=dict(protocol=planned_cv,effective=int(cv_folds))
    lower,upper=protocol['cross_validation']['configs_per_family_range']
    if not lower <= trials <= upper:
        overrides['search_budget']=dict(protocol_range=[lower,upper],effective=int(trials))
    if trials != requested_trials or cv_folds != requested_cv:
        overrides['quick_mode']=dict(requested_trials=requested_trials,requested_cv=requested_cv,
                                    effective_trials=int(trials),effective_cv=int(cv_folds))
    runtime=dict(
        run_id=execution['run_id'], requested=dict(
            data=str(data_path),output_dir=str(out),trials=requested_trials,cv_folds=requested_cv,
            quick=bool(quick),demo=bool(demo)),
        effective=dict(
            mode=status,target=TARGET,split=dict(ratios=[0.70,0.15,0.15],stratify=True,
                rounding='round(0.15*N) for validation and test; remainder train',
                source_rows=source_rows,modeling_rows=len(X),index_space='positions in source_row_indices.npy'),
            cross_validation=dict(type='StratifiedKFold',folds=int(cv_folds),shuffle=True,
                                  scoring='average_precision',n_jobs=1),
            search_backend='RandomizedSearchCV',search_spaces=spaces,
            fitted_model_configuration=model_configuration(models),
            xgboost_fixed_configuration=dict(tree_method='hist',eval_metric='logloss'),
            weighting='A1/A3 balanced on full train; B3 search unweighted',
            feature_selection=dict(method='SelectKBest(f_classif)',k_values=list(FEATURE_COUNTS),
                                   fit_scope='full train; k selected on validation; no A3 CV',
                                   ordinal_representation='numeric codes; LR scales all inputs'),
            threshold=dict(reference=0.5,selection='max F1 on validation',tie_rule='highest threshold'),
            final_model_tie_rule=FINAL_MODEL_TIE_RULE),
        seeds=dict(split=SEED,cv=SEED,search=SEED,model=SEED),
        protocol_overrides=overrides,search_counts=search_counts)
    write_json(out/'runtime_config.json',runtime)
    if source_identity()['file_hashes'] != identity['file_hashes']:
        raise ValueError('Component source changed during A4 execution')
    if sha256_file(data_path) != source_hash:
        raise ValueError('CSV changed during A4 execution')
    hashes=artifact_hashes(out,A4_CRITICAL_ARTIFACTS)
    actual_counts={name:counts['actual_configurations'] for name,counts in search_counts.items()}
    manifest=dict(
        **finish_execution(execution,started),manifest_version=MANIFEST_VERSION,
        protocol='A2_protocol_v2',protocol_sha256=protocol_hash,status=status,
        target=TARGET,seed=SEED,source_file=str(data_path),dataset_sha256=source_hash,
        source_rows=source_rows,total_rows=int(len(X)),features=input_features,
        split_ratios=[0.70,0.15,0.15],cv_folds=int(cv_folds),
        requested_trials_per_family={name:requested_trials for name in search_counts},
        effective_requested_trials_per_family={name:int(trials) for name in search_counts},
        actual_trials_per_family=actual_counts,trials_per_family=actual_counts,
        trained_on='train only',tuned_on='stratified CV inside train',
        threshold_and_feature_k_selected_on='validation only',test_used_after_selection_locked=True,
        split_counts={key:len(ix) for key,ix in splits.items()},
        split_class_counts=diagnostics['class_counts'],model_ids=list(models),
        source_files_sha256=identity['source_files_sha256'],artifact_hashes=hashes,run_dir=str(out))
    write_json(out/'manifest.json',manifest)
    print('A4 complete:',out)
    print('Data mode:',manifest['status'])
    print('Actual search configurations:',actual_counts)
    print('FINAL:',final_id,'threshold=',round(final_threshold,5),'selected_k=',selected_k)
    print('Validation AP:',round(val_AP[final_id],5))
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,help='Official diabetes_binary...csv')
    ap.add_argument('--output-dir',default='results/a4_run')
    ap.add_argument('--trials',type=int,default=10)
    ap.add_argument('--cv',type=int,default=3)
    ap.add_argument('--quick',action='store_true',help='260-row SMOKE ONLY, results not for research')
    ap.add_argument('--demo',action='store_true',help='Allow nonofficial CSV for code QA only')
    a=ap.parse_args()
    if not 1<=a.trials<=100:ap.error('--trials must be 1..100')
    if a.cv<2:ap.error('--cv must be >=2')
    run(a.data,a.output_dir,a.trials,a.cv,a.quick,a.demo)

if __name__=='__main__':main()
