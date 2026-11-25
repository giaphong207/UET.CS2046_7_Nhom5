"""A4 V3: AP-ranked XGBoost/baselines and validation positive-class F2.

No test labels/scores are used for model, hyperparameter, feature-count or threshold
selection. Test runs once after the validation decisions have been locked.
Primary CV is stratified 3-fold, seed 42; F2 exact ties favor the smallest grid
threshold. The primary selector is implemented; official evaluation release
remains separate. ANOVA is a separately saved synthetic/quick QA experiment.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .core import (TARGET, SEED, CORE_BOOTSTRAP_PAIRS, FEATURE_COUNTS, FINAL_MODEL_TIE_RULE,
                   dataset_audit, feature_selection_policy, f2_threshold_curve, load_dataset,
                   metrics_at_threshold, optimal_f2_threshold, threshold_policy,
                   select_feature_budget, select_final_model, sha256_file, split_diagnostics,
                   stratified_indices, versions, write_json)
from .provenance import (A4_CRITICAL_ARTIFACTS, MANIFEST_VERSION, SELECTION_LOCK_ARTIFACTS, artifact_hashes,
                         finish_execution, require_official_release, snapshot_protocol, source_identity, start_execution)
from .paths import COMPONENT_ROOT
from .shared_splits import (CANONICAL_DATASET_SHA256, LEGACY_QA_MODE, SHARED_MODE,
                            array_sha256, verify_shared_split)

class FoldAwareXGB(ClassifierMixin, BaseEstimator):
    """Calculate `scale_pos_weight` on labels passed to *each* fit, not all CV labels.

    Useful for a fair comparison between weighted and unweighted XGBoost.
    """
    def __init__(self, n_estimators=160, max_depth=4, learning_rate=0.08,
                 min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, weight_mode='none', n_jobs=2, random_state=SEED,
                 importance_type='gain'):
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
        self.importance_type=importance_type

    def fit(self, X, y):
        y=np.asarray(y).astype(int)
        if set(np.unique(y))!={0,1}:
            raise ValueError('XGBoost train fold must include both classes')
        ratio=(y==0).sum()/(y==1).sum()
        if self.weight_mode not in ('balanced','none'):
            raise ValueError('weight_mode must be balanced or none')
        self.weight_ratio_ = float(ratio if self.weight_mode=='balanced' else 1.0)
        self.core_=self._build_core(self.weight_ratio_)
        self.core_.fit(X,y)
        self.classes_=self.core_.classes_
        self.n_features_in_=X.shape[1]
        self.n_fit_samples_=len(y)
        self.fit_label_sha256_=array_sha256(y)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.fit_source_row_indices_=np.asarray(X.index,dtype=np.int64).copy()
        elif hasattr(self, 'feature_names_in_'):
            del self.feature_names_in_
        return self

    def _build_core(self, weight_ratio):
        """Construct the native estimator without fitting or changing its settings."""
        return XGBClassifier(n_estimators=self.n_estimators,
            max_depth=self.max_depth,learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,reg_lambda=self.reg_lambda,
            scale_pos_weight=weight_ratio,tree_method='hist',
            eval_metric='logloss',random_state=self.random_state,
            n_jobs=self.n_jobs,importance_type=self.importance_type)

    def predict_proba(self,X):
        return self.core_.predict_proba(X)

    def predict(self,X):
        return self.core_.predict(X)

    @property
    def feature_importances_(self):
        return self.core_.feature_importances_


class ExactTopKSelectFromModel(SelectFromModel):
    """sklearn SelectFromModel with explicit stable ties and exact support size.

    Gain scores are nonnegative. The -inf threshold admits even zero-gain
    columns; stable descending sorting resolves ties by original column order.
    """
    def __init__(self, estimator, k):
        self.k=k
        super().__init__(estimator=estimator,threshold=-np.inf,max_features=k,
                         prefit=False,importance_getter='feature_importances_',norm_order=1)

    def fit(self, X, y=None, **fit_params):
        if (isinstance(self.k,(bool,np.bool_)) or not isinstance(self.k,(int,np.integer))
                or not 1 <= self.k <= X.shape[1]):
            raise ValueError('Selector k must be an integer within the input feature count')
        if (not isinstance(X,pd.DataFrame) or not X.columns.is_unique
                or X.index.dtype.kind not in 'iu' or not X.index.is_unique):
            raise ValueError('Primary selector requires unique columns and original integer source-row IDs')
        super().fit(X,y,**fit_params)
        self.fit_source_row_indices_=X.index.to_numpy(dtype=np.int64,copy=True)
        self.get_support()  # Fail immediately on invalid ranking/settings.
        return self

    def _get_support_mask(self):
        if (self.prefit or self.threshold != -np.inf or self.max_features != self.k
                or self.importance_getter != 'feature_importances_' or self.norm_order != 1):
            raise ValueError('Primary selector settings differ from exact top-k SelectFromModel')
        scores=np.asarray(self.estimator_.feature_importances_,dtype=float)
        if (scores.shape != (self.n_features_in_,) or not np.isfinite(scores).all()
                or (scores < 0).any()):
            raise ValueError('Gain importance must be finite, nonnegative and match original columns')
        support=np.zeros(len(scores),dtype=bool)
        support[np.argsort(-scores,kind='stable')[:self.k]]=True
        return support


def make_primary_pipeline(b3_params, k):
    params={**b3_params,'weight_mode':'balanced','importance_type':'gain'}
    selector=ExactTopKSelectFromModel(FoldAwareXGB(**params),k=k)
    selector.set_output(transform='pandas')
    return Pipeline([('selection',selector),('model',FoldAwareXGB(**params))])


def fit_primary_pipeline(pipeline, X, y, allowed_source_ids):
    """Check exact ordered fit scope before either ranking or downstream fitting."""
    if not np.array_equal(np.asarray(X.index),np.asarray(allowed_source_ids)) or len(X) != len(y):
        raise ValueError('Primary selector fit rows differ from the allowed training fold')
    pipeline.fit(X,y)
    return pipeline


def primary_ranking_record(pipeline):
    selector=pipeline.named_steps['selection']
    ranker=selector.estimator_
    scores=np.asarray(ranker.feature_importances_,dtype=float)
    names=list(selector.feature_names_in_)
    order=np.argsort(-scores,kind='stable')
    support=selector.get_support()
    return dict(input_feature_names=names,ranking_scores=scores.tolist(),
                ranked_feature_names=list(np.asarray(names)[order]),
                selected_feature_names=list(np.asarray(names)[support]),
                support=support.tolist(),requested_k=int(selector.k),actual_feature_count=int(support.sum()),
                fit_source_indices_sha256=array_sha256(selector.fit_source_row_indices_),
                ranker_weight_ratio=ranker.weight_ratio_,
                downstream_weight_ratio=pipeline.named_steps['model'].weight_ratio_,
                configuration=model_configuration({'pipeline':pipeline})['pipeline'])


def evaluate_primary_feature_cv(X, y, b3_params, cv, *, run_id, identity, dataset_hash, split_hash, cv_hash):
    """Only primary-training data enters this diagnostic; no reserved labels accepted."""
    records=[]; rows=[]; fitted={}
    for k in FEATURE_COUNTS:
        for fold,(tr,va) in enumerate(cv.split(X,y)):
            train_ids=X.index.to_numpy()[tr]; valid_ids=X.index.to_numpy()[va]
            model=make_primary_pipeline(b3_params,k)
            started=time.perf_counter()
            fit_primary_pipeline(model,X.iloc[tr],y.iloc[tr],train_ids)
            ap=float(average_precision_score(y.iloc[va],model.predict_proba(X.iloc[va])[:,1]))
            key=f'A3_k{k}_fold_{fold}'
            fitted[key]=model
            record=dict(model=f'A3_k{k}',fold=fold,scope='primary_training_cv',
                        train_source_row_indices=train_ids.tolist(),validation_source_row_indices=valid_ids.tolist(),
                        validation_source_indices_sha256=array_sha256(valid_ids),
                        ap=ap,duration_seconds=time.perf_counter()-started,**primary_ranking_record(model))
            records.append(record)
            rows.append(dict(model=record['model'],fold=fold,requested_k=k,actual_feature_count=record['actual_feature_count'],
                             average_precision=ap,train_rows=len(tr),validation_rows=len(va),
                             ranker_weight_ratio=record['ranker_weight_ratio'],downstream_weight_ratio=record['downstream_weight_ratio']))
    context=dict(run_id=run_id,source_files_sha256=identity['source_files_sha256'],dataset_sha256=dataset_hash,
                 split_archive_sha256=split_hash,cv_fold_archive_sha256=cv_hash,policy=feature_selection_policy())
    return dict(**context,records=records,successful_folds_per_k={str(k):cv.n_splits for k in FEATURE_COUNTS}), pd.DataFrame(rows), dict(**context,models=fitted)


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
    def normalize(value):
        if isinstance(value,BaseEstimator):
            return dict(type=type(value).__name__,parameters=normalize(value.get_params(deep=False)))
        if isinstance(value,dict):
            return {key:normalize(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)):
            return [normalize(item) for item in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value):
            return '-inf' if value == -np.inf else ('inf' if value == np.inf else 'nan')
        if isinstance(value,np.generic):
            return value.item()
        return value.__name__ if callable(value) else value
    result = {}
    for name, pipeline in models.items():
        if isinstance(pipeline, Pipeline):
            steps = {}
            for step_name, estimator in pipeline.named_steps.items():
                params = estimator.get_params(deep=False)
                if isinstance(estimator,SelectFromModel):
                    params.update(threshold=estimator.threshold,max_features=estimator.max_features,
                                  prefit=estimator.prefit,importance_getter=estimator.importance_getter,
                                  norm_order=estimator.norm_order)
                params=normalize(params)
                steps[step_name] = dict(type=type(estimator).__name__, parameters=params)
            result[name] = steps
        else:
            result[name] = dict(type=type(pipeline).__name__, parameters=pipeline.get_params(deep=False))
    return result


def run(data_path,output_dir, trials=10, cv_folds=3, quick=False, demo=False,*,
        shared_indices=None,shared_manifest=None,split_mode=None):
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
    if demo and source_hash==CANONICAL_DATASET_SHA256 and not quick:
        raise ValueError('--demo is for synthetic QA; it cannot release full official training/evaluation')
    if not (quick or demo) and requested_cv != 3:
        raise ValueError('Primary A4 requires stratified 3-fold CV; fold overrides are QA only')
    split_mode=split_mode or (LEGACY_QA_MODE if (quick or demo) else SHARED_MODE)
    if split_mode not in (SHARED_MODE,LEGACY_QA_MODE):
        raise ValueError('Unknown split mode')
    if split_mode==LEGACY_QA_MODE and not (quick or demo):
        raise ValueError('The old test-first sorted split is a specifically labeled QA mode only')
    if split_mode==SHARED_MODE and quick:
        raise ValueError('Canonical full-population shared splits cannot index a quick subset')
    if not (quick or demo):
        require_official_release()
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
    if split_mode==SHARED_MODE:
        splits,split_record=verify_shared_split(data_path,shared_indices,shared_manifest)
    else:
        splits=stratified_indices(y)
        split_record=dict(status='QA_ONLY',mode=LEGACY_QA_MODE,
                          recipe='V2 test-first round(0.15*N), validation second; sorted local indices',
                          index_space='positions in source_row_indices.npy',
                          shared_archive_sha256=None,input_manifest_sha256=None,
                          verified_B_v9_recipe_and_order=False,
                          normalized_array_sha256={key:array_sha256(ix) for key,ix in splits.items()})
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
    if split_mode==SHARED_MODE:
        shutil.copyfile(split_record['archive_path'],out/'split_indices.npz')
        shutil.copyfile(split_record['input_manifest_path'],out/'shared_split_manifest.json')
    else:
        np.savez_compressed(out/'split_indices.npz',**splits)
    split_record['archive_sha256']=sha256_file(out/'split_indices.npz')
    split_record['shared_archive_sha256']=(split_record['archive_sha256'] if split_mode==SHARED_MODE else None)
    split_record['class_counts']=diagnostics['class_counts']
    write_json(out/'split_manifest.json',split_record)
    amendments=json.loads((out/'protocol_amendments.json').read_text())
    write_json(out/'test_exposure_log.json',amendments['test_exposure'])
    (out/'dataset_sha256.txt').write_text(source_hash+'\n',encoding='ascii')
    write_json(out/'dataset_audit.json',audit)
    write_json(out/'versions.json',versions())
    write_json(out/'source_identity.json',identity)

    spaces=random_spaces(quick)
    cv=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=SEED)
    cv_arrays={}
    train_source_ids=source_indices[splits['train']]
    Xt=Xt.copy()
    Xt.index=train_source_ids
    for fold,(tr,va) in enumerate(cv.split(Xt,yt)):
        cv_arrays[f'fold_{fold}_train']=train_source_ids[tr]
        cv_arrays[f'fold_{fold}_validation']=train_source_ids[va]
    np.savez_compressed(out/'cv_fold_indices.npz',**cv_arrays)
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
    chosen_thresholds={key:optimal_f2_threshold(yv,p) for key,p in val_prob.items()}
    result_rows=[]
    for key,p in val_prob.items():
        threshold=chosen_thresholds[key] if key=='A2' else 0.5
        result_rows.append({'model':key,'split':'validation',
                            **metrics_at_threshold(yv,p,threshold)})
    write_json(out/'a2_reference_lock.json',dict(
        run_id=execution['run_id'],source_files_sha256=identity['source_files_sha256'],
        model='A2',average_precision=float(average_precision_score(yv,val_prob['A2'])),
        train_source_indices_sha256=array_sha256(train_source_ids),
        validation_source_indices_sha256=array_sha256(source_indices[splits['validation']]),
        locked_before_primary_selector=True))
    fold_record,fold_table,fold_models=evaluate_primary_feature_cv(
        Xt,yt,xgb_params,cv,run_id=execution['run_id'],identity=identity,
        dataset_hash=source_hash,split_hash=split_record['archive_sha256'],
        cv_hash=sha256_file(out/'cv_fold_indices.npz'))
    write_json(out/'feature_selection_folds.json',fold_record)
    fold_table.to_csv(out/'feature_selection_cv.csv',index=False)
    joblib.dump(fold_models,out/'feature_selection_models.joblib',compress=3)
    full_rankings={}
    # Fresh ranker and downstream fit on full primary train for every k.
    for k in FEATURE_COUNTS:
        m=fit_primary_pipeline(make_primary_pipeline(xgb_params,k),Xt,yt,train_source_ids)
        key=f'A3_k{k}'
        models[key]=m
        p=m.predict_proba(Xv)[:,1]
        val_prob[key]=p
        chosen_thresholds[key]=optimal_f2_threshold(yv,p)
        result_rows.append({'model':key,'split':'validation',
                            **metrics_at_threshold(yv,p,chosen_thresholds[key])})
        full_rankings[key]=dict(scope='full_primary_training_refit',**primary_ranking_record(m))
    write_json(out/'feature_rankings.json',dict(
        run_id=execution['run_id'],source_files_sha256=identity['source_files_sha256'],
        policy=feature_selection_policy(),models=full_rankings))
    # Preserve ANOVA exclusively as a separate QA experiment; it never enters FINAL.
    secondary=dict(run_id=execution['run_id'],scope='SECONDARY_QA_ONLY',method='SelectKBest(f_classif)',
                   training_cv_performed=False,used_for_primary_selection=False,models={})
    if quick or demo:
        for k in FEATURE_COUNTS:
            m=Pipeline([('selection',SelectKBest(f_classif,k=k)),
                        ('model',FoldAwareXGB(**{**xgb_params,'weight_mode':'balanced'}))]).fit(Xt,yt)
            p=m.predict_proba(Xv)[:,1]
            secondary['models'][f'ANOVA_k{k}']=dict(
                selected_feature_names=list(Xt.columns[m.named_steps['selection'].get_support()]),
                actual_feature_count=k,**metrics_at_threshold(yv,p,optimal_f2_threshold(yv,p)))
        secondary_k,secondary_eligible=select_feature_budget(
            float(average_precision_score(yv,val_prob['A2'])),
            {k:secondary['models'][f'ANOVA_k{k}']['average_precision'] for k in FEATURE_COUNTS})
        secondary.update(selected_k=secondary_k,eligible_feature_counts=secondary_eligible)
    write_json(out/'secondary_anova.json',secondary)
    validation=pd.DataFrame(result_rows)
    validation.to_csv(out/'validation_metrics.csv',index=False)
    pd.DataFrame([dict(model=key,split='validation',**metrics_at_threshold(yv,p,0.5))
                  for key,p in val_prob.items()]).to_csv(out/'validation_reference_metrics.csv',index=False)
    threshold_search={}
    for key,p in val_prob.items():
        curve=f2_threshold_curve(yv,p)
        threshold_search[key]=dict(f2_values=curve.tolist(),winner_grid_index=int(np.argmax(curve)),
                                   selected_threshold=chosen_thresholds[key],max_f2=float(curve.max()))
    write_json(out/'threshold_search.json',dict(run_id=execution['run_id'],
                                               policy=threshold_policy(),models=threshold_search))
    base_ap=float(validation.loc[validation.model=='A2','average_precision'].iloc[0])
    feature_ap={f'A3_k{k}':float(validation.loc[
        validation.model==f'A3_k{k}','average_precision'].iloc[0]) for k in FEATURE_COUNTS}
    selected_k,eligible=select_feature_budget(base_ap,{k:feature_ap[f'A3_k{k}'] for k in FEATURE_COUNTS})
    selected_arm=f'A3_k{selected_k}' if eligible else 'A2'
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
        validation_f2_thresholds=chosen_thresholds, test_thresholds=test_thresholds,
        threshold_policy=threshold_policy(),threshold_selection='max positive-class F2 on validation; exact ties smallest threshold',
        feature_selection=feature_selection_policy(),primary_feature_selection_status='IMPLEMENTED',
        feature_selection_cv_performed=True,selected_reduction_model=selected_arm,
        separately_reported_a3_model=f'A3_k{selected_k}',full_feature_fallback='A2',
        interpretation=('SMOKE NOT RESEARCH' if quick else ('SYNTHETIC DEMO' if demo else 'LOCKED TEST PROTOCOL')),
        note='Do not change choices based on test scores.')
    write_json(out/'selection.json',selection)
    joblib.dump(dict(
        schema_version=MANIFEST_VERSION, run_id=execution['run_id'],
        source_files_sha256=identity['source_files_sha256'],
        models=models, model_ids=list(models), final_model_id=final_id,
        final_threshold=final_threshold, feature_names=input_features,
        model_feature_sets=model_feature_sets, selection=selection),
        out/'fitted_models.joblib',compress=3)
    lock=dict(run_id=execution['run_id'],locked_at_utc=datetime.now(timezone.utc).isoformat(),
              dataset_sha256=source_hash,threshold_policy=threshold_policy(),
              bootstrap_comparisons=[dict(id=key,left=left,right=right) for key,left,right in CORE_BOOTSTRAP_PAIRS],
              feature_selection_status='IMPLEMENTED',
              artifact_hashes=artifact_hashes(out,SELECTION_LOCK_ARTIFACTS))
    lock['lock_id']=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(out/'selection_lock.json',lock)
    if artifact_hashes(out,SELECTION_LOCK_ARTIFACTS)!=lock['artifact_hashes']:
        raise ValueError('Frozen A4 artifacts changed before test evaluation')

    # All model/k/threshold decisions above are now locked. No test-based refit/selection.
    rows=[];reference_rows=[]
    preds=pd.DataFrame({'row_index':source_indices[splits['test']],'y_true':ytest.to_numpy()})
    for key in test_model_ids:
        prob=models[key].predict_proba(Xtest)[:,1]
        preds[f'prob_{key}']=prob
        rows.append({'model':key,'split':'test',
                     **metrics_at_threshold(ytest,prob,test_thresholds[key])})
        reference_rows.append(dict(model=key,split='test',**metrics_at_threshold(ytest,prob,0.5)))
    preds['prob_FINAL']=models[final_id].predict_proba(Xtest)[:,1]
    rows.append({'model':'FINAL','split':'test',
                 **metrics_at_threshold(ytest,preds['prob_FINAL'],final_threshold)})
    reference_rows.append(dict(model='FINAL',split='test',
                               **metrics_at_threshold(ytest,preds['prob_FINAL'],0.5)))
    test_metrics=pd.DataFrame(rows)
    test_metrics.to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(reference_rows).to_csv(out/'test_reference_metrics.csv',index=False)
    preds.to_csv(out/'test_predictions.csv',index=False)
    ablation=pd.concat([validation,test_metrics],ignore_index=True)
    ablation['selected_feature_count']=[
        len(model_feature_sets[final_id if key=='FINAL' else key]) for key in ablation.model]
    ablation.to_csv(out/'ablation.csv',index=False)
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
            quick=bool(quick),demo=bool(demo),split_mode=split_mode,
            shared_indices=None if shared_indices is None else str(shared_indices),
            shared_manifest=None if shared_manifest is None else str(shared_manifest)),
        effective=dict(
            mode=status,target=TARGET,split=dict(ratios=[0.70,0.15,0.15],stratify=True,
                mode=split_mode,recipe=split_record['recipe'],
                source_rows=source_rows,modeling_rows=len(X),index_space=split_record['index_space'],
                archive_sha256=split_record['archive_sha256'],
                shared_archive_sha256=split_record['shared_archive_sha256'],
                input_manifest_sha256=split_record.get('input_manifest_sha256'),
                normalized_array_sha256=split_record['normalized_array_sha256']),
            cross_validation=dict(type='StratifiedKFold',folds=int(cv_folds),shuffle=True,
                                  scoring='average_precision',n_jobs=1),
            search_backend='RandomizedSearchCV',search_spaces=spaces,
            fitted_model_configuration=model_configuration(models),
            xgboost_fixed_configuration=dict(tree_method='hist',eval_metric='logloss'),
            weighting='A1/A2/A3 balanced on labels available to each fit; B3 search unweighted',
            feature_selection=feature_selection_policy(),threshold=threshold_policy(),
            explanations=protocol['explanations'],
            group_sensitivity_mode='separate_train_only_cv_and_outer_diagnostics',
            bootstrap_comparisons=[dict(id=key,left=left,right=right) for key,left,right in CORE_BOOTSTRAP_PAIRS],
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
        protocol='A2_protocol_v3',protocol_sha256=protocol_hash,status=status,
        target=TARGET,seed=SEED,source_file=str(data_path),dataset_sha256=source_hash,
        source_rows=source_rows,total_rows=int(len(X)),features=input_features,
        split_ratios=[0.70,0.15,0.15],cv_folds=int(cv_folds),
        split_mode=split_mode,split_archive_sha256=split_record['archive_sha256'],
        shared_split_archive_sha256=split_record['shared_archive_sha256'],
        shared_split_manifest_sha256=split_record.get('input_manifest_sha256'),
        threshold_policy=threshold_policy(),feature_selection_status='IMPLEMENTED',
        primary_selector_cv_fits=len(fold_record['records']),primary_selector_refit_fits=len(FEATURE_COUNTS),
        shap_samples=protocol['explanations']['shap']['n'],
        shap_background_samples=protocol['explanations']['shap']['background_n'],
        shap_configuration=protocol['explanations']['shap'],
        group_sensitivity_mode='separate_train_only_cv_and_outer_diagnostics',
        group_sensitivity_performed=False,selection_lock_id=lock['lock_id'],
        bootstrap_comparisons=[dict(id=key,left=left,right=right) for key,left,right in CORE_BOOTSTRAP_PAIRS],
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
    ap.add_argument('--output-dir',default=COMPONENT_ROOT/'results'/'a4_v3_run')
    ap.add_argument('--trials',type=int,default=10)
    ap.add_argument('--cv',type=int,default=3,help='3 stratified folds on primary train; overrides are QA only')
    ap.add_argument('--shared-split-indices',type=Path,help='Exact B v9 original-row NPZ; verified before use')
    ap.add_argument('--shared-split-manifest',type=Path,help='Manifest bound to the exact CSV/archive hashes')
    ap.add_argument('--split-mode',choices=[SHARED_MODE,LEGACY_QA_MODE],
                    help='Default: canonical shared B v9; quick/demo use explicitly labeled V2 QA mode')
    ap.add_argument('--quick',action='store_true',help='260-row SMOKE ONLY, results not for research')
    ap.add_argument('--demo',action='store_true',help='Allow nonofficial CSV for code QA only')
    a=ap.parse_args()
    if not 1<=a.trials<=100:ap.error('--trials must be 1..100')
    if a.cv<2:ap.error('--cv must be >=2')
    run(a.data,a.output_dir,a.trials,a.cv,a.quick,a.demo,
        shared_indices=a.shared_split_indices,shared_manifest=a.shared_split_manifest,split_mode=a.split_mode)

if __name__=='__main__':main()
