"""Exact-X sensitivity: separate outer diagnostics and primary-training-only CV.

Adapted from the supplied reference. X equality does not establish a respondent
identity. GroupShuffleSplit ratios target groups and are approximate by rows.
There is no held-out group evaluation or pipeline-selection path in this module.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold

from .core import (SEED, TARGET, load_dataset, sha256_file, stratified_indices,
                   validate_split_indices, versions, write_json)
from .paths import COMPONENT_ROOT, PROTOCOL_PATH
from .provenance import (artifact_hashes, finish_execution, snapshot_protocol,
                         source_identity, start_execution, validate_protocol)
from .modeling import FoldAwareXGB
from .sensitivity_contract import validate_sensitivity_native, validate_sensitivity_wrapper
from xgboost import XGBClassifier
from .shared_splits import CANONICAL_DATASET_SHA256, LEGACY_QA_MODE, array_sha256, verify_shared_split

GROUP_DEFINITION = 'Exact equality of every input feature X, excluding y; not respondent identity'
OUTER_DESIGN_ID = 'exact-x-group-outer-v3'
CV_DESIGN_ID = 'exact-x-primary-train-cv-v3'


class FrozenSensitivityXGB(FoldAwareXGB):
    """Validate the approved wrapper and actual native settings before every fit."""
    def fit(self, X, y):
        validate_sensitivity_wrapper(self.get_params(deep=False))
        return super().fit(X, y)

    def _build_core(self, weight_ratio):
        core = super()._build_core(weight_ratio)
        if type(core) is not XGBClassifier:
            raise ValueError('sensitivity.native.family: expected XGBClassifier')
        validate_sensitivity_native(core.get_params(deep=False))
        return core


def frozen_sensitivity_pipeline(protocol):
    """Preflight protocol/amendment/native agreement without fitting or output writes."""
    from sklearn.pipeline import Pipeline
    validate_protocol(protocol)
    estimator = FrozenSensitivityXGB(**protocol['sensitivity']['frozen_estimator']['parameters'])
    validate_sensitivity_wrapper(estimator.get_params(deep=False))
    estimator._build_core(1.0)  # Check native defaults now, and again inside every fold fit.
    return Pipeline([('model', estimator)])


@dataclass(frozen=True)
class SplitSensitivity:
    stratified_cv_ap_mean: float
    stratified_cv_ap_std: float
    group_cv_ap_mean: float
    group_cv_ap_std: float
    delta_ap_group_minus_stratified: float
    n_rows: int
    n_unique_feature_groups: int
    n_folds: int
    positive_prevalence: float
    fold_details: list[dict]
    status: str = 'SENSITIVITY_EXPLORATORY'
    scope: str = 'primary_training_only'
    std_ddof: int = 0
    sign_convention: str = 'mean AP Group minus mean AP Stratified; positive favors Group'
    used_for_model_selection: bool = False

    def to_dict(self):
        return asdict(self)


def exact_feature_groups(X):
    """Factorize full tuples with equality checks, rather than treating a hash as identity."""
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError('X must be a nonempty pandas DataFrame')
    if X.columns.has_duplicates or X.isna().any().any() or TARGET in X.columns:
        raise ValueError('X needs unique feature names, no missing values and no target column')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('X must contain finite numeric features')
    return np.asarray(pd.factorize(pd.MultiIndex.from_frame(X), sort=False)[0], dtype=np.int64)


def validate_group_partitions(X, splits):
    checked = validate_split_indices(splits, len(X))
    groups = exact_feature_groups(X)
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        if np.intersect1d(groups[checked[left]],groups[checked[right]]).size:
            raise ValueError(f'Exact feature group crosses group partitions: {left}/{right}')
    return checked


def group_shuffle_split_indices(X, seed=SEED):
    groups = exact_feature_groups(X)
    if np.unique(groups).size < 8:
        raise ValueError('At least eight unique groups are required')
    train,rest = next(GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=seed).split(X,groups=groups))
    val_rel,test_rel = next(GroupShuffleSplit(n_splits=1,test_size=0.50,random_state=seed).split(
        X.iloc[rest],groups=groups[rest]))
    splits=dict(train=np.sort(train),validation=np.sort(rest[val_rel]),test=np.sort(rest[test_rel]))
    return validate_group_partitions(X,splits)


def _partition_counts(groups, y, indices):
    positive=int(y[indices].sum())
    return dict(rows=len(indices),groups=int(np.unique(groups[indices]).size),
                positive_count=positive,negative_count=len(indices)-positive,
                positive_prevalence=float(y[indices].mean()))


def group_partition_summary(X, y, splits):
    splits=validate_group_partitions(X,splits)
    y=np.asarray(y)
    if y.ndim!=1 or len(y)!=len(X) or not set(np.unique(y)).issubset({0,1}):
        raise ValueError('Original binary labels must align with X')
    groups=exact_feature_groups(X)
    return dict(status='SENSITIVITY_OUTER_DIAGNOSTICS_ONLY',design_id=OUTER_DESIGN_ID,
                group_definition=GROUP_DEFINITION,seed=SEED,
                requested_ratios=[0.70,0.15,0.15],ratios_target='numbers of groups',
                row_ratios_are_approximate=True,total_rows=len(X),total_groups=int(np.unique(groups).size),
                partitions={key:_partition_counts(groups,y,ix) for key,ix in splits.items()},
                actual_row_ratios={key:len(ix)/len(X) for key,ix in splits.items()},
                groups_cross_partitions={'train__validation':0,'train__test':0,'validation__test':0},
                heldout_evaluation=False,performance_metrics_computed=False,
                heldout_design_requirement='Any group-heldout evaluation requires a separately prespecified design and run ID')


def sensitivity_cv_plans(X_train,y_train,n_folds=3,seed=SEED):
    y=np.asarray(y_train)
    if (isinstance(n_folds,bool) or not isinstance(n_folds,int) or n_folds<2
            or y.ndim!=1 or len(y)!=len(X_train) or set(np.unique(y))!={0,1}):
        raise ValueError('CV requires aligned binary training labels and at least two integer folds')
    groups=exact_feature_groups(X_train)
    if np.unique(groups).size<n_folds or min(np.bincount(y.astype(int)))<n_folds:
        raise ValueError('Too few training feature groups or minority-class rows for the requested CV')
    plans=dict(
        stratified=list(StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=seed).split(X_train,y)),
        group=list(StratifiedGroupKFold(n_splits=n_folds,shuffle=True,random_state=seed).split(X_train,y,groups)))
    for mode,folds in plans.items():
        valid_groups=[]
        for fold,(train,valid) in enumerate(folds):
            if (np.intersect1d(train,valid).size
                    or not np.array_equal(np.sort(np.concatenate([train,valid])),np.arange(len(y)))):
                raise ValueError(f'{mode} fold {fold} has overlapping/incomplete row partitions')
            if len(np.unique(y[train]))!=2 or len(np.unique(y[valid]))!=2:
                raise ValueError(f'{mode} fold {fold} is single-class; no fold is silently excluded')
            if mode=='group' and np.intersect1d(groups[train],groups[valid]).size:
                raise ValueError('Exact feature group leaked within group-CV fold')
            valid_groups.append(np.unique(groups[valid]))
        if not np.array_equal(np.sort(np.concatenate([valid for _,valid in folds])),np.arange(len(y))):
            raise ValueError('CV validation folds overlap or fail to cover primary training')
        if mode=='group' and any(np.intersect1d(valid_groups[i],valid_groups[j]).size
                                for i in range(n_folds) for j in range(i+1,n_folds)):
            raise ValueError('Exact feature group crosses group-CV validation folds')
    return plans


def compare_train_only_cv_ap(X_train,y_train,estimator,n_folds=3,seed=SEED,*,source_row_indices=None):
    """Training-only comparison; generic non-XGBoost estimators support synthetic QA.

    The scientific XGBoost path requires the guarded frozen pipeline. This helper
    grants no authorization to run an alternative real-data estimator.
    """
    from sklearn.pipeline import Pipeline
    steps = estimator.steps if isinstance(estimator, Pipeline) else [('model', estimator)]
    xgb_steps = [model for _, model in steps if isinstance(model, FoldAwareXGB)]
    for model in xgb_steps:
        validate_sensitivity_wrapper(model.get_params(deep=False))
        if type(model) is not FrozenSensitivityXGB or len(steps) != 1:
            raise ValueError('sensitivity.wrapper: use the guarded frozen sensitivity pipeline without added preprocessing')
        model._build_core(1.0)
        if n_folds != 3 or seed != SEED:
            raise ValueError('sensitivity.cv: frozen XGBoost requires three folds and seed 42')
    y=np.asarray(y_train)
    plans=sensitivity_cv_plans(X_train,y,n_folds,seed)
    groups=exact_feature_groups(X_train)
    source_ids=np.arange(len(y)) if source_row_indices is None else np.asarray(source_row_indices)
    if (source_ids.ndim!=1 or len(source_ids)!=len(y) or source_ids.dtype.kind not in 'iu'
            or (source_ids<0).any() or len(np.unique(source_ids))!=len(source_ids)):
        raise ValueError('Training source-row IDs must be aligned unique original positions')
    details=[];results={}
    for mode,folds in plans.items():
        aps=[]
        for fold,(train,valid) in enumerate(folds):
            fitted=clone(estimator).fit(X_train.iloc[train],y[train])
            prob=np.asarray(fitted.predict_proba(X_train.iloc[valid]))
            if (list(np.asarray(fitted.classes_))!=[0,1] or prob.shape!=(len(valid),2)
                    or not np.isfinite(prob).all() or (prob<0).any() or (prob>1).any()):
                raise ValueError('Frozen sensitivity estimator must return binary probabilities')
            ap=float(average_precision_score(y[valid],prob[:,1]))
            aps.append(ap)
            tr,va=_partition_counts(groups,y,train),_partition_counts(groups,y,valid)
            details.append(dict(method=mode,fold=fold,ap=ap,
                n_train=tr['rows'],n_valid=va['rows'],train_groups=tr['groups'],valid_groups=va['groups'],
                train_positive_count=tr['positive_count'],valid_positive_count=va['positive_count'],
                train_positive_prevalence=tr['positive_prevalence'],valid_positive_prevalence=va['positive_prevalence'],
                groups_cross_train_valid=int(np.intersect1d(groups[train],groups[valid]).size),
                train_source_indices_sha256=array_sha256(source_ids[train]),
                valid_source_indices_sha256=array_sha256(source_ids[valid])))
        results[mode]=(float(np.mean(aps)),float(np.std(aps,ddof=0)))
    return SplitSensitivity(
        stratified_cv_ap_mean=results['stratified'][0],stratified_cv_ap_std=results['stratified'][1],
        group_cv_ap_mean=results['group'][0],group_cv_ap_std=results['group'][1],
        delta_ap_group_minus_stratified=results['group'][0]-results['stratified'][0],
        n_rows=len(y),n_unique_feature_groups=int(np.unique(groups).size),n_folds=n_folds,
        positive_prevalence=float(y.mean()),fold_details=details)


def run(data_path,output_dir,*,train_cv=False,demo=False,shared_indices=None,shared_manifest=None):
    protocol = yaml.safe_load(PROTOCOL_PATH.read_bytes())
    validate_protocol(protocol)
    estimator = frozen_sensitivity_pipeline(protocol) if train_cv else None
    execution,started=start_execution('group_sensitivity')
    data_path=Path(data_path).expanduser().resolve()
    out=Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Group sensitivity requires a new output directory')
    digest=sha256_file(data_path)
    X,y=load_dataset(data_path)
    if sha256_file(data_path)!=digest:
        raise ValueError('CSV changed while loading sensitivity input')
    if demo:
        if digest==CANONICAL_DATASET_SHA256:
            raise ValueError('--demo is synthetic QA, not the original official population')
        primary=stratified_indices(y)
        primary_record=dict(mode=LEGACY_QA_MODE,status='SYNTHETIC_QA_ONLY',
                            normalized_array_sha256={k:array_sha256(v) for k,v in primary.items()})
    else:
        primary,primary_record=verify_shared_split(data_path,shared_indices,shared_manifest)
    outer=group_shuffle_split_indices(X)
    outer_summary=group_partition_summary(X,y,outer)
    train_ids=primary['train']
    Xtrain=X.iloc[train_ids].reset_index(drop=True)
    ytrain=y.iloc[train_ids].to_numpy()
    plans=sensitivity_cv_plans(Xtrain,ytrain) if train_cv else None
    out.mkdir(parents=True,exist_ok=True)
    protocol,protocol_hash=snapshot_protocol(out)
    identity=source_identity()
    write_json(out/'source_identity.json',identity)
    write_json(out/'versions.json',versions())
    write_json(out/'primary_split_identity.json',primary_record)
    np.savez_compressed(out/'group_outer_indices.npz',**outer)
    np.save(out/'source_group_ids.npy',exact_feature_groups(X),allow_pickle=False)
    np.save(out/'primary_train_source_ids.npy',train_ids,allow_pickle=False)
    write_json(out/'outer_partition_diagnostics.json',outer_summary)
    report=None
    frozen=protocol['sensitivity']['frozen_estimator']
    if train_cv:
        from .modeling import model_configuration
        write_json(out/'frozen_estimator.json',model_configuration({'sensitivity':estimator}))
        arrays={f'{mode}_fold_{fold}_{part}_source_ids':train_ids[ix]
                for mode,folds in plans.items() for fold,pair in enumerate(folds)
                for part,ix in zip(('train','validation'),pair)}
        np.savez_compressed(out/'primary_train_cv_indices.npz',**arrays)
        report=compare_train_only_cv_ap(Xtrain,ytrain,estimator,source_row_indices=train_ids)
        write_json(out/'train_cv_ap_sensitivity.json',report.to_dict())
        pd.DataFrame(report.fold_details).to_csv(out/'train_cv_fold_details.csv',index=False)
    if sha256_file(data_path)!=digest or source_identity()['file_hashes']!=identity['file_hashes']:
        raise ValueError('Sensitivity data/source changed during execution')
    manifest=dict(**finish_execution(execution,started),manifest_version=3,
        status='SYNTHETIC_QA_ONLY' if demo else 'SENSITIVITY_EXPLORATORY',
        source_sha256=digest,source_files_sha256=identity['source_files_sha256'],
        protocol_sha256=protocol_hash,outer_design_id=OUTER_DESIGN_ID,cv_design_id=CV_DESIGN_ID,
        group_definition=GROUP_DEFINITION,primary_train_rows=len(train_ids),
        primary_train_groups=int(np.unique(exact_feature_groups(Xtrain)).size),
        primary_train_positive_prevalence=float(ytrain.mean()),train_cv_performed=bool(train_cv),
        cv_folds=3,frozen_estimator=frozen,heldout_group_evaluation=False,
        no_pipeline_selection=True,historical_test_exposure_acknowledged=True,
        outer_split_ratios_approximate_by_rows=True,artifact_hashes=artifact_hashes(out))
    write_json(out/'manifest_group_sensitivity.json',manifest)
    print('Group sensitivity complete:',out)
    print('Status:',manifest['status'],'training CV:',bool(train_cv),'group-heldout evaluation: False')
    return dict(manifest=manifest,outer=outer_summary,cv=None if report is None else report.to_dict())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output-dir',default=COMPONENT_ROOT/'results'/'group_sensitivity_v3_run')
    parser.add_argument('--shared-split-indices',type=Path)
    parser.add_argument('--shared-split-manifest',type=Path)
    parser.add_argument('--train-cv',action='store_true',help='Fit the frozen XGBoost for 3 folds per CV method on primary train only')
    parser.add_argument('--demo',action='store_true',help='Synthetic QA only; explicitly labeled legacy primary split')
    args=parser.parse_args()
    run(args.data,args.output_dir,train_cv=args.train_cv,demo=args.demo,
        shared_indices=args.shared_split_indices,shared_manifest=args.shared_split_manifest)


if __name__=='__main__':
    main()
