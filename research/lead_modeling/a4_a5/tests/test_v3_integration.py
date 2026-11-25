"""V3 scientific/integrity checks on synthetic data; no official model scores."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from diabetes_research import analysis, group_sensitivity, modeling, shared_splits
from diabetes_research.artifact_validation import validate_a4_artifacts
from diabetes_research.core import (
    CORE_BOOTSTRAP_PAIRS, F2_GRID, TARGET, f2_threshold_curve, metrics_at_threshold,
    optimal_f1_threshold, optimal_f2_threshold, stratified_indices, write_json,
)
from diabetes_research.paths import COMPONENT_ROOT, PROTOCOL_PATH, REPOSITORY_ROOT
from diabetes_research.provenance import A4_CRITICAL_ARTIFACTS, artifact_hashes, read_json, source_identity
from test_audit_regressions import refresh_hashes
from test_protocol import toy_data


def grouped_sample():
    X=pd.DataFrame({'a':np.repeat(np.arange(100),3),'b':np.repeat(np.arange(100)%7,3)})
    y=(X.a.to_numpy()%4==0).astype(int)
    y[1]=1-y[0]
    return X,y


def test_exact_groups_ignore_conflicting_labels_and_dataframe_index():
    X,y=grouped_sample()
    X=X.astype(float)
    X.index=np.arange(10000,10300)
    groups=group_sensitivity.exact_feature_groups(X)
    assert len(np.unique(groups))==100
    assert groups[0]==groups[1]==groups[2] and y[0]!=y[1]
    X.loc[X.index[2],'b']=0.001
    changed=group_sensitivity.exact_feature_groups(X)
    assert changed[2]!=changed[0]


@pytest.mark.parametrize('problem',['target','missing','infinite','duplicate_names'])
def test_group_identity_rejects_invalid_feature_inputs(problem):
    X,y=grouped_sample()
    X=X.astype(float)
    if problem=='target': X[TARGET]=y
    elif problem=='missing': X.loc[0,'a']=np.nan
    elif problem=='infinite': X.loc[0,'a']=np.inf
    else: X.columns=['a','a']
    with pytest.raises(ValueError):
        group_sensitivity.exact_feature_groups(X)


def test_unequal_size_group_outer_split_reports_actual_counts_and_no_crossings():
    values=np.repeat(np.arange(30),np.arange(1,31))
    X=pd.DataFrame({'a':values,'b':values%3})
    y=(np.arange(len(X))%4==0).astype(int)
    splits=group_sensitivity.group_shuffle_split_indices(X)
    report=group_sensitivity.group_partition_summary(X,y,splits)
    assert report['total_rows']==465 and report['total_groups']==30
    assert sum(p['groups'] for p in report['partitions'].values())==30
    assert sum(p['rows'] for p in report['partitions'].values())==465
    assert set(report['groups_cross_partitions'].values())=={0}
    assert report['row_ratios_are_approximate'] and not report['heldout_evaluation']
    for name,ix in splits.items():
        assert report['partitions'][name]['positive_prevalence']==float(y[ix].mean())
        assert report['actual_row_ratios'][name]==len(ix)/465
    assert any(report['actual_row_ratios'][k]!=v for k,v in zip(splits,[0.7,0.15,0.15]))
    repeated=group_sensitivity.group_shuffle_split_indices(X)
    for key in splits: np.testing.assert_array_equal(splits[key],repeated[key])


def test_disjoint_rows_can_still_violate_exact_group_boundaries():
    X,_=grouped_sample()
    splits=dict(train=np.arange(210),validation=np.arange(210,255),test=np.arange(255,300))
    # Exchange two rows from different groups while retaining perfect row coverage.
    splits['train'][0],splits['test'][0]=splits['test'][0],splits['train'][0]
    with pytest.raises(ValueError,match='group crosses'):
        group_sensitivity.validate_group_partitions(X,splits)


def test_group_cv_fold_global_invariants_and_ap_delta_with_original_ids():
    X,y=grouped_sample()
    primary=stratified_indices(y)['train']
    Xt=X.iloc[primary]
    yt=y[primary]
    plans=group_sensitivity.sensitivity_cv_plans(Xt,yt)
    groups=group_sensitivity.exact_feature_groups(Xt)
    heldout=[]
    for train,valid in plans['group']:
        assert not np.intersect1d(groups[train],groups[valid]).size
        heldout.append(set(groups[valid]))
    assert all(not heldout[i]&heldout[j] for i in range(3) for j in range(i+1,3))
    estimator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,random_state=42))
    report=group_sensitivity.compare_train_only_cv_ap(Xt,yt,estimator,source_row_indices=primary)
    assert report.n_rows==len(primary) and len(report.fold_details)==6
    assert report.scope=='primary_training_only' and not report.used_for_model_selection
    assert not hasattr(estimator[-1],'classes_'), 'Input estimator must remain unfitted'
    for method in ['group','stratified']:
        rows=[row for row in report.fold_details if row['method']==method]
        aps=np.array([row['ap'] for row in rows])
        mean=getattr(report,method+'_cv_ap_mean')
        std=getattr(report,method+'_cv_ap_std')
        assert mean==aps.mean() and std==aps.std(ddof=0)
        for row,(train,valid) in zip(rows,plans[method]):
            assert row['train_groups']==len(np.unique(groups[train]))
            assert row['valid_groups']==len(np.unique(groups[valid]))
            assert row['train_positive_prevalence']==float(yt[train].mean())
            assert row['valid_positive_prevalence']==float(yt[valid].mean())
            assert row['valid_source_indices_sha256']==shared_splits.array_sha256(primary[valid])
    assert report.delta_ap_group_minus_stratified==report.group_cv_ap_mean-report.stratified_cv_ap_mean


def test_sensitivity_never_silently_falls_back_when_groups_are_insufficient():
    X=pd.DataFrame({'a':[1]*30})
    with pytest.raises(ValueError,match='Too few'):
        group_sensitivity.sensitivity_cv_plans(X,np.arange(30)%2)


def test_group_runner_uses_only_primary_train_and_frozen_xgboost(tmp_path,monkeypatch):
    base=toy_data(n=100)
    data_frame=base.iloc[np.repeat(np.arange(len(base)),3)].reset_index(drop=True)
    data_frame.loc[1,TARGET]=1-data_frame.loc[0,TARGET]
    path=tmp_path/'synthetic_group.csv';data_frame.to_csv(path,index=False)
    expected=stratified_indices(data_frame[TARGET])['train']
    original_compare=group_sensitivity.compare_train_only_cv_ap
    captured=[]
    def check_training(X,y,estimator,**kwargs):
        np.testing.assert_array_equal(X.to_numpy(),data_frame.drop(columns=TARGET).iloc[expected].to_numpy())
        np.testing.assert_array_equal(y,data_frame[TARGET].iloc[expected].to_numpy())
        np.testing.assert_array_equal(kwargs['source_row_indices'],expected)
        assert estimator.named_steps['model'].weight_mode=='none'
        assert estimator.named_steps['model'].n_estimators==160
        captured.append(True)
        return original_compare(X,y,estimator,**kwargs)
    monkeypatch.setattr(group_sensitivity,'compare_train_only_cv_ap',check_training)
    monkeypatch.setattr(modeling,'search_model_family',lambda *a:pytest.fail('Sensitivity cannot select/tune models'))
    result=group_sensitivity.run(path,tmp_path/'group_QA',train_cv=True,demo=True)
    assert captured==[True]
    assert result['manifest']['status']=='SYNTHETIC_QA_ONLY'
    assert result['manifest']['primary_train_rows']==len(expected)
    assert result['manifest']['no_pipeline_selection'] and not result['manifest']['heldout_group_evaluation']
    assert result['cv']['n_rows']==len(expected) and result['cv']['status']=='SENSITIVITY_EXPLORATORY'
    output=tmp_path/'group_QA'
    assert result['manifest']['artifact_hashes']==artifact_hashes(output)
    assert not (output/'test_predictions.csv').exists() and not (output/'test_metrics.csv').exists()
    with np.load(output/'primary_train_cv_indices.npz') as archive:
        assert len(archive.files)==12
        assert all(set(archive[key])<=set(expected) for key in archive.files)


def test_f2_fixed_grid_matches_independent_sklearn_evaluation_and_smallest_tie():
    y=np.array([1,0,1,0,0,1])
    p=np.array([0.8,0.2,0.6,0.1,0.05,0.8])
    expected=np.array([fbeta_score(y,p>=t,beta=2,zero_division=0) for t in np.linspace(.01,.99,199)])
    np.testing.assert_allclose(f2_threshold_curve(y,p),expected,atol=1e-15,rtol=0)
    perfect=np.flatnonzero(expected==1)
    assert len(perfect)>1
    assert optimal_f2_threshold(y,p)==float(F2_GRID[perfect[0]])


def test_f2_increases_recall_with_precision_tradeoff_without_changing_ap():
    y=np.array([1,1,1,0,0,0,0,0])
    p=np.array([.9,.6,.1,.5,.4,.3,.2,.05])
    f1=metrics_at_threshold(y,p,optimal_f1_threshold(y,p))
    f2=metrics_at_threshold(y,p,optimal_f2_threshold(y,p))
    assert f2['recall']>f1['recall'] and f2['precision']<f1['precision']
    assert f2['f2']>f1['f2']
    assert f2['average_precision']==f1['average_precision']==average_precision_score(y,p)
    assert f2['roc_auc']==f1['roc_auc']


def test_f2_ties_are_exact_not_tolerance_based(monkeypatch):
    curve=np.zeros(199);curve[1]=1-1e-14;curve[2]=1
    monkeypatch.setattr('diabetes_research.core.f2_threshold_curve',lambda y,p:curve)
    assert optimal_f2_threshold([0,1],[.1,.9])==float(F2_GRID[2])


@pytest.mark.parametrize('kind',['order','membership'])
def test_shared_recipe_rejects_seed_equal_but_different_indices(kind):
    y=toy_data()[TARGET].to_numpy()
    canonical=shared_splits.b_v9_indices(y)
    shared_splits.validate_b_v9_recipe(canonical,y)
    wrong={k:v.copy() for k,v in canonical.items()}
    if kind=='order': wrong['test']=wrong['test'][::-1]
    else: wrong=stratified_indices(y)
    with pytest.raises(ValueError,match='recipe or source-row order'):
        shared_splits.validate_b_v9_recipe(wrong,y)


def test_shared_loader_has_no_fallback_for_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError,match='no generated fallback'):
        shared_splits.verify_shared_split(tmp_path/'missing.csv')


def test_shared_loader_checks_full_bytes_archive_hash_labels_recipe_and_order(tmp_path,monkeypatch):
    data=tmp_path/'synthetic.csv'
    toy_data().to_csv(data,index=False)
    y=pd.read_csv(data)[TARGET]
    splits=shared_splits.b_v9_indices(y)
    archive=tmp_path/'indices.npz';np.savez_compressed(archive,**splits)
    dataset_hash=hashlib.sha256(data.read_bytes()).hexdigest()
    archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(shared_splits,'CANONICAL_DATASET_SHA256',dataset_hash)
    monkeypatch.setattr(shared_splits,'CANONICAL_ARCHIVE_SHA256',archive_hash)
    monkeypatch.setattr(shared_splits,'CANONICAL_ROWS',len(y))
    counts={k:len(v) for k,v in splits.items()}
    monkeypatch.setattr(shared_splits,'CANONICAL_COUNTS',counts)
    manifest=tmp_path/'manifest.json'
    record=dict(expected_dataset_full_sha256_from_user_terminal=dataset_hash,sizes=counts,
                original_n_rows=len(y),uses_original_row_positions=True,archive_sha256=archive_hash,
                arrays_in_zip_sha256={k:shared_splits.npy_sha256(v) for k,v in splits.items()})
    write_json(manifest,record)
    got,verification=shared_splits.verify_shared_split(data,archive,manifest)
    assert verification['verified_B_v9_recipe_and_order'] and verification['cross_split_overlap']==0
    for key in splits: np.testing.assert_array_equal(got[key],splits[key])
    archive.write_bytes(archive.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='archive SHA256'):
        shared_splits.verify_shared_split(data,archive,manifest)
    data.write_bytes(data.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Dataset SHA256 mismatch'):
        shared_splits.verify_shared_split(data,archive,manifest)


def bootstrap_input():
    pred=pd.DataFrame(dict(row_index=[90,12],y_true=[0,1],
                           prob_B1=[.9,.1],prob_B2=[.1,.9],prob_B3=[.2,.8],prob_FINAL=[.2,.8],
                           prob_A1=[.01,.99]))
    context=dict(a4_run_id='a4_unit_QA',prediction_run_id='a4_unit_QA',
                 expected_case_indices=np.array([90,12]),original_labels=np.array([0,1]),
                 final_model_id='B3',split_archive_sha256=hashlib.sha256(b'unit-QA-split').hexdigest())
    return pred,context


def test_three_core_bootstraps_share_draws_exclusions_signs_and_self_notice():
    pred,context=bootstrap_input()
    frame,summary,excluded=analysis.paired_ap_comparisons(pred,**context)
    assert summary['comparison_order']==[k for k,_,_ in CORE_BOOTSTRAP_PAIRS]
    assert len(summary['comparisons'])==3 and len(excluded)>0
    rng=np.random.default_rng(42);valid=[]
    for attempt in range(1000):
        ix=rng.integers(0,2,size=2)
        if len(np.unique(pred.y_true.to_numpy()[ix]))==2: valid.append(attempt)
    assert sorted(frame.replicate.unique())==valid
    for key,comparison in summary['comparisons'].items():
        rows=frame.loc[frame.comparison_key==key]
        assert len(rows)==comparison['n_valid_bootstrap']==len(valid)
        assert comparison['n_attempted_bootstrap']==1000
        assert comparison['n_excluded_bootstrap']==len(excluded)==1000-len(valid)
        assert comparison['excluded_replicate_indices']==excluded.replicate.tolist()
        assert comparison['confidence_level']==.95 and comparison['random_seed']==42
        assert 'positive' in comparison['sign_convention'] and len(comparison['comparison_id'])==64
        assert comparison['ci95_low']==np.quantile(rows.delta_ap,.025)
        assert comparison['ci95_high']==np.quantile(rows.delta_ap,.975)
        expected=0 if key=='FINAL_minus_B3' else .5
        assert comparison['delta_ap_point']==expected
    assert frame.groupby('replicate').sample_source_indices_sha256.nunique().eq(1).all()
    final=summary['comparisons']['FINAL_minus_B3']
    assert final['self_comparison'] and 'Same fitted model' in final['self_comparison_notice']
    assert final['ci95_low']==final['ci95_high']==0


@pytest.mark.parametrize('fault',['run','cases','labels','duplicate_cases','alias_scores'])
def test_generic_family_rejects_cross_run_or_mismatched_pairing(fault):
    pred,context=bootstrap_input()
    if fault=='run': context['prediction_run_id']='a4_other_QA'
    elif fault=='cases': pred=pred.iloc[::-1]
    elif fault=='labels': pred.loc[0,'y_true']=1
    elif fault=='duplicate_cases':
        pred['row_index']=[12,12];context['expected_case_indices']=np.array([12,12])
    else: pred['prob_FINAL']=[.8,.2]
    with pytest.raises(ValueError):
        analysis.paired_ap_comparisons(pred,**context,n_boot=10)


def test_comparison_ids_bind_run_and_source_cases_reproducibly():
    pred,context=bootstrap_input()
    a,sa,ea=analysis.paired_ap_comparisons(pred,**context,n_boot=40)
    b,sb,eb=analysis.paired_ap_comparisons(pred,**context,n_boot=40)
    pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(ea,eb)
    assert sa==sb
    context.update(a4_run_id='a4_other_QA',prediction_run_id='a4_other_QA')
    _,other,_=analysis.paired_ap_comparisons(pred,**context,n_boot=40)
    assert all(sa['comparisons'][k]['comparison_id']!=other['comparisons'][k]['comparison_id']
               for k in sa['comparisons'])


def test_explicit_generic_self_comparison_supports_same_model_names():
    _,summary=analysis.bootstrap_paired_ap([0,1],[.1,.9],[.1,.9],
                                         left_id='B3',right_id='B3',n_boot=40)
    assert summary['self_comparison'] and summary['delta_ap_point']==0


def test_new_a4_metadata_and_reference_thresholds_are_self_consistent(clean_a4):
    _,out,manifest=clean_a4
    assert manifest['manifest_version']==3 and manifest['feature_selection_status']=='IMPLEMENTED'
    assert manifest['split_mode']==shared_splits.LEGACY_QA_MODE
    assert manifest['shap_samples']==500 and manifest['shap_background_samples']==100
    assert manifest['bootstrap_comparisons']==[dict(id=k,left=l,right=r) for k,l,r in CORE_BOOTSTRAP_PAIRS]
    selection=read_json(out/'selection.json')
    assert 'validation_f2_thresholds' in selection and 'validation_f1_thresholds' not in selection
    assert selection['threshold_policy']['beta']==2
    assert selection['threshold_policy']['tie_rule']=='smallest_threshold_exact_tie'
    for name in ['validation','test']:
        primary=pd.read_csv(out/f'{name}_metrics.csv')
        reference=pd.read_csv(out/f'{name}_reference_metrics.csv')
        assert reference.threshold.eq(.5).all()
        np.testing.assert_allclose(primary.average_precision,reference.average_precision,atol=1e-12,rtol=0)
        assert {'f1','f2'}<=set(primary.columns)
    predictions=pd.read_csv(out/'test_predictions.csv')
    np.testing.assert_array_equal(predictions.prob_A1,predictions.prob_A2)
    lock=read_json(out/'selection_lock.json')
    assert lock['lock_id']==manifest['selection_lock_id']


@pytest.mark.parametrize('fault,reason',[
    ('folds','CV folds/source positions'),
    ('f2_tie','smallest exact-tie'),
    ('xgb_core','actually contain XGBClassifier'),
    ('pending_feature_status','modeling methodology'),
    ('split_manifest','split manifest/archive'),
    ('shap_runtime','SHAP/LIME effective settings'),
    ('group_mode','group sensitivity mode'),
    ('lock_policy','selection lock'),
    ('reference_metrics','0.5 reference metrics'),
])
def test_new_semantic_tamper_checks_survive_rehashed_artifacts(clean_a4,tmp_path,fault,reason):
    data,original,_=clean_a4
    broken=tmp_path/'broken';shutil.copytree(original,broken)
    if fault=='folds':
        p=broken/'cv_fold_indices.npz'
        with np.load(p,allow_pickle=False) as archive: values={k:archive[k] for k in archive.files}
        key=next(iter(values));values[key]=values[key][::-1]
        np.savez_compressed(p,**values)
    elif fault=='f2_tie':
        p=broken/'threshold_search.json';value=read_json(p);row=value['models']['B0']
        assert row['f2_values'][0]==row['f2_values'][1]
        row['winner_grid_index']=1;row['selected_threshold']=float(F2_GRID[1]);write_json(p,value)
    elif fault=='xgb_core':
        p=broken/'fitted_models.joblib';registry=joblib.load(p)
        registry['models']['B3'].named_steps['model'].core_=modeling.RandomForestClassifier()
        joblib.dump(registry,p)
    elif fault=='pending_feature_status':
        manifest=read_json(broken/'manifest.json');manifest['feature_selection_status']='APPROVED'
        write_json(broken/'manifest.json',manifest)
    elif fault=='split_manifest':
        p=broken/'split_manifest.json';record=read_json(p);record['mode']='unknown';write_json(p,record)
    elif fault=='shap_runtime':
        p=broken/'runtime_config.json';record=read_json(p)
        record['effective']['explanations']['shap']['n']=12;write_json(p,record)
    elif fault=='group_mode':
        p=broken/'manifest.json';record=read_json(p);record['group_sensitivity_mode']='primary';write_json(p,record)
    elif fault=='lock_policy':
        p=broken/'selection_lock.json';record=read_json(p);record['threshold_policy']['beta']=1;write_json(p,record)
    else:
        p=broken/'test_reference_metrics.csv';record=pd.read_csv(p);record.loc[0,'f2']=.99
        record.to_csv(p,index=False)
    refresh_hashes(broken)
    output=tmp_path/'no_statistics'
    with pytest.raises(ValueError,match=reason):
        analysis.run(data,broken,output,allow_smoke=True,bootstraps=10)
    assert not output.exists()


def test_selection_lock_and_legacy_version_fail_before_unpickle(clean_a4,tmp_path,monkeypatch):
    data,original,_=clean_a4
    monkeypatch.setattr(joblib,'load',lambda *args:pytest.fail('Must reject before loading model pickle'))
    for fault in ['lock','legacy']:
        broken=tmp_path/fault;shutil.copytree(original,broken)
        manifest=read_json(broken/'manifest.json')
        if fault=='legacy': manifest['manifest_version']=2
        else:
            lock=read_json(broken/'selection_lock.json');lock['locked_at_utc']='tampered'
            write_json(broken/'selection_lock.json',lock)
            manifest['artifact_hashes']=artifact_hashes(broken,A4_CRITICAL_ARTIFACTS)
        write_json(broken/'manifest.json',manifest)
        with pytest.raises(ValueError,match='selection lock|Unsupported A4 provenance'):
            validate_a4_artifacts(data,broken,manifest)


def fake_shap(monkeypatch):
    calls=[]
    class Independent:
        def __init__(self,data,max_samples=100):
            self.data=np.asarray(data)[:max_samples]
            self.max_samples=max_samples
            calls.append(('masker',len(self.data),{'max_samples':max_samples}))
    class Explanation:
        def __init__(self,values,base_values,data,feature_names=None):
            self.values=np.asarray(values);self.base_values=np.asarray(base_values);self.data=np.asarray(data)
        def __getitem__(self,index):
            return Explanation(self.values[index:index+1],self.base_values[index:index+1],self.data[index:index+1])
    class Explainer:
        def __init__(self,predict,masker,**settings):
            self.masker=masker if isinstance(masker,Independent) else Independent(masker)
            self.predict=predict;calls.append(('background',len(self.masker.data),settings))
        def __call__(self,batch,**settings):
            calls.append(('chunk',len(batch),settings))
            self.predict(pd.concat([batch]*5,ignore_index=True).to_numpy())
            return Explanation(np.zeros(batch.shape),np.full(len(batch),.2),batch.to_numpy())
    def plot(*args,**kwargs): analysis.plt.figure()
    monkeypatch.setitem(sys.modules,'shap',types.SimpleNamespace(
        Explainer=Explainer,Explanation=Explanation,
        maskers=types.SimpleNamespace(Independent=Independent),
        plots=types.SimpleNamespace(beeswarm=plot,bar=plot,waterfall=plot)))
    return calls


def test_shap_500_100_defaults_deterministic_source_ids_and_bounded_processing(tmp_path,monkeypatch):
    calls=fake_shap(monkeypatch)
    batches=[]
    class Model:
        def predict_proba(self,X):
            batches.append(len(X));return np.tile([.8,.2],(len(X),1))
    Xtrain=pd.DataFrame({'a':np.arange(160),'b':np.arange(160)%2})
    Xtest=pd.DataFrame({'a':np.arange(600),'b':np.arange(600)%2})
    settings=dict(source_train_indices=np.arange(10000,10160),source_test_indices=np.arange(20000,20600),
                  a4_run_id='a4_stub_QA',final_model_id='B3',prediction_batch_size=64)
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
    for out in [first,second]: analysis.plot_shap(Model(),Xtrain,Xtest,out,**settings)
    metadata=read_json(first/'shap_metadata.json')
    assert metadata['requested_explained_rows']==metadata['explained_rows']==500
    assert metadata['requested_background_rows']==metadata['background_rows']==100
    assert metadata['random_seed']==metadata['explainer_seed']==42
    assert metadata['chunks_processed']==20 and metadata['chunk_size']==25
    assert metadata['mask_batch_size']==1
    assert metadata['masker']=='Independent' and metadata['masker_max_samples']==100
    assert all(row[2]['batch_size']==1 for row in calls if row[0]=='chunk')
    assert max(batches)<=64 and all(row[1]<=25 for row in calls if row[0]=='chunk')
    assert all(row[1]==100 for row in calls if row[0]=='background')
    assert metadata==read_json(second/'shap_metadata.json')
    with np.load(first/'shap_selected_source_rows.npz') as archive:
        np.testing.assert_array_equal(archive['explained_source_row_indices'],metadata['explained_source_row_indices'])
        np.testing.assert_array_equal(archive['background_source_row_indices'],metadata['background_source_row_indices'])
        assert np.intersect1d(archive['explained_source_row_indices'],archive['background_source_row_indices']).size==0
    with np.load(first/'shap_attributions.npz') as archive:
        assert archive['values'].shape==(500,2)
    assert inspect.signature(analysis.plot_lime).parameters['num_samples'].default==1000


@pytest.mark.parametrize('background_n',[9,150,300])
def test_shap_background_overrides_reach_masker_and_saved_source_ids(tmp_path,monkeypatch,background_n):
    calls=fake_shap(monkeypatch)
    class Model:
        def predict_proba(self,X):
            return np.tile([.8,.2],(len(X),1))
    Xtrain=pd.DataFrame({'a':np.arange(200),'b':np.arange(200)%2})
    Xtest=pd.DataFrame({'a':np.arange(50),'b':np.arange(50)%2})
    analysis.plot_shap(Model(),Xtrain,Xtest,tmp_path,n=35,background_n=background_n,
        source_train_indices=np.arange(1000,1200),source_test_indices=np.arange(2000,2050))
    actual=min(background_n,len(Xtrain))
    metadata=read_json(tmp_path/'shap_metadata.json')
    assert metadata['requested_background_rows']==background_n
    assert metadata['background_rows']==metadata['masker_max_samples']==actual
    assert [row[1] for row in calls if row[0]=='background']==[actual]
    assert [row[2]['max_samples'] for row in calls if row[0]=='masker']==[actual]
    with np.load(tmp_path/'shap_selected_source_rows.npz') as archive:
        assert len(archive['background_source_row_indices'])==actual
        np.testing.assert_array_equal(archive['background_source_row_indices'],metadata['background_source_row_indices'])
    assert metadata['explained_rows']==35 and metadata['status']=='COMPLETED'


def test_a5_cli_overrides_forward_shap_budgets_without_changing_lime(clean_a4,tmp_path,monkeypatch):
    data,a4,_=clean_a4
    captured={}
    monkeypatch.setattr(analysis,'plot_shap',lambda *args,**kwargs:captured.update(shap=kwargs))
    monkeypatch.setattr(analysis,'plot_lime',lambda *args,**kwargs:captured.update(lime=kwargs))
    out=tmp_path/'a5_QA'
    analysis.run(data,a4,out,allow_smoke=True,bootstraps=10,shap=True,lime=True,
                 shap_n=17,shap_background_n=9,shap_chunk_size=4,shap_prediction_batch_size=64)
    assert captured['shap']['n']==17 and captured['shap']['background_n']==9
    assert captured['shap']['chunk_size']==4 and captured['shap']['prediction_batch_size']==64
    assert captured['lime']['num_samples']==1000
    manifest=read_json(out/'manifest_a5.json')
    assert manifest['shap_samples']==17 and manifest['shap_background_samples']==9
    assert len(manifest['bootstrap_comparisons'])==3


@pytest.mark.parametrize('script',[
    COMPONENT_ROOT/'scripts/run_a4.py',COMPONENT_ROOT/'scripts/run_a5.py',
    COMPONENT_ROOT/'scripts/run_group_sensitivity.py',REPOSITORY_ROOT/'shared_splits/verify_shared_split.py',
])
def test_all_wrappers_import_root_package_from_unrelated_cwd(script,tmp_path):
    environment=os.environ.copy();environment.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,str(script),'--help'],cwd=tmp_path,env=environment,
                          text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert '--help' in result.stdout


def test_single_canonical_package_and_extended_source_provenance():
    assert Path(modeling.__file__).resolve()==REPOSITORY_ROOT/'src/diabetes_research/modeling.py'
    assert not (COMPONENT_ROOT/'src').exists()
    files=source_identity()['file_hashes']
    for rel in [
        'src/diabetes_research/paths.py','src/diabetes_research/group_sensitivity.py',
        'src/diabetes_research/shared_splits.py','pyproject.toml','requirements.txt',
        'shared_splits/shared_split_indices.npz',
        'research/lead_modeling/a4_a5/config/A2_protocol_v3.yaml',
        'research/lead_modeling/a4_a5/config/PROTOCOL_AMENDMENTS_V3.json',
        'research/lead_modeling/a4_a5/notebooks/A4_Training_Protocol_v2.ipynb',
    ]:
        assert files[rel]==hashlib.sha256((REPOSITORY_ROOT/rel).read_bytes()).hexdigest()


@pytest.mark.parametrize('name',['A4_Training_Protocol_v2.ipynb','A5_Evaluation_XAI_v2.ipynb'])
def test_updated_notebook_cells_compile_without_running_experiments(name):
    import nbformat
    path=COMPONENT_ROOT/'notebooks'/name
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is None and cell.outputs==[]
            compile(cell.source,str(path),'exec')
