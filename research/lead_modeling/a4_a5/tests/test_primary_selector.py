"""Primary selector acceptance on constructed data; no official evaluation."""
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold

from diabetes_research import analysis, modeling, provenance
from diabetes_research.artifact_validation import validate_a4_artifacts
from diabetes_research.core import FEATURE_COUNTS, TARGET, feature_selection_policy, select_feature_budget, write_json
from diabetes_research.paths import AMENDMENTS_PATH, PROTOCOL_PATH
from diabetes_research.provenance import artifact_hashes, read_json, source_identity, validate_protocol
from test_audit_regressions import refresh_hashes
from test_protocol import toy_data


class FixedImportance(BaseEstimator):
    def __init__(self, scores):
        self.scores=scores

    def fit(self,X,y):
        self.feature_importances_=np.asarray(self.scores,dtype=float)
        self.n_features_in_=X.shape[1]
        self.feature_names_in_=np.asarray(X.columns,dtype=object)
        return self


@pytest.mark.parametrize('k',FEATURE_COUNTS)
@pytest.mark.parametrize('scores',[np.zeros(21),np.ones(21),np.array([3.,2.,2.,2.,1.,1.,0.]*3)])
def test_exact_budget_stable_gain_ties_and_original_transform_order(k,scores):
    X=pd.DataFrame(np.arange(84).reshape(4,21),columns=[f'c{i}' for i in range(21)],index=[90,12,34,77])
    selector=modeling.ExactTopKSelectFromModel(FixedImportance(scores),k=k).set_output(transform='pandas')
    selector.fit(X,[0,1,0,1])
    assert isinstance(selector,SelectFromModel)
    # Independent lexicographic specification: gain descending, column position ascending.
    chosen=sorted(sorted(range(21),key=lambda i:(-scores[i],i))[:k])
    np.testing.assert_array_equal(selector.get_support(indices=True),chosen)
    pd.testing.assert_frame_equal(selector.transform(X),X.iloc[:,chosen])
    assert selector.get_support().sum()==k
    # Repeat on the identical original column order, with a fresh selector.
    again=modeling.ExactTopKSelectFromModel(FixedImportance(scores),k=k).fit(X,[0,1,0,1])
    np.testing.assert_array_equal(selector.get_support(),again.get_support())


@pytest.mark.parametrize('k',[0,22,-1,True,5.5])
def test_selector_rejects_incorrect_budget_before_fit(k):
    with pytest.raises(ValueError,match='Selector k'):
        modeling.ExactTopKSelectFromModel(FixedImportance(np.ones(21)),k=k).fit(toy_data(20).drop(columns=TARGET),np.arange(20)%2)


@pytest.mark.parametrize('scores',[np.full(21,np.nan),np.full(21,np.inf),np.full(21,-1.),np.ones(20)])
def test_selector_rejects_invalid_gain_scores(scores):
    with pytest.raises(ValueError,match='Gain importance'):
        modeling.ExactTopKSelectFromModel(FixedImportance(scores),k=5).fit(toy_data(20).drop(columns=TARGET),np.arange(20)%2)


def test_importance_near_ties_are_not_rounded_or_tolerance_tied():
    scores=np.zeros(21);scores[12]=1.;scores[1]=1.-1e-14
    selector=modeling.ExactTopKSelectFromModel(FixedImportance(scores),k=1).fit(toy_data(20).drop(columns=TARGET),np.arange(20)%2)
    assert selector.get_support(indices=True).tolist()==[12]


def test_each_selector_ranker_and_downstream_fit_is_fold_local(monkeypatch):
    frame=toy_data(90)
    X=frame.drop(columns=TARGET);X.index=np.arange(10000,10090)
    y=frame[TARGET]
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    plan=list(cv.split(X,y));selector_calls=[];learner_calls=[]
    real_selector=modeling.ExactTopKSelectFromModel.fit
    real_learner=modeling.FoldAwareXGB.fit

    def selector_fit(self,Xfit,yfit,**kwargs):
        selector_calls.append((self.k,Xfit.index.to_numpy().copy()))
        return real_selector(self,Xfit,yfit,**kwargs)

    def learner_fit(self,Xfit,yfit):
        learner_calls.append((Xfit.index.to_numpy().copy(),np.asarray(yfit).copy(),self))
        return real_learner(self,Xfit,yfit)

    monkeypatch.setattr(modeling.ExactTopKSelectFromModel,'fit',selector_fit)
    monkeypatch.setattr(modeling.FoldAwareXGB,'fit',learner_fit)
    params=modeling.FoldAwareXGB(n_estimators=4,n_jobs=1).get_params()
    records,table,registry=modeling.evaluate_primary_feature_cv(
        X,y,params,cv,run_id='a4_fold_QA',identity=source_identity(),dataset_hash='synthetic',split_hash='synthetic',cv_hash='synthetic')
    assert len(selector_calls)==12 and len(learner_calls)==24
    for i,(k,ids) in enumerate(selector_calls):
        train,valid=plan[i%3]
        np.testing.assert_array_equal(ids,X.index.to_numpy()[train])
        assert not set(ids)&set(X.index.to_numpy()[valid])
        for fitted_ids,labels,learner in learner_calls[2*i:2*i+2]:
            np.testing.assert_array_equal(fitted_ids,ids)
            np.testing.assert_array_equal(labels,y.iloc[train])
            assert learner.weight_ratio_==float((labels==0).sum()/(labels==1).sum())
            assert learner.core_.get_params()['importance_type']=='gain'
            assert learner.get_params()=={**params,'weight_mode':'balanced','importance_type':'gain'}
    assert table.groupby('requested_k').size().to_dict()==dict.fromkeys(FEATURE_COUNTS,3)
    assert len(records['records'])==len(registry['models'])==12


def test_fit_scope_guard_rejects_validation_contamination_before_any_fit(monkeypatch):
    frame=toy_data(30);X=frame.drop(columns=TARGET);y=frame[TARGET]
    pipeline=modeling.make_primary_pipeline(modeling.FoldAwareXGB(n_estimators=2).get_params(),5)
    monkeypatch.setattr(modeling.ExactTopKSelectFromModel,'fit',lambda *a,**kw:pytest.fail('No ranking fit allowed'))
    with pytest.raises(ValueError,match='allowed training fold'):
        modeling.fit_primary_pipeline(pipeline,X.iloc[:21],y.iloc[:21],np.arange(20))


@pytest.mark.parametrize('aps,expected,eligible',[
    ({21:.5,15:.499,10:.49,5:.489},10,[10,15,21]),
    ({21:.5,15:.51,10:.6,5:.7},5,[5,10,15,21]),
    ({21:.5,15:.48,10:.48,5:.48},21,[21]),
    ({21:.48,15:.48,10:.48,5:.48},21,[]),
    ({21:.5,15:.5,10:.49-2e-12,5:.48},15,[15,21]),
])
def test_reserved_validation_smallest_budget_boundary_and_fallback(aps,expected,eligible):
    assert select_feature_budget(.5,aps)==(expected,eligible)


def test_ready_protocol_does_not_release_official_evaluation(tmp_path,monkeypatch):
    validate_protocol(yaml.safe_load(PROTOCOL_PATH.read_bytes()))
    assert feature_selection_policy()['primary_method_implemented']
    assert read_json(AMENDMENTS_PATH)['decisions']['official_run_release']['status']=='PENDING'
    path=tmp_path/'diabetes_binary_health_indicators_BRFSS2015.csv';path.write_text('synthetic preflight sentinel')
    class OfficialShapeSentinel:
        def __len__(self):return 253680
    monkeypatch.setattr(modeling,'load_dataset',lambda p:(OfficialShapeSentinel(),None))
    monkeypatch.setattr(modeling,'dataset_audit',lambda *a:pytest.fail('Release must fail before scientific work'))
    with pytest.raises(ValueError,match='authorization remains PENDING; primary SelectFromModel implementation is READY'):
        modeling.run(path,tmp_path/'never_created')
    assert not (tmp_path/'never_created').exists()


@pytest.mark.parametrize('field,value',[
    ('importance_type','weight'),('weight_mode','none'),('threshold','mean'),
    ('prefit',True),('tie_rule','unstable'),('selection_scope','test AP'),
])
def test_protocol_rejects_primary_ranking_drift(field,value):
    protocol=yaml.safe_load(PROTOCOL_PATH.read_bytes())
    protocol['feature_reduction']['primary_configuration'][field]=value
    with pytest.raises(ValueError,match='feature_reduction'):
        validate_protocol(protocol)


TAMPERS=[
    ('fold_fit_ids','fit identities'),('fold_ranker_weight','class weighting'),
    ('fold_gain_type','ranking settings'),('fold_ap','rankings/selected columns/AP'),
    ('fold_columns','rankings/selected columns/AP'),('fold_scores','rankings/selected columns/AP'),
    ('fold_source','fold/source identities'),('missing_fold','fold coverage'),
    ('cv_count','CV AP/feature counts'),('full_scores','full-training rankings'),
    ('full_columns','full-training rankings'),('full_k','configuration|selector'),
    ('primary_anova','configuration|selector'),('a2_reference','Locked full-feature A2'),
    ('secondary_primary','Secondary ANOVA'),('a4_a5_run','registry provenance'),
]


@pytest.mark.parametrize('fault,reason',TAMPERS)
def test_a5_rejects_primary_tampering_even_after_rehash(clean_a4,tmp_path,fault,reason):
    data,original,_=clean_a4
    broken=tmp_path/'broken';shutil.copytree(original,broken)
    if fault in ('fold_fit_ids','fold_ranker_weight','fold_gain_type'):
        p=broken/'feature_selection_models.joblib';registry=joblib.load(p)
        model=next(iter(registry['models'].values()))
        ranker=model.named_steps['selection'].estimator_
        if fault=='fold_fit_ids':ranker.fit_source_row_indices_[0]=-1
        elif fault=='fold_ranker_weight':ranker.core_.set_params(scale_pos_weight=99.)
        else:ranker.importance_type='weight'
        joblib.dump(registry,p)
    elif fault in ('fold_ap','fold_columns','fold_scores','fold_source','missing_fold'):
        p=broken/'feature_selection_folds.json';record=read_json(p)
        if fault=='fold_ap':record['records'][0]['ap']=.123
        elif fault=='fold_columns':record['records'][0]['selected_feature_names'].reverse()
        elif fault=='fold_scores':record['records'][0]['ranking_scores'][0]=.123
        elif fault=='fold_source':record['source_files_sha256']='0'*64
        else:record['records'].pop()
        write_json(p,record)
    elif fault=='cv_count':
        p=broken/'feature_selection_cv.csv';table=pd.read_csv(p);table.loc[0,'actual_feature_count']=20
        table.to_csv(p,index=False)
    elif fault in ('full_scores','full_columns'):
        p=broken/'feature_rankings.json';record=read_json(p);row=record['models']['A3_k5']
        if fault=='full_scores':row['ranking_scores'][0]=.987
        else:row['selected_feature_names'].reverse()
        write_json(p,record)
    elif fault in ('full_k','primary_anova','a4_a5_run'):
        p=broken/'fitted_models.joblib';registry=joblib.load(p)
        if fault=='full_k':registry['models']['A3_k5'].named_steps['selection'].k=10
        elif fault=='primary_anova':registry['models']['A3_k5'].steps[0]=('selection',modeling.SelectKBest(k=5))
        else:registry['run_id']='a4_other_QA'
        joblib.dump(registry,p)
    elif fault=='a2_reference':
        p=broken/'a2_reference_lock.json';record=read_json(p);record['average_precision']=.987;write_json(p,record)
    else:
        p=broken/'secondary_anova.json';record=read_json(p);record['used_for_primary_selection']=True;write_json(p,record)
    refresh_hashes(broken)
    out=tmp_path/'no_statistics'
    with pytest.raises(ValueError,match=reason):
        analysis.run(data,broken,out,allow_smoke=True,bootstraps=10)
    assert not out.exists()


def test_primary_integrity_guard_precedes_any_pickle(clean_a4,tmp_path,monkeypatch):
    data,original,_=clean_a4
    broken=tmp_path/'broken';shutil.copytree(original,broken)
    (broken/'feature_selection_folds.json').write_text('{}')
    monkeypatch.setattr(joblib,'load',lambda *a:pytest.fail('Tampering must be detected before unpickle'))
    with pytest.raises(ValueError,match='checksum mismatch: feature_selection_folds'):
        validate_a4_artifacts(data,broken,read_json(broken/'manifest.json'))


def test_primary_full_budget_parity_and_separate_selected_a3(clean_a4):
    data,out,manifest=clean_a4
    verified=validate_a4_artifacts(data,out,manifest)
    registry=joblib.load(out/'fitted_models.joblib')
    models=registry['models'];X=verified['Xtrain']
    np.testing.assert_array_equal(models['A2'].predict_proba(X),models['A3_k21'].predict_proba(X))
    selection=read_json(out/'selection.json')
    assert selection['separately_reported_a3_model']==f"A3_k{selection['selected_feature_k']}"
    assert selection['full_feature_fallback']=='A2'
    assert not any(key.startswith('ANOVA') for key in models)
    assert read_json(out/'secondary_anova.json')['used_for_primary_selection'] is False
    assert verified['verification']['primary_selector_folds_verified']
    json.dumps(modeling.model_configuration(models),allow_nan=False)


def test_a4_a5_feature_metadata_matches_locked_selector(clean_a4,tmp_path):
    data,out,manifest=clean_a4
    a5=tmp_path/'a5';analysis.run(data,out,a5,allow_smoke=True,bootstraps=10)
    saved=read_json(a5/'manifest_a5.json');selection=read_json(out/'selection.json')
    assert saved['source_a4_run_id']==manifest['run_id']
    assert saved['feature_selection_status']=='IMPLEMENTED'
    assert saved['selected_feature_k']==selection['selected_feature_k']
    assert saved['selected_a3_model']==selection['separately_reported_a3_model']
    assert saved['final_feature_names']==selection['final_feature_names']
    assert saved['source_a4_artifact_hashes']==artifact_hashes(out,list(manifest['artifact_hashes']))


def test_test_label_changes_cannot_change_rankings_k_or_thresholds(tmp_path,monkeypatch):
    # Fixed constructed outer partitions isolate outcome use from stratified split construction.
    frame=toy_data(100)
    splits=modeling.stratified_indices(frame[TARGET])
    monkeypatch.setattr(modeling,'stratified_indices',lambda y:splits)
    spaces=modeling.random_spaces(quick=True)
    spaces['B3']['model__n_estimators']=[3];spaces['B2']['model__n_estimators']=[3]
    monkeypatch.setattr(modeling,'random_spaces',lambda quick:spaces)
    source=modeling.FoldAwareXGB.predict_proba
    seen_test=[]
    def locked_predict(self,X):
        # Original test positions only; training CV/validation never use these IDs.
        if isinstance(X,pd.DataFrame) and set(X.index)==set(splits['test']):
            assert (active_out/'selection_lock.json').is_file()
            seen_test.append(True)
        return source(self,X)
    monkeypatch.setattr(modeling.FoldAwareXGB,'predict_proba',locked_predict)
    selections=[];rankings=[]
    for changed in [False,True]:
        data=tmp_path/f'synthetic_{changed}.csv';active_out=tmp_path/f'a4_{changed}'
        sample=frame.copy()
        if changed:sample.loc[splits['test'],TARGET]=1-sample.loc[splits['test'],TARGET]
        sample.to_csv(data,index=False)
        modeling.run(data,active_out,trials=1,cv_folds=3,demo=True)
        selection=read_json(active_out/'selection.json');selection.pop('run_id')
        selections.append(selection)
        rankings.append(read_json(active_out/'feature_rankings.json')['models'])
    assert seen_test
    assert selections[0]==selections[1]
    assert rankings[0]==rankings[1]
