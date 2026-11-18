"""QA only. No claims about BRFSS metrics are based on these tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from diabetes_research.core import (BINARY,EXPECTED,TARGET,
    stratified_indices,optimal_f1_threshold,metrics_at_threshold,
    load_dataset,sha256_file)
from diabetes_research.modeling import FoldAwareXGB
from diabetes_research.analysis import bootstrap_paired_ap


def toy_data(n=200,seed=42):
    rng=np.random.default_rng(seed)
    data={c:rng.integers(0,2,n) for c in BINARY}
    data.update(BMI=rng.integers(18,45,n),MentHlth=rng.integers(0,31,n),
        PhysHlth=rng.integers(0,31,n),GenHlth=rng.integers(1,6,n),
        Age=rng.integers(1,14,n),Education=rng.integers(1,7,n),Income=rng.integers(1,9,n))
    data[TARGET]=np.array([0]*int(.8*n)+[1]*(n-int(.8*n)))
    df=pd.DataFrame(data)
    return df.sample(frac=1,random_state=seed).reset_index(drop=True)


def test_expected_schema():
    assert len(EXPECTED)==21
    assert len(set(BINARY))==len(BINARY)


def test_splits_are_disjoint_and_stratified():
    y=toy_data()[TARGET]
    s=stratified_indices(y)
    assert [len(s[k]) for k in ['train','validation','test']]==[140,30,30]
    assert not (set(s['train'])&set(s['test']))
    assert not (set(s['train'])&set(s['validation']))
    assert abs(y.iloc[s['test']].mean()-y.mean())<.03


def test_threshold_tie_rule_and_metrics():
    y=np.array([1,0,1,0]);p=np.array([.8,.4,.6,.2])
    t=optimal_f1_threshold(y,p)
    assert np.isclose(t,.6)
    m=metrics_at_threshold(y,p,t)
    assert m['tp']==2 and m['fp']==0
    assert m['f1']==1.0


def test_paired_bootstrap_identical_predictions():
    y=np.array([0,0,1,1,0,1,0,1,0,0,1,0])
    p=np.array([.1,.2,.7,.8,.2,.6,.4,.5,.1,.3,.7,.2])
    df,s=bootstrap_paired_ap(y,p,p,n_boot=50)
    assert len(df)>0
    assert np.isclose(s['delta_ap_point'],0)
    assert np.isclose(s['ci95_low'],0)
    assert np.isclose(s['ci95_high'],0)


def test_loader_rejects_bad_schema(tmp_path):
    df=toy_data().drop(columns=['HighBP'])
    p=tmp_path/'bad.csv';df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='21 biến'):
        load_dataset(p)


def test_fold_xgb_ratio_depends_on_train_y_only():
    df=toy_data(n=50)
    X=df.drop(columns=[TARGET]);y=df[TARGET]
    model=FoldAwareXGB(n_estimators=5,n_jobs=1,weight_mode='balanced')
    model.fit(X,y)
    assert np.isclose(model.weight_ratio_,(y==0).sum()/(y==1).sum())


def test_only_research_file_can_claim_full_research(tmp_path):
    from diabetes_research.modeling import run
    df=toy_data(n=120)
    p=tmp_path/'toy.csv';df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='Chế độ nghiên cứu yêu cầu'):
        run(p,tmp_path/'invalid_full',trials=1,cv_folds=2)


def test_qa_hash_and_loader(tmp_path):
    df=toy_data()
    p=tmp_path/'toy.csv';df.to_csv(p,index=False)
    X,y=load_dataset(p)
    assert X.shape==(200,21)
    assert y.sum()==40
    assert len(sha256_file(p))==64
