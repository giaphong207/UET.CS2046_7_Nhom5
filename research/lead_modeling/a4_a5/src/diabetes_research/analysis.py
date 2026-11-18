"""A5: locked-test evaluation, paired bootstrap and optional SHAP/LIME."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve)
from .core import CATEGORICAL, metrics_at_threshold, sha256_file, write_json, SEED
from .artifact_validation import A4_STATUSES, validate_a4_artifacts
from .provenance import (MANIFEST_VERSION, artifact_hashes, finish_execution, read_json,
                         save_analysis_provenance, source_identity, start_execution)

BOOTSTRAP_ATTEMPTS = 1000
SUBGROUP_COLUMNS = [
    'feature', 'group_code', 'n', 'positive_n', 'status', 'withholding_reason',
    'average_precision', 'roc_auc', 'accuracy', 'precision', 'recall',
    'specificity', 'f1', 'threshold', 'tn', 'fp', 'fn', 'tp',
]


def bootstrap_paired_ap(y,p_final,p_baseline,n_boot=1000,seed=SEED):
    """Paired resampling: the same sampled positions for BOTH models.

    CI is conditional on the fitted models and this particular test split.
    """
    y=np.asarray(y)
    p_final=np.asarray(p_final,dtype=float)
    p_baseline=np.asarray(p_baseline,dtype=float)
    if any(a.ndim != 1 for a in (y, p_final, p_baseline)) or not len(y)==len(p_final)==len(p_baseline):
        raise ValueError('Arrays must be one-dimensional and have same length')
    if set(np.unique(y))!={0,1}:
        raise ValueError('Test must have both classes')
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError('Bootstrap attempts must be a positive integer')
    if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()
           for a in (p_final, p_baseline)):
        raise ValueError('Bootstrap scores must be finite probabilities in [0, 1]')
    y=y.astype(int)
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(n_boot):
        ix=rng.integers(0,len(y),size=len(y))
        if len(np.unique(y[ix]))<2:
            continue
        f=average_precision_score(y[ix],p_final[ix])
        b=average_precision_score(y[ix],p_baseline[ix])
        rows.append({'replicate':i,'ap_final':f,'ap_baseline':b,'delta_ap':f-b})
    if not rows:raise RuntimeError('All bootstrap replications were single-class.')
    df=pd.DataFrame(rows)
    delta=float(average_precision_score(y,p_final)-average_precision_score(y,p_baseline))
    low,high=np.quantile(df.delta_ap,[0.025,0.975])
    return df,{'delta_ap_point':delta,'ci95_low':float(low),'ci95_high':float(high),
               'n_valid_bootstrap':len(df),
               'n_attempted_bootstrap':int(n_boot),
               'n_excluded_bootstrap':int(n_boot)-len(df),
               'exclusion_policy':'Exclude single-class samples; do not replace excluded attempts',
               'random_seed':int(seed),'confidence_level':0.95,
               'interval_method':'percentile (2.5%, 97.5%)',
               'paired_same_indices':True,'comparison':'AP_FINAL minus AP_B3',
               'interpretation':('CI includes zero: no clear difference under this bootstrap'
                    if low<=0<=high else 'CI excludes zero under this bootstrap'),
               'scope':'Test-set resampling uncertainty conditional on already-fitted models, not full training uncertainty'}


def save_curves(y,pred,out):
    out=Path(out)
    selected=['prob_B1','prob_B2','prob_B3','prob_FINAL']
    fig,ax=plt.subplots(figsize=(7.6,5.1))
    for key in selected:
        p,r,_=precision_recall_curve(y,pred[key])
        ap=average_precision_score(y,pred[key])
        ax.plot(r,p,label=f'{key.replace("prob_","")} — AP {ap:.3f}')
    ax.axhline(np.mean(y),ls='--',lw=1,label='prevalence')
    ax.set(xlabel='Recall',ylabel='Precision',title='Locked-test Precision–Recall curves')
    ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/'precision_recall_test.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.6,5.1))
    for key in selected:
        fpr,tpr,_=roc_curve(y,pred[key]);ax.plot(fpr,tpr,label=key.replace('prob_',''))
    ax.plot([0,1],[0,1],ls='--',color='gray',lw=1)
    ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate',title='Locked-test ROC curves')
    ax.legend();fig.tight_layout();fig.savefig(out/'roc_test.png',dpi=160);plt.close(fig)


def save_confusion(y,p,threshold,out):
    m=metrics_at_threshold(y,p,threshold)
    values=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
    fig,ax=plt.subplots(figsize=(5.6,4.6))
    ax.imshow(values,cmap='Blues')
    for (i,j),v in np.ndenumerate(values):
        ax.text(j,i,f'{v:,}',ha='center',va='center',fontsize=12,
                color=('white' if v>values.max()*.5 else 'black'))
    ax.set_xticks([0,1],['Predicted 0','Predicted 1']);ax.set_yticks([0,1],['True 0','True 1'])
    ax.set_title(f'Confusion matrix — locked threshold {threshold:.3f}')
    fig.tight_layout();fig.savefig(Path(out)/'confusion_matrix_final.png',dpi=160);plt.close(fig)


def save_calibration(y,p,out):
    frac,mean=calibration_curve(y,p,n_bins=10,strategy='quantile')
    fig,ax=plt.subplots(figsize=(5.8,5.1))
    ax.plot([0,1],[0,1],'--',color='gray',label='perfect calibration')
    ax.plot(mean,frac,marker='o',label='final classifier')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted probability',
           ylabel='Observed positive fraction',title='Calibration (descriptive only)')
    ax.legend();fig.tight_layout();fig.savefig(Path(out)/'calibration_final.png',dpi=160);plt.close(fig)
    return float(brier_score_loss(y,p))


def subgroup_metrics(Xtest,y,p,threshold,min_n=50):
    out=[]
    for feature in ['Sex','Age']:
        if feature not in Xtest:continue
        for group,idx in Xtest.groupby(feature).indices.items():
            yy=np.asarray(y)[idx]
            reasons=[]
            if len(idx)<min_n:reasons.append(f'n below minimum {min_n}')
            if len(np.unique(yy))<2:reasons.append('single-class group')
            row={'feature':feature,'group_code':group,'n':len(idx),
                 'positive_n':int(yy.sum()),'threshold':float(threshold),
                 'status':'withheld' if reasons else 'evaluated',
                 'withholding_reason':'; '.join(reasons)}
            if not reasons:row.update(metrics_at_threshold(yy,np.asarray(p)[idx],threshold))
            out.append(row)
    return pd.DataFrame(out,columns=SUBGROUP_COLUMNS)


def error_cases(Xtest,pred,threshold):
    y=pred.y_true.to_numpy(dtype=int)
    p=pred.prob_FINAL.to_numpy(dtype=float)
    guess=(p>=threshold).astype(int)
    df=pd.DataFrame({'row_index':pred.row_index,'truth':y,'prediction':guess,
                     'probability':p})
    condition={'FN':(y==1)&(guess==0),'FP':(y==0)&(guess==1),
               'TP':(y==1)&(guess==1),'TN':(y==0)&(guess==0)}
    output=[];chosen={}
    for kind,mask in condition.items():
        sub=df.loc[mask].copy()
        if kind=='FN':sub=sub.sort_values('probability',ascending=True)
        elif kind=='FP':sub=sub.sort_values('probability',ascending=False)
        else:sub=sub.sort_values('probability',ascending=False)
        samples=sub.head(5)
        if not samples.empty:
            chosen[kind]=int(samples.index[0])  # POSITION within test, never an ID for a respondent
            for _,row in samples.iterrows():output.append({'error_type':kind,**row.to_dict()})
    return pd.DataFrame(output),chosen


def plot_shap(final_model,Xtrain,Xtest,out,seed=SEED,n=15,background_n=25,*,
              source_train_indices=None,source_test_indices=None,a4_run_id=None,final_model_id=None):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError('SHAP is optional; from the A4/A5 component install requirements-explain.txt') from exc
    rng=np.random.default_rng(seed)
    sample_ix=np.sort(rng.choice(len(Xtest),size=min(n,len(Xtest)),replace=False))
    background_ix=np.sort(rng.choice(len(Xtrain),size=min(background_n,len(Xtrain)),replace=False))
    sample=Xtest.iloc[sample_ix]
    background=Xtrain.iloc[background_ix]
    cols=list(Xtrain.columns)
    def predict_positive(a):
        a=pd.DataFrame(np.asarray(a),columns=cols)
        return final_model.predict_proba(a)[:,1]
    max_evals=max(2*len(cols)+1,50)
    explainer=shap.Explainer(predict_positive,background,algorithm='permutation',seed=seed)
    explanation=explainer(sample,max_evals=max_evals)
    np.savez_compressed(Path(out)/'shap_attributions.npz',values=explanation.values,
                        base_values=explanation.base_values,data=sample.to_numpy())
    shap.plots.beeswarm(explanation,max_display=14,show=False)
    plt.gcf().set_size_inches(9,6);plt.tight_layout();plt.savefig(Path(out)/'shap_beeswarm_final.png',dpi=160,bbox_inches='tight');plt.close('all')
    shap.plots.bar(explanation,max_display=14,show=False)
    plt.gcf().set_size_inches(8,6);plt.tight_layout();plt.savefig(Path(out)/'shap_bar_final.png',dpi=160,bbox_inches='tight');plt.close('all')
    shap.plots.waterfall(explanation[0],max_display=12,show=False)
    plt.gcf().set_size_inches(8,6);plt.tight_layout();plt.savefig(Path(out)/'shap_waterfall_final.png',dpi=160,bbox_inches='tight');plt.close('all')
    # No test-based refitting; explainability only, output is probability of positive class.
    write_json(Path(out)/'shap_metadata.json',{'model':'final model pipeline',
        'a4_run_id':a4_run_id,'final_model_id':final_model_id,
        'output':'P(label=1), NOT log-odds','algorithm':'permutation',
        'background_rows':len(background),'explained_rows':len(sample),
        'random_seed':int(seed),'explainer_seed':int(seed),'max_evals':max_evals,
        'feature_names':cols,'explained_test_positions':sample_ix,
        'background_train_positions':background_ix,
        'explained_source_row_indices':(None if source_test_indices is None else
                                         np.asarray(source_test_indices)[sample_ix]),
        'background_source_row_indices':(None if source_train_indices is None else
                                         np.asarray(source_train_indices)[background_ix]),
        'waterfall_test_position':int(sample_ix[0]),
        'attributions_artifact':'shap_attributions.npz',
        'interpretation':'Associations with predictions; not clinical/causal effects'})


def plot_lime(final_model,Xtrain,Xtest,chosen,out,seed=SEED,num_samples=1000,*,
              source_train_indices=None,source_test_indices=None,a4_run_id=None,
              final_model_id=None,case_details=None):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as e:
        raise RuntimeError('LIME is optional; from the A4/A5 component install requirements-explain.txt') from e
    cols=list(Xtrain.columns)
    categorical=[i for i,c in enumerate(cols) if c in CATEGORICAL]
    explainer=LimeTabularExplainer(Xtrain.to_numpy(dtype=float),feature_names=cols,
          categorical_features=categorical,mode='classification',
          class_names=['No survey diabetes label','Survey diabetes/prediabetes label'],
          discretize_continuous=True,random_state=seed)
    def proba(a):return final_model.predict_proba(pd.DataFrame(a,columns=cols))
    cases={}
    for case in ['TP','FP','FN']:
        if case not in chosen:continue
        i=chosen[case]
        ex=explainer.explain_instance(Xtest.iloc[i].to_numpy(dtype=float),proba,
                          num_features=min(10,len(cols)),num_samples=num_samples,labels=(1,))
        ex.save_to_file(str(Path(out)/f'lime_{case}.html'))
        fig=ex.as_pyplot_figure(label=1)
        fig.tight_layout();fig.savefig(Path(out)/f'lime_{case}.png',dpi=160,bbox_inches='tight');plt.close(fig)
        cases[case]={'test_position':int(i),
                     'source_row_index':(None if source_test_indices is None else
                                          int(np.asarray(source_test_indices)[i])),
                     'local_weights':ex.as_list(label=1),'local_surrogate_score':float(ex.score),
                     'local_prediction':np.asarray(ex.local_pred).tolist(),
                     'details':(case_details or {}).get(case,{})}
    if source_train_indices is not None:
        np.save(Path(out)/'lime_training_source_indices.npy',source_train_indices,allow_pickle=False)
    write_json(Path(out)/'lime_metadata.json',{
        'a4_run_id':a4_run_id,'final_model_id':final_model_id,'random_seed':int(seed),
        'num_samples_per_case':int(num_samples),'num_features':min(10,len(cols)),
        'feature_names':cols,'categorical_feature_indices':categorical,
        'discretize_continuous':True,'output':'P(label=1), classification',
        'background_scope':'full A4 training partition','background_rows':len(Xtrain),
        'background_source_indices_artifact':('lime_training_source_indices.npy'
                                               if source_train_indices is not None else None),
        'case_order':['TP','FP','FN'],'cases':cases,
        'interpretation':'Local surrogate associations with predictions; not clinical/causal effects'})


def run(data_path,a4_dir,output_dir,bootstraps=1000,shap=False,lime=False,allow_smoke=False):
    execution,started=start_execution('a5')
    if isinstance(bootstraps,bool) or not isinstance(bootstraps,(int,np.integer)) or bootstraps<1:
        raise ValueError('Bootstrap attempts must be a positive integer')
    a4=Path(a4_dir).expanduser().resolve()
    out=Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}')
    manifest=read_json(a4/'manifest.json')
    status=manifest.get('status')
    if status not in A4_STATUSES:
        raise ValueError('Unknown A4 research/QA status')
    if status!='RESEARCH_TEST_COMPLETE' and not allow_smoke:
        raise ValueError(f'A4 run has status {status}. A5 refuses to treat smoke as research.')
    if status=='RESEARCH_TEST_COMPLETE' and bootstraps!=BOOTSTRAP_ATTEMPTS:
        raise ValueError('Research A5 requires the protocol\'s 1000 bootstrap attempts; overrides are QA only')
    data_path=Path(data_path).expanduser().resolve()
    verified=validate_a4_artifacts(data_path,a4,manifest)
    selection=verified['selection']
    threshold=verified['final_threshold']
    final=verified['final_model']
    pred=verified['predictions']
    ytest=verified['ytest']
    p=pred.prob_FINAL.to_numpy(dtype=float)
    Xtest,Xtrain=verified['Xtest'],verified['Xtrain']
    runtime={
        'run_id':execution['run_id'],'source_a4_run_id':manifest['run_id'],
        'requested':{'data':str(data_path),'a4_dir':str(a4),'output_dir':str(out),
                     'bootstraps':int(bootstraps),'shap':bool(shap),'lime':bool(lime),
                     'allow_smoke':bool(allow_smoke)},
        'effective':{'status':status,'bootstrap':{
            'attempts':int(bootstraps),'random_seed':SEED,'confidence_level':0.95,
            'interval_method':'percentile','paired_same_indices':True,
            'exclusion_policy':'Exclude single-class samples; do not replace excluded attempts'},
            'subgroups':{'features':['Sex','Age'],'min_n':50,'require_both_classes':True},
            'explanations':{'shap':bool(shap),'lime':bool(lime),'random_seed':SEED},
            'final_model_id':selection['final_model'],'locked_threshold':threshold,
            'final_feature_names':selection['final_feature_names'],
            'probability_source':'recomputed from serialized A4 models; no refit'},
        'protocol_overrides':({} if bootstraps==BOOTSTRAP_ATTEMPTS else
                              {'bootstrap_attempts':{'protocol':BOOTSTRAP_ATTEMPTS,
                                                     'effective':int(bootstraps),'scope':'QA only'}}),
    }
    # Validation finishes before any analysis directory or statistics are created.
    out.mkdir(parents=True,exist_ok=True)
    analysis_provenance=save_analysis_provenance(out,runtime)
    write_json(out/'artifact_verification.json',{
        **verified['verification'],
        'source_manifest_sha256':verified['provenance']['source_manifest_sha256'],
        'verified_a4_artifact_hashes':verified['provenance']['verified_artifact_hashes'],
    })
    save_curves(ytest,pred,out)
    save_confusion(ytest,p,threshold,out)
    brier=save_calibration(ytest,p,out)
    bootstrap,summary=bootstrap_paired_ap(ytest,p,pred.prob_B3.to_numpy(),n_boot=bootstraps)
    bootstrap.to_csv(out/'paired_bootstrap_ap.csv',index=False)
    write_json(out/'paired_bootstrap_summary.json',summary)
    pd.DataFrame([{'metric':'brier_final','value':brier},
                  {'metric':'ap_delta_final_minus_B3','value':summary['delta_ap_point']}]).to_csv(
                   out/'additional_metrics.csv',index=False)
    subgroups=subgroup_metrics(Xtest,ytest,p,threshold)
    subgroups.to_csv(out/'subgroups_test.csv',index=False)
    examples,chosen=error_cases(Xtest,pred,threshold)
    examples.to_csv(out/'error_cases.csv',index=False)
    write_json(out/'case_selection.json',chosen)
    explanation_context={
        'source_train_indices':verified['source_train_indices'],
        'source_test_indices':verified['source_test_indices'],
        'a4_run_id':manifest['run_id'],'final_model_id':selection['final_model'],
    }
    if shap:plot_shap(final,Xtrain,Xtest,out,**explanation_context)
    if lime:
        details={case:{'truth':int(ytest[i]),'probability':float(p[i]),
                       'prediction':int(p[i]>=threshold),'locked_threshold':threshold}
                 for case,i in chosen.items()}
        plot_lime(final,Xtrain,Xtest,chosen,out,case_details=details,**explanation_context)
    provenance=verified['provenance']
    if (sha256_file(a4/'manifest.json')!=provenance['source_manifest_sha256']
            or artifact_hashes(a4,list(provenance['verified_artifact_hashes']))!=provenance['verified_artifact_hashes']
            or sha256_file(data_path)!=manifest['dataset_sha256']):
        raise ValueError('A4 artifacts or source CSV changed during A5 execution')
    if source_identity()['file_hashes']!=analysis_provenance['source_identity']['file_hashes']:
        raise ValueError('Component source changed during A5 execution')
    write_json(out/'manifest_a5.json',{
       **finish_execution(execution,started),'manifest_version':MANIFEST_VERSION,
       'source_a4_run_id':manifest['run_id'],
       'source_manifest':str(a4/'manifest.json'),'source_sha256':manifest['dataset_sha256'],
       'source_manifest_sha256':provenance['source_manifest_sha256'],
       'source_a4_artifact_hashes':provenance['verified_artifact_hashes'],
       'source_files_sha256':analysis_provenance['source_identity']['source_files_sha256'],
       'status':status,'final_model_id':selection['final_model'],
       'locked_threshold':threshold,'bootstrap_repeats':int(bootstraps),
       'bootstrap_valid_replicates':summary['n_valid_bootstrap'],
       'bootstrap_excluded_replicates':summary['n_excluded_bootstrap'],
       'shap':bool(shap),'lime':bool(lime),'artifact_hashes':artifact_hashes(out),
       'no_model_selection_based_on_test':True,
       'notes':'Analysis on locked test; no diagnostic or causal claims'})
    print('A5 complete:',out)
    print('Paired AP delta final - B3:',summary['delta_ap_point'],
          '95% CI:',summary['ci95_low'],summary['ci95_high'])
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--a4-dir',required=True)
    ap.add_argument('--output-dir',default='results/a5_run')
    ap.add_argument('--bootstraps',type=int,default=1000,help='1000 attempts for research; positive overrides for QA only')
    ap.add_argument('--shap',action='store_true')
    ap.add_argument('--lime',action='store_true')
    ap.add_argument('--allow-smoke',action='store_true',help='For QA only; real A5 rejects smoke')
    a=ap.parse_args()
    if a.bootstraps<1:ap.error('--bootstraps must be positive')
    run(a.data,a.a4_dir,a.output_dir,a.bootstraps,a.shap,a.lime,a.allow_smoke)

if __name__=='__main__':main()
