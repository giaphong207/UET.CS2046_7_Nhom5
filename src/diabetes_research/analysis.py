"""A5 V3: same-run AP bootstrap B2-B1, B3-B1, FINAL-B3; no refit/selection.

Positive-class F2 thresholds are locked on validation. Optional SHAP defaults:
500 explained rows / 100 training background rows, deterministic and chunked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve)
from .core import (CATEGORICAL, CORE_BOOTSTRAP_PAIRS, SHAP_DEFAULT_N,
                   SHAP_DEFAULT_BACKGROUND_N, SHAP_DEFAULT_CHUNK_SIZE,
                   metrics_at_threshold, sha256_file, threshold_policy, write_json, SEED)
from .shared_splits import array_sha256
from .paths import COMPONENT_ROOT
from .artifact_validation import A4_STATUSES, validate_a4_artifacts
from .provenance import (MANIFEST_VERSION, artifact_hashes, finish_execution, read_json,
                         save_analysis_provenance, source_identity, start_execution)

BOOTSTRAP_ATTEMPTS = 1000
SUBGROUP_COLUMNS = [
    'feature', 'group_code', 'n', 'positive_n', 'status', 'withholding_reason',
    'average_precision', 'roc_auc', 'accuracy', 'precision', 'recall',
    'specificity', 'f1', 'f2', 'threshold', 'tn', 'fp', 'fn', 'tp',
]


def _bootstrap_family(y, probabilities, pairs, *, n_boot, seed, case_indices,
                      a4_run_id, aliases=None, split_archive_sha256=None):
    """One bounded draw stream for a frozen family of same-case AP comparisons."""
    y=np.asarray(y)
    scores={key:np.asarray(value,dtype=float) for key,value in probabilities.items()}
    if y.ndim != 1 or set(np.unique(y))!={0,1}:
        raise ValueError('Test must have both classes')
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError('Bootstrap attempts must be a positive integer')
    for value in scores.values():
        if value.ndim != 1 or len(value) != len(y):
            raise ValueError('Arrays must be one-dimensional and have same length')
        if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError('Bootstrap scores must be finite probabilities in [0, 1]')
    cases=np.asarray(case_indices)
    if (cases.ndim != 1 or len(cases) != len(y) or cases.dtype.kind not in 'iu'
            or (cases < 0).any() or len(np.unique(cases)) != len(cases)):
        raise ValueError('Bootstrap case indices must be unique aligned source-row positions')
    aliases=aliases or {key:key for key in scores}
    alias_scores={}
    for key,value in scores.items():
        resolved=aliases[key]
        if resolved in alias_scores and not np.array_equal(value,alias_scores[resolved]):
            raise ValueError('The same fitted model identity has inconsistent scores')
        alias_scores[resolved]=value
    if len({key for key,_,_ in pairs}) != len(pairs):
        raise ValueError('Bootstrap comparison keys must be unique')
    for _,left,right in pairs:
        if left not in scores or right not in scores:
            raise ValueError('Missing probability column for a prespecified comparison')
        if aliases[left] == aliases[right] and not np.array_equal(scores[left],scores[right]):
            raise ValueError('Self-comparison model identity has inconsistent scores')
    y=y.astype(int)
    binding=dict(a4_run_id=a4_run_id,case_indices_sha256=array_sha256(cases),
                 original_labels_sha256=array_sha256(y),seed=int(seed),
                 split_archive_sha256=split_archive_sha256,
                 algorithm='numpy.default_rng.integers; N row draws with replacement per attempt')
    stream_id=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    score_hashes={key:hashlib.sha256(np.asarray(value,dtype='<f8').tobytes()).hexdigest()
                  for key,value in scores.items()}
    comparison_ids={key:hashlib.sha256(
        json.dumps({**binding,'comparison_key':key,'left_model':aliases[left],
                    'right_model':aliases[right],'left_scores_sha256':score_hashes[left],
                    'right_scores_sha256':score_hashes[right]},sort_keys=True).encode()).hexdigest()
        for key,left,right in pairs}
    rng=np.random.default_rng(seed)
    rows=[];excluded=[]
    for i in range(n_boot):
        ix=rng.integers(0,len(y),size=len(y))
        draw=dict(replicate=i,draw_stream_id=stream_id,
                  sample_source_indices_sha256=array_sha256(cases[ix]))
        if len(np.unique(y[ix]))<2:
            excluded.append({**draw,'reason':'single_class','only_class':int(y[ix][0])})
            continue
        aps={}
        for key,value in scores.items():
            resolved=aliases[key]
            if resolved not in aps:
                aps[resolved]=float(average_precision_score(y[ix],value[ix]))
        for key,left,right in pairs:
            left_ap,right_ap=aps[aliases[left]],aps[aliases[right]]
            rows.append({**draw,'comparison_key':key,'comparison_id':comparison_ids[key],
                         'a4_run_id':a4_run_id,'left_model_id':left,'right_model_id':right,
                         'ap_left':left_ap,'ap_right':right_ap,'delta_ap':left_ap-right_ap})
    if not rows:raise RuntimeError('All bootstrap replications were single-class.')
    df=pd.DataFrame(rows)
    summaries={}
    for key,left,right in pairs:
        paired=df.loc[df.comparison_key==key]
        ap_left=float(average_precision_score(y,scores[left]))
        ap_right=float(average_precision_score(y,scores[right]))
        low,high=np.quantile(paired.delta_ap,[0.025,0.975],method='linear')
        is_self=aliases[left]==aliases[right]
        summaries[key]=dict(
            comparison_key=key,comparison_id=comparison_ids[key],a4_run_id=a4_run_id,
            left_model_id=left,right_model_id=right,
            resolved_left_model_id=aliases[left],resolved_right_model_id=aliases[right],
            left_scores_sha256=score_hashes[left],right_scores_sha256=score_hashes[right],
            observed_ap_left=ap_left,observed_ap_right=ap_right,
            delta_ap_point=ap_left-ap_right,ci95_low=float(low),ci95_high=float(high),
            n_valid_bootstrap=len(paired),n_attempted_bootstrap=int(n_boot),
            n_excluded_bootstrap=len(excluded),
            excluded_replicate_indices=[row['replicate'] for row in excluded],
            exclusion_policy='Exclude single-class samples; do not replace excluded attempts',
            random_seed=int(seed),confidence_level=0.95,
            interval_method='percentile (2.5%, 97.5%); linear quantiles',
            paired_same_indices=True,comparison=f'AP_{left} minus AP_{right}',
            sign_convention='positive delta favors the left model; AP_left - AP_right',
            self_comparison=is_self,identical_scores=bool(np.array_equal(scores[left],scores[right])),
            self_comparison_notice=('Same fitted model on the same cases: zero delta and CI; no evidence of a model improvement'
                                    if is_self else None),
            interpretation=('CI includes zero: no clear difference under this bootstrap'
                            if low<=0<=high else 'CI excludes zero under this bootstrap'),
            scope='Conditional on already-fitted models and this test split; marginal CI, not simultaneous coverage',
            draw_stream_id=stream_id,**{k:v for k,v in binding.items() if k not in ('a4_run_id','seed')})
    exclusion_frame=pd.DataFrame(excluded,columns=[
        'replicate','draw_stream_id','sample_source_indices_sha256','reason','only_class'])
    return df,summaries,exclusion_frame


def bootstrap_paired_ap(y,p_left,p_right,n_boot=1000,seed=SEED,*,left_id='FINAL',
                        right_id='B3',case_indices=None,a4_run_id='standalone_QA'):
    """Generic two-score helper; A5 dispatches exactly the three prespecified pairs."""
    key=f'{left_id}_minus_{right_id}'
    # Distinct storage keys also permit explicitly named self-comparisons.
    frame,summaries,_=_bootstrap_family(
        y,{'left':p_left,'right':p_right},[(key,'left','right')],n_boot=n_boot,seed=seed,
        case_indices=np.arange(len(y)) if case_indices is None else case_indices,
        a4_run_id=a4_run_id,aliases={'left':left_id,'right':right_id})
    summary=summaries[key]
    summary.update(left_model_id=left_id,right_model_id=right_id,
                   comparison=f'AP_{left_id} minus AP_{right_id}')
    frame['left_model_id']=left_id
    frame['right_model_id']=right_id
    return frame,summary


def paired_ap_comparisons(predictions,*,a4_run_id,prediction_run_id,
                          expected_case_indices,original_labels,final_model_id,
                          split_archive_sha256,n_boot=1000,seed=SEED):
    """Use original locked labels/order; reject cross-run or mismatched same-case pairing."""
    if not a4_run_id or prediction_run_id != a4_run_id:
        raise ValueError('Paired comparisons require predictions from the same locked A4 run')
    if ('row_index' not in predictions or 'y_true' not in predictions
            or not np.array_equal(predictions.row_index.to_numpy(),expected_case_indices)):
        raise ValueError('Paired comparisons require the same original test case indices and order')
    if not np.array_equal(predictions.y_true.to_numpy(),original_labels):
        raise ValueError('Paired comparisons require original test labels')
    ids=['B1','B2','B3','FINAL']
    if any(f'prob_{key}' not in predictions for key in ids):
        raise ValueError('Missing probability column for a prespecified comparison')
    frame,summaries,excluded=_bootstrap_family(
        original_labels,{key:predictions[f'prob_{key}'].to_numpy() for key in ids},
        CORE_BOOTSTRAP_PAIRS,n_boot=n_boot,seed=seed,
        case_indices=expected_case_indices,a4_run_id=a4_run_id,
        aliases={**{key:key for key in ids},'FINAL':final_model_id},
        split_archive_sha256=split_archive_sha256)
    return frame,dict(schema_version=3,a4_run_id=a4_run_id,
                      comparison_order=[key for key,_,_ in CORE_BOOTSTRAP_PAIRS],
                      comparisons=summaries,exclusion_log='paired_bootstrap_exclusions.csv',
                      no_refitting=True,no_model_selection=True),excluded


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


def plot_shap(final_model,Xtrain,Xtest,out,seed=SEED,n=SHAP_DEFAULT_N,
              background_n=SHAP_DEFAULT_BACKGROUND_N,*,chunk_size=SHAP_DEFAULT_CHUNK_SIZE,
              prediction_batch_size=2048,
              source_train_indices=None,source_test_indices=None,a4_run_id=None,final_model_id=None):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError('SHAP is optional; from the A4/A5 component install requirements-explain.txt') from exc
    settings=dict(n=n,background_n=background_n,chunk_size=chunk_size,
                  prediction_batch_size=prediction_batch_size)
    if any(isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1
           for value in settings.values()):
        raise ValueError('SHAP sample/background/chunk/prediction budgets must be positive integers')
    if not len(Xtrain) or not len(Xtest) or list(Xtrain.columns)!=list(Xtest.columns):
        raise ValueError('SHAP requires nonempty train/test with matching feature order')
    source_ids={}
    for name,ids,data in [('train',source_train_indices,Xtrain),('test',source_test_indices,Xtest)]:
        values=np.asarray(ids)
        if (values.ndim!=1 or len(values)!=len(data) or values.dtype.kind not in 'iu'
                or (values<0).any() or len(np.unique(values))!=len(values)):
            raise ValueError('SHAP requires the aligned original source-row positions for both partitions')
        source_ids[name]=values
    if np.intersect1d(source_ids['train'],source_ids['test']).size:
        raise ValueError('SHAP train/test source-row positions overlap')
    rng=np.random.default_rng(seed)
    sample_ix=np.sort(rng.choice(len(Xtest),size=min(n,len(Xtest)),replace=False))
    background_ix=np.sort(rng.choice(len(Xtrain),size=min(background_n,len(Xtrain)),replace=False))
    sample=Xtest.iloc[sample_ix]
    background=Xtrain.iloc[background_ix]
    cols=list(Xtrain.columns)
    def predict_positive(a):
        a=pd.DataFrame(np.asarray(a),columns=cols)
        return np.concatenate([final_model.predict_proba(a.iloc[start:start+prediction_batch_size])[:,1]
                               for start in range(0,len(a),prediction_batch_size)])
    max_evals=max(2*len(cols)+1,50)
    mask_batch_size=max(1,prediction_batch_size//len(background))
    selected_test_ids=source_ids['test'][sample_ix]
    selected_train_ids=source_ids['train'][background_ix]
    metadata={'status':'SELECTED_ROWS_PENDING_ATTRIBUTIONS','model':'final model pipeline',
        'a4_run_id':a4_run_id,'final_model_id':final_model_id,
        'output':'P(Diabetes_binary=1), NOT log-odds','algorithm':'permutation',
        'requested_explained_rows':int(n),'requested_background_rows':int(background_n),
        'defaults':{'n':SHAP_DEFAULT_N,'background_n':SHAP_DEFAULT_BACKGROUND_N},
        'background_rows':len(background),'explained_rows':len(sample),
        'random_seed':int(seed),'explainer_seed':int(seed),'max_evals':max_evals,
        'chunk_size':int(chunk_size),'prediction_batch_size':int(prediction_batch_size),
        'mask_batch_size':int(mask_batch_size),
        'masker':'Independent','masker_max_samples':len(background),
        'sampling':'default_rng(seed); sorted choices without replacement; test then train',
        'feature_names':cols,'explained_test_positions':sample_ix,
        'background_train_positions':background_ix,
        'explained_source_row_indices':selected_test_ids,
        'background_source_row_indices':selected_train_ids,
        'waterfall_test_position':int(sample_ix[0]),
        'waterfall_source_row_index':int(selected_test_ids[0]),
        'selected_rows_artifact':'shap_selected_source_rows.npz',
        'attributions_artifact':'shap_attributions.npz',
        'interpretation':'Associations with predictions; not clinical/causal effects'}
    np.savez_compressed(Path(out)/'shap_selected_source_rows.npz',
                        explained_source_row_indices=selected_test_ids,
                        background_source_row_indices=selected_train_ids,
                        explained_test_positions=sample_ix,background_train_positions=background_ix)
    write_json(Path(out)/'shap_metadata.json',metadata)
    # SHAP's automatic Independent masker defaults to max_samples=100.
    # Keep configurable backgrounds and their recorded source rows identical.
    masker=shap.maskers.Independent(background,max_samples=len(background))
    explainer=shap.Explainer(predict_positive,masker,algorithm='permutation',seed=seed)
    chunks=[]
    for start in range(0,len(sample),chunk_size):
        chunks.append(explainer(sample.iloc[start:start+chunk_size],max_evals=max_evals,
                                batch_size=mask_batch_size))
    values=np.concatenate([np.asarray(part.values) for part in chunks],axis=0)
    base_values=np.concatenate([np.atleast_1d(part.base_values) for part in chunks],axis=0)
    if values.shape!=(len(sample),len(cols)) or base_values.shape!=(len(sample),):
        raise RuntimeError('Unexpected positive-probability SHAP attribution shapes')
    explanation=shap.Explanation(values=values,base_values=base_values,
                                  data=sample.to_numpy(),feature_names=cols)
    np.savez_compressed(Path(out)/'shap_attributions.npz',values=explanation.values,
                        base_values=explanation.base_values,data=sample.to_numpy(),
                        explained_source_row_indices=selected_test_ids,
                        background_source_row_indices=selected_train_ids)
    shap.plots.beeswarm(explanation,max_display=14,show=False)
    plt.gcf().set_size_inches(9,6);plt.tight_layout();plt.savefig(Path(out)/'shap_beeswarm_final.png',dpi=160,bbox_inches='tight');plt.close('all')
    shap.plots.bar(explanation,max_display=14,show=False)
    plt.gcf().set_size_inches(8,6);plt.tight_layout();plt.savefig(Path(out)/'shap_bar_final.png',dpi=160,bbox_inches='tight');plt.close('all')
    shap.plots.waterfall(explanation[0],max_display=12,show=False)
    plt.gcf().set_size_inches(8,6);plt.tight_layout();plt.savefig(Path(out)/'shap_waterfall_final.png',dpi=160,bbox_inches='tight');plt.close('all')
    metadata.update(status='COMPLETED',chunks_processed=len(chunks))
    write_json(Path(out)/'shap_metadata.json',metadata)


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
          class_names=['Diabetes_binary=0','Diabetes_binary=1'],
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


def run(data_path,a4_dir,output_dir,bootstraps=1000,shap=False,lime=False,allow_smoke=False,*,
        shap_n=None,shap_background_n=None,shap_chunk_size=None,shap_prediction_batch_size=None):
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
    planned=verified['provenance']['runtime']['effective']['explanations']
    shap_settings={key:(planned['shap'][key] if value is None else value) for key,value in dict(
        n=shap_n,background_n=shap_background_n,chunk_size=shap_chunk_size,
        prediction_batch_size=shap_prediction_batch_size).items()}
    if any(isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1
           for value in shap_settings.values()):
        raise ValueError('SHAP sample/background/chunk/prediction budgets must be positive integers')
    runtime={
        'run_id':execution['run_id'],'source_a4_run_id':manifest['run_id'],
        'requested':{'data':str(data_path),'a4_dir':str(a4),'output_dir':str(out),
                     'bootstraps':int(bootstraps),'shap':bool(shap),'lime':bool(lime),
                     'allow_smoke':bool(allow_smoke),'shap_samples':shap_n,
                     'shap_background_samples':shap_background_n,'shap_chunk_size':shap_chunk_size,
                     'shap_prediction_batch_size':shap_prediction_batch_size},
        'effective':{'status':status,'bootstrap':{
            'attempts':int(bootstraps),'random_seed':SEED,'confidence_level':0.95,
            'interval_method':'percentile','paired_same_indices':True,
            'exclusion_policy':'Exclude single-class samples; do not replace excluded attempts'},
            'subgroups':{'features':['Sex','Age'],'min_n':50,'require_both_classes':True},
            'explanations':{'shap_enabled':bool(shap),'lime_enabled':bool(lime),'random_seed':SEED,
                            'shap':shap_settings,'lime':planned['lime']},
            'bootstrap_comparisons':manifest['bootstrap_comparisons'],
            'threshold_policy':threshold_policy(),
            'group_sensitivity_mode':manifest['group_sensitivity_mode'],
            'split_archive_sha256':manifest['split_archive_sha256'],
            'final_model_id':selection['final_model'],'locked_threshold':threshold,
            'final_feature_names':selection['final_feature_names'],
            'feature_selection':selection['feature_selection'],
            'selected_a3_model':selection['separately_reported_a3_model'],
            'selected_feature_k':selection['selected_feature_k'],
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
    bootstrap,summary,exclusions=paired_ap_comparisons(
        pred,a4_run_id=manifest['run_id'],prediction_run_id=verified['verification']['run_id'],
        expected_case_indices=verified['source_test_indices'],original_labels=ytest,
        final_model_id=selection['final_model'],split_archive_sha256=manifest['split_archive_sha256'],
        n_boot=bootstraps)
    bootstrap.to_csv(out/'paired_bootstrap_ap.csv',index=False)
    exclusions.to_csv(out/'paired_bootstrap_exclusions.csv',index=False)
    write_json(out/'paired_bootstrap_summary.json',summary)
    pd.DataFrame([{'metric':'brier_final','value':brier}]).to_csv(
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
    if shap:plot_shap(final,Xtrain,Xtest,out,**shap_settings,**explanation_context)
    if lime:
        details={case:{'truth':int(ytest[i]),'probability':float(p[i]),
                       'prediction':int(p[i]>=threshold),'locked_threshold':threshold}
                 for case,i in chosen.items()}
        plot_lime(final,Xtrain,Xtest,chosen,out,num_samples=planned['lime']['num_samples'],
                  case_details=details,**explanation_context)
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
       'feature_selection_status':manifest['feature_selection_status'],
       'selected_a3_model':selection['separately_reported_a3_model'],
       'selected_feature_k':selection['selected_feature_k'],
       'final_feature_names':selection['final_feature_names'],
       'threshold_policy':threshold_policy(),'split_archive_sha256':manifest['split_archive_sha256'],
       'shared_split_archive_sha256':manifest['shared_split_archive_sha256'],
       'bootstrap_comparisons':manifest['bootstrap_comparisons'],
       'bootstrap_valid_replicates':{key:value['n_valid_bootstrap'] for key,value in summary['comparisons'].items()},
       'bootstrap_excluded_replicates':{key:value['n_excluded_bootstrap'] for key,value in summary['comparisons'].items()},
       'bootstrap_comparison_ids':{key:value['comparison_id'] for key,value in summary['comparisons'].items()},
       'shap_samples':int(shap_settings['n']),'shap_background_samples':int(shap_settings['background_n']),
       'shap_configuration':shap_settings,'group_sensitivity_mode':manifest['group_sensitivity_mode'],
       'shap':bool(shap),'lime':bool(lime),'artifact_hashes':artifact_hashes(out),
       'no_model_selection_based_on_test':True,
       'notes':'Analysis on locked test; no diagnostic or causal claims'})
    print('A5 complete:',out)
    print('Paired AP comparisons:',summary['comparison_order'])
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True)
    ap.add_argument('--a4-dir',required=True)
    ap.add_argument('--output-dir',default=COMPONENT_ROOT/'results'/'a5_v3_run')
    ap.add_argument('--bootstraps',type=int,default=1000,help='1000 attempts for research; positive overrides for QA only')
    ap.add_argument('--shap',action='store_true')
    ap.add_argument('--shap-samples',type=int,help='Explained rows; V3 config default 500, deterministic subsampling')
    ap.add_argument('--shap-background-samples',type=int,help='Training background rows; V3 config default 100')
    ap.add_argument('--shap-chunk-size',type=int,help='Explained rows per chunk; V3 config default 25')
    ap.add_argument('--shap-prediction-batch-size',type=int,help='Model prediction rows per batch; V3 config default 2048')
    ap.add_argument('--lime',action='store_true')
    ap.add_argument('--allow-smoke',action='store_true',help='For QA only; real A5 rejects smoke')
    a=ap.parse_args()
    if a.bootstraps<1:ap.error('--bootstraps must be positive')
    for value in (a.shap_samples,a.shap_background_samples,a.shap_chunk_size,a.shap_prediction_batch_size):
        if value is not None and value<1:ap.error('SHAP budgets must be positive')
    run(a.data,a.a4_dir,a.output_dir,a.bootstraps,a.shap,a.lime,a.allow_smoke,
        shap_n=a.shap_samples,shap_background_n=a.shap_background_samples,
        shap_chunk_size=a.shap_chunk_size,shap_prediction_batch_size=a.shap_prediction_batch_size)

if __name__=='__main__':main()
