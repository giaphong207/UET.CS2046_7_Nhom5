"""Dataset checks, reproducible splits and threshold-aware evaluation."""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
    confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

TARGET = 'Diabetes_binary'
SEED = 42
BINARY = ['HighBP','HighChol','CholCheck','Smoker','Stroke',
          'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
          'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex']
CATEGORICAL = BINARY + ['GenHlth','Age','Education','Income']
EXPECTED = set(BINARY + ['BMI','MentHlth','PhysHlth','GenHlth','Age','Education','Income'])
SPLIT_NAMES = ('train', 'validation', 'test')
FEATURE_COUNTS = (21, 15, 10, 5)
FINAL_MODEL_TIE_RULE = 'highest validation AP; exact ties follow candidate_order'
F2_GRID = np.linspace(0.01, 0.99, 199)
SHAP_DEFAULT_N = 500
SHAP_DEFAULT_BACKGROUND_N = 100
SHAP_DEFAULT_CHUNK_SIZE = 25
CORE_BOOTSTRAP_PAIRS = (
    ('B2_minus_B1', 'B2', 'B1'),
    ('B3_minus_B1', 'B3', 'B1'),
    ('FINAL_minus_B3', 'FINAL', 'B3'),
)


def threshold_policy():
    return dict(reference=0.5, selection='maximize_f2_validation',
                positive_label=1, beta=2, grid=F2_GRID.tolist(),
                grid_construction='np.linspace(0.01, 0.99, 199)',
                grid_order='ascending', tie_rule='smallest_threshold_exact_tie',
                decision_rule='probability >= threshold')


def feature_selection_policy():
    """Implementation authorization is distinct from official evaluation release."""
    return dict(method='fold-local SelectFromModel', status='IMPLEMENTED',
                scope='PRIMARY', k_values=list(FEATURE_COUNTS),
                primary_method_implemented=True, training_cv_performed=True,
                estimator='FoldAwareXGB', estimator_configuration='selected B3 parameters',
                weight_mode='balanced', weight_scope='labels passed to each fit only',
                importance_type='gain', importance_getter='feature_importances_',
                threshold='-inf', prefit=False, norm_order=1,
                max_features='exact k', tie_rule='descending unrounded gain; original column order on exact ties',
                transform_order='original column order',
                fit_scope='each training CV fold; full primary train refit for each k',
                cv_interpretation='descriptive conditional on B3 selected using the same training CV; not nested CV',
                selection_scope='reserved validation AP only', reference='A2 full-feature',
                tolerance_ap=0.01, eligibility_numeric_tolerance=1e-12,
                fallback='A2 full-feature', secondary_qa_method='SelectKBest(f_classif)',
                secondary_used_for_primary_selection=False,
                ordinal_representation='numeric codes; LR scales all inputs')


def select_feature_budget(reference_ap, ap_by_k):
    """Smallest validation-eligible budget; the full-feature A2 remains fallback."""
    if set(ap_by_k) != set(FEATURE_COUNTS):
        raise ValueError('Feature AP must cover exactly 21/15/10/5')
    values = [reference_ap, *ap_by_k.values()]
    if any(isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Validation AP must be finite in [0, 1]')
    eligible = [k for k in sorted(FEATURE_COUNTS) if reference_ap-ap_by_k[k] <= 0.01+1e-12]
    return (min(eligible) if eligible else 21), eligible


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def load_dataset(path: str | Path, strict_columns: bool=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy CSV: {path}. Tải từ Kaggle và xem data/README.md')
    df = pd.read_csv(path)
    if TARGET not in df:
        raise ValueError(f'Thiếu nhãn {TARGET}. Đã chọn nhầm phiên bản CSV?')
    if df.columns.duplicated().any():
        raise ValueError('Tên cột trùng nhau trong CSV.')
    X = df.drop(columns=TARGET)
    if 'Unnamed: 0' in X.columns or 'index' in X.columns:
        raise ValueError('CSV chứa cột index không hợp lệ; kiểm tra phiên bản dữ liệu.')
    if strict_columns and (len(X.columns) != 21 or set(X.columns) != EXPECTED):
        missing = sorted(EXPECTED - set(X.columns))
        extra = sorted(set(X.columns) - EXPECTED)
        raise ValueError(f'Phải có đúng 21 biến BRFSS. Thiếu: {missing}; thừa: {extra}')
    if df.isna().any().any():
        raise ValueError('Dữ liệu có missing values: cần đưa imputation vào pipeline CV, không điền toàn bộ trước split.')
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError('Dữ liệu chứa Inf/-Inf hoặc giá trị không phải số.')
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError('Nhãn phải có đủ 0 và 1 (file diabetes_binary).')
    if (df[TARGET] % 1 != 0).any():
        raise ValueError('Nhãn không nguyên.')
    for c in BINARY:
        if c in X and not X[c].isin([0, 1]).all():
            raise ValueError(f'Giá trị không hợp lệ tại {c}, yêu cầu 0/1')
    ranges = {'GenHlth':(1,5),'Age':(1,13),'Education':(1,6),
              'Income':(1,8),'MentHlth':(0,30),'PhysHlth':(0,30)}
    for c, (lo,hi) in ranges.items():
        if c in X and (~X[c].between(lo,hi)).any():
            raise ValueError(f'{c} nằm ngoài miền mã hoá {lo}..{hi}')
        if c in X and (X[c] % 1 != 0).any():
            raise ValueError(f'{c} phải là mã/số ngày nguyên trong {lo}..{hi}')
    # BMI is positive by definition. No unsupported clinical upper/lower cutoff.
    if 'BMI' in X and (X['BMI'] <= 0).any():
        raise ValueError('BMI phải lớn hơn 0; không áp dụng ngưỡng loại trừ lâm sàng.')
    return X, df[TARGET].astype(int)


def dataset_audit(X:pd.DataFrame,y:pd.Series):
    # A vector of answers is not a respondent identifier.
    vc = X.value_counts(dropna=False)
    contradictory = (pd.concat([X,y.rename(TARGET)],axis=1)
                     .groupby(list(X.columns),dropna=False)[TARGET].nunique())
    return dict(rows=int(len(X)),features=int(X.shape[1]),
       columns=list(X.columns),positive_count=int(y.sum()),
       positive_fraction=float(y.mean()),missing_cells=int(X.isna().sum().sum()),
       duplicate_feature_vectors=int(X.duplicated().sum()),
       unique_feature_vectors=int(len(vc)),
       vectors_with_disagreeing_labels=int((contradictory>1).sum()),
       notes='Trùng vector đặc trưng không đồng nghĩa cùng người; không tự động xóa.')


def stratified_indices(y, seed=SEED):
    idx=np.arange(len(y))
    # Integer sizes avoid floating-point ceil changing a 70/15/15 split.
    n_test=int(round(len(y)*0.15))
    n_val=int(round(len(y)*0.15))
    remain,test=train_test_split(idx,test_size=n_test,random_state=seed,
                                stratify=np.asarray(y))
    train,val=train_test_split(remain,test_size=n_val,
                     random_state=seed,stratify=np.asarray(y)[remain])
    assert len(set(train)&set(val))==0
    assert len(set(train)&set(test))==0
    assert len(set(val)&set(test))==0
    assert len(train)+len(val)+len(test)==len(y)
    return {'train':np.sort(train), 'validation':np.sort(val),'test':np.sort(test)}


def validate_split_indices(splits, n_rows, expected_counts=None):
    """Check stored positions before indexing data, including hostile/corrupt arrays."""
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError('Split archive must contain train, validation and test only')
    checked = {}
    for name in SPLIT_NAMES:
        ix = np.asarray(splits[name])
        if ix.ndim != 1 or ix.dtype.kind not in 'iu':
            raise ValueError(f'{name} split indices must be a one-dimensional integer array')
        if not len(ix) or (ix < 0).any() or (ix >= n_rows).any():
            raise ValueError(f'{name} split indices are empty or out of bounds')
        if len(np.unique(ix)) != len(ix):
            raise ValueError(f'{name} split indices contain duplicates')
        checked[name] = ix
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if np.intersect1d(checked[left], checked[right]).size:
            raise ValueError(f'Split indices overlap: {left}/{right}')
    all_ix = np.concatenate([checked[name] for name in SPLIT_NAMES])
    if not np.array_equal(np.sort(all_ix), np.arange(n_rows)):
        raise ValueError('Split indices are not complete for the modeling dataset')
    if expected_counts is not None:
        if set(expected_counts) != set(SPLIT_NAMES):
            raise ValueError('Manifest split counts are incomplete')
        for name in SPLIT_NAMES:
            if len(checked[name]) != expected_counts[name]:
                raise ValueError(f'{name} split count differs from A4 manifest')
    return checked


def split_diagnostics(X, y, splits):
    """Exact vector overlap and class counts; vectors are not respondent IDs."""
    splits = validate_split_indices(splits, len(X))
    labels = np.asarray(y)
    counts, vectors = {}, {}
    for name, ix in splits.items():
        positive = int((labels[ix] == 1).sum())
        counts[name] = dict(rows=len(ix), negative_count=len(ix) - positive,
                            positive_count=positive, positive_fraction=positive / len(ix))
        vectors[name] = pd.MultiIndex.from_frame(X.iloc[ix])
    overlaps = {}
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        shared = vectors[left].unique().intersection(vectors[right].unique(), sort=False)
        overlaps[f'{left}__{right}'] = dict(
            shared_unique_vectors=len(shared),
            left_rows_with_shared_vector=int(vectors[left].isin(shared).sum()),
            right_rows_with_shared_vector=int(vectors[right].isin(shared).sum()))
    return dict(class_counts=counts, identical_vectors_across_splits=overlaps,
                group_definition='Exact equality of all input features, without target',
                interpretation='Identical feature vectors do not establish duplicate individuals')


def select_final_model(candidates, validation_ap):
    """Keep the existing predeclared candidate-order rule for exact AP ties."""
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('Final-model candidate order must be nonempty and unique')
    if set(validation_ap) != set(candidates):
        raise ValueError('Validation AP candidates do not match candidate order')
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in validation_ap.values()):
        raise ValueError('Validation AP must be finite and in [0, 1]')
    return max(candidates, key=lambda name: validation_ap[name])


def optimal_f1_threshold(y, scores):
    """Historical V2 QA helper; active A4 uses optimal_f2_threshold."""
    from sklearn.metrics import precision_recall_curve
    p,r,t=precision_recall_curve(y,np.asarray(scores))
    if not len(t):
        return 0.5
    f1=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],
                 out=np.zeros(len(t),dtype=float),where=(p[:-1]+r[:-1])>0)
    winners=np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0, atol=1e-12))
    return float(np.max(t[winners]))  # tie -> highest threshold


def f2_threshold_curve(y, scores):
    """Evaluate the fixed ascending grid without allocating a rows-by-grid matrix."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores)
            or set(np.unique(y)) != {0, 1}):
        raise ValueError('Validation labels/scores must align and contain both binary classes')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Validation scores must be finite probabilities in [0, 1]')
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    cumulative = np.concatenate(([0], np.cumsum(y[order], dtype=np.int64)))
    below = np.searchsorted(sorted_scores, F2_GRID, side='left')
    positives = int(y.sum())
    tp = positives - cumulative[below]
    fn = positives - tp
    fp = len(y) - below - tp
    denominator = 5 * tp + 4 * fn + fp
    values = np.divide(5 * tp, denominator, out=np.zeros(len(F2_GRID), dtype=float),
                       where=denominator != 0)
    return values


def optimal_f2_threshold(y, scores):
    # np.argmax returns the first exact maximum: smallest threshold, without tolerance.
    return float(F2_GRID[int(np.argmax(f2_threshold_curve(y, scores)))])


def metrics_at_threshold(y, prob, threshold=0.5):
    y=np.asarray(y,dtype=int)
    prob=np.asarray(prob,dtype=float)
    pred=(prob>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(average_precision=float(average_precision_score(y,prob)),
        roc_auc=float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else float('nan'),
        accuracy=float(accuracy_score(y,pred)),
        precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),
        specificity=float(tn/(tn+fp)) if tn+fp else float('nan'),
        f1=float(f1_score(y,pred,zero_division=0)),
        f2=float(fbeta_score(y,pred,beta=2,zero_division=0)),
        threshold=float(threshold),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def write_json(path, value):
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o,Path): return str(o)
        raise TypeError(type(o).__name__)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=convert),encoding='utf-8')


def versions():
    from importlib.metadata import version, PackageNotFoundError
    items={'python':sys.version.split()[0], 'platform':platform.platform(),
           'architecture':platform.machine()}
    for p in ['numpy','pandas','scikit-learn','xgboost','scipy','matplotlib','joblib',
              'PyYAML','nbformat','pytest','shap','lime','optuna']:
        try: items[p]=version(p)
        except PackageNotFoundError:items[p]='not installed'
    return items
