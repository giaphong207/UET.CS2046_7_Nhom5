
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- CONFIG
SEED     = 42
N_FOLDS  = 5
N_BOOT   = 1000
FAST     = False    # True = bo qua bootstrap (smoke test)
N_JOBS   = 2        # Kaggle: 2 CPU
OUT      = Path("report/results/B2_B3")   # CSV ket qua B2/B3 + idx_*.npy
OUT.mkdir(parents=True, exist_ok=True)
PLOTS    = Path("report/figures/B2_B3")  # bieu do B2/B3
PLOTS.mkdir(parents=True, exist_ok=True)

F_2L = "diabetes_binary_health_indicators_BRFSS2015.csv"      # bai chinh 86/14
F_3L = "diabetes_012_health_indicators_BRFSS2015.csv"        # nghiem phu 3-lop
RESAMPLES = ["none", "class_weight", "smote", "adasyn", "undersample"]

# ---- probe GPU + LightGBM ----
def gpu_available():
    try:
        out = subprocess.run("nvidia-smi -L", shell=True, capture_output=True,
                             text=True, timeout=15)
        return "NVIDIA" in out.stdout
    except Exception:
        return False

GPU = gpu_available()

try:
    import lightgbm as lgb
    LGBM_OK = True
except Exception:
    lgb, LGBM_OK = None, False

LGBM_GPU = False
if GPU and LGBM_OK:
    try:
        _p = lgb.LGBMClassifier(n_estimators=1, num_leaves=4,
                                device="gpu", gpu_use_dp=True, verbose=-1)
        _Xp = np.random.default_rng(0).integers(0, 2, (32, 3))
        _yp = np.random.default_rng(0).integers(0, 2, 32)
        _p.fit(_Xp, _yp)
        _p.predict_proba(np.zeros((4, 3)))
        LGBM_GPU = True
    except Exception as e:
        print(f"[gpu] lightgbm GPU khong dung duoc -> CPU: {e}")
print(f"[deps] lightgbm={'OK' if LGBM_OK else 'THIEU'} | "
      f"GPU={'LIGHTGBM-GPU' if LGBM_GPU else 'CPU (fallback)'}")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.neighbors import NearestNeighbors
except ImportError as e:
    raise SystemExit(f"[ERROR] thieu scikit-learn: {e}")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    IML_OK = True
except Exception:
    IML_OK = False
print(f"[deps] imbalanced-learn={'OK' if IML_OK else 'fallback numpy'}")

# ============================================================
#  PHAN 1 - TIM DU LIEU + TIM COT TARGET
# ============================================================
def find_csv(name):
    here = Path.cwd()
    roots = [here]
    for _ in range(4):
        here = here.parent
        roots.append(here)
    roots += [Path("/kaggle/input")]          # <- Them cho Kaggle
    for r in roots:
        d = r / "input"
        if d.exists():
            m = list(d.rglob(name))
            if m:
                return m[0]
    for c in (Path(name), Path("data") / name):
        if c.exists():
            return c
    return None

def find_target(df, prefer=None):
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in (prefer, "diabetes_binary", "diabetes_012", "diabetes"):
        if cand and cand in low:
            return low[cand]
    for c in df.columns:
        if "diab" in str(c).lower() and 1 < df[c].nunique(dropna=True) <= 5:
            print(f"  [WARN] dung cot fallback '{c}' - kiem tra!")
            return c
    raise SystemExit("[ERROR] khong tim thay cot target")

# ============================================================
#  PHAN 2 - TIEN XU LY
# ============================================================
def prepare(df, tcol, dedup=True, noise="majority"):
    stats = {"n_raw": len(df), "n_changed_by_noise": 0,
             "n_noisy_groups": 0, "n_collapsed_after_noise": 0}
    df = df.copy()
    if dedup:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
    stats["n_after_dedup"] = len(df)

    feats = [c for c in df.columns if c != tcol]
    if noise == "majority":
        grp = pd.util.hash_pandas_object(df[feats], index=False)
        nu = df.groupby(grp)[tcol].transform("nunique")
        noisy_mask = nu > 1
        if noisy_mask.any():
            noisy_grp = grp[noisy_mask]
            mode_map = (df.loc[noisy_mask].assign(_g=noisy_grp)
                       .groupby("_g")[tcol].agg(lambda s: s.mode().iloc[0]))
            new_vals = noisy_grp.map(mode_map)
            old = df.loc[noisy_mask, tcol].to_numpy()
            df.loc[noisy_mask, tcol] = new_vals.to_numpy()
            stats["n_changed_by_noise"] = int((old != new_vals.to_numpy()).sum())
            stats["n_noisy_groups"] = int(mode_map.size)
            n_before = len(df)
            df = df.drop_duplicates(keep="first").reset_index(drop=True)
            stats["n_collapsed_after_noise"] = n_before - len(df)
    stats["n_final"] = len(df)
    return df, feats, stats

def encode(df, feats):
    X = df[feats].copy()
    for c in feats:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X.astype(np.float64)

# ============================================================
#  PHAN 3 - RESAMPLE (imblearn chuan, fallback numpy)
# ============================================================
def resample(X, y, strategy, seed=SEED):
    X = np.asarray(X); y = np.asarray(y)
    if strategy in ("none", "class_weight"):
        return X, y
    if IML_OK:
        s = {"smote": SMOTE(random_state=seed),
             "adasyn": ADASYN(random_state=seed),
             "undersample": RandomUnderSampler(random_state=seed)}[strategy]
        return s.fit_resample(X, y)
    # ---- fallback thuan numpy ----
    rng = np.random.default_rng(seed)
    if strategy == "undersample":
        im, ma = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(ma) <= len(im):
            return X, y
        keep = np.sort(rng.choice(ma, len(im), replace=False))
        idx = np.sort(np.concatenate([keep, im]))
        return X[idx], y[idx]
    Xm, Xa = X[y == 1], X[y == 0]
    extra = len(Xa) - len(Xm)
    if extra <= 0 or len(Xm) < 2:
        return X, y
    k = min(5, len(Xm) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(Xm)
    nn_idx = nn.kneighbors(Xm, return_distance=False)
    p = np.ones(len(Xm))
    if strategy == "adasyn":
        d = np.linalg.norm(Xm[nn_idx[:, 1:]] - Xm[:, None, :], axis=2).mean(axis=1)
        p = (1.0 / (d + 1e-6)) ** 2
    p /= p.sum()
    pick = rng.choice(len(Xm), size=extra, replace=True, p=p)
    which = nn_idx[pick, 1 + rng.integers(0, k, size=extra)]
    r = rng.random(extra)[:, None]
    gen = Xm[pick] + r * (Xm[which] - Xm[pick])
    return (np.vstack([X, gen]),
            np.concatenate([y, np.ones(extra, dtype=y.dtype)]))

# ============================================================
#  PHAN 4 - DANH GIA
# ============================================================
def metrics_at_threshold(y, prob, thr):
    yhat = (np.asarray(prob) >= thr).astype(int)
    out = {"AP": round(float(average_precision_score(y, prob)), 4),
           "accuracy": round(float(accuracy_score(y, yhat)), 4),
           "balanced_acc": round(float(balanced_accuracy_score(y, yhat)), 4),
           "f1_macro": round(float(f1_score(y, yhat, average="macro")), 4),
           "f1_class1": round(float(f1_score(y, yhat, pos_label=1, zero_division=0)), 4),
           "precision_1": round(float(precision_score(y, yhat, pos_label=1, zero_division=0)), 4),
           "recall_1": round(float(recall_score(y, yhat, pos_label=1, zero_division=0)), 4),
           "specificity": round(float(recall_score(y, yhat, pos_label=0, zero_division=0)), 4),
           "mcc": round(float(matthews_corrcoef(y, yhat)), 4),
           "auc_roc": round(float(roc_auc_score(y, prob)), 4),
           "threshold": round(float(thr), 4)}
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out.update({"n_tp": int(tp), "n_fp": int(fp), "n_fn": int(fn), "n_tn": int(tn)})
    return out

def best_f1_threshold(y, prob):
    """Nguong toi uu F1 lop benh TREN VALIDATION.
    Grid 0.01-0.99 (199 muc). Bien 'tie-break' lay nguong NHO nhat khi F1 bang
    nhau -> uu tien recall (it bo sot nguoi benh) vi muc tieu la so phat hien
    moi, khong phai do chinh xac.
    """
    grid = np.linspace(0.01, 0.99, 199)
    best, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y, (prob >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best, best_f1 = float(t), float(f1)
    return best

def paired_bootstrap(y, prob_a, prob_b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            deltas[b] = 0.0
            continue
        deltas[b] = (average_precision_score(yb, prob_a[idx])
                     - average_precision_score(yb, prob_b[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return float(deltas.mean()), float(lo), float(hi), float(min(p, 1.0))

# ============================================================
#  PHAN 5 - MO HINH (A/C THAY HAM NAY)
#  GPU: chi dung cho LightGBM. Data ~70k x 43 (chieu hoc)
#  -> force_row_wise=1 hieu qua hon col-parallel tren GPU.
# ============================================================
def lgb_kwargs(strategy):
    cw = "balanced" if strategy == "class_weight" else None
    kw = dict(n_estimators=300, learning_rate=0.08, random_state=SEED,
              n_jobs=N_JOBS, verbose=-1)
    if cw:
        kw["class_weight"] = "balanced"
    if LGBM_GPU:
        kw.update(device="gpu", gpu_use_dp=True,
                  gpu_platform_id=0, gpu_device_id=0,
                  force_row_wise=True)
    return kw

def build_model(model_id="primary", strategy="none"):
    cw = "balanced" if strategy == "class_weight" else None
    if model_id == "logistic":
        return LogisticRegression(max_iter=5000, class_weight=cw, random_state=SEED)
    if model_id == "decision_tree":
        return DecisionTreeClassifier(class_weight=cw, random_state=SEED)
    if model_id == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight=cw,
                                      random_state=SEED, n_jobs=N_JOBS)
    if model_id == "stacking":
        from sklearn.ensemble import StackingClassifier
        base = [("rf", RandomForestClassifier(n_estimators=200, random_state=SEED,
                                               n_jobs=N_JOBS, class_weight=cw))]
        if LGBM_OK:
            base.append(("lgbm", lgb.LGBMClassifier(**lgb_kwargs(strategy))))
        return StackingClassifier(estimators=base,
                                  final_estimator=LogisticRegression(max_iter=5000),
                                  cv=3)
    # ---- model chinh (default: primary) ----
    if LGBM_OK:
        return lgb.LGBMClassifier(**lgb_kwargs(strategy))
    return RandomForestClassifier(n_estimators=300, class_weight=cw,
                                  random_state=SEED, n_jobs=N_JOBS)

# ============================================================
#  PHAN 6 - CV (5-fold stratified, resample TRONG fold)
# ============================================================
def cv_evaluate(X, y, strategy, model_id="primary", n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aps, rec1, f1m, aucs = [], [], [], []
    for tr_i, va_i in skf.split(X, y):
        Xr, yr = resample(X[tr_i], y[tr_i], strategy, seed)
        m = build_model(model_id, strategy).fit(Xr, yr)
        p = m.predict_proba(X[va_i])[:, 1]
        yhat = (p >= 0.5).astype(int)
        aps.append(average_precision_score(y[va_i], p))
        rec1.append(recall_score(y[va_i], yhat, pos_label=1, zero_division=0))
        f1m.append(f1_score(y[va_i], yhat, average="macro"))
        aucs.append(roc_auc_score(y[va_i], p))
    return {"cv_AP": round(float(np.mean(aps)), 4), "cv_AP_std": round(float(np.std(aps)), 4),
            "cv_recall_1": round(float(np.mean(rec1)), 4),
            "cv_f1_macro": round(float(np.mean(f1m)), 4),
            "cv_auc": round(float(np.mean(aucs)), 4)}

# ============================================================
#  PHAN 7 - CHAY TOAN BUOI
# ============================================================
def main():
    t0 = time.time()
    path = find_csv(F_2L)
    if path is None:
        raise SystemExit(f"[ERROR] khong tim thay {F_2L}. "
                         "Add Input dataset 'diabetes-health-indicators-dataset'.")
    print(f"[OK] Du lieu: {path}")
    df_raw = pd.read_csv(path)
    tcol = find_target(df_raw)
    vc = df_raw[tcol].value_counts(normalize=True).round(4).to_dict()
    print(f"[data] shape={df_raw.shape} | target='{tcol}' | phan bo: {vc}")

    # ---- 7.1 chu bien so sanh (doi DUNG 1 yeu to) ----
    variants = {"dedup+noise-majority": dict(dedup=True, noise="majority"),
                "dedup+noise-keep":     dict(dedup=True, noise="keep"),
                "nodup+noise-majority": dict(dedup=False, noise="majority")}
    prepared = {}
    for name, kw in variants.items():
        d, feats, st = prepare(df_raw, tcol, **kw)
        prepared[name] = (d, feats, st)
        print(f"[prep] {name:22s} raw={st['n_raw']:,} -> final={st['n_final']:,} "
              f"| doi {st['n_changed_by_noise']:,} nhan/{st['n_noisy_groups']:,} nhom "
              f"| gop {st['n_collapsed_after_noise']:,}")

    BASE = "dedup+noise-majority"
    d, feats, st = prepared[BASE]
    X_all = encode(d, feats).to_numpy()
    y_all = d[tcol].to_numpy()

    # ---- 7.2 split stratified 70/15/15 seed 42 ----
    tr_i, tmp_i = train_test_split(np.arange(len(y_all)), test_size=0.30,
                                   stratify=y_all, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=y_all[tmp_i],
                                  random_state=SEED)
    Xtr, ytr = X_all[tr_i], y_all[tr_i]
    Xva, yva = X_all[va_i], y_all[va_i]
    Xte, yte = X_all[te_i], y_all[te_i]
    print(f"[split] train={len(ytr):,} | val={len(yva):,} | test={len(yte):,} (seed={SEED})")
    print(f"[split] ty le lop benh (train/val/test): "
          f"{ytr.mean():.4f} / {yva.mean():.4f} / {yte.mean():.4f}")
    # luu chi so split vao report/results (ban .gitignore bo *_out/ nen
    # chi so nam o b2_out se khong duoc commit -> khong tai lap duoc)
    np.save(OUT / "idx_train.npy", tr_i)
    np.save(OUT / "idx_val.npy", va_i)
    np.save(OUT / "idx_test.npy", te_i)
    pd.DataFrame({"feature": feats, "index_order": range(len(feats))}).to_csv(
        OUT / "feature_order.csv", index=False)

    # ---- 7.3 ABLATION tien xu ly (CV tren train) ----
    print("\n" + "=" * 72 + "\nABLACTION - YEU TO TIEN XU LY (CV tren TRAIN)\n" + "=" * 72)
    rows = []
    for name in variants:
        dd, ff, _ = prepared[name]
        XX = encode(dd, ff).to_numpy(); yy = dd[tcol].to_numpy()
        nv = len(yy)
        tr_v, _ = train_test_split(np.arange(nv), test_size=0.30,
                                   stratify=yy, random_state=SEED)
        row = {"factor": "preprocessing", "setting": name, "n_final": int(nv)}
        row.update(cv_evaluate(XX[tr_v], yy[tr_v], "none", "logistic"))
        rows.append(row)
    prep_df = pd.DataFrame(rows)
    ref = prep_df[prep_df["setting"] == BASE].iloc[0]["cv_AP"]
    prep_df["delta_AP"] = (prep_df["cv_AP"] - ref).round(4)
    prep_df.to_csv(OUT / "ablation_preprocessing.csv", index=False)
    print(prep_df[["setting", "n_final", "cv_AP", "delta_AP",
                   "cv_recall_1", "cv_f1_macro"]].to_string(index=False))

    # ---- 7.4 MAT TRAN model x resample (CV) ----
    print("\n" + "=" * 72 + "\nMAT TRAN: 5 model x 5 resample (CV tren TRAIN)\n" + "=" * 72)
    models = ["logistic", "decision_tree", "random_forest", "primary", "stacking"]
    grid_rows = []
    for mid in models:
        for strat in RESAMPLES:
            t_s = time.time()
            r = {"model": mid, "strategy": strat}
            r.update(cv_evaluate(Xtr, ytr, strat, mid))
            r["sec"] = round(time.time() - t_s, 1)
            grid_rows.append(r)
            print(f"  [cv] {mid:14s} x {strat:12s} "
                  f"AP={r['cv_AP']:.4f} R1={r['cv_recall_1']:.3f} "
                  f"F1m={r['cv_f1_macro']:.3f} ({r['sec']}s)")
    grid = pd.DataFrame(grid_rows)
    prim = grid[grid["model"] == "primary"]
    cap = prim["cv_f1_macro"].max() - 0.01
    cand = prim[prim["cv_f1_macro"] >= cap].sort_values(["cv_AP", "cv_recall_1"],
                                                        ascending=False)
    best_strategy = cand.iloc[0]["strategy"]
    print(f"\n[CHON] strategy = '{best_strategy}' (CV AP={cand.iloc[0]['cv_AP']}, "
          f"recall_1={cand.iloc[0]['cv_recall_1']}) - trong nhom F1macro gan toi da")
    grid.to_csv(OUT / "grid_model_x_resample.csv", index=False)
    pd.DataFrame([{"chosen_strategy": best_strategy,
                    "rule": "AP cao nhat trong nhom F1macro >= max-0.01, tie-break recall_1",
                    "cv_AP": float(cand.iloc[0]["cv_AP"]),
                    "cv_recall_1": float(cand.iloc[0]["cv_recall_1"]),
                    "seed": SEED, "n_folds": N_FOLDS}]).to_csv(
        OUT / "chosen_strategy.csv", index=False)

    # ---- 7.5 FINAL: fit TRAIN -> nguong VAL (khoa) -> TEST ----
    print("\n" + "=" * 72 + "\nFINAL: fit TRAIN -> nguong VAL (khoa) -> TEST\n" + "=" * 72)
    final_rows = []
    proba_cache = {}
    for mid in models:
        Xr, yr = resample(Xtr, ytr, best_strategy, SEED)
        m = build_model(mid, best_strategy).fit(Xr, yr)
        p_va = m.predict_proba(Xva)[:, 1]
        p_te = m.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, p_va)      # chi dung VAL
        proba_cache[mid] = p_te
        for name, t in (("thr_0.5", 0.5), ("thr_val_maxF1", thr)):
            met = metrics_at_threshold(yte, p_te, t)
            met.update({"model": mid, "strategy": best_strategy,
                        "threshold_rule": name, "n_train_resampled": int(len(yr)),
                        "seed": SEED, "n_folds": N_FOLDS})
            final_rows.append(met)
            print(f"[test] {mid:14s} {name:14s} AP={met['AP']:.4f} "
                  f"R1={met['recall_1']:.3f} P1={met['precision_1']:.3f} "
                  f"F1m={met['f1_macro']:.3f} AUC={met['auc_roc']:.3f} "
                  f"MCC={met['mcc']:.3f} (n_resample={met['n_train_resampled']:,})")
    pd.DataFrame(final_rows).to_csv(OUT / "final_test_metrics.csv", index=False)

    # ---- 7.6 BOOTSTRAP: mo hinh chinh vs baseline ----
    if not FAST:
        print("\n" + "=" * 72 + f"PAIRED BOOTSTRAP ({N_BOOT} lan) tren TEST\n" + "=" * 72)
        pairs = [("primary_vs_logistic", "primary", "logistic"),
                 ("primary_vs_rf", "primary", "random_forest"),
                 ("primary_vs_stacking", "primary", "stacking")]
        boot_rows = []
        for name, a, b in pairs:
            dlt, lo, hi, p = paired_bootstrap(yte, proba_cache[a], proba_cache[b])
            sig = "DANG KE" if (lo > 0 or hi < 0) else "KHONG DU BA CHUNG"
            boot_rows.append({"comparison": name, "delta_AP": round(dlt, 4),
                             "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                             "p_value": round(p, 4), "significant": sig})
            print(f"[boot] {name:24s} dAP={dlt:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] "
                  f"p={p:.4f} -> {sig}")
        pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap_ci.csv", index=False)

    # ---- 7.7 NGHIEM PHU: 3-lop ----
    print("\n" + "=" * 72 + "\nNGHIEM PHU: 3-lop (lop thieu ~1,8%)\n" + "=" * 72)
    p3 = find_csv(F_3L)
    if p3 is not None:
        df3 = pd.read_csv(p3)
        t3 = find_target(df3, prefer="diabetes_012")
        d3, f3, s3 = prepare(df3, t3, dedup=True, noise="majority")
        X3 = encode(d3, f3).to_numpy(); y3 = d3[t3].to_numpy()
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        rows3 = []
        for strat in ("none", "class_weight"):
            ap3, f1m3 = [], []
            for a, b in skf.split(X3, y3):
                Xr, yr = resample(X3[a], y3[a], strat, SEED)
                m = build_model("primary", strat).fit(Xr, yr)
                p3_ = m.predict_proba(X3[b])[:, 2]
                ap3.append(average_precision_score((y3[b] == 2).astype(int), p3_))
                f1m3.append(f1_score(y3[b], m.predict(X3[b]), average="macro"))
            row3 = {"strategy": strat,
                    "AP_class2": round(float(np.mean(ap3)), 4),
                    "AP_class2_std": round(float(np.std(ap3)), 4),
                    "F1_macro": round(float(np.mean(f1m3)), 4),
                    "n_rows": int(len(y3)),
                    "note": "SMOTE/ADASYN khong dung: lop 1 (tien duong) chi ~1,8%"}
            rows3.append(row3)
            print(f"[3-lop] strategy={strat:12s} AP(cua lop 2)={row3['AP_class2']:.4f} "
                  f"| F1macro={row3['F1_macro']:.4f}")
        pd.DataFrame(rows3).to_csv(OUT / "aux_3class_results.csv", index=False)
        print("     -> luu y: SMOTE 3-lop khong dung (lop ~1,8%); "
              "ket luan de xuat 'loi nuoc' trong SP2.")
    else:
        print("[3-lop] file khong ton tai - bo qua")

    print("\n" + "=" * 72)
    print(f"[OK] XONG sau {time.time()-t0:.0f}s ({time.time()-t0:.0f}s) -> {OUT}/: "
          + ", ".join(f.name for f in sorted(OUT.glob("*.csv"))))
    print("=" * 72)

if __name__ == "__main__":
    main()