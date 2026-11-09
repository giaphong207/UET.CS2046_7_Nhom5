
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- CONFIG
SEED      = 42
N_FOLDS   = 5
N_BOOT    = 1000
FAST      = False          # True = bo bootstrap (smoke test nhanh)
K_GRID    = [6, 10, 14, 18, 21]     # so dac trung can thu
STRATEGY  = "class_weight"          # strategy da chon o B2/B3
RESULTS   = Path("report/results")
FIGURES   = Path("report/figures")
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

F_2L = "diabetes_binary_health_indicators_BRFSS2015.csv"

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import (f_classif, mutual_info_classif,
                                           SelectKBest, VarianceThreshold)
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold
    print("[deps] scikit-learn OK")
except ImportError as e:
    raise SystemExit(f"[ERROR] thieu scikit-learn: {e}")

try:
    import lightgbm as lgb
    LGBM_OK = True
    print("[deps] lightgbm OK")
except Exception:
    LGBM_OK = False
    print("[deps] lightgbm chua co -> dung RandomForest (cai: !pip install lightgbm)")

# =========================================================================
#  PHAN 1 - TIM DU LIEU (tu dong)
# =========================================================================
def find_csv(name=F_2L):
    here = Path.cwd()
    roots = [here]
    for _ in range(4):
        here = here.parent
        roots.append(here)
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

def find_target(df):
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in ("diabetes_binary", "diabetes_012", "diabetes"):
        if cand in low:
            return low[cand]
    for c in df.columns:
        if "diab" in str(c).lower() and 1 < df[c].nunique(dropna=True) <= 5:
            print(f"  [WARN] dung cot fallback '{c}' - kiem tra!")
            return c
    raise SystemExit("[ERROR] khong tim thay cot target")

# =========================================================================
#  PHAN 2 - TIEN XU LY (GIONG HET B2/B3 - phai khop so)
# =========================================================================
def prepare(df, tcol, dedup=True, noise="majority"):
    stats = {"n_raw": len(df)}
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
            df.loc[noisy_mask, tcol] = new_vals.to_numpy()
            stats["n_noisy_groups"] = int(mode_map.size)
            n_before = len(df)
            df = df.drop_duplicates(keep="first").reset_index(drop=True)
            stats["n_collapsed"] = n_before - len(df)
    stats["n_final"] = len(df)
    return df, feats, stats

def encode(df, feats):
    X = df[feats].copy()
    for c in feats:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X.astype(np.float64)

# =========================================================================
#  PHAN 3 - CAC BO CHON DAC TRUNG (fit tren data duoc dua vao)
# =========================================================================
def _corr_dedup(X: pd.DataFrame, thr: float = 0.95):
    """Loai dac trung trung lap: voi moi cap |r|>=thr, giu 1 bien (theo thu tu
    cot) va bo bien con lai. Tra (mask bo, danh sach ten giu)."""
    corr = X.corr().abs()
    n = len(corr.columns)
    drop = np.zeros(n, dtype=bool)
    cols = list(corr.columns)
    for i in range(n):
        if drop[i]:
            continue
        for j in range(i + 1, n):
            if drop[j]:
                continue
            if corr.iat[i, j] >= thr:
                drop[j] = True
    kept = [c for c, d in zip(cols, drop) if not d]
    return ~drop, kept

def select_features(method: str, X: pd.DataFrame, y: np.ndarray, k: int):
    """Chon 'k' dac trung. X, y = TAP TRAIN CUA FOLD (fit selector o day).
    Tra (list ten dac trung, giai thich)."""
    cols = list(X.columns)
    if method == "baseline" or k >= len(cols):
        return cols, "tat ca dac trung"
    if method == "variance":
        sel = VarianceThreshold(threshold=0.0).fit(X)
        keep = [c for c, v in zip(cols, sel.get_support()) if v]
        if len(keep) <= k:
            return keep, "variance (khong loai duoc)"
        # uu tien giu cac bien co bien phan lon nhat
        var = X[keep].var().sort_values(ascending=False)
        return list(var.index[:k]), "variance"
    if method == "corr_dedup":
        mask, kept = _corr_dedup(X, 0.95)
        if len(kept) <= k:
            return kept, "corr-dedup 0.95"
        # uu tien theo bien phan lon nhat trong nhom trung lap
        var = X[kept].var().sort_values(ascending=False)
        return list(var.index[:k]), "corr-dedup 0.95 + top-k variance"
    if method == "mutual_info":
        mi = mutual_info_classif(X, y, random_state=SEED, discrete_features=False)
        order = np.argsort(-mi)[:k]
        return [cols[i] for i in order], "mutual information top-k"
    if method == "anova_f":
        f, _ = f_classif(X, y)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0)
        order = np.argsort(-f)[:k]
        return [cols[i] for i in order], "ANOVA F-test top-k"
    if method in ("lgbm_gain", "rfe_lgbm"):
        m = build_model(STRATEGY)
        m.fit(X, y)
        imp = getattr(m, "feature_importances_", None)
        if imp is None:
            imp = m.coef_[0]
        imp = np.abs(np.asarray(imp))
        order = np.argsort(-imp)
        keep = [cols[i] for i in order[:k]]
        if method == "rfe_lgbm":
            return keep, "RFE theo thu tu gain importance cua LightGBM"
        return keep, "LightGBM gain importance top-k"
    raise ValueError(method)

METHODS = ["baseline", "variance", "corr_dedup", "mutual_info", "anova_f",
           "lgbm_gain", "rfe_lgbm"]

# =========================================================================
#  PHAN 4 - DANH GIA
# =========================================================================
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

# =========================================================================
#  PHAN 5 - MO HINH (A THAY HAM NAY)
# =========================================================================
def build_model(strategy="none"):
    """Model de xuat cua nhom. A sua ham nay de thay mo hinh; B khong sua."""
    cw = "balanced" if strategy == "class_weight" else None
    if LGBM_OK:
        kw = dict(n_estimators=300, learning_rate=0.08, random_state=SEED,
                  n_jobs=-1, verbose=-1)
        if cw:
            kw["class_weight"] = "balanced"
        return lgb.LGBMClassifier(**kw)
    return RandomForestClassifier(n_estimators=300, class_weight=cw,
                                  random_state=SEED, n_jobs=-1)

# =========================================================================
#  PHAN 6 - CHAY
# =========================================================================
def cv_feature_selection(Xtr, ytr, method, k, n_folds=N_FOLDS, seed=SEED):
    """5-fold stratified CV tren TRAIN. Selector FIT TRONG TUNG FOLD
    (khong dung toan bo tap train -> khong leakage).
    Tra (AP, recall, F1macro, std) trung binh qua cac fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aps, recs, f1s = [], [], []
    for a, b in skf.split(Xtr, ytr):
        Xa, ya = Xtr.iloc[a], ytr[a]
        Xb, yb = Xtr.iloc[b], ytr[b]
        try:
            feats, _ = select_features(method, Xa, ya, k)
        except Exception as e:
            print(f"    [WARN] selector '{method}' k={k} loi: {e}")
            continue
        if not feats:
            continue
        m = build_model(STRATEGY).fit(Xa[feats], ya)
        p = m.predict_proba(Xb[feats])[:, 1]
        yhat = (p >= 0.5).astype(int)
        aps.append(average_precision_score(yb, p))
        recs.append(recall_score(yb, yhat, pos_label=1, zero_division=0))
        f1s.append(f1_score(yb, yhat, average="macro"))
    if not aps:
        return None
    return {"cv_AP": round(float(np.mean(aps)), 4),
            "cv_AP_std": round(float(np.std(aps)), 4),
            "cv_recall_1": round(float(np.mean(recs)), 4),
            "cv_f1_macro": round(float(np.mean(f1s)), 4)}


def main():
    t0 = time.time()
    path = find_csv()
    if path is None:
        raise SystemExit(f"[ERROR] khong tim thay {F_2L}. Add Input dataset vao notebook.")
    print(f"[OK] Du lieu: {path}")
    df_raw = pd.read_csv(path)
    tcol = find_target(df_raw)
    d, feats, st = prepare(df_raw, tcol, dedup=True, noise="majority")
    print(f"[prep] raw={st['n_raw']:,} -> final={st['n_final']:,} "
          f"(dedup + majority-vote, giong B2/B3)")
    X = encode(d, feats)
    y = d[tcol].to_numpy()
    print(f"[data] X={X.shape} | ty le lop benh={y.mean():.4f}")

    # ---- 6.1 DUNG LAI DUNG SPLIT CUA B2/B3 (neu co) ----
    idx_files = {k: RESULTS / f"idx_{k}.npy" for k in ("train", "val", "test")}
    if all(p.exists() for p in idx_files.values()):
        tr_i = np.load(idx_files["train"]); va_i = np.load(idx_files["val"])
        te_i = np.load(idx_files["test"])
        print(f"[split] DUNG LAI idx_*.npy tu B2/B3 -> train={len(tr_i):,} "
              f"val={len(va_i):,} test={len(te_i):,} (so lieu so sanh duoc)")
    else:
        from sklearn.model_selection import train_test_split
        tr_i, tmp_i = train_test_split(np.arange(len(y)), test_size=0.30,
                                       stratify=y, random_state=SEED)
        va_i, te_i = train_test_split(tmp_i, test_size=0.50,
                                      stratify=y[tmp_i], random_state=SEED)
        print(f"[split] khong co idx_*.npy -> tach moi (seed={SEED}): "
              f"train={len(tr_i):,} val={len(va_i):,} test={len(te_i):,}")
        np.save(idx_files["train"], tr_i); np.save(idx_files["val"], va_i)
        np.save(idx_files["test"], te_i)
    Xtr, ytr = X.iloc[tr_i], y[tr_i]
    Xva, yva = X.iloc[va_i], y[va_i]
    Xte, yte = X.iloc[te_i], y[te_i]
    print(f"[split] ty le lop benh (train/val/test): "
          f"{ytr.mean():.4f} / {yva.mean():.4f} / {yte.mean():.4f}")

    # ---- 6.2 QUET TAC DUNG - PHUONG PHAP x k (CV tren train) ----
    print("\n" + "=" * 74)
    print("B4 - CHON LOC DAC TRUNG: phuong phap x so dac trung (CV tren TRAIN)")
    print("=" * 74)
    rows = []
    for method in METHODS:
        for k in K_GRID:
            if method == "baseline" and k != K_GRID[-1]:
                continue
            t1 = time.time()
            r = cv_feature_selection(Xtr, ytr, method, k)
            if r is None:
                continue
            r.update({"method": method, "k": k,
                      "n_features_kept": k if method != "baseline" else len(feats),
                      "sec": round(time.time() - t1, 1)})
            rows.append(r)
            print(f"  [cv] {method:14s} k={k:>2d} | AP={r['cv_AP']:.4f} "
                  f"R1={r['cv_recall_1']:.3f} F1m={r['cv_f1_macro']:.3f} "
                  f"({r['sec']}s)")
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "b4_feature_selection_cv.csv", index=False)

    # ---- 6.3 CHON tap dac trung toi uu ----
    # Tieu chuan: AP cao nhat trong nhom F1macro >= max-0.01, tie-break recall
    cap = grid["cv_f1_macro"].max() - 0.01
    cand = grid[grid["cv_f1_macro"] >= cap].sort_values(
        ["cv_AP", "cv_recall_1"], ascending=False)
    best = cand.iloc[0]
    best_method, best_k = best["method"], int(best["k"])
    print(f"\n[CHON] method='{best_method}' k={best_k} | CV AP={best['cv_AP']} "
          f"recall_1={best['cv_recall_1']} F1macro={best['cv_f1_macro']}")
    pd.DataFrame([{"selected_method": best_method, "k": best_k,
                   "cv_AP": best["cv_AP"], "cv_recall_1": best["cv_recall_1"],
                   "cv_f1_macro": best["cv_f1_macro"],
                   "rule": "AP cao nhat trong nhom F1macro >= max-0.01, "
                           "tie-break recall_1",
                   "strategy": STRATEGY, "seed": SEED, "n_folds": N_FOLDS}]
                  ).to_csv(RESULTS / "b4_selected_config.csv", index=False)

    # ---- 6.4 FIT selector tren TOAN BO TRAIN -> tap dac trung cuoi ----
    final_feats, how = select_features(best_method, Xtr, ytr, best_k)
    print(f"[final] selector '{best_method}' ({how}) tren TOAN BO train "
          f"-> {len(final_feats)} dac trung:")
    print(f"        {final_feats}")
    pd.DataFrame({"feature": final_feats,
                  "selected_by": best_method, "k": best_k,
                  "n_selected": len(final_feats)}
                 ).to_csv(RESULTS / "b4_selected_features.csv", index=False)

    # ---- 6.5 FINAL tren TEST: baseline (21 dac trung) vs tap da chon ----
    print("\n" + "=" * 74)
    print("FINAL tren TEST: 21 dac trung vs tap da chon")
    print("=" * 74)
    proba = {}
    final_rows = []
    for name, fset in (("all_features", feats), ("selected", final_feats)):
        m = build_model(STRATEGY).fit(Xtr[fset], ytr)
        p_va = m.predict_proba(Xva[fset])[:, 1]
        p_te = m.predict_proba(Xte[fset])[:, 1]
        thr = best_f1_threshold(yva, p_va)      # chi dung VAL
        proba[name] = p_te
        for rule, t in (("thr_0.5", 0.5), ("thr_val_maxF1", thr)):
            met = metrics_at_threshold(yte, p_te, t)
            met.update({"feature_set": name, "n_features": len(fset),
                        "strategy": STRATEGY, "threshold_rule": rule})
            final_rows.append(met)
            print(f"[test] {name:13s} n_feat={len(fset):>2d} {rule:14s} "
                  f"AP={met['AP']:.4f} R1={met['recall_1']:.3f} "
                  f"P1={met['precision_1']:.3f} F1m={met['f1_macro']:.3f} "
                  f"AUC={met['auc_roc']:.3f} MCC={met['mcc']:.3f}")
    pd.DataFrame(final_rows).to_csv(RESULTS / "b4_final_test_metrics.csv", index=False)

    # ---- 6.6 BOOTSTRAP: tap da chon vs 21 dac trung ----
    if not FAST:
        print("\n" + "=" * 74 + f"\nPAIRED BOOTSTRAP ({N_BOOT} lan) tren TEST\n" + "=" * 74)
        dlt, lo, hi, p = paired_bootstrap(yte, proba["selected"], proba["all_features"])
        sig = "DANG KE" if (lo > 0 or hi < 0) else "KHONG DU BA CHUNG"
        print(f"[boot] selected - all_features: dAP={dlt:+.4f} "
              f"95%CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} -> {sig}")
        pd.DataFrame([{"comparison": "selected_vs_all_features",
                       "delta_AP": round(dlt, 4), "ci_low": round(lo, 4),
                       "ci_high": round(hi, 4), "p_value": round(p, 4),
                       "significant": sig, "n_boot": N_BOOT}]
                      ).to_csv(RESULTS / "b4_bootstrap_ci.csv", index=False)

    # ---- 6.7 PERMUTATION IMPORTANCE tren VAL (giai thich, khong dung de chon) ----
    print("\n" + "=" * 74 + "\nPERMUTATION IMPORTANCE (tập val, 15 lần lặp)\n" + "=" * 74)
    try:
        from sklearn.inspection import permutation_importance
        m = build_model(STRATEGY).fit(Xtr[final_feats], ytr)
        pi = permutation_importance(m, Xva[final_feats], yva, n_repeats=15,
                                    random_state=SEED, scoring="average_precision",
                                    n_jobs=-1)
        imp = pd.DataFrame({"feature": final_feats,
                            "imp_mean": pi.importances_mean.round(4),
                            "imp_std": pi.importances_std.round(4)})
        imp = imp.sort_values("imp_mean", ascending=False)
        imp.to_csv(RESULTS / "b4_permutation_importance.csv", index=False)
        print(imp.to_string(index=False))
    except Exception as e:
        print(f"[perm] bo qua: {e}")

    # ---- 6.8 BIEU DO ----
    try:
        make_figures(grid, final_feats, imp if 'imp' in dir() else None)
    except Exception as e:
        print(f"[fig] bo qua: {e}")

    print("\n" + "=" * 74)
    print(f"[OK] XONG sau {time.time()-t0:.0f}s -> {RESULTS}/")
    print(f"     tap dac trung cuoi ({len(final_feats)}): {final_feats}")
    print("=" * 74)


def make_figures(grid, final_feats, imp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # (a) AP theo phuong phap x k
    fig, ax = plt.subplots(figsize=(11, 5))
    for method in METHODS:
        sub = grid[grid["method"] == method].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["cv_AP"], marker="o", lw=1.8, label=method)
    ax.set_xlabel("Số đặc trưng giữ lại (k)")
    ax.set_ylabel("CV Average Precision (AP)")
    ax.set_title("(a) B4 – AP theo phương pháp chọn lọc × số đặc trưng\n"
                 "(5-fold stratified CV trên tập train; selector fit trong từng fold)")
    ax.legend(fontsize=9, ncol=2)
    fig.savefig(FIGURES / "b4_feature_selection.png"); plt.close(fig)
    print(f"  [OK] {FIGURES / 'b4_feature_selection.png'}")

    # (b) permutation importance cua tap da chon
    if imp is not None and len(imp):
        fig, ax = plt.subplots(figsize=(9, 5))
        sub = imp.sort_values("imp_mean")
        ax.barh(sub["feature"], sub["imp_mean"], xerr=sub["imp_std"],
                color="#C44E52", alpha=0.9, capsize=3)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Giảm AP khi hoán đổi ngẫu nhiên (permutation importance)")
        ax.set_title(f"(b) B4 – Tầm quan trọng của {len(imp)} đặc trưng được chọn\n"
                     "(đánh giá trên tập validation, scoring = average_precision)")
        fig.savefig(FIGURES / "b4_selected_importance.png"); plt.close(fig)
        print(f"  [OK] {FIGURES / 'b4_selected_importance.png'}")


if __name__ == "__main__":
    main()