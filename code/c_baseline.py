
"""c_baseline.py (v2) - C2/C3: baseline B0 Dummy, B1 Logistic Regression, B2 Random Forest
theo de cuong 4.3-4.4, tren DUNG pipeline va DUNG cach chia cua B (b2_b3_pipeline.py + idx_*.npy).
 
Khac ban v1:
  - Bo Decision Tree (de cuong 4.4 chi co B0-B2 + B3).
  - TIM SIEU THAM SO cho B1 va B2 theo de cuong 4.3: CV 3 fold phan tang tren TRAIN,
    cung fold (StratifiedKFold, seed 42) va cung tieu chi (AP) nhu B, 10 cau hinh moi ho.
    Khoa tham so TRUOC khi chon nguong tren validation.
  - Strategy = "none": baseline KHONG dung class_weight / resample
    (A1 = B3 + trong so lop moi la phuong an cai tien, de cuong 4.4).
 
Cach chay (Kaggle, sau khi upload repo):
    python c_baseline.py --b-code <repo>/code --idx-dir <repo>/report/results/B2_B3
    python c_baseline.py ... --final       # them TEST: CHI chay MOT LAN sau khi nhom chot
Kiem thu nhanh khong can du lieu that:
    python c_baseline.py --b-code <repo>/code --synthetic --quick
"""
import argparse
import itertools
import sys
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
 
ap = argparse.ArgumentParser()
ap.add_argument("--b-code", required=True, help="thu muc code/ cua B (chua b2_b3_pipeline.py)")
ap.add_argument("--idx-dir", help="thu muc chua idx_train/val/test.npy cua B")
ap.add_argument("--data", help="duong dan CSV (mac dinh: B tu tim trong /kaggle/input)")
ap.add_argument("--synthetic", action="store_true", help="du lieu gia de kiem thu ma")
ap.add_argument("--quick", action="store_true",
                help="luoi nho (CHI de kiem thu toc do; KHONG dung cho ket qua nop)")
ap.add_argument("--skip-tune", action="store_true",
                help="bo qua tune, dung tham so mac dinh (CHI de kiem thu)")
ap.add_argument("--final", action="store_true", help="chay them tren TEST (mot lan)")
ap.add_argument("--out-dir", default=".", help="thu muc luu file ket qua")
args = ap.parse_args()
 
sys.path.insert(0, str(Path(args.b_code).resolve()))
import b2_b3_pipeline as B                      # noqa: E402  (import se in vai dong kiem tra thu vien)
from sklearn.metrics import average_precision_score   # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
 
SEED = B.SEED
N_FOLDS = B.N_FOLDS                              # nhom chot 3 (de cuong 4.3)
STRATEGY = "none"
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
 
# ---- Luoi tham so (10 cau hinh moi ho, trong khoang 10-15 cua de cuong 4.3) ----
LOGIT_GRID = [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)]
RF_GRID = [{"n_estimators": n, "max_depth": d}
           for n, d in itertools.product((200, 300), (8, 12, 16, 20, None))]
if args.quick:
    LOGIT_GRID = LOGIT_GRID[::3]
    RF_GRID = [{"n_estimators": 100, "max_depth": d} for d in (8, 12)]
    print("[CANH BAO] --quick: luoi nho, KHONG dung ket qua nay de nop.")
 
 
def make_synthetic(n=30000, seed=SEED):
    """Du lieu gia co ca dong trung va nhan mau thuan - CHI de kiem thu ma."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 12))
    logit = X[:, :4].sum(axis=1) - 3.2 + rng.normal(0, 1, n)
    y = (logit > 0.2).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["Diabetes_binary"] = y
    return df
 
 
# ---- 1. Du lieu + tien xu ly dung nhu B -----------------------------------
if args.synthetic:
    df_raw = make_synthetic()
else:
    path = Path(args.data) if args.data else B.find_csv(B.F_2L)
    if path is None:
        raise SystemExit("Khong tim thay CSV. Add Input dataset tren Kaggle hoac dung --data.")
    df_raw = pd.read_csv(path)
 
tcol = B.find_target(df_raw)
d, feats, st = B.prepare(df_raw, tcol, dedup=True, noise="majority")
X_all = B.encode(d, feats).to_numpy()
y_all = d[tcol].to_numpy()
print(f"[prepare] raw={st['n_raw']:,} -> final={st['n_final']:,} "
      f"(B bao cao: 253.680 -> 227.908 tren du lieu that)")
 
# ---- 2. Chi so chia tap: doc cua B (hoac tu tao khi kiem thu) -------------
if args.synthetic:
    tr_i, tmp_i = train_test_split(np.arange(len(y_all)), test_size=0.30,
                                   stratify=y_all, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=y_all[tmp_i],
                                  random_state=SEED)
else:
    idx = Path(args.idx_dir)
    tr_i, va_i, te_i = (np.load(idx / f"idx_{n}.npy") for n in ("train", "val", "test"))
 
n_idx = len(tr_i) + len(va_i) + len(te_i)
assert n_idx == len(y_all), (
    f"idx cua B co {n_idx:,} dong nhung prepare() cho {len(y_all):,} dong -> "
    "du lieu hoac tien xu ly khac B, DUNG LAI va hoi B.")
assert len(set(tr_i) | set(va_i) | set(te_i)) == n_idx, "cac tap chia bi chong lan"
 
Xtr, ytr = X_all[tr_i], y_all[tr_i]
Xva, yva = X_all[va_i], y_all[va_i]
Xte, yte = X_all[te_i], y_all[te_i]
print(f"[split] train/val/test = {len(ytr):,}/{len(yva):,}/{len(yte):,}; "
      f"ty le duong = {ytr.mean():.4f}/{yva.mean():.4f}/{yte.mean():.4f}")
 
 
# ---- 3. Tim sieu tham so (CHI tren TRAIN; khong dung val/test) -------------
def make_logistic(p):
    m = B.build_model("logistic", STRATEGY)       # Pipeline(StandardScaler, LogisticRegression)
    m.set_params(clf__C=p["C"])
    return m
 
 
def make_rf(p):
    m = B.build_model("random_forest", STRATEGY)
    m.set_params(n_estimators=p["n_estimators"], max_depth=p["max_depth"])
    return m
 
 
def tune(name, make, grid):
    """CV N_FOLDS phan tang tren TRAIN, chon theo AP trung binh (cung fold + tieu chi voi B)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for p in grid:
        t0 = time.time()
        aps = []
        for tr, va in skf.split(Xtr, ytr):
            m = make(p).fit(Xtr[tr], ytr[tr])
            aps.append(average_precision_score(ytr[va], m.predict_proba(Xtr[va])[:, 1]))
        rows.append({**p, "cv_AP": float(np.mean(aps)), "cv_AP_std": float(np.std(aps)),
                     "seconds": round(time.time() - t0, 1)})
        print(f"[tune {name}] {p} -> cv_AP={rows[-1]['cv_AP']:.4f} "
              f"(sd {rows[-1]['cv_AP_std']:.4f}, {rows[-1]['seconds']}s)", flush=True)
    df = pd.DataFrame(rows)
    best = df.loc[df["cv_AP"].idxmax()]            # hoa -> cau hinh dung truoc trong luoi
    best_p = {k: (None if pd.isna(best[k]) else (int(best[k]) if k != "C" else float(best[k])))
              for k in grid[0]}
    df.round(4).to_csv(OUT / f"tune_C_{name}.csv", index=False)
    print(f"[tune {name}] CHON {best_p} (cv_AP={best['cv_AP']:.4f}) -> khoa lai")
    return best_p
 
 
if args.skip_tune:
    best_logit, best_rf = {"C": 1.0}, {"n_estimators": 300, "max_depth": None}
    print("[CANH BAO] --skip-tune: dung tham so mac dinh, KHONG dung cho ket qua nop.")
else:
    best_logit = tune("logistic", make_logistic, LOGIT_GRID)
    best_rf = tune("random_forest", make_rf, RF_GRID)
 
# ---- 4. Mo hinh baseline (strategy = none) ---------------------------------
models = {
    "B0_dummy": (B.build_model("dummy", STRATEGY), "prior"),
    "B1_logistic": (make_logistic(best_logit), str(best_logit)),
    "B2_random_forest": (make_rf(best_rf), str(best_rf)),
}
 
rows = []
for name, (m, params) in models.items():
    m.fit(Xtr, ytr)
    p_va = m.predict_proba(Xva)[:, 1]
    thr = B.best_f1_threshold(yva, p_va)           # chon nguong CHI tren validation
    r = B.metrics_at_threshold(yva, p_va, thr)
    rows.append({"model": name, "strategy": STRATEGY, "params": params, "split": "val",
                 "threshold_rule": "thr_val_maxF1", "seed": SEED, "n_folds": N_FOLDS, **r})
    if args.final:
        p_te = m.predict_proba(Xte)[:, 1]
        for rule, t in (("thr_0.5", 0.5), ("thr_val_maxF1", thr)):
            r = B.metrics_at_threshold(yte, p_te, t)
            rows.append({"model": name, "strategy": STRATEGY, "params": params, "split": "test",
                         "threshold_rule": rule, "seed": SEED, "n_folds": N_FOLDS, **r})
 
res = pd.DataFrame(rows)
res.to_csv(OUT / "results_C_baseline_none.csv", index=False)
show = ["model", "split", "threshold_rule", "AP", "auc_roc", "f1_macro", "mcc",
        "recall_1", "precision_1", "threshold"]
print(res[show].to_string(index=False))
print(f"Da luu {OUT / 'results_C_baseline_none.csv'}")
