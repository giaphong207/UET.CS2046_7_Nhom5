"""c_xie_reimpl.py - PHU LUC "doi chung theo Xie et al. (2019)", cai lai tren DU LIEU + PROTOCOL cua nhom.

Xie 2019 dung: SVM (tuyen tinh/da thuc/RBF), Gaussian Naive Bayes, Logistic, Neural network,
Decision Tree, Random Forest; SMOTE tren train; do accuracy/sensitivity/specificity/AUC.
O day cai lai tren protocol cua nhom (cung du lieu 253.680 dong, cung idx, CV 3-fold theo AP,
nguong F-beta tren validation, tap test chi dung voi --final):

  X1 Gaussian Naive Bayes        (khong xu ly mat can bang)
  X2 Neural network (MLP)        (khong xu ly mat can bang)
  X3 SVM tuyen tinh (+ hieu chinh xac suat)  (khong xu ly mat can bang)
  X4 Logistic + SMOTE            (cong thuc cua Xie, SMOTE TRONG fold CV, chi tren train)
  X5 Random Forest + SMOTE       (nt)

KHONG cai: SVM nhan RBF / da thuc (chi phi O(n^2) tren 177.576 dong qua lon cho Kaggle).
=> ghi ro gioi han nay trong SP2. Logistic / RF / Decision Tree khong-SMOTE da co o c_baseline.py.
KHONG so sanh truc tiep so lieu cong bo cua Xie (BRFSS 2014, dinh nghia nhan khac).

Cach chay (Kaggle):
    python c_xie_reimpl.py --b-code <thu muc co b2_b3_pipeline.py> --idx-dir <thu muc idx> --out-dir out_xie
Kiem thu: python c_xie_reimpl.py --b-code <...> --synthetic --quick
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from c_run_guard import RunGuard, validate_split, provenance

ap = argparse.ArgumentParser()
ap.add_argument("--b-code", required=True)
ap.add_argument("--idx-dir")
ap.add_argument("--data")
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--beta", type=float, default=2.0, help="F-beta chon nguong tren validation (nhom: 2)")
ap.add_argument("--only", help="chi chay cac ma, vd X1,X3 (mac dinh: tat ca)")
ap.add_argument("--quick", action="store_true", help="luoi nho, CHI de kiem thu")
ap.add_argument("--final", action="store_true", help="them TEST - chi khi nhom da khoa; chay 1 LAN")
ap.add_argument("--out-dir", default="out_xie")
ap.add_argument("--approved-final", action="store_true", help="Only after documented group approval")
args = ap.parse_args()
if args.final and not args.approved_final:
    ap.error("FINAL blocked: group approval required")
if not args.synthetic and not args.idx_dir:
    ap.error("--idx-dir is required for real data")
if not args.synthetic and args.final and (args.quick or getattr(args, 'skip_tune', False) or args.beta != 2):
    ap.error("Real FINAL requires full F2 configuration")

sys.path.insert(0, str(Path(args.b_code).resolve()))
import b2_b3_pipeline as B                                   # noqa: E402
from sklearn.calibration import CalibratedClassifierCV       # noqa: E402
from sklearn.metrics import average_precision_score, fbeta_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB                   # noqa: E402
from sklearn.neural_network import MLPClassifier             # noqa: E402
from sklearn.pipeline import Pipeline                        # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402
from sklearn.svm import LinearSVC                            # noqa: E402

SEED, N_FOLDS = B.SEED, B.N_FOLDS
OUT = Path(args.out_dir)

RULE = f"thr_val_maxF{args.beta:g}"
if args.quick:
    print("[CANH BAO] --quick: luoi nho, KHONG dung ket qua nay de nop.")


def best_fbeta_threshold(y, prob, beta):
    best, best_s = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 199):          # cung luoi + luat hoa voi B
        sc = fbeta_score(y, (prob >= t).astype(int), beta=beta, pos_label=1, zero_division=0)
        if sc > best_s:
            best, best_s = float(t), float(sc)
    return best


def metrics(y, prob, thr):
    r = B.metrics_at_threshold(y, prob, thr)
    r["f2_class1"] = round(float(fbeta_score(y, (prob >= thr).astype(int), beta=2,
                                             pos_label=1, zero_division=0)), 4)
    return r


def make_synthetic(n=30000, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 12))
    y = ((X[:, :4].sum(axis=1) - 3.2 + rng.normal(0, 1, n)) > 0.2).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["Diabetes_binary"] = y
    return df


# ---- du lieu + chia tap: GIONG c_baseline.py ----
if args.synthetic:
    path = None
    df_raw = make_synthetic(n=1500 if args.quick else 30000)
else:
    path = Path(args.data) if args.data else B.find_csv(B.F_2L)
    if path is None:
        raise SystemExit("Khong tim thay CSV. Add Input dataset hoac dung --data.")
    df_raw = pd.read_csv(path)
tcol = B.find_target(df_raw)
if not args.synthetic and (tcol != "Diabetes_binary" or df_raw.shape != (253680, 22)
        or not all(pd.api.types.is_numeric_dtype(df_raw[c]) for c in df_raw.columns)):
    raise ValueError("Unexpected real-data schema")
if set(df_raw[tcol].unique()) != {0, 1} or df_raw.isna().any().any():
    raise ValueError("Expected binary labels and no missing data")
d, feats, st = B.prepare(df_raw, tcol, dedup=False, noise="keep")     # nhom chot: du dong, giu nhan
X_all = B.encode(d, feats).to_numpy()
y_all = d[tcol].to_numpy()
if args.synthetic:
    tr_i, tmp_i = train_test_split(np.arange(len(y_all)), test_size=0.30, stratify=y_all, random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=y_all[tmp_i], random_state=SEED)
else:
    idx = Path(args.idx_dir)
    tr_i, va_i, te_i = (np.load(idx / f"idx_{n}.npy", allow_pickle=False) for n in ("train", "val", "test"))
validate_split((tr_i, va_i, te_i), len(y_all), args.synthetic)
if not np.isfinite(X_all).all():
    raise ValueError("Non-finite input")
Xtr, ytr = X_all[tr_i], y_all[tr_i]
Xva, yva = X_all[va_i], y_all[va_i]
print(f"[split] train/val/test sizes = {len(tr_i):,}/{len(va_i):,}/{len(te_i):,}")


# ---- mo hinh ----
def mk_gnb(p):
    return GaussianNB(var_smoothing=p["var_smoothing"])


def mk_mlp(p):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=tuple(p["hidden"]), alpha=p["alpha"],
                                           early_stopping=True, max_iter=60 if args.quick else 200,
                                           random_state=SEED))])


def mk_svm(p):   # SVM tuyen tinh + hieu chinh xac suat (sigmoid) de co predict_proba cho AP / nguong
    base = Pipeline([("scaler", StandardScaler()),
                     ("clf", LinearSVC(C=p["C"], dual=False, max_iter=5000, random_state=SEED))])
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def mk_logit(p):
    return B.build_model("logistic", "none").set_params(clf__C=p["C"])


def mk_rf(p):
    return B.build_model("random_forest", "none").set_params(n_estimators=p["n_estimators"],
                                                              max_depth=p["max_depth"], n_jobs=1)


G_GNB = [{"var_smoothing": v} for v in np.logspace(-9, -1, 10)]
G_MLP = [{"hidden": h, "alpha": a} for h, a in
         itertools.product(((32,), (64,), (128,), (64, 32), (128, 64)), (1e-4, 1e-2))]
G_SVM = [{"C": c} for c in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)]
G_LOGIT = [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)]
G_RF = [{"n_estimators": n, "max_depth": dd} for n, dd in itertools.product((200, 300), (8, 12, 16, 20, None))]
if args.quick:
    G_GNB, G_MLP, G_SVM, G_LOGIT = G_GNB[::4], G_MLP[::5], G_SVM[::4], G_LOGIT[::4]
    G_RF = [{"n_estimators": 100, "max_depth": dd} for dd in (8, 12)]

SPECS = {  # ma: (ten, ham tao, luoi, chien luoc xu ly mat can bang)
    "X1": ("X1_gaussian_nb", mk_gnb, G_GNB, "none"),
    "X2": ("X2_mlp", mk_mlp, G_MLP, "none"),
    "X3": ("X3_svm_linear", mk_svm, G_SVM, "none"),
    "X4": ("X4_logistic_SMOTE", mk_logit, G_LOGIT, "smote"),
    "X5": ("X5_random_forest_SMOTE", mk_rf, G_RF, "smote"),
}
chosen = [c.strip() for c in args.only.split(",")] if args.only else list(SPECS)


def key(v):
    return v if not isinstance(v, (np.floating, np.integer)) else v.item()


def tune(name, make, grid, strat):
    """CV 3-fold phan tang tren TRAIN, theo AP; resample (neu co) CHI tren phan fit cua fold."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for p in grid:
        t0 = time.time()
        aps = []
        for a, b in skf.split(Xtr, ytr):
            Xa, ya = B.resample(Xtr[a], ytr[a], strat, SEED)
            m = make(p).fit(Xa, ya)
            aps.append(average_precision_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))
        rows.append({**{k: str(v) for k, v in p.items()}, "cv_AP": float(np.mean(aps)),
                     "cv_AP_std": float(np.std(aps)), "seconds": round(time.time() - t0, 1)})
        print(f"[tune {name}] {p} -> cv_AP={rows[-1]['cv_AP']:.4f} ({rows[-1]['seconds']}s)", flush=True)
    df = pd.DataFrame(rows)
    i = int(df["cv_AP"].idxmax())                 # hoa -> cau hinh dung truoc trong luoi
    df.round(4).to_csv(OUT / f"tune_C_{name}.csv", index=False)
    print(f"[tune {name}] CHON {grid[i]} (cv_AP={df.loc[i, 'cv_AP']:.4f}) -> khoa lai")
    return grid[i]


if len(set(chosen)) != len(chosen) or not all(c in SPECS for c in chosen):
    raise ValueError("Invalid or repeated --only model code")
if any(SPECS[c][3] == "smote" for c in chosen) and not B.IML_OK:
    raise ValueError("imbalanced-learn required for SMOTE; no silent fallback")
meta = provenance(args, B, __file__, path, feats, (tr_i, va_i, te_i),
                  dict(beta=args.beta, quick=args.quick, chosen=chosen))
guard = RunGuard(args, meta, [SPECS[c][0] for c in chosen])
OUT = guard.out
rows, fitted = [], []
for code in chosen:
    name, make, grid, strat = SPECS[code]
    best = guard.params(name, lambda: tune(name, make, grid, strat))
    Xr, yr = B.resample(Xtr, ytr, strat, SEED)
    m = make(best).fit(Xr, yr)
    p_va = m.predict_proba(Xva)[:, 1]
    thr = guard.register(name, best, p_va, lambda: best_fbeta_threshold(yva, p_va, args.beta))
    base = {"model":name,"strategy":strat,"params":str(best),"seed":SEED,"n_folds":N_FOLDS}
    for rule,t in (("thr_0.5",0.5),(RULE,thr)):
        rows.append({**base,"split":"val","threshold_rule":rule,**metrics(yva,p_va,t)})
    fitted.append((name,m,thr,base))
if args.final:
    guard.begin_test()
    Xte,yte = X_all[te_i], y_all[te_i]
    preds = {}
    for name,m,thr,base in fitted:
        p_te = m.predict_proba(Xte)[:,1]
        preds[name] = p_te
        for rule,t in (("thr_0.5",0.5),(RULE,thr)):
            rows.append({**base,"split":"test","threshold_rule":rule,**metrics(yte,p_te,t)})
else:
    guard.finish_dev(va_i,yva)

res = pd.DataFrame(rows)
res.to_csv(OUT / "results_C_xie_appendix.csv", index=False)
show = ["model", "split", "threshold_rule", "AP", "auc_roc", "f1_class1", "f2_class1", "recall_1",
        "precision_1", "specificity", "threshold"]
print(res[show].to_string(index=False))
print(f"Da luu {OUT / 'results_C_xie_appendix.csv'}")

if args.final:
    guard.finish_test("test_predictions_C_xie.npz", te_i, yte, preds)
