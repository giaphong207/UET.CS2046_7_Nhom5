import os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

FILES = {
    "3-lp (012)":   "diabetes_012_health_indicators_BRFSS2015.csv",
    "2-lp (5050)":  "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
    "2-lp (unbal)": "diabetes_binary_health_indicators_BRFSS2015.csv",
}
SENTINELS = (777, 888, 999, -1, -9, -99, -999)
MAX_LEVELS = 15 
TOP_K = 4
def find_csvs():
    here = Path.cwd()
    roots = [here]
    for _ in range(4):
        here = here.parent
        roots.append(here)
    for root in roots:
        input_dir = root / "input"
        if not input_dir.exists():
            continue
        hits = []
        for tag, fname in FILES.items():
            m = list(input_dir.rglob(fname))
            if m:
                hits.append((tag, m[0]))
        if hits:
            return hits
    for tag, fname in FILES.items():
        c1, c2 = Path(fname), Path("data") / fname
        if c1.exists() or c2.exists():
            return [(tag, c1 if c1.exists() else c2)]
    return []

_founds = find_csvs()
if not _founds:
    raise SystemExit(
        "[ERROR] Khong tim thay CSV.\n"
        "  Cac: Notebook -> File -> Add Input -> chon dataset\n"
        "  'diabetes-health-indicators-dataset' roi chay lai.\n"
        f"  (Da tu dong tim thu muc 'input' quanh {Path.cwd()})")
print(f"[OK] Tim thay {len(_founds)} CSV (noi dung: {_founds[0][1].parent})")

OUT = Path("report/results")
OUT.mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

# ------------------------------------------------------------------
def find_target(df, tag):
    """Uu tien theo ten thuc cua dataset; fallback an toan (khong bao gio
    tra ve cot income/education... ma chua 'diab' trong ten, neu 2-3
    gia tri ma ten chua 'diab' thi bao cao WARN de du xac nhan)."""
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in ("diabetes_012", "diabetes_binary", "diabetes"):
        if cand in low:
            print(f"  [TARGET] {tag}: dung cot '{low[cand]}'")
            return low[cand]
    # fallback: cot co ten chua 'diab'
    for c in df.columns:
        if "diab" in str(c).lower() and 1 < df[c].nunique(dropna=True) <= 5:
            print(f"  [TARGET] {tag}: WARN - dung cot fallback '{c}', KIEM TRA lai ten cot!")
            return c
    raise SystemExit(f"[ERROR] {tag}: khong tim thay cot target trong {list(df.columns)}")

def _is_num(s):
    return pd.api.types.is_numeric_dtype(s)

# =========================================================================
print("\n" + "=" * 72)
print("B1 EDA - DIABETES BRFSS2015 (fix target v2)")
print("=" * 72)

all_insights = {}

for tag, path in _founds:
    df = pd.read_csv(path)
    tcol = find_target(df, tag)
    feats = [c for c in df.columns if c != tcol]
    print(f"\n{'~' * 36} FILE: {tag} ({path.name}) {'~' * 36}")
    print(f"  Shape: {df.shape[0]:,} x {df.shape[1]} | Memory: "
          f"{df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    # ---- 1. duplicates + label noise (chinh xac) ----
    n_exact = int(df.duplicated().sum())
    nu_rows = df.groupby(feats)[tcol].transform("nunique")
    n_lnoise = int((nu_rows > 1).sum())
    n_lgroups = int((df.groupby(feats)[tcol].nunique() > 1).sum())
    print(f"\n  Duplicate (toan dong): {n_exact:,} ({100*n_exact/len(df):.2f}%)")
    print(f"  Label noise (feature giong, target khac): {n_lnoise:,} dong"
          f" trong {n_lgroups:,} nhom -> {100*n_lnoise/len(df):.2f}%")

    # ---- 2. missing ----
    na = df.isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    if len(na):
        print(f"\n  Missing values ({len(na)} cols, top 10):")
        print(na.head(10).to_string())
    else:
        print("\n  Missing values: 0 dong/col -> khong can imputation (B3)")

    # ---- 3. sentinel BRFSS ----
    sent = []
    for c in feats:
        if not _is_num(df[c]):
            continue
        cnt = int(df[c].isin(SENTINELS).sum())
        if cnt:
            sent.append((c, cnt, round(100*cnt/len(df), 3)))
    if sent:
        print(f"\n  Sentinel 777/888/999: {len(sent)} cols -> doi ra MISSING (B3)")
        for c, cnt, pct in sent:
            print(f"    {c:24s} n={cnt:>7,} ({pct:.2f}%)")
    else:
        print("\n  Sentinel 777/888/999: 0 col (dataset da sach) [OK]")

    # ---- 4. target / imbalance ----
    vc = df[tcol].value_counts().sort_index()
    ratio = vc.max() / max(vc.min(), 1)
    print(f"\n  Target '{tcol}' ({vc.size} classes), kiem tra phan bo:")
    for v, c in vc.items():
        print(f"    {v} : {c:>9,} ({100*c/vc.sum():5.1f}%)")
    print(f"  Imbalance ratio (max/min): {ratio:.1f}x")

    # ---- 5. numeric per-class + correlation ----
    corr_rows = []
    for c in feats:
        if not _is_num(df[c]):
            continue
        x = df[c][~df[c].isin(SENTINELS)]
        pair = pd.concat([x, pd.to_numeric(df[tcol], errors="coerce")], axis=1).dropna()
        r = np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1] if len(pair) > 2 else np.nan
        corr_rows.append({"feature": c, "corr_with_target":
                          round(r, 4) if np.isfinite(r) else np.nan})
    corr_df = pd.DataFrame(corr_rows).sort_values(
        "corr_with_target", key=lambda s: s.abs(), ascending=False)
    corr_df.to_csv(OUT / f"{tag}_corr.csv", index=False)
    print(f"\n  Correlation |r| top 8 (target={tcol}):")
    print(corr_df.head(8).to_string(index=False))

    # ---- 6. categorical lift ----
    base = df[tcol].value_counts(normalize=True)
    lift_rows = []
    for c in feats:
        s = df[c]
        if _is_num(s) and s.nunique(dropna=True) > MAX_LEVELS:
            continue
        for lvl, cnt in s.value_counts(dropna=False).items():
            mask = s.isna() if pd.isna(lvl) else (s == lvl)
            if cnt < 0.005 * len(df):
                continue
            sub_p = df.loc[mask, tcol].value_counts(normalize=True)
            for tv, p_tv in sub_p.items():
                p_b = base.get(tv, 0.0)
                lift_rows.append({"feature": c, "level": str(lvl), "n": int(cnt),
                                  "target_class": int(tv),
                                  "p_class_given_level": round(float(p_tv), 4),
                                  "p_class_overall": round(float(p_b), 4),
                                  "lift": round(float(p_tv / p_b), 3) if p_b > 0 else np.nan})
    cat_df = pd.DataFrame(lift_rows)
    if len(cat_df):
        cat_df.to_csv(OUT / f"{tag}_lift.csv", index=False)
        cls_max = int(df[tcol].max())   # lop quan tam (khong phu)
        top = cat_df[cat_df["target_class"] == cls_max].nlargest(10, "lift")
        print(f"\n  Lift top 10 (lop {cls_max} = lop dich vu/du'oc quan tam nhat):")
        print(top.to_string(index=False))

    # ---- PLOTS ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(v) for v in vc.index], vc.values, color="#4C72B0")
    for i, (v, c) in enumerate(vc.items()):
        ax.text(i, c, f"{100*c/vc.sum():.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_title(f"{tag}: target '{tcol}' distribution")
    ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig(OUT / "plots" / f"{tag}_target.png", dpi=150); plt.close(fig)

    if len(na):
        top_na = na.head(12)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_na.index, 100*top_na.values/len(df), color="#C44E59")
        ax.set_title(f"{tag}: missing % (top 12)"); ax.set_xlabel("% rows missing")
        ax.invert_yaxis()
        plt.tight_layout(); plt.savefig(OUT / "plots" / f"{tag}_missing.png", dpi=150); plt.close(fig)

    for fc in corr_df.head(TOP_K)["feature"]:
        clean = df[fc][~df[fc].isin(SENTINELS)].dropna()
        classes = sorted(clean.groupby(df[tcol]).count().index.tolist())
        data = [clean[df[tcol] == c].values for c in classes]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(data, labels=[str(c) for c in classes])
        ax.set_title(f"{tag}: {fc} per {tcol}")
        ax.set_xlabel(tcol)
        plt.tight_layout(); plt.savefig(OUT / "plots" / f"{tag}_{fc}_box.png", dpi=150); plt.close(fig)

    if len(cat_df):
        cls_max = int(df[tcol].max())
        feats_int = list(dict.fromkeys(
            cat_df[cat_df["target_class"] == cls_max]["feature"].tolist()))[:TOP_K]
        for fc in feats_int:
            sub = cat_df[cat_df["feature"] == fc].nlargest(6, "lift")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(sub["level"].astype(str), sub["lift"], color="#55A868")
            ax.axhline(1.0, color="grey", ls="--", lw=0.8)
            ax.set_title(f"{tag}: lift theo muc - {fc} (lop {cls_max})")
            ax.set_xlabel("level"); ax.set_ylabel("lift")
            plt.tight_layout(); plt.savefig(OUT / "plots" / f"{tag}_{fc}_lift.png", dpi=150); plt.close(fig)

    # ---- INSIGHTS (input cho B3 + SP1) ----
    ins = []
    if ratio >= 3:
        ins.append(f"Mat can bang {ratio:.1f}x (lop thieu {100*vc.min()/vc.sum():.1f}%) "
                   f"-> B3: thu SMOTE/ADASYN/undersample/class-weight, chon theo recall lop thieu")
    if 1 < ratio < 3:
        ins.append(f"Mat can bang nhua ({ratio:.1f}x) -> de class-weight, khong can resample")
    if n_exact / len(df) > 0.01:
        ins.append(f"Exact duplicates {100*n_exact/len(df):.1f}% -> B3: drop_duplicates (keep first) TRUOC split")
    if n_lnoise:
        ins.append(f"Label noise {n_lnoise:,} dong/{n_lgroups:,} nhom -> B3: majority-vote hoac bo nhom, "
                   f"ghi ruiet bien doan vao bao cao SP2 (phai co giai thich)")
    if len(na):
        cols5 = [c for c in na.index if na[c] / len(df) > 0.01]
        if cols5:
            ins.append(f"Missing >1%: {cols5[:6]} -> B3: imputation (fit train chi) de tránh leakage")
    if sent:
        ins.append("Sentinel: " + ", ".join(c for c,_,_ in sent) + " -> doi MISSING/level Unknown truoc khi model")
    ins.append(f"Feature |corr| cao nhat: {corr_df.iloc[0]['feature']} "
               f"(r={corr_df.iloc[0]['corr_with_target']}); "
               f"luon kiem chung voi lift (corr Pearson chi bat quan he tu' tuyen)")
    all_insights[tag] = ins
    print(f"\n{'-'*34} INSIGHTS & B3 ACTIONS - {tag} {'-'*34}")
    for i in ins:
        print(f"  * {i}")

# =========================================================================
print("\n" + "=" * 72)
print("GHI CHU PHU PHUC CHO Nhom (SP1/SP2):")
print("  - File '2-lp (5050)' la con duong stratified cua file '2-lp (unbal)' (goc BRFSS)")
print("    -> chi dung de sanity-check; ket thuc chinh phai chay tren 'unbal' (86/14)")
print("  - File '3-lp (012)': 0=khong, 1=tien duong, 2=duong; 2 lop 1&2 co luong bang nhau")
print("    -> bai toan 3-lop mat can bang; SMOTE 3-lop hoac class-weight")
n_csv = len(list(OUT.glob("*.csv")))
n_png = len(list((OUT / "plots").glob("*.png")))
print(f"\n[OK] B1 EDA XONG -> {OUT}/ : {n_csv} CSV + {n_png} plots + insights Tren duong log.")
print("     Mang ve cuoi cao, hoi cao SP1: (1) ratio mat can bang 3-lp, (2) % duplicate + label noise, (3) top feature.")
