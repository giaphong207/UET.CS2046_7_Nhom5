"""
Sinh biểu đồ cho báo cáo B2/B3 từ các CSV kết quả trong report/results/.
Chạy sau khi đã có b2_b3_pipeline.py tạo ra CSV:
    python Src/make_report_figures.py
Tất cả hình ghi vào report/figures/ với tiền tố b2b3_ để không đè hình của B1.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_curve, roc_auc_score)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    # Fallback thuan numpy (khong can sklearn) - cung dung ve do
    def average_precision_score(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        order = np.argsort(-y_score)
        y = y_true[order]
        tp = np.cumsum(y)
        prec = tp / np.arange(1, len(y) + 1)
        n_pos = y.sum()
        return float((prec * y).sum() / n_pos) if n_pos else 0.0

    def precision_recall_curve(y_true, y_score):
        y_true = np.asarray(y_true); y_score = np.asarray(y_score)
        order = np.argsort(-y_score)
        y = y_true[order]
        tp = np.cumsum(y); fp = np.cumsum(1 - y)
        rec = tp / max(y.sum(), 1)
        prec = tp / np.maximum(tp + fp, 1)
        return np.concatenate([prec, [1.0]]), np.concatenate([rec, [0.0]]), order

    def roc_curve(y_true, y_score):
        y_true = np.asarray(y_true); y_score = np.asarray(y_score)
        order = np.argsort(-y_score)
        y = y_true[order]
        tps = np.cumsum(y); fps = np.cumsum(1 - y)
        tpr = np.concatenate([[0.0], tps / max(tps[-1], 1)])
        fpr = np.concatenate([[0.0], fps / max(fps[-1], 1)])
        return fpr, tpr, order

    def roc_auc_score(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
            else float(np.trapz(tpr, fpr))

RES = Path("report/results/B2_B3")   # CSV ket qua B2/B3
FIG = Path("report/figures/B2_B3")   # bieu do B2/B3
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})

STRAT_ORDER = ["none", "class_weight", "smote", "adasyn", "undersample"]
MODEL_LABEL = {"logistic": "Logistic", "decision_tree": "Decision Tree",
               "random_forest": "Random Forest", "primary": "LightGBM",
               "stacking": "Stacking", "dummy": "Dummy (B0)"}


def _need(fname):
    p = RES / fname
    if not p.exists():
        print(f"  [SKIP] thieu {fname}")
        return None
    return pd.read_csv(p)


def fig_ap_by_strategy(grid):
    """F1: AP CV theo 5 resample strategy, 5 model -> nhom truc."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    models = [m for m in MODEL_LABEL if m in set(grid["model"])]
    w = 0.16
    x = np.arange(len(STRAT_ORDER))
    for i, m in enumerate(models):
        sub = grid[grid["model"] == m].set_index("strategy").reindex(STRAT_ORDER)
        ax.bar(x + (i - (len(models) - 1) / 2) * w, sub["cv_AP"].values,
               width=w, label=MODEL_LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels(STRAT_ORDER)
    ax.set_ylabel("CV Average Precision")
    ax.set_title("B3 - AP theo chiến lược mất cân bằng (5-fold CV trên train)")
    ax.legend(ncol=len(models), fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_ap_by_strategy.png", bbox_inches="tight")
    plt.close(fig)


def fig_recall_vs_ap(grid):
    """F2: tradeoff recall lop benh vs AP cho model 'primary'."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = grid[grid["model"] == "primary"].set_index("strategy").reindex(STRAT_ORDER)
    for _, r in sub.iterrows():
        ax.scatter(r["cv_recall_1"], r["cv_AP"], s=110, zorder=3)
        ax.annotate(r.name, (r["cv_recall_1"], r["cv_AP"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel("CV recall lớp bệnh (ngưỡng 0.5)")
    ax.set_ylabel("CV Average Precision")
    ax.set_title("B3 - LightGBM: đánh đổi recall ↔ AP theo chiến lược")
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_recall_vs_ap.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation(prep):
    """F3: ablation tiền xử lý - delta AP."""
    fig, ax = plt.subplots(figsize=(7, 4))
    p = prep.sort_values("cv_AP")
    colors = ["#55A868" if d >= 0 else "#C44E59" for d in p["delta_AP"]]
    ax.barh(p["setting"], p["cv_AP"], color=colors)
    for i, (_, r) in enumerate(p.iterrows()):
        ax.text(r["cv_AP"], i, f"  ΔAP={r['delta_AP']:+.4f}", va="center", fontsize=9)
    ax.set_xlabel("CV Average Precision")
    ax.set_title("B3 - Ablation tiền xử lý (baseline = dedup + majority-vote)")
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_ablation_preprocessing.png", bbox_inches="tight")
    plt.close(fig)


def fig_bootstrap(boot):
    """F4: paired bootstrap - delta AP + 95% CI."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(boot))[::-1]
    for i, (_, r) in enumerate(boot.iterrows()):
        yy = y[i]
        ax.plot([r["ci_low"], r["ci_high"]], [yy, yy], lw=2.5,
                color="#C44E59" if r["ci_low"] > 0 or r["ci_high"] < 0 else "#888888")
        ax.scatter(r["delta_AP"], yy, zorder=3, s=60,
                   color="#C44E59" if r["ci_low"] > 0 or r["ci_high"] < 0 else "#888888")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(boot["comparison"], fontsize=9)
    ax.set_xlabel("ΔAP (LightGBM − baseline) với 95% CI")
    ax.set_title("B2 - Paired bootstrap 1000 lần trên tập test")
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_bootstrap_ci.png", bbox_inches="tight")
    plt.close(fig)


def fig_final(final):
    """F5: kết quả test theo model, 2 quy tắc ngưỡng."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    models = [m for m in MODEL_LABEL if m in set(final["model"])]
    w = 0.35
    x = np.arange(len(models))
    for i, rule in enumerate(["thr_0.5", "thr_val_maxF1"]):
        sub = (final[final["threshold_rule"] == rule]
               .set_index("model").reindex(models))
        ax.bar(x + (i - 0.5) * w, sub["f1_macro"].values, width=w,
               label=f"{rule} (F1-macro)")
        ax.bar(x + (i - 0.5) * w, sub["recall_1"].values, width=w,
               bottom=sub["f1_macro"].values, alpha=0.35,
               label=f"{rule} (recall bệnh)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.set_ylabel("Giá trị chỉ số")
    ax.set_title("B2 - Kết quả TEST theo mô hình và quy tắc ngưỡng")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_final_test_metrics.png", bbox_inches="tight")
    plt.close(fig)


def fig_confusion(final):
    """F6: confusion matrix của model chính (primary) theo 2 quy tắc ngưỡng."""
    prim = final[final["model"] == "primary"]
    if prim.empty:
        return
    fig, axes = plt.subplots(1, len(prim), figsize=(4.2 * len(prim), 3.8))
    if len(prim) == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, prim.iterrows()):
        m = np.array([[r["n_tn"], r["n_fp"]], [r["n_fn"], r["n_tp"]]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center",
                        fontsize=13,
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        ax.set_title(f"{r['threshold_rule']}\nthr={r['threshold']}")
        ax.grid(False)
    fig.suptitle("B2 - Confusion matrix trên TEST (LightGBM + class_weight)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_confusion_primary.png", bbox_inches="tight")
    plt.close(fig)


def fig_3class(aux3):
    """F7: kết quả nghiệm phụ 3 lớp."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(aux3))
    ax.bar(x - 0.2, aux3["AP_class2"], width=0.4, label="AP lớp 2 (tiểu đường)")
    ax.bar(x + 0.2, aux3["F1_macro"], width=0.4, label="F1-macro")
    ax.set_xticks(x)
    ax.set_xticklabels(aux3["strategy"])
    ax.set_ylim(0, 1)
    ax.set_title("Nghiệm phụ - bài 3 lớp (lớp 1 chỉ ~1,8%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_aux_3class.png", bbox_inches="tight")
    plt.close(fig)


def fig_pr_roc():
    """F8/F9: duong Precision-Recall va ROC cua cac model tren TEST
    (de cuong 3: 'duong Precision-Recall va ROC')."""
    npz_path = RES / "test_predictions.npz"
    if not npz_path.exists():
        print("  [SKIP] thieu test_predictions.npz (chay b2_b3_pipeline.py truoc)")
        return []
    z = np.load(npz_path)
    y = z["y_test"]
    keys = [k for k in z.files if k.startswith("proba_")]
    made = []

    # --- Precision-Recall ---
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for k in keys:
        mid = k.replace("proba_", "")
        p = z[k]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, lw=1.8, label=f"{MODEL_LABEL.get(mid, mid)} (AP={ap:.3f})")
    base = float(np.mean(y))
    ax.axhline(base, color="k", ls="--", lw=1, label=f"base rate = {base:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (TEST)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_precision_recall_curve.png", bbox_inches="tight")
    plt.close(fig); made.append("b2b3_precision_recall_curve.png")

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for k in keys:
        mid = k.replace("proba_", "")
        p = z[k]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.8,
                label=f"{MODEL_LABEL.get(mid, mid)} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="k", ls="--", lw=1, label="random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (TEST)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_roc_curve.png", bbox_inches="tight")
    plt.close(fig); made.append("b2b3_roc_curve.png")
    return made


def fig_group_split():
    """F10: do nhay cua ket qua toi trung lap (split theo nhom vector trung)."""
    df = _need("group_split_sensitivity.csv")
    if df is None:
        return []
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(df))
    colors = ["#4C72B0" if "stratified" in s else "#C44E59"
              for s in df["split_type"]]
    ax.bar(x, df["cv_AP"], yerr=df["cv_AP_std"], color=colors, capsize=5)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, r["cv_AP"] + r["cv_AP_std"] + 0.002, f"{r['cv_AP']:.4f}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" (", "\n(") for s in df["split_type"]], fontsize=8)
    ax.set_ylabel("CV Average Precision")
    ax.set_ylim(0, max(df["cv_AP"]) * 1.15)
    ax.set_title("Độ nhạy kết quả với trùng lặp: split theo nhóm vector trùng")
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_group_split_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    return ["b2b3_group_split_sensitivity.png"]


def fig_subgroup():
    """F11: chi so theo nhom tuoi / gioi."""
    df = _need("subgroup_age_sex.csv")
    if df is None:
        return []
    fig, axes = plt.subplots(1, df["group_type"].nunique(),
                             figsize=(5.2 * df["group_type"].nunique(), 4))
    if df["group_type"].nunique() == 1:
        axes = [axes]
    for ax, (gtype, g) in zip(axes, df.groupby("group_type")):
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["recall_1"], width=0.4, label="Recall lớp bệnh")
        ax.bar(x + 0.2, g["AP"], width=0.4, label="AP")
        for i, (_, r) in enumerate(g.iterrows()):
            ax.text(i, max(r["recall_1"], r["AP"]) + 0.02,
                    f"n={int(r['n_test']):,}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{gtype}={int(v)}" for v in g["group_value"]])
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Kết quả theo nhóm {gtype} (TEST)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "b2b3_subgroup_age_sex.png", bbox_inches="tight")
    plt.close(fig)
    return ["b2b3_subgroup_age_sex.png"]


def main():
    made = []
    grid = _need("grid_model_x_resample.csv")
    if grid is not None:
        fig_ap_by_strategy(grid); made.append("b2b3_ap_by_strategy.png")
        fig_recall_vs_ap(grid); made.append("b2b3_recall_vs_ap.png")
    prep = _need("ablation_preprocessing.csv")
    if prep is not None:
        fig_ablation(prep); made.append("b2b3_ablation_preprocessing.png")
    boot = _need("bootstrap_ci.csv")
    if boot is not None:
        fig_bootstrap(boot); made.append("b2b3_bootstrap_ci.png")
    final = _need("final_test_metrics.csv")
    if final is not None:
        fig_final(final); made.append("b2b3_final_test_metrics.png")
        fig_confusion(final); made.append("b2b3_confusion_primary.png")
    aux3 = _need("aux_3class_results.csv")
    if aux3 is not None:
        fig_3class(aux3); made.append("b2b3_aux_3class.png")
    made += fig_pr_roc()
    made += fig_group_split()
    made += fig_subgroup()
    print(f"[OK] da sinh {len(made)} hinh -> {FIG}/")
    for m in made:
        print("   ", m)


if __name__ == "__main__":
    sys.exit(main())
