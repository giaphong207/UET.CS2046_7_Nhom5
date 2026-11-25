
"""
Visualization for paired binary classification.
Creates PR, ROC, bootstrap distribution and CI plots.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score
)

from c_bootstrap import (
    load_paired_predictions,
    sha256_file
)


# ==========================================
# 1. SAVE FIGURE
# ==========================================

def save_figure(fig, path):

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return Path(path)


# ==========================================
# 2. PRECISION-RECALL CURVE
# ==========================================

def plot_pr_curve(
    y, scores_a, scores_b,
    model_a, model_b,
    path, scope
):

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, scores in [
        (model_a, scores_a),
        (model_b, scores_b)
    ]:

        precision, recall, _ = (
            precision_recall_curve(y, scores)
        )

        ap = average_precision_score(y, scores)

        ax.step(
            recall,
            precision,
            where="post",
            label=f"{name} (AP={ap:.4f})"
        )

    prevalence = float(np.mean(y))

    ax.axhline(
        prevalence,
        linestyle="--",
        label=f"No skill ({prevalence:.4f})"
    )

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.05),
        title=f"{scope} - Precision-Recall Curve"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    return save_figure(fig, path)


# ==========================================
# 3. ROC CURVE
# ==========================================

def plot_roc_curve(
    y, scores_a, scores_b,
    model_a, model_b,
    path, scope
):

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, scores in [
        (model_a, scores_a),
        (model_b, scores_b)
    ]:

        fpr, tpr, _ = roc_curve(y, scores)

        auc = roc_auc_score(y, scores)

        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.4f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0, 1),
        ylim=(0, 1.05),
        title=f"{scope} - ROC Curve"
    )

    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    return save_figure(fig, path)


# ==========================================
# 4. BOOTSTRAP DISTRIBUTION
# ==========================================

def plot_bootstrap_distribution(
    draws, summary, path, scope
):

    ap_row = summary.loc[
        summary["metric"] == "AP"
    ].iloc[0]

    delta = draws["delta_AP"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        delta,
        bins=40,
        alpha=0.8,
        edgecolor="white"
    )

    ax.axvline(
        ap_row["observed_delta"],
        linewidth=2,
        label="Observed delta"
    )

    ax.axvline(
        ap_row["ci_lower"],
        linestyle="--",
        label="Lower CI"
    )

    ax.axvline(
        ap_row["ci_upper"],
        linestyle="--",
        label="Upper CI"
    )

    ax.set(
        xlabel="Delta AP (B - A)",
        ylabel="Frequency",
        title=(
            f"{scope} - Bootstrap Distribution"
            f" (n={len(delta)})"
        )
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    return save_figure(fig, path)


# ==========================================
# 5. CONFIDENCE INTERVAL PLOT
# ==========================================

def plot_confidence_intervals(
    summary, path, scope
):

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (_, row) in enumerate(
        summary.iterrows()
    ):

        lower = row["ci_lower"]
        upper = row["ci_upper"]
        observed = row["observed_delta"]

        ax.hlines(
            y=i,
            xmin=lower,
            xmax=upper,
            linewidth=3
        )

        ax.vlines(
            [lower, upper],
            i - 0.08,
            i + 0.08
        )

        ax.scatter(
            observed,
            i,
            s=50,
            zorder=3
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_yticks(
        np.arange(len(summary))
    )

    ax.set_yticklabels(
        summary["metric"]
    )

    confidence = float(
        summary["confidence"].iloc[0]
    )

    ax.set(
        xlabel="Observed delta (B - A)",
        title=(
            f"{scope} - "
            f"{confidence:.0%} Bootstrap CI"
        )
    )

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.margins(x=0.15)

    return save_figure(fig, path)


# ==========================================
# 6. VALIDATE BOOTSTRAP RESULTS
# ==========================================

def validate_results(
    summary, draws, metadata,
    y, scores_a, scores_b
):

    if len(summary) != 2 or set(
        summary["metric"]
    ) != {"AP", "ROC_AUC"}:
        raise ValueError(
            "Invalid summary metrics"
        )

    if len(draws) != metadata[
        "n_bootstrap_valid"
    ]:
        raise ValueError(
            "Bootstrap count mismatch"
        )

    expected_iterations = np.arange(
        1, len(draws) + 1
    )

    np.testing.assert_array_equal(
        draws["iteration"],
        expected_iterations
    )

    confidence = metadata["confidence"]
    alpha = 1 - confidence

    metrics = {
        "AP": average_precision_score,
        "ROC_AUC": roc_auc_score
    }

    for metric, metric_fn in metrics.items():

        row = summary.loc[
            summary["metric"] == metric
        ].iloc[0]

        expected_a = metric_fn(y, scores_a)
        expected_b = metric_fn(y, scores_b)

        np.testing.assert_allclose(
            [
                row["observed_A"],
                row["observed_B"],
                row["observed_delta"]
            ],
            [
                expected_a,
                expected_b,
                expected_b - expected_a
            ],
            rtol=0,
            atol=1e-12
        )

        a = draws[f"{metric}_A"].to_numpy()
        b = draws[f"{metric}_B"].to_numpy()
        delta = draws[
            f"delta_{metric}"
        ].to_numpy()

        for values in [a, b, delta]:

            if not np.isfinite(values).all():
                raise ValueError(
                    "Non-finite bootstrap result"
                )

        np.testing.assert_allclose(
            delta,
            b - a,
            rtol=0,
            atol=1e-12
        )

        expected_ci = np.quantile(
            delta,
            [alpha / 2, 1 - alpha / 2]
        )

        np.testing.assert_allclose(
            [
                row["ci_lower"],
                row["ci_upper"]
            ],
            expected_ci,
            rtol=0,
            atol=1e-12
        )

        if not np.isclose(
            row["confidence"], confidence
        ):
            raise ValueError(
                "Confidence level mismatch"
            )


# ==========================================
# 7. MAIN VISUALIZATION PIPELINE
# ==========================================

def make_figures(
    file_a, model_a,
    file_b, model_b,
    bootstrap_dir,
    out_dir,
    scope
):

    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(
            "Explicit figure scope required"
        )

    idx, y, scores_a, scores_b = (
        load_paired_predictions(
            file_a, model_a,
            file_b, model_b
        )
    )

    bootstrap_dir = Path(bootstrap_dir)

    summary = pd.read_csv(
        bootstrap_dir / "bootstrap_summary.csv"
    )

    draws = pd.read_csv(
        bootstrap_dir / "bootstrap_draws.csv"
    )

    metadata = json.loads(
        (
            bootstrap_dir
            / "bootstrap_metadata.json"
        ).read_text(encoding="utf-8")
    )

    # Verify source identity
    if metadata["file_A_sha256"] != (
        sha256_file(file_a)
    ):
        raise ValueError(
            "Source A hash mismatch"
        )

    if metadata["file_B_sha256"] != (
        sha256_file(file_b)
    ):
        raise ValueError(
            "Source B hash mismatch"
        )

    if (
        metadata["model_A"] != model_a
        or metadata["model_B"] != model_b
    ):
        raise ValueError(
            "Model identity mismatch"
        )

    if not (
        (summary["model_A"] == model_a).all()
        and
        (summary["model_B"] == model_b).all()
    ):
        raise ValueError(
            "Summary model mismatch"
        )

    validate_results(
        summary, draws, metadata,
        y, scores_a, scores_b
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True, exist_ok=True
    )

    files = [
        plot_pr_curve(
            y, scores_a, scores_b,
            model_a, model_b,
            out_dir / "fig_pr_curve.png",
            scope
        ),

        plot_roc_curve(
            y, scores_a, scores_b,
            model_a, model_b,
            out_dir / "fig_roc_curve.png",
            scope
        ),

        plot_bootstrap_distribution(
            draws, summary,
            out_dir / "fig_bootstrap_delta_ap.png",
            scope
        ),

        plot_confidence_intervals(
            summary,
            out_dir / "fig_bootstrap_ci.png",
            scope
        )
    ]

    return files


# ==========================================
# 8. COMMAND-LINE INTERFACE
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file-a", required=True
    )
    parser.add_argument(
        "--model-a", required=True
    )
    parser.add_argument(
        "--file-b", required=True
    )
    parser.add_argument(
        "--model-b", required=True
    )
    parser.add_argument(
        "--bootstrap-dir", required=True
    )
    parser.add_argument(
        "--out-dir", required=True
    )
    parser.add_argument(
        "--scope", required=True
    )

    args = parser.parse_args()

    files = make_figures(
        file_a=args.file_a,
        model_a=args.model_a,
        file_b=args.file_b,
        model_b=args.model_b,
        bootstrap_dir=args.bootstrap_dir,
        out_dir=args.out_dir,
        scope=args.scope
    )

    print("Generated figures:")

    for path in files:
        print(path)


if __name__ == "__main__":
    main()
