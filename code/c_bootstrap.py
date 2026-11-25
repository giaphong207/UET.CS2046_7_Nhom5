
"""
Paired bootstrap for binary classification.

Primary metric: Average Precision.
Secondary metric: ROC-AUC.

Delta = metric(B) - metric(A).
Confidence interval: percentile bootstrap.
"""

import argparse
import hashlib
import json

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)


# ==========================================
# 1. FILE HASH
# ==========================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


# ==========================================
# 2. VALIDATE PREDICTION ARRAYS
# ==========================================

def validate_predictions(idx, y, scores):

    if idx.ndim != 1:
        raise ValueError("Index must be 1D")

    if y.ndim != 1 or scores.ndim != 1:
        raise ValueError(
            "Labels and scores must be 1D"
        )

    if not len(idx) == len(y) == len(scores):
        raise ValueError("Length mismatch")

    if len(idx) == 0:
        raise ValueError("Empty dataset")

    if idx.dtype.kind not in "iu":
        raise ValueError("Index must be integer")

    if np.any(idx < 0):
        raise ValueError("Negative index")

    if len(np.unique(idx)) != len(idx):
        raise ValueError("Duplicate row indices")

    if y.dtype.kind not in "biuf":
        raise ValueError("Invalid label dtype")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")

    if len(np.unique(y)) != 2:
        raise ValueError("Both classes required")

    if scores.dtype.kind not in "iuf":
        raise ValueError("Invalid score dtype")

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")

    if not (
        (scores >= 0) & (scores <= 1)
    ).all():
        raise ValueError(
            "Scores must be in [0, 1]"
        )


# ==========================================
# 3. LOAD NPZ
# ==========================================

def load_predictions(path, model_name):

    path = Path(path)
    score_key = f"proba_{model_name}"

    required = [
        "idx_test",
        "y_test",
        score_key
    ]

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        missing = [
            key for key in required
            if key not in data.files
        ]

        if missing:
            raise ValueError(
                f"Missing NPZ keys: {missing}"
            )

        idx = data["idx_test"]
        y = data["y_test"]
        scores = data[score_key]

    validate_predictions(idx, y, scores)

    return (
        idx,
        y.astype(np.int64),
        scores.astype(np.float64)
    )


# ==========================================
# 4. VERIFY PAIRED ALIGNMENT
# ==========================================

def load_paired_predictions(
    file_a, model_a,
    file_b, model_b
):

    idx_a, y_a, scores_a = load_predictions(
        file_a, model_a
    )

    idx_b, y_b, scores_b = load_predictions(
        file_b, model_b
    )

    if not np.array_equal(idx_a, idx_b):
        raise ValueError(
            "Row indices or row order mismatch"
        )

    if not np.array_equal(y_a, y_b):
        raise ValueError(
            "Ground-truth labels mismatch"
        )

    return idx_a, y_a, scores_a, scores_b


# ==========================================
# 5. PAIRED BOOTSTRAP CORE
# ==========================================

def paired_bootstrap(
    y,
    scores_a,
    scores_b,
    n_bootstrap=1000,
    seed=42,
    confidence=0.95
):

    y = np.asarray(y)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    n = len(y)
    idx = np.arange(n)

    validate_predictions(idx, y, scores_a)
    validate_predictions(idx, y, scores_b)

    if not isinstance(
        n_bootstrap, (int, np.integer)
    ) or n_bootstrap < 1:
        raise ValueError(
            "n_bootstrap must be positive integer"
        )

    if not 0 < confidence < 1:
        raise ValueError(
            "Confidence must be between 0 and 1"
        )

    rng = np.random.default_rng(seed)

    # Original observed metrics
    observed = {
        "AP": (
            average_precision_score(y, scores_a),
            average_precision_score(y, scores_b)
        ),

        "ROC_AUC": (
            roc_auc_score(y, scores_a),
            roc_auc_score(y, scores_b)
        )
    }

    rows = []

    attempted = 0
    skipped = 0

    max_attempts = n_bootstrap * 100

    while len(rows) < n_bootstrap:

        if attempted >= max_attempts:
            raise RuntimeError(
                "Too many invalid bootstrap samples"
            )

        attempted += 1

        # ONE shared sample for both models
        sample = rng.integers(
            low=0,
            high=n,
            size=n
        )

        y_b = y[sample]

        # Skip samples containing one class
        if np.unique(y_b).size != 2:
            skipped += 1
            continue

        p_a = scores_a[sample]
        p_b = scores_b[sample]

        ap_a = average_precision_score(
            y_b, p_a
        )

        ap_b = average_precision_score(
            y_b, p_b
        )

        auc_a = roc_auc_score(
            y_b, p_a
        )

        auc_b = roc_auc_score(
            y_b, p_b
        )

        rows.append({
            "iteration": len(rows) + 1,

            "AP_A": ap_a,
            "AP_B": ap_b,
            "delta_AP": ap_b - ap_a,

            "ROC_AUC_A": auc_a,
            "ROC_AUC_B": auc_b,
            "delta_ROC_AUC": auc_b - auc_a
        })

    draws = pd.DataFrame(rows)

    # Percentile confidence interval
    alpha = 1 - confidence

    summary_rows = []

    for metric, values in observed.items():

        observed_a, observed_b = values

        delta = draws[
            f"delta_{metric}"
        ].to_numpy()

        ci_lower, ci_upper = np.quantile(
            delta,
            [alpha / 2, 1 - alpha / 2]
        )

        summary_rows.append({
            "metric": metric,

            "observed_A": observed_a,
            "observed_B": observed_b,

            "observed_delta":
                observed_b - observed_a,

            "bootstrap_mean_delta":
                float(delta.mean()),

            "bootstrap_std":
                float(delta.std(ddof=1))
                if len(delta) > 1 else 0.0,

            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),

            "ci_contains_zero": bool(
                ci_lower <= 0 <= ci_upper
            ),

            "confidence": confidence,
            "n_bootstrap": n_bootstrap
        })

    summary = pd.DataFrame(summary_rows)

    metadata = {
        "n_records": n,
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),

        "seed": int(seed),

        "n_bootstrap_requested":
            int(n_bootstrap),

        "n_bootstrap_valid": len(draws),
        "n_bootstrap_attempted": attempted,
        "n_bootstrap_skipped": skipped,

        "confidence": float(confidence),
        "ci_method": "percentile",
        "sampling": "paired_row_with_replacement",

        "delta_direction": "B_minus_A",

        "primary_metric": "AP",

        "single_class_policy":
            "skip_and_redraw",

        "model_refitting": False
    }

    return summary, draws, metadata


# ==========================================
# 6. RUN COMPARISON AND SAVE RESULTS
# ==========================================

def run_comparison(
    file_a, model_a,
    file_b, model_b,
    out_dir,
    n_bootstrap=1000,
    seed=42,
    confidence=0.95
):

    idx, y, scores_a, scores_b = (
        load_paired_predictions(
            file_a, model_a,
            file_b, model_b
        )
    )

    summary, draws, metadata = (
        paired_bootstrap(
            y,
            scores_a,
            scores_b,
            n_bootstrap=n_bootstrap,
            seed=seed,
            confidence=confidence
        )
    )

    metadata.update({
        "model_A": model_a,
        "model_B": model_b,

        "file_A": str(file_a),
        "file_B": str(file_b),

        "file_A_sha256":
            sha256_file(file_a),

        "file_B_sha256":
            sha256_file(file_b),

        "index_alignment": "PASS",
        "label_alignment": "PASS"
    })

    summary.insert(0, "model_A", model_a)
    summary.insert(1, "model_B", model_b)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        out_dir / "bootstrap_summary.csv",
        index=False
    )

    draws.to_csv(
        out_dir / "bootstrap_draws.csv",
        index=False
    )

    with open(
        out_dir / "bootstrap_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    return summary, draws, metadata


# ==========================================
# 7. COMMAND-LINE INTERFACE
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Paired bootstrap comparison"
    )

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
        "--out-dir", required=True
    )

    parser.add_argument(
        "--n-bootstrap", type=int, default=1000
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--confidence", type=float, default=0.95
    )

    args = parser.parse_args()

    summary, draws, metadata = run_comparison(
        file_a=args.file_a,
        model_a=args.model_a,
        file_b=args.file_b,
        model_b=args.model_b,
        out_dir=args.out_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        confidence=args.confidence
    )

    print("\n=== PAIRED BOOTSTRAP SUMMARY ===")

    print(
        summary.round(6).to_string(index=False)
    )

    print("\n=== SAMPLING INFORMATION ===")

    print(
        "Valid:",
        metadata["n_bootstrap_valid"]
    )

    print(
        "Attempted:",
        metadata["n_bootstrap_attempted"]
    )

    print(
        "Skipped:",
        metadata["n_bootstrap_skipped"]
    )


if __name__ == "__main__":
    main()
