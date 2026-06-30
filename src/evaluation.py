"""Model evaluation and error-analysis utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_ALERT_RATES = (0.001, 0.005, 0.01)


def select_threshold_max_f1(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Select a decision threshold using only validation data."""

    target = np.asarray(y_true, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(target, scores_array)
    if len(thresholds) == 0:
        return 0.5, {
            "validation_precision": 0.0,
            "validation_recall": 0.0,
            "validation_f1": 0.0,
        }

    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    f1_values = np.divide(
        2 * threshold_precision * threshold_recall,
        threshold_precision + threshold_recall,
        out=np.zeros_like(threshold_precision, dtype=float),
        where=(threshold_precision + threshold_recall) > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), {
        "validation_precision": float(threshold_precision[best_index]),
        "validation_recall": float(threshold_recall[best_index]),
        "validation_f1": float(f1_values[best_index]),
    }


def evaluate_binary_scores(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    model_name: str,
    split_name: str,
) -> dict[str, float | int | str]:
    """Evaluate fraud scores with class-imbalance-aware metrics."""

    target = np.asarray(y_true, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    predictions = (scores_array >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    if len(np.unique(target)) < 2:
        roc_auc = np.nan
    else:
        roc_auc = float(roc_auc_score(target, scores_array))

    return {
        "model": model_name,
        "split": split_name,
        "threshold": float(threshold),
        "fraud_prevalence": float(target.mean()),
        "average_precision": float(average_precision_score(target, scores_array)),
        "roc_auc": roc_auc,
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alert_count": int(predictions.sum()),
    }


def alert_counts_from_rates(
    row_count: int,
    *,
    rates: Iterable[float] = DEFAULT_ALERT_RATES,
) -> list[int]:
    """Convert alert-budget rates into unique positive top-k counts."""

    counts = sorted(
        {
            max(1, min(row_count, int(round(row_count * rate))))
            for rate in rates
        }
    )
    return counts


def compute_precision_recall_at_k(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    *,
    model_name: str,
    k_values: Iterable[int],
) -> pd.DataFrame:
    """Measure ranking quality under fixed investigation budgets."""

    target = np.asarray(y_true, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if len(target) != len(scores_array):
        raise ValueError("y_true and scores must have the same length.")

    order = np.argsort(-scores_array, kind="mergesort")
    ranked_target = target[order]
    total_fraud = int(target.sum())

    rows: list[dict[str, float | int | str]] = []
    for k_value in k_values:
        k = max(1, min(int(k_value), len(target)))
        fraud_found = int(ranked_target[:k].sum())
        rows.append(
            {
                "model": model_name,
                "k": k,
                "alert_rate": k / len(target),
                "fraud_found": fraud_found,
                "precision_at_k": fraud_found / k,
                "recall_at_k": fraud_found / total_fraud if total_fraud else np.nan,
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_ALERT_RATES",
    "alert_counts_from_rates",
    "compute_precision_recall_at_k",
    "evaluate_binary_scores",
    "select_threshold_max_f1",
]
