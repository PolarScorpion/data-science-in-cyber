"""Reproducible baseline fraud-modeling workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import (
    alert_counts_from_rates,
    compute_precision_recall_at_k,
    evaluate_binary_scores,
    select_threshold_max_f1,
)
from src.modeling import (
    RANDOM_SEED,
    build_baseline_estimators,
    fit_baseline_estimators,
    predict_fraud_scores,
)
from src.preprocessing import (
    BASELINE_EXCLUDED_RAW_COLUMNS,
    BASELINE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    assert_no_leakage_columns,
    build_baseline_feature_matrix,
    describe_temporal_split,
    extract_target,
    make_temporal_train_validation_test_split,
)


@dataclass
class BaselineExperimentResult:
    """In-memory outputs from the baseline modeling experiment."""

    split_summary: pd.DataFrame
    feature_columns: list[str]
    excluded_raw_columns: tuple[str, ...]
    thresholds: pd.DataFrame
    test_metrics: pd.DataFrame
    precision_recall_at_k: pd.DataFrame
    fitted_estimators: dict[str, object]
    validation_scores: dict[str, np.ndarray]
    test_scores: dict[str, np.ndarray]
    y_validation: pd.Series
    y_test: pd.Series


def run_baseline_experiment(
    transactions: pd.DataFrame,
    *,
    results_dir: str | Path | None = None,
    random_seed: int = RANDOM_SEED,
) -> BaselineExperimentResult:
    """Train and evaluate leakage-safe baseline fraud models."""

    split = make_temporal_train_validation_test_split(transactions)
    split_summary = describe_temporal_split(split)

    X_train = build_baseline_feature_matrix(split.train)
    X_validation = build_baseline_feature_matrix(split.validation)
    X_test = build_baseline_feature_matrix(split.test)
    feature_columns = list(X_train.columns)
    assert_no_leakage_columns(feature_columns)

    y_train = extract_target(split.train)
    y_validation = extract_target(split.validation)
    y_test = extract_target(split.test)

    estimators = build_baseline_estimators(random_seed=random_seed)
    fitted_estimators = fit_baseline_estimators(estimators, X_train, y_train)

    validation_scores = {
        model_name: predict_fraud_scores(estimator, X_validation)
        for model_name, estimator in fitted_estimators.items()
    }
    test_scores = {
        model_name: predict_fraud_scores(estimator, X_test)
        for model_name, estimator in fitted_estimators.items()
    }

    threshold_rows: list[dict[str, float | str]] = []
    metric_rows: list[dict[str, float | int | str]] = []
    precision_recall_rows: list[pd.DataFrame] = []
    k_values = alert_counts_from_rates(len(y_test))

    for model_name in fitted_estimators:
        if model_name == "no_fraud_prior":
            threshold = 0.5
            threshold_source = "fixed no-fraud decision threshold"
            validation_metrics = evaluate_binary_scores(
                y_validation,
                validation_scores[model_name],
                threshold=threshold,
                model_name=model_name,
                split_name="validation",
            )
            selection_stats = {
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_metrics["f1"],
            }
        else:
            threshold, selection_stats = select_threshold_max_f1(
                y_validation,
                validation_scores[model_name],
            )
            threshold_source = "max validation F1"

        threshold_rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "threshold_source": threshold_source,
                **selection_stats,
            }
        )
        metric_rows.append(
            evaluate_binary_scores(
                y_test,
                test_scores[model_name],
                threshold=threshold,
                model_name=model_name,
                split_name="test",
            )
        )
        precision_recall_rows.append(
            compute_precision_recall_at_k(
                y_test,
                test_scores[model_name],
                model_name=model_name,
                k_values=k_values,
            )
        )

    thresholds = pd.DataFrame(threshold_rows).set_index("model")
    test_metrics = (
        pd.DataFrame(metric_rows)
        .set_index("model")
        .sort_values("average_precision", ascending=False)
    )
    precision_recall_at_k = pd.concat(
        precision_recall_rows,
        ignore_index=True,
    ).sort_values(["k", "model"])

    result = BaselineExperimentResult(
        split_summary=split_summary,
        feature_columns=feature_columns,
        excluded_raw_columns=BASELINE_EXCLUDED_RAW_COLUMNS,
        thresholds=thresholds,
        test_metrics=test_metrics,
        precision_recall_at_k=precision_recall_at_k,
        fitted_estimators=fitted_estimators,
        validation_scores=validation_scores,
        test_scores=test_scores,
        y_validation=y_validation,
        y_test=y_test,
    )

    if results_dir is not None:
        save_baseline_tables(result, results_dir)

    return result


def save_baseline_tables(
    result: BaselineExperimentResult,
    results_dir: str | Path,
) -> list[Path]:
    """Save small reproducibility tables without serializing fitted models."""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "baseline_split_summary.csv",
        output_dir / "baseline_thresholds.csv",
        output_dir / "baseline_test_metrics.csv",
        output_dir / "baseline_precision_recall_at_k.csv",
    ]

    result.split_summary.to_csv(output_paths[0])
    result.thresholds.to_csv(output_paths[1])
    result.test_metrics.to_csv(output_paths[2])
    result.precision_recall_at_k.to_csv(output_paths[3], index=False)
    return output_paths


__all__ = [
    "BaselineExperimentResult",
    "run_baseline_experiment",
    "save_baseline_tables",
]
