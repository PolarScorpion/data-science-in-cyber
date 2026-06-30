"""Model comparison with leakage-safe behavioral features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavioral_features import (
    PAST_LABEL_FEATURE_COLUMNS,
    UNLABELED_BEHAVIORAL_FEATURE_COLUMNS,
    build_behavioral_feature_matrix,
    describe_behavioral_feature_sets,
)
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
    BASELINE_FEATURE_COLUMNS,
    assert_no_leakage_columns,
    build_baseline_feature_matrix,
    describe_temporal_split,
    extract_target,
    make_temporal_train_validation_test_split,
)


FEATURE_SET_ORDER = (
    "phase5_baseline",
    "behavior_unlabeled",
    "behavior_label_history",
)


@dataclass
class BehavioralModelingResult:
    """In-memory outputs from the behavioral feature comparison."""

    split_summary: pd.DataFrame
    feature_set_descriptions: pd.DataFrame
    feature_columns: dict[str, list[str]]
    thresholds: pd.DataFrame
    test_metrics: pd.DataFrame
    metric_comparison: pd.DataFrame
    precision_recall_at_k: pd.DataFrame
    logistic_coefficients: pd.DataFrame
    fitted_estimators: dict[str, dict[str, object]]
    validation_scores: dict[str, dict[str, np.ndarray]]
    test_scores: dict[str, dict[str, np.ndarray]]
    y_validation: pd.Series
    y_test: pd.Series


def _evaluate_feature_set(
    *,
    feature_set: str,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Fit and evaluate all baseline estimators for one feature set."""

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
                "feature_set": feature_set,
                "model": model_name,
                "threshold": threshold,
                "threshold_source": threshold_source,
                **selection_stats,
            }
        )
        metric_row = evaluate_binary_scores(
            y_test,
            test_scores[model_name],
            threshold=threshold,
            model_name=model_name,
            split_name="test",
        )
        metric_row["feature_set"] = feature_set
        metric_rows.append(metric_row)

        precision_recall_at_k = compute_precision_recall_at_k(
            y_test,
            test_scores[model_name],
            model_name=model_name,
            k_values=k_values,
        )
        precision_recall_at_k.insert(0, "feature_set", feature_set)
        precision_recall_rows.append(precision_recall_at_k)

    thresholds = pd.DataFrame(threshold_rows)
    test_metrics = pd.DataFrame(metric_rows)
    precision_recall_at_k = pd.concat(precision_recall_rows, ignore_index=True)
    return (
        thresholds,
        test_metrics,
        precision_recall_at_k,
        fitted_estimators,
        validation_scores,
        test_scores,
    )


def _compare_to_phase5_baseline(test_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute metric deltas against Phase 5 features for the same model."""

    metric_columns = ["average_precision", "roc_auc", "precision", "recall", "f1"]
    baseline = (
        test_metrics[test_metrics["feature_set"] == "phase5_baseline"]
        .set_index("model")[metric_columns]
        .add_prefix("phase5_")
    )
    comparison = test_metrics.merge(
        baseline,
        left_on="model",
        right_index=True,
        how="left",
    )

    for metric in metric_columns:
        comparison[f"delta_{metric}"] = (
            comparison[metric] - comparison[f"phase5_{metric}"]
        )

    return comparison.sort_values(
        ["model", "feature_set"],
        key=lambda series: series.map(
            {name: position for position, name in enumerate(FEATURE_SET_ORDER)}
        ).fillna(series),
    )


def _extract_logistic_coefficients(
    fitted_estimators: dict[str, dict[str, object]],
    feature_columns: dict[str, list[str]],
) -> pd.DataFrame:
    """Return scaled logistic-regression coefficients for model interpretation."""

    rows: list[dict[str, float | str]] = []
    for feature_set, estimators in fitted_estimators.items():
        estimator = estimators.get("logistic_regression")
        if estimator is None:
            continue
        classifier = estimator.named_steps["classifier"]
        coefficients = classifier.coef_[0]
        for feature, coefficient in zip(feature_columns[feature_set], coefficients):
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "abs_coefficient": abs(float(coefficient)),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["feature_set", "abs_coefficient"],
        ascending=[True, False],
    )


def run_behavioral_modeling_experiment(
    transactions: pd.DataFrame,
    *,
    results_dir: str | Path | None = None,
    random_seed: int = RANDOM_SEED,
) -> BehavioralModelingResult:
    """Compare Phase 5 baseline features with strict past behavioral features."""

    split = make_temporal_train_validation_test_split(transactions)
    split_summary = describe_temporal_split(split)
    feature_set_descriptions = describe_behavioral_feature_sets()

    baseline_features = build_baseline_feature_matrix(transactions)
    behavioral_label_features = build_behavioral_feature_matrix(
        transactions,
        include_past_fraud_labels=True,
    )
    feature_matrices = {
        "phase5_baseline": baseline_features.loc[:, list(BASELINE_FEATURE_COLUMNS)],
        "behavior_unlabeled": behavioral_label_features.loc[
            :,
            list(UNLABELED_BEHAVIORAL_FEATURE_COLUMNS),
        ],
        "behavior_label_history": behavioral_label_features.loc[
            :,
            [
                *UNLABELED_BEHAVIORAL_FEATURE_COLUMNS,
                *PAST_LABEL_FEATURE_COLUMNS,
            ],
        ],
    }
    feature_columns = {
        feature_set: list(feature_matrix.columns)
        for feature_set, feature_matrix in feature_matrices.items()
    }
    for columns in feature_columns.values():
        assert_no_leakage_columns(columns)

    y_train = extract_target(split.train)
    y_validation = extract_target(split.validation)
    y_test = extract_target(split.test)

    threshold_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    precision_recall_frames: list[pd.DataFrame] = []
    fitted_estimators: dict[str, dict[str, object]] = {}
    validation_scores: dict[str, dict[str, np.ndarray]] = {}
    test_scores: dict[str, dict[str, np.ndarray]] = {}

    for feature_set in FEATURE_SET_ORDER:
        feature_matrix = feature_matrices[feature_set]
        (
            thresholds,
            test_metrics,
            precision_recall_at_k,
            fitted,
            validation_predictions,
            test_predictions,
        ) = _evaluate_feature_set(
            feature_set=feature_set,
            X_train=feature_matrix.loc[split.train.index],
            X_validation=feature_matrix.loc[split.validation.index],
            X_test=feature_matrix.loc[split.test.index],
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
            random_seed=random_seed,
        )
        threshold_frames.append(thresholds)
        metric_frames.append(test_metrics)
        precision_recall_frames.append(precision_recall_at_k)
        fitted_estimators[feature_set] = fitted
        validation_scores[feature_set] = validation_predictions
        test_scores[feature_set] = test_predictions

    thresholds = pd.concat(threshold_frames, ignore_index=True)
    test_metrics = pd.concat(metric_frames, ignore_index=True).sort_values(
        ["feature_set", "average_precision"],
        ascending=[True, False],
        key=lambda series: series.map(
            {name: position for position, name in enumerate(FEATURE_SET_ORDER)}
        ).fillna(series)
        if series.name == "feature_set"
        else series,
    )
    metric_comparison = _compare_to_phase5_baseline(test_metrics)
    precision_recall_at_k = pd.concat(
        precision_recall_frames,
        ignore_index=True,
    ).sort_values(["feature_set", "k", "model"])
    logistic_coefficients = _extract_logistic_coefficients(
        fitted_estimators,
        feature_columns,
    )

    result = BehavioralModelingResult(
        split_summary=split_summary,
        feature_set_descriptions=feature_set_descriptions,
        feature_columns=feature_columns,
        thresholds=thresholds,
        test_metrics=test_metrics,
        metric_comparison=metric_comparison,
        precision_recall_at_k=precision_recall_at_k,
        logistic_coefficients=logistic_coefficients,
        fitted_estimators=fitted_estimators,
        validation_scores=validation_scores,
        test_scores=test_scores,
        y_validation=y_validation,
        y_test=y_test,
    )

    if results_dir is not None:
        save_behavioral_tables(result, results_dir)

    return result


def save_behavioral_tables(
    result: BehavioralModelingResult,
    results_dir: str | Path,
) -> list[Path]:
    """Save small Phase 6 tables without serializing fitted models."""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        output_dir / "behavioral_split_summary.csv",
        output_dir / "behavioral_feature_sets.csv",
        output_dir / "behavioral_thresholds.csv",
        output_dir / "behavioral_test_metrics.csv",
        output_dir / "behavioral_metric_comparison.csv",
        output_dir / "behavioral_precision_recall_at_k.csv",
        output_dir / "behavioral_logistic_coefficients.csv",
    ]

    result.split_summary.to_csv(output_paths[0])
    result.feature_set_descriptions.to_csv(output_paths[1])
    result.thresholds.to_csv(output_paths[2], index=False)
    result.test_metrics.to_csv(output_paths[3], index=False)
    result.metric_comparison.to_csv(output_paths[4], index=False)
    result.precision_recall_at_k.to_csv(output_paths[5], index=False)
    result.logistic_coefficients.to_csv(output_paths[6], index=False)
    return output_paths


__all__ = [
    "FEATURE_SET_ORDER",
    "BehavioralModelingResult",
    "run_behavioral_modeling_experiment",
    "save_behavioral_tables",
]
