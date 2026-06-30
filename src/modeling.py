"""Model training utilities."""

from __future__ import annotations

import os
from collections.abc import Mapping

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_SEED = 42


class PriorProbabilityClassifier(BaseEstimator, ClassifierMixin):
    """Trivial baseline that assigns every row the training fraud prevalence."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PriorProbabilityClassifier":
        target = pd.Series(y).astype(int)
        if target.empty:
            raise ValueError("Cannot fit the prior baseline on an empty target.")

        self.classes_ = np.array([0, 1])
        self.prior_probability_ = float(target.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "prior_probability_"):
            raise ValueError("PriorProbabilityClassifier must be fitted before scoring.")

        positive_scores = np.full(len(X), self.prior_probability_, dtype=float)
        return np.column_stack([1.0 - positive_scores, positive_scores])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        positive_scores = self.predict_proba(X)[:, 1]
        return (positive_scores >= 0.5).astype(int)


def build_baseline_estimators(
    *,
    random_seed: int = RANDOM_SEED,
) -> dict[str, BaseEstimator]:
    """Create reproducible baseline estimators for imbalanced fraud detection."""

    return {
        "no_fraud_prior": PriorProbabilityClassifier(),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            l2_regularization=0.01,
            max_iter=80,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            random_state=random_seed,
        ),
    }


def fit_baseline_estimators(
    estimators: Mapping[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, BaseEstimator]:
    """Fit all baseline estimators without mutating the input estimator objects."""

    fitted_estimators: dict[str, BaseEstimator] = {}
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    for model_name, estimator in estimators.items():
        fitted_estimator = clone(estimator)
        if model_name == "hist_gradient_boosting":
            fitted_estimator.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            fitted_estimator.fit(X_train, y_train)
        fitted_estimators[model_name] = fitted_estimator

    return fitted_estimators


def predict_fraud_scores(
    estimator: BaseEstimator,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return fraud-class scores from a fitted binary classifier."""

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("predict_proba must return two binary-class columns.")

        classes = getattr(estimator, "classes_", np.array([0, 1]))
        positive_index = int(np.where(classes == 1)[0][0])
        return probabilities[:, positive_index]

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return np.asarray(scores, dtype=float)

    raise TypeError("Estimator must expose predict_proba or decision_function.")


__all__ = [
    "RANDOM_SEED",
    "PriorProbabilityClassifier",
    "build_baseline_estimators",
    "fit_baseline_estimators",
    "predict_fraud_scores",
]
