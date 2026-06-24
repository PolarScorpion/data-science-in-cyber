"""Reusable exploratory data analysis utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from matplotlib.figure import Figure


CORRELATION_METHODS = {"pearson", "spearman", "kendall"}


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when an EDA input column is unavailable."""

    missing_columns = sorted(set(columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def compute_class_prevalence(
    frame: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Return class counts and proportions for a target column."""

    _require_columns(frame, [target_col])
    counts = frame[target_col].value_counts(dropna=False).sort_index()
    result = counts.rename("count").to_frame()
    result["proportion"] = result["count"] / len(frame)
    result["percent"] = result["proportion"].mul(100)
    result.index.name = target_col
    return result


def summarize_numeric_features(
    frame: pd.DataFrame,
    numeric_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return distribution statistics for selected numerical columns."""

    columns = (
        list(numeric_cols)
        if numeric_cols is not None
        else frame.select_dtypes(include="number").columns.tolist()
    )
    _require_columns(frame, columns)
    if not columns:
        return pd.DataFrame()

    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Summary columns must be numeric: {non_numeric}")

    numeric_frame = frame[columns]
    summary = numeric_frame.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T
    summary["missing_count"] = numeric_frame.isna().sum()
    summary["zero_count"] = numeric_frame.eq(0).sum()
    summary["skew"] = numeric_frame.skew()
    return summary


def summarize_categorical_features(
    frame: pd.DataFrame,
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    """Return cardinality and dominant-value statistics for categorical fields."""

    columns = list(categorical_cols)
    _require_columns(frame, columns)
    summaries: list[dict[str, object]] = []

    for column in columns:
        counts = frame[column].value_counts(dropna=False)
        top_value = counts.index[0] if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        summaries.append(
            {
                "column": column,
                "distinct_values": int(frame[column].nunique(dropna=False)),
                "top_value": top_value,
                "top_count": top_count,
                "top_percent": top_count / len(frame) * 100 if len(frame) else 0.0,
            }
        )

    return pd.DataFrame(summaries).set_index("column")


def compute_missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for every column."""

    result = pd.DataFrame(
        {
            "missing_count": frame.isna().sum(),
            "missing_percent": frame.isna().mean().mul(100),
        }
    )
    result.index.name = "column"
    return result.sort_values(["missing_count", "missing_percent"], ascending=False)


def compute_outlier_summary_iqr(
    frame: pd.DataFrame,
    numeric_cols: Sequence[str],
) -> pd.DataFrame:
    """Summarize Tukey 1.5-IQR outliers without removing observations."""

    columns = list(numeric_cols)
    _require_columns(frame, columns)
    summaries: list[dict[str, float | int | str]] = []

    for column in columns:
        values = frame[column].dropna()
        if not pd.api.types.is_numeric_dtype(values):
            raise TypeError(f"Column {column!r} must be numeric for IQR analysis.")

        first_quartile = float(values.quantile(0.25))
        third_quartile = float(values.quantile(0.75))
        iqr = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * iqr
        upper_bound = third_quartile + 1.5 * iqr
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = int(outlier_mask.sum())

        summaries.append(
            {
                "column": column,
                "q1": first_quartile,
                "q3": third_quartile,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percent": outlier_count / len(values) * 100,
            }
        )

    return pd.DataFrame(summaries).set_index("column")


def compute_daily_transaction_summary(
    frame: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Aggregate daily transaction volume, fraud count, and fraud prevalence."""

    _require_columns(frame, [date_col, target_col])
    dates = pd.to_datetime(frame[date_col], errors="raise").dt.floor("D")
    daily = (
        frame.assign(_EDA_DATE=dates)
        .groupby("_EDA_DATE", observed=True)[target_col]
        .agg(transactions="size", fraud_count="sum", fraud_rate="mean")
    )
    daily.index.name = "date"
    return daily


def compute_hourly_fraud_summary(
    frame: pd.DataFrame,
    datetime_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Aggregate volume and fraud prevalence by hour of day."""

    _require_columns(frame, [datetime_col, target_col])
    hours = pd.to_datetime(frame[datetime_col], errors="raise").dt.hour
    hourly = (
        frame.assign(_EDA_HOUR=hours)
        .groupby("_EDA_HOUR", observed=True)[target_col]
        .agg(transactions="size", fraud_count="sum", fraud_rate="mean")
        .reindex(range(24), fill_value=0)
    )
    hourly.index.name = "hour"
    return hourly


def compute_grouped_fraud_rates(
    frame: pd.DataFrame,
    group_cols: str | Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    """Aggregate transaction and fraud counts for meaningful groups."""

    columns = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    _require_columns(frame, [*columns, target_col])
    return (
        frame.groupby(columns, dropna=False, observed=True)[target_col]
        .agg(transactions="size", fraud_count="sum", fraud_rate="mean")
        .sort_index()
    )


def compute_correlation_matrix(
    frame: pd.DataFrame,
    numeric_cols: Sequence[str],
    method: str,
) -> pd.DataFrame:
    """Compute a justified correlation matrix for explicitly selected columns."""

    columns = list(numeric_cols)
    _require_columns(frame, columns)
    normalized_method = method.casefold()
    if normalized_method not in CORRELATION_METHODS:
        allowed = ", ".join(sorted(CORRELATION_METHODS))
        raise ValueError(f"Correlation method must be one of: {allowed}.")

    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"Correlation columns must be numeric: {non_numeric}")

    return frame[columns].corr(method=normalized_method)


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 160,
) -> Path:
    """Save a figure deterministically after creating its parent directory."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


__all__ = [
    "compute_class_prevalence",
    "compute_correlation_matrix",
    "compute_daily_transaction_summary",
    "compute_grouped_fraud_rates",
    "compute_hourly_fraud_summary",
    "compute_missing_summary",
    "compute_outlier_summary_iqr",
    "save_figure",
    "summarize_categorical_features",
    "summarize_numeric_features",
]
