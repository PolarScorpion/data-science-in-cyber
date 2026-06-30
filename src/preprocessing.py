"""Preprocessing and feature-engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


TARGET_COLUMN = "TX_FRAUD"
DATETIME_COLUMN = "TX_DATETIME"
BASELINE_FEATURE_COLUMNS = (
    "tx_amount",
    "log_tx_amount",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
)
LEAKAGE_COLUMNS = (
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)
BASELINE_EXCLUDED_RAW_COLUMNS = (
    "TRANSACTION_ID",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)


@dataclass(frozen=True)
class TemporalSplit:
    """Container for a chronological train/validation/test split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when required columns are unavailable."""

    missing_columns = sorted(set(columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def make_temporal_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    datetime_col: str = DATETIME_COLUMN,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
) -> TemporalSplit:
    """Split transactions by calendar day without shuffling across time."""

    _require_columns(frame, [datetime_col])
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must leave a test period.")

    timestamps = pd.to_datetime(frame[datetime_col], errors="raise")
    normalized_dates = timestamps.dt.floor("D")
    unique_dates = pd.Index(sorted(normalized_dates.unique()))
    if len(unique_dates) < 3:
        raise ValueError("At least three distinct dates are required for temporal splits.")

    train_days = max(1, int(round(len(unique_dates) * train_ratio)))
    validation_days = max(1, int(round(len(unique_dates) * validation_ratio)))
    if train_days + validation_days >= len(unique_dates):
        validation_days = max(1, len(unique_dates) - train_days - 1)
    if train_days + validation_days >= len(unique_dates):
        raise ValueError("Temporal split ratios do not leave a non-empty test set.")

    train_dates = unique_dates[:train_days]
    validation_dates = unique_dates[train_days : train_days + validation_days]
    test_dates = unique_dates[train_days + validation_days :]

    train_mask = normalized_dates.isin(train_dates)
    validation_mask = normalized_dates.isin(validation_dates)
    test_mask = normalized_dates.isin(test_dates)

    sort_columns = [datetime_col]
    if "TRANSACTION_ID" in frame.columns:
        sort_columns.append("TRANSACTION_ID")

    def _subset(mask: pd.Series) -> pd.DataFrame:
        return frame.loc[mask].sort_values(sort_columns, kind="mergesort").copy()

    train = _subset(train_mask)
    validation = _subset(validation_mask)
    test = _subset(test_mask)
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition.")

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        train_start=pd.Timestamp(train_dates[0]),
        train_end=pd.Timestamp(train_dates[-1]),
        validation_start=pd.Timestamp(validation_dates[0]),
        validation_end=pd.Timestamp(validation_dates[-1]),
        test_start=pd.Timestamp(test_dates[0]),
        test_end=pd.Timestamp(test_dates[-1]),
    )


def describe_temporal_split(
    split: TemporalSplit,
    *,
    datetime_col: str = DATETIME_COLUMN,
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Return row counts, date spans, and fraud prevalence for each split."""

    rows: list[dict[str, object]] = []
    for split_name in ("train", "validation", "test"):
        partition = getattr(split, split_name)
        _require_columns(partition, [datetime_col, target_col])
        dates = pd.to_datetime(partition[datetime_col], errors="raise").dt.floor("D")
        target = partition[target_col].astype(int)
        rows.append(
            {
                "split": split_name,
                "rows": len(partition),
                "days": int(dates.nunique()),
                "start_date": dates.min().date().isoformat(),
                "end_date": dates.max().date().isoformat(),
                "fraud_count": int(target.sum()),
                "fraud_rate": float(target.mean()),
            }
        )

    return pd.DataFrame(rows).set_index("split")


def build_baseline_feature_matrix(
    frame: pd.DataFrame,
    *,
    datetime_col: str = DATETIME_COLUMN,
) -> pd.DataFrame:
    """Build leakage-safe baseline features available at transaction time."""

    _require_columns(frame, ["TX_AMOUNT", datetime_col])
    amount = pd.to_numeric(frame["TX_AMOUNT"], errors="raise").astype(float)
    if amount.lt(0).any():
        raise ValueError("TX_AMOUNT contains negative values; log1p is not valid.")

    timestamps = pd.to_datetime(frame[datetime_col], errors="raise")
    hour = timestamps.dt.hour.astype(float)
    dayofweek = timestamps.dt.dayofweek.astype(float)

    features = pd.DataFrame(index=frame.index)
    features["tx_amount"] = amount
    features["log_tx_amount"] = np.log1p(amount)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    features["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
    return features.loc[:, list(BASELINE_FEATURE_COLUMNS)]


def extract_target(
    frame: pd.DataFrame,
    *,
    target_col: str = TARGET_COLUMN,
) -> pd.Series:
    """Return the binary fraud target as integers."""

    _require_columns(frame, [target_col])
    target = frame[target_col].astype(int)
    unexpected_targets = set(target.unique()).difference({0, 1})
    if unexpected_targets:
        raise ValueError(f"Unexpected target labels: {sorted(unexpected_targets)}")
    return target


def assert_no_leakage_columns(columns: Sequence[str]) -> None:
    """Validate that a feature list does not include explicit target leakage."""

    unsafe_columns = sorted(set(columns).intersection(LEAKAGE_COLUMNS))
    if unsafe_columns:
        raise ValueError(f"Leakage columns must not be used as predictors: {unsafe_columns}")


__all__ = [
    "BASELINE_EXCLUDED_RAW_COLUMNS",
    "BASELINE_FEATURE_COLUMNS",
    "DATETIME_COLUMN",
    "LEAKAGE_COLUMNS",
    "TARGET_COLUMN",
    "TemporalSplit",
    "assert_no_leakage_columns",
    "build_baseline_feature_matrix",
    "describe_temporal_split",
    "extract_target",
    "make_temporal_train_validation_test_split",
]
