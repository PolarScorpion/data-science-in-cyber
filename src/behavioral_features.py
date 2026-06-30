"""Leakage-safe behavioral feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.preprocessing import (
    BASELINE_FEATURE_COLUMNS,
    DATETIME_COLUMN,
    TARGET_COLUMN,
    build_baseline_feature_matrix,
)


AMOUNT_COLUMN = "TX_AMOUNT"
CUSTOMER_COLUMN = "CUSTOMER_ID"
TERMINAL_COLUMN = "TERMINAL_ID"
TRANSACTION_COLUMN = "TRANSACTION_ID"
ONE_DAY_SECONDS = 24 * 60 * 60
SEVEN_DAYS_SECONDS = 7 * ONE_DAY_SECONDS

UNLABELED_BEHAVIORAL_FEATURE_COLUMNS = (
    *BASELINE_FEATURE_COLUMNS,
    "customer_tx_count_past",
    "customer_mean_amount_past",
    "customer_median_amount_past",
    "customer_std_amount_past",
    "customer_tx_count_1d_past",
    "customer_tx_count_7d_past",
    "customer_seconds_since_prev_tx",
    "terminal_tx_count_past",
    "terminal_mean_amount_past",
    "terminal_tx_count_1d_past",
    "terminal_tx_count_7d_past",
    "terminal_seconds_since_prev_tx",
    "customer_terminal_tx_count_past",
)

PAST_LABEL_FEATURE_COLUMNS = (
    "customer_fraud_count_past",
    "customer_fraud_rate_past",
    "customer_fraud_count_7d_past",
    "terminal_fraud_count_past",
    "terminal_fraud_rate_past",
    "terminal_fraud_count_7d_past",
)


@dataclass(frozen=True)
class BehavioralFeatureSpec:
    """Description of behavioral feature availability assumptions."""

    feature_set: str
    include_past_fraud_labels: bool
    operational_assumption: str


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error when required columns are unavailable."""

    missing_columns = sorted(set(columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _sort_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return transactions sorted into deterministic prediction order."""

    sort_columns = [DATETIME_COLUMN]
    if TRANSACTION_COLUMN in frame.columns:
        sort_columns.append(TRANSACTION_COLUMN)
    sorted_frame = frame.sort_values(sort_columns, kind="mergesort").copy()
    if not sorted_frame.index.is_unique:
        raise ValueError("Behavioral feature construction requires a unique index.")
    return sorted_frame


def _aggregate_by_entity_time(
    frame: pd.DataFrame,
    entity_cols: Sequence[str],
    *,
    include_target: bool,
) -> pd.DataFrame:
    """Aggregate transactions to entity-time blocks for strict past histories."""

    columns = [*entity_cols, DATETIME_COLUMN, AMOUNT_COLUMN]
    if include_target:
        columns.append(TARGET_COLUMN)
    _require_columns(frame, columns)

    working = frame.loc[:, columns].copy()
    working[AMOUNT_COLUMN] = pd.to_numeric(
        working[AMOUNT_COLUMN],
        errors="raise",
    ).astype(float)
    working["_amount_square"] = working[AMOUNT_COLUMN] ** 2

    aggregations: dict[str, tuple[str, str]] = {
        "block_count": (AMOUNT_COLUMN, "size"),
        "block_amount_sum": (AMOUNT_COLUMN, "sum"),
        "block_amount_square_sum": ("_amount_square", "sum"),
    }
    if include_target:
        aggregations["block_fraud_count"] = (TARGET_COLUMN, "sum")

    return (
        working.groupby([*entity_cols, DATETIME_COLUMN], sort=False, observed=True)
        .agg(**aggregations)
        .reset_index()
    )


def _recent_count_from_blocks(
    block_stats: pd.DataFrame,
    entity_cols: Sequence[str],
    *,
    value_col: str,
    window_seconds: int,
) -> pd.Series:
    """Count prior entity events inside a strict lookback window."""

    result = pd.Series(0.0, index=block_stats.index)
    window_ns = np.int64(window_seconds) * np.int64(1_000_000_000)

    for _, group in block_stats.groupby(list(entity_cols), sort=False, observed=True):
        positions = group.index.to_numpy()
        times = pd.to_datetime(group[DATETIME_COLUMN], errors="raise").astype("int64")
        times_array = times.to_numpy()
        values = group[value_col].to_numpy(dtype=float)
        cumulative_values = np.cumsum(values)
        prior_values = cumulative_values - values

        lower_positions = np.searchsorted(
            times_array,
            times_array - window_ns,
            side="left",
        )
        before_lower = np.where(
            lower_positions > 0,
            cumulative_values[lower_positions - 1],
            0.0,
        )
        result.loc[positions] = prior_values - before_lower

    return result


def _strict_past_entity_stats(
    frame: pd.DataFrame,
    entity_cols: Sequence[str],
    *,
    prefix: str,
    include_target: bool,
) -> pd.DataFrame:
    """Compute strict prior count, amount, recency, and optional fraud histories."""

    entity_cols = tuple(entity_cols)
    block_stats = _aggregate_by_entity_time(
        frame,
        entity_cols,
        include_target=include_target,
    )
    group = block_stats.groupby(list(entity_cols), sort=False, observed=True)

    cumulative_count = group["block_count"].cumsum()
    cumulative_amount = group["block_amount_sum"].cumsum()
    cumulative_amount_square = group["block_amount_square_sum"].cumsum()

    block_stats[f"{prefix}_tx_count_past"] = (
        cumulative_count - block_stats["block_count"]
    ).astype(float)
    past_amount_sum = cumulative_amount - block_stats["block_amount_sum"]
    past_amount_square_sum = (
        cumulative_amount_square - block_stats["block_amount_square_sum"]
    )
    past_count = block_stats[f"{prefix}_tx_count_past"]
    block_stats[f"{prefix}_mean_amount_past"] = past_amount_sum / past_count.replace(
        0,
        np.nan,
    )

    variance_numerator = (
        past_amount_square_sum - (past_amount_sum**2 / past_count.replace(0, np.nan))
    )
    variance = variance_numerator / (past_count - 1).where(past_count > 1)
    block_stats[f"{prefix}_std_amount_past"] = np.sqrt(variance.clip(lower=0))

    previous_time = group[DATETIME_COLUMN].shift(1)
    seconds_since_previous = (
        pd.to_datetime(block_stats[DATETIME_COLUMN], errors="raise")
        - pd.to_datetime(previous_time, errors="raise")
    ).dt.total_seconds()
    block_stats[f"{prefix}_seconds_since_prev_tx"] = seconds_since_previous

    block_stats[f"{prefix}_tx_count_1d_past"] = _recent_count_from_blocks(
        block_stats,
        entity_cols,
        value_col="block_count",
        window_seconds=ONE_DAY_SECONDS,
    )
    block_stats[f"{prefix}_tx_count_7d_past"] = _recent_count_from_blocks(
        block_stats,
        entity_cols,
        value_col="block_count",
        window_seconds=SEVEN_DAYS_SECONDS,
    )

    if include_target:
        cumulative_fraud = group["block_fraud_count"].cumsum()
        past_fraud = cumulative_fraud - block_stats["block_fraud_count"]
        block_stats[f"{prefix}_fraud_count_past"] = past_fraud.astype(float)
        block_stats[f"{prefix}_fraud_rate_past"] = past_fraud / past_count.replace(
            0,
            np.nan,
        )
        block_stats[f"{prefix}_fraud_count_7d_past"] = _recent_count_from_blocks(
            block_stats,
            entity_cols,
            value_col="block_fraud_count",
            window_seconds=SEVEN_DAYS_SECONDS,
        )

    feature_columns = [
        column
        for column in block_stats.columns
        if column.startswith(f"{prefix}_")
    ]
    return frame.loc[:, [*entity_cols, DATETIME_COLUMN]].merge(
        block_stats.loc[:, [*entity_cols, DATETIME_COLUMN, *feature_columns]],
        on=[*entity_cols, DATETIME_COLUMN],
        how="left",
        sort=False,
    ).set_index(frame.index)[feature_columns]


def _strict_past_customer_median(frame: pd.DataFrame) -> pd.Series:
    """Compute exact prior customer median amount excluding same-time events."""

    _require_columns(frame, [CUSTOMER_COLUMN, DATETIME_COLUMN, AMOUNT_COLUMN])
    row_median = (
        frame.groupby(CUSTOMER_COLUMN, sort=False, observed=True)[AMOUNT_COLUMN]
        .expanding()
        .median()
        .groupby(level=0)
        .shift(1)
        .droplevel(0)
        .reindex(frame.index)
    )

    return row_median.groupby(
        [frame[CUSTOMER_COLUMN], frame[DATETIME_COLUMN]],
        sort=False,
        observed=True,
    ).transform("first")


def _strict_past_interaction_count(frame: pd.DataFrame) -> pd.Series:
    """Count prior customer-terminal interactions before each transaction time."""

    block_stats = _aggregate_by_entity_time(
        frame,
        [CUSTOMER_COLUMN, TERMINAL_COLUMN],
        include_target=False,
    )
    group = block_stats.groupby(
        [CUSTOMER_COLUMN, TERMINAL_COLUMN],
        sort=False,
        observed=True,
    )
    block_stats["customer_terminal_tx_count_past"] = (
        group["block_count"].cumsum() - block_stats["block_count"]
    ).astype(float)

    return frame.loc[:, [CUSTOMER_COLUMN, TERMINAL_COLUMN, DATETIME_COLUMN]].merge(
        block_stats.loc[
            :,
            [
                CUSTOMER_COLUMN,
                TERMINAL_COLUMN,
                DATETIME_COLUMN,
                "customer_terminal_tx_count_past",
            ],
        ],
        on=[CUSTOMER_COLUMN, TERMINAL_COLUMN, DATETIME_COLUMN],
        how="left",
        sort=False,
    ).set_index(frame.index)["customer_terminal_tx_count_past"]


def build_behavioral_feature_matrix(
    frame: pd.DataFrame,
    *,
    include_past_fraud_labels: bool = False,
) -> pd.DataFrame:
    """Build baseline plus strict past-only behavioral features.

    Past fraud-label features are optional. When enabled, they encode only labels
    from transactions that occurred before the transaction being scored.
    """

    required_columns = [
        DATETIME_COLUMN,
        AMOUNT_COLUMN,
        CUSTOMER_COLUMN,
        TERMINAL_COLUMN,
        TARGET_COLUMN,
    ]
    _require_columns(frame, required_columns)
    sorted_frame = _sort_transactions(frame)

    baseline_features = build_baseline_feature_matrix(sorted_frame)
    customer_features = _strict_past_entity_stats(
        sorted_frame,
        [CUSTOMER_COLUMN],
        prefix="customer",
        include_target=include_past_fraud_labels,
    )
    terminal_features = _strict_past_entity_stats(
        sorted_frame,
        [TERMINAL_COLUMN],
        prefix="terminal",
        include_target=include_past_fraud_labels,
    )
    customer_features["customer_median_amount_past"] = (
        _strict_past_customer_median(sorted_frame)
    )
    interaction_count = _strict_past_interaction_count(sorted_frame)

    features = pd.concat(
        [
            baseline_features,
            customer_features,
            terminal_features,
            interaction_count.rename("customer_terminal_tx_count_past"),
        ],
        axis=1,
    )

    if not include_past_fraud_labels:
        features = features.loc[:, list(UNLABELED_BEHAVIORAL_FEATURE_COLUMNS)]
    else:
        features = features.loc[
            :,
            [
                *UNLABELED_BEHAVIORAL_FEATURE_COLUMNS,
                *PAST_LABEL_FEATURE_COLUMNS,
            ],
        ]

    count_columns = [
        column
        for column in features.columns
        if column.endswith("_count_past")
        or column.endswith("_count_1d_past")
        or column.endswith("_count_7d_past")
    ]
    features.loc[:, count_columns] = features.loc[:, count_columns].fillna(0.0)
    return features.reindex(frame.index)


def describe_behavioral_feature_sets() -> pd.DataFrame:
    """Document feature-set assumptions for reporting and notebooks."""

    rows = [
        BehavioralFeatureSpec(
            feature_set="behavior_unlabeled",
            include_past_fraud_labels=False,
            operational_assumption=(
                "Uses only prior transaction amounts, timestamps, customers, "
                "terminals, and customer-terminal interactions."
            ),
        ),
        BehavioralFeatureSpec(
            feature_set="behavior_label_history",
            include_past_fraud_labels=True,
            operational_assumption=(
                "Adds prior fraud-label counts and rates. This assumes earlier "
                "fraud labels are already known when a later transaction is scored."
            ),
        ),
    ]
    return pd.DataFrame([row.__dict__ for row in rows]).set_index("feature_set")


__all__ = [
    "PAST_LABEL_FEATURE_COLUMNS",
    "UNLABELED_BEHAVIORAL_FEATURE_COLUMNS",
    "BehavioralFeatureSpec",
    "build_behavioral_feature_matrix",
    "describe_behavioral_feature_sets",
]
