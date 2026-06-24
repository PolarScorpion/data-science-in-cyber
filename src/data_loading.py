"""Dataset loading and initial inspection utilities."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


DATASET_REPOSITORY_URL = (
    "https://github.com/Fraud-Detection-Handbook/simulated-data-raw"
)
DATASET_CLONE_COMMAND = (
    "git clone --depth 1 "
    f"{DATASET_REPOSITORY_URL}.git data/raw/simulated-data-raw"
)
RAW_TRANSACTION_COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)


def get_project_root() -> Path:
    """Return the repository root without depending on the current directory."""

    return Path(__file__).resolve().parents[1]


def get_data_paths(project_root: str | Path | None = None) -> dict[str, Path]:
    """Return the documented project data locations without creating them."""

    root = Path(project_root).resolve() if project_root else get_project_root()
    data_directory = root / "data"
    raw_directory = data_directory / "raw"

    return {
        "project": root,
        "data": data_directory,
        "raw": raw_directory,
        "processed": data_directory / "processed",
        "transactions": raw_directory / "simulated-data-raw" / "data",
    }


def _display_path(path: Path) -> str:
    """Format a project path without exposing a machine-specific prefix."""

    try:
        return path.resolve().relative_to(get_project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _as_date(value: str | date | pd.Timestamp | None) -> date | None:
    """Normalize an optional date-like value for filename filtering."""

    if value is None:
        return None

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Date filters must not be missing values.")
    return timestamp.date()


def load_available_transaction_files(
    data_directory: str | Path | None = None,
    *,
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
) -> list[Path]:
    """List available daily pickle files in an inclusive date range."""

    directory = (
        Path(data_directory)
        if data_directory is not None
        else get_data_paths()["transactions"]
    )
    start = _as_date(start_date)
    end = _as_date(end_date)

    if start and end and start > end:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Transaction directory not found: {_display_path(directory)}. "
            f"Acquire the official data with: {DATASET_CLONE_COMMAND}"
        )

    dated_files: list[tuple[date, Path]] = []
    for path in sorted(directory.glob("*.pkl")):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError as error:
            raise ValueError(
                f"Unexpected transaction filename {path.name!r}; "
                "expected YYYY-MM-DD.pkl."
            ) from error
        dated_files.append((file_date, path))

    selected_files = [
        path
        for file_date, path in dated_files
        if (start is None or file_date >= start) and (end is None or file_date <= end)
    ]

    if not selected_files:
        date_description = (
            f" between {start or 'the first date'} and {end or 'the last date'}"
        )
        raise FileNotFoundError(
            f"No daily transaction files were found{date_description} in "
            f"{_display_path(directory)}."
        )

    return selected_files


def load_transactions(
    data_directory: str | Path | None = None,
    *,
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
    validate_schema: bool = True,
) -> pd.DataFrame:
    """Load official daily transaction pickles without transforming features.

    Date filters are inclusive. Pickle files can execute code while loading, so this
    function should only be used with files acquired from the documented source.
    """

    transaction_files = load_available_transaction_files(
        data_directory,
        start_date=start_date,
        end_date=end_date,
    )

    daily_frames: list[pd.DataFrame] = []
    for path in transaction_files:
        try:
            daily_frame = pd.read_pickle(path)
        except Exception as error:
            raise ValueError(f"Could not read transaction file {path.name!r}.") from error

        if not isinstance(daily_frame, pd.DataFrame):
            raise TypeError(
                f"Transaction file {path.name!r} does not contain a DataFrame."
            )
        daily_frames.append(daily_frame)

    transactions = pd.concat(daily_frames, ignore_index=True)
    if validate_schema:
        validate_transaction_schema(transactions)
    return transactions


def validate_transaction_schema(frame: pd.DataFrame) -> None:
    """Raise a clear error when the raw transaction schema is not usable."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The transaction DataFrame is empty.")

    missing_columns = sorted(set(RAW_TRANSACTION_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required transaction columns: {missing_columns}")

    if not is_datetime64_any_dtype(frame["TX_DATETIME"]):
        raise TypeError("TX_DATETIME must use a pandas datetime dtype.")

    if frame["TRANSACTION_ID"].isna().any():
        raise ValueError("TRANSACTION_ID contains missing values.")
    if frame["TX_FRAUD"].isna().any():
        raise ValueError("TX_FRAUD contains missing target labels.")

    unexpected_targets = set(frame["TX_FRAUD"].unique()).difference({0, 1})
    if unexpected_targets:
        raise ValueError(
            "TX_FRAUD contains unexpected labels: "
            f"{sorted(map(repr, unexpected_targets))}"
        )

    unexpected_scenarios = (
        set(frame["TX_FRAUD_SCENARIO"].dropna().unique()).difference({0, 1, 2, 3})
    )
    if unexpected_scenarios:
        raise ValueError(
            "TX_FRAUD_SCENARIO contains unexpected values: "
            f"{sorted(map(repr, unexpected_scenarios))}"
        )


def summarize_dataframe(frame: pd.DataFrame) -> pd.Series:
    """Return compact structural statistics for an inspection report."""

    return pd.Series(
        {
            "rows": len(frame),
            "columns": frame.shape[1],
            "memory_mib": round(frame.memory_usage(deep=True).sum() / (1024**2), 2),
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "index_type": type(frame.index).__name__,
            "index_name": frame.index.name if frame.index.name is not None else "<unnamed>",
        },
        name="value",
    )


def detect_constant_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns containing at most one distinct value, including missingness."""

    distinct_counts = frame.nunique(dropna=False)
    return distinct_counts[distinct_counts <= 1].index.tolist()


def detect_duplicate_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map each exact duplicate column to the earlier matching column."""

    duplicate_columns: dict[str, str] = {}
    columns = list(frame.columns)

    for position, column in enumerate(columns):
        for earlier_column in columns[:position]:
            if frame[column].equals(frame[earlier_column]):
                duplicate_columns[column] = earlier_column
                break

    return duplicate_columns


__all__ = [
    "DATASET_CLONE_COMMAND",
    "DATASET_REPOSITORY_URL",
    "RAW_TRANSACTION_COLUMNS",
    "detect_constant_columns",
    "detect_duplicate_columns",
    "get_data_paths",
    "get_project_root",
    "load_available_transaction_files",
    "load_transactions",
    "summarize_dataframe",
    "validate_transaction_schema",
]
