"""Configurable CSV and Parquet loading at the raw-data boundary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from transaction_forecasting.config import DatasetConfig


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Read and validate a CSV or Parquet file according to its configured contract."""
    if config.path is None or config.format is None:
        raise ValueError("Dataset path and format must be configured")
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(path)
    format_name = config.format.lower()
    if format_name == "csv":
        frame = pd.read_csv(path)
    elif format_name == "parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Dataset format must be 'csv' or 'parquet'")
    for column, expected_type in config.column_types.items():
        if expected_type == "datetime" and column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="raise", utc=True)
    return validate_frame(frame, config.required_columns, config.column_types)


def validate_frame(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...] = (),
    column_types: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate only the explicit, source-independent schema contract."""
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    for column, expected_type in (column_types or {}).items():
        if column not in frame.columns:
            raise ValueError(f"Configured type column is missing: {column}")
        series = frame[column]
        if expected_type == "numeric" and not is_numeric_dtype(series):
            raise TypeError(f"Column '{column}' must be numeric")
        if expected_type == "datetime" and not is_datetime64_any_dtype(series):
            raise TypeError(f"Column '{column}' must be datetime")
        if expected_type == "string" and not (
            pd.api.types.is_string_dtype(series) or series.dtype == object
        ):
            raise TypeError(f"Column '{column}' must be string-like")
        if expected_type not in {"numeric", "datetime", "string"}:
            raise ValueError(f"Unsupported configured type '{expected_type}' for '{column}'")
    return frame.copy()
