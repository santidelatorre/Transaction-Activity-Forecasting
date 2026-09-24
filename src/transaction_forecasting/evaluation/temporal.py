"""Leakage-safe temporal evaluation helpers."""

from __future__ import annotations

import pandas as pd


def temporal_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically; the test set is strictly after train."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if "timestamp" not in frame:
        raise ValueError("temporal_split requires a timestamp column")
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise", utc=True)
    if ordered["timestamp"].isna().any():
        raise ValueError("timestamp cannot contain missing values")
    ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)
    test_size = max(1, int(round(len(ordered) * test_fraction)))
    if test_size >= len(ordered):
        raise ValueError("At least two rows are required for a temporal split")
    # Keep all events at the boundary on the same side. Row slicing alone can
    # place identical timestamps in both sets, violating the strict time order.
    boundary = ordered.iloc[-test_size]["timestamp"]
    train = ordered[ordered["timestamp"] < boundary].copy()
    test = ordered[ordered["timestamp"] >= boundary].copy()
    if train.empty:
        raise ValueError("No nonempty strictly chronological split at this boundary")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def train_validation_test_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    embargo_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make ordered train, validation, and test partitions with optional row embargoes.

    The caller must define the temporal field explicitly. No random partition is
    available here, because it could expose future observations during training.
    """
    if time_column not in frame.columns:
        raise ValueError(f"Missing temporal split column: {time_column}")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    if embargo_rows < 0:
        raise ValueError("embargo_rows cannot be negative")
    ordered = frame.sort_values(time_column).reset_index(drop=True)
    validation_size = max(1, int(round(len(ordered) * validation_fraction)))
    test_size = max(1, int(round(len(ordered) * test_fraction)))
    train_end = len(ordered) - validation_size - test_size - (2 * embargo_rows)
    validation_start = train_end + embargo_rows
    test_start = validation_start + validation_size + embargo_rows
    if train_end < 1:
        raise ValueError("Not enough rows for train, validation, and test partitions")
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[validation_start : validation_start + validation_size].copy(),
        ordered.iloc[test_start:].copy(),
    )
