"""Canonical transaction contract used between hackathon modules."""

from __future__ import annotations

from typing import Final

import pandas as pd

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "customer_id",
    "transaction_id",
    "timestamp",
    "amount",
    "currency",
    "merchant",
)
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "merchant_normalized",
    "category",
    "account_id",
    "debit_credit",
)


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy of the canonical transaction table.

    Source-specific loaders should map their raw columns to this contract before
    calling downstream modules. Unknown extra columns are preserved.
    """
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing canonical transaction columns: {', '.join(missing)}")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise", utc=True)
    result["amount"] = pd.to_numeric(result["amount"], errors="raise")
    if result[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Canonical transaction columns cannot contain null values")
    return result.sort_values(["customer_id", "timestamp", "transaction_id"]).reset_index(drop=True)


def mock_transactions() -> pd.DataFrame:
    """Small deterministic dataset for parallel development and smoke tests."""
    rows = [
        ("c-001", "t-001", "2026-01-01", 120.0, "CHF", "Health Club"),
        ("c-001", "t-002", "2026-02-01", 120.0, "CHF", "Health Club"),
        ("c-001", "t-003", "2026-03-01", 121.0, "CHF", "Health Club"),
        ("c-001", "t-004", "2026-03-15", 45.0, "CHF", "Grocery Market"),
        ("c-002", "t-101", "2026-01-10", 80.0, "CHF", "City Transit"),
        ("c-002", "t-111", "2026-02-10", 81.0, "CHF", "City Transit"),
        ("c-002", "t-121", "2026-03-10", 79.0, "CHF", "City Transit"),
    ]
    return validate_transactions(pd.DataFrame(rows, columns=REQUIRED_COLUMNS))
