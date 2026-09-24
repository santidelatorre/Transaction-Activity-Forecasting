"""Dataset contract and local loading for the UBS challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
LABELS = ("cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none")
TARGET_COLUMN = "target_next_recurring_merchant"
PREDICTION_COLUMN = "predicted_next_recurring_merchant"
TRANSACTION_COLUMNS = {
    "client_id",
    "timestamp",
    "amount",
    "currency",
    "description",
    "direction",
    "fee",
    "mcc",
    "type",
}


@dataclass(frozen=True)
class UBSData:
    """All official partitions, with test labels deliberately absent."""

    train_transactions: pd.DataFrame
    train_labels: pd.DataFrame
    valid_transactions: pd.DataFrame
    valid_labels: pd.DataFrame
    test_transactions: pd.DataFrame
    sample_submission: pd.DataFrame


def read_transactions(path: str | Path) -> pd.DataFrame:
    """Load one JSONL transaction partition and enforce the pre-cutoff contract."""
    frame = pd.read_json(path, lines=True, convert_dates=False)
    missing = TRANSACTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing transaction columns in {path}: {sorted(missing)}")
    frame = frame.loc[:, sorted(TRANSACTION_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["client_id"].isna().any() or frame["timestamp"].isna().any():
        raise ValueError(f"Null client_id/timestamp values in {path}")
    if (frame["timestamp"] >= CUTOFF).any():
        raise ValueError(f"Leakage detected: transactions at or after {CUTOFF} in {path}")
    if frame.duplicated().any():
        raise ValueError(f"Exact duplicate transactions in {path}")
    frame["description"] = frame["description"].astype(str).str.lower().str.strip()
    frame["currency"] = frame["currency"].astype(str).str.lower()
    frame["direction"] = frame["direction"].astype(str).str.lower()
    frame["type"] = frame["type"].astype(str).str.lower()
    frame["mcc"] = frame["mcc"].astype(str)
    return frame.sort_values(["client_id", "timestamp"], kind="stable").reset_index(drop=True)


def read_labels(path: str | Path) -> pd.DataFrame:
    """Load labels and validate the fixed challenge target vocabulary."""
    frame = pd.read_csv(path, dtype={"client_id": str})
    required = {"client_id", "cutoff_date", TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing label columns in {path}: {sorted(missing)}")
    if frame["client_id"].duplicated().any():
        raise ValueError(f"Duplicate client labels in {path}")
    unknown = set(frame[TARGET_COLUMN]).difference(LABELS)
    if unknown:
        raise ValueError(f"Unknown target labels in {path}: {sorted(unknown)}")
    cutoff = pd.to_datetime(frame["cutoff_date"], utc=True, errors="raise")
    if not cutoff.eq(CUTOFF).all():
        raise ValueError(f"Unexpected cutoff date in {path}")
    return frame


def load_ubs_data(data_dir: str | Path) -> UBSData:
    """Load and cross-check train, validation, test, and submission template."""
    root = Path(data_dir)
    data = UBSData(
        train_transactions=read_transactions(root / "train_transactions.jsonl"),
        train_labels=read_labels(root / "train_labels.csv"),
        valid_transactions=read_transactions(root / "valid_transactions.jsonl"),
        valid_labels=read_labels(root / "valid_labels.csv"),
        test_transactions=read_transactions(root / "test_transactions.jsonl"),
        sample_submission=pd.read_csv(root / "sample_submission.csv", dtype={"client_id": str}),
    )
    transaction_sets = {
        "train": set(data.train_transactions["client_id"]),
        "valid": set(data.valid_transactions["client_id"]),
        "test": set(data.test_transactions["client_id"]),
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = transaction_sets[left].intersection(transaction_sets[right])
        if overlap:
            raise ValueError(f"Client leakage between {left} and {right}: {len(overlap)}")
    for name, labels, transactions in (
        ("train", data.train_labels, data.train_transactions),
        ("valid", data.valid_labels, data.valid_transactions),
    ):
        if set(labels["client_id"]) != set(transactions["client_id"]):
            raise ValueError(f"{name} label/transaction client sets do not match")
    if set(data.sample_submission["client_id"]) != transaction_sets["test"]:
        raise ValueError("sample_submission client IDs do not match test")
    return data


def validate_submission(
    submission: pd.DataFrame, sample: pd.DataFrame, test_transactions: pd.DataFrame
) -> None:
    """Strictly verify schema, row order, IDs, cardinality, and allowed predictions."""
    expected_columns = ["client_id", PREDICTION_COLUMN]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"Submission columns must be exactly {expected_columns}")
    if len(submission) != len(sample) or submission["client_id"].duplicated().any():
        raise ValueError("Submission row count or client uniqueness is invalid")
    if not submission["client_id"].equals(sample["client_id"]):
        raise ValueError("Submission must preserve sample_submission client order")
    if set(submission["client_id"]) != set(test_transactions["client_id"]):
        raise ValueError("Submission client IDs do not match test")
    if submission[PREDICTION_COLUMN].isna().any():
        raise ValueError("Submission contains missing predictions")
    unknown = set(submission[PREDICTION_COLUMN]).difference(LABELS)
    if unknown:
        raise ValueError(f"Submission contains unknown labels: {sorted(unknown)}")
