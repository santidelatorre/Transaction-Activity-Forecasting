"""Dataset contract and local loading for the UBS challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from transaction_forecasting.evaluation.official import (
    CUTOFF_DATE,
    PREDICTION,
    TARGET,
    validate_client_ids,
    validate_labels,
)
from transaction_forecasting.evaluation.official import LABELS as LABELS
from transaction_forecasting.evaluation.official import validate_submission as validate_client_set

CUTOFF = pd.Timestamp(CUTOFF_DATE, tz="UTC")
PREDICTION_COLUMN = PREDICTION
TARGET_COLUMN = TARGET
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


def validate_history(frame: pd.DataFrame) -> None:
    """Reject invalid IDs and future/unknown times at every feature entry point."""
    validate_client_ids(frame[["client_id"]].drop_duplicates(), "transactions")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise", format="mixed")
    if timestamps.isna().any():
        raise ValueError("Null timestamp values in transactions")
    if (timestamps >= CUTOFF).any():
        raise ValueError(f"Leakage detected: transactions at or after cutoff {CUTOFF}")


def read_transactions(path: str | Path) -> pd.DataFrame:
    """Load one JSONL transaction partition and enforce the pre-cutoff contract."""
    frame = pd.read_json(path, lines=True, convert_dates=False, dtype=False)
    missing = TRANSACTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing transaction columns in {path}: {sorted(missing)}")
    frame = frame.loc[:, sorted(TRANSACTION_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="raise", format="mixed"
    )
    validate_history(frame)
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
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_labels(frame)
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
        sample_submission=pd.read_csv(
            root / "sample_submission.csv", dtype=str, keep_default_na=False
        ),
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
    # Validate the template before training; its placeholders are not target labels.
    probe = data.sample_submission.assign(**{PREDICTION: "none"})
    validate_submission(probe, data.sample_submission, data.test_transactions)
    return data


def split_training_clients(
    labels: pd.DataFrame, *, holdout_fraction: float = 0.25, seed: int = 42
) -> tuple[pd.Index, pd.Index]:
    """Create one reproducible stratified selection holdout inside official train."""
    validate_labels(labels)
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    ordered = labels.sort_values("client_id").set_index("client_id")
    fit_ids, selection_ids = train_test_split(
        ordered.index,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=ordered[TARGET_COLUMN],
    )
    return pd.Index(sorted(fit_ids), name="client_id"), pd.Index(
        sorted(selection_ids), name="client_id"
    )


def validate_submission(
    submission: pd.DataFrame, sample: pd.DataFrame, test_transactions: pd.DataFrame
) -> None:
    """Strictly verify schema, row order, IDs, cardinality, and allowed predictions."""
    # Share schema, vocabulary and coverage checks with the CSV CLI while
    # retaining UBS V1's stricter order check and None return contract.
    validate_client_set(submission, sample)
    if not submission["client_id"].equals(sample["client_id"]):
        raise ValueError("Submission must preserve sample_submission client order")
    if set(submission["client_id"]) != set(test_transactions["client_id"]):
        raise ValueError("Submission client IDs do not match test")
