from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS = [
    "cloud",
    "gym",
    "insurance",
    "mobile",
    "music",
    "software",
    "streaming",
    "none",
]
TARGET = "target_next_recurring_merchant"
PREDICTION = "predicted_next_recurring_merchant"
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")


def transactions(split, *, use_cache=True, data_dir=None):
    if split not in {"train", "valid", "test", "unlabeled_pretrain"}:
        raise ValueError(split)
    cache = ROOT / f"data/cache/{split}.parquet"
    if use_cache and data_dir is None and cache.exists():
        df = pd.read_parquet(cache)
    else:
        raw = ROOT / "data/raw" if data_dir is None else Path(data_dir)
        df = pd.read_json(
            raw / f"{split}_transactions.jsonl", lines=True, dtype={"mcc": str}
        )
        df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    if df.timestamp.max() >= CUTOFF:
        raise ValueError("Transactions cross the prediction cutoff")
    return df.sort_values(["client_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def labels(split="train", *, allow_holdout=False, data_dir=None):
    if split == "valid" and not allow_holdout:
        raise ValueError("Official holdout requires an explicit frozen evaluation")
    if split not in {"train", "valid"}:
        raise ValueError("No labels are available for this split")
    raw = ROOT / "data/raw" if data_dir is None else Path(data_dir)
    df = pd.read_csv(raw / f"{split}_labels.csv").set_index("client_id")
    if (
        not df.index.is_unique
        or df[TARGET].isna().any()
        or not set(df[TARGET]) <= set(LABELS)
    ):
        raise ValueError("Invalid labels")
    if not (df.cutoff_date == "2026-01-01").all():
        raise ValueError("Unexpected cutoff")
    return df


def aligned_target(ids, split="train", *, allow_holdout=False, data_dir=None):
    table = labels(split, allow_holdout=allow_holdout, data_dir=data_dir)
    if set(ids) != set(table.index) or len(ids) != len(table):
        raise ValueError("Client alignment mismatch")
    return table.loc[ids, TARGET].map({v: k for k, v in enumerate(LABELS)}).to_numpy()


def validate_submission(df, sample=None):
    sample = (
        pd.read_csv(ROOT / "data/raw/sample_submission.csv")
        if sample is None
        else sample
    )
    if list(df.columns) != ["client_id", PREDICTION]:
        raise ValueError("Incorrect submission columns")
    if df.isna().any().any() or not df.client_id.is_unique:
        raise ValueError("Missing values or duplicate clients")
    if set(df.client_id) != set(sample.client_id) or len(df) != len(sample):
        raise ValueError("Submission ID set does not equal sample")
    if not set(df[PREDICTION]) <= set(LABELS):
        raise ValueError("Illegal prediction label")
    return True
