"""Download, extract, and prepare the UBS hackathon training data."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


def download_and_extract(zip_url: str, data_dir: Path) -> None:
    """Download a dataset ZIP and safely extract it under ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "dataset.zip"

    response = requests.get(zip_url, stream=True, timeout=(15, 120))
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type or "text/html" in response.url:
        raise ValueError(
            "The URL returned an HTML page, not a ZIP archive. Supply the direct dataset ZIP URL."
        )

    with archive_path.open("wb") as archive_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                archive_file.write(chunk)

    root = data_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (data_dir / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
        archive.extractall(data_dir)


def find_data_file(data_dir: Path, filename: str) -> Path:
    """Find a required file at the extraction root or in one nested directory."""
    matches = list(data_dir.rglob(filename))
    matches = [path for path in matches if path.name == filename]
    if not matches:
        raise FileNotFoundError(f"{filename} was not found under {data_dir}")
    if len(matches) > 1:
        raise ValueError(f"Found multiple copies of {filename} under {data_dir}")
    return matches[0]


def load_training_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, clean, sort, and join training transactions with client labels."""
    transaction_path = find_data_file(data_dir, "train_transactions.jsonl")
    labels_path = find_data_file(data_dir, "train_labels.csv")

    transactions = pd.read_json(transaction_path, lines=True)
    labels = pd.read_csv(labels_path)

    required_transaction_columns = {"client_id", "timestamp", "description", "type", "direction"}
    missing_transaction_columns = required_transaction_columns - set(transactions.columns)
    if missing_transaction_columns:
        raise ValueError(f"Transactions are missing columns: {sorted(missing_transaction_columns)}")
    required_label_columns = {"client_id", "cutoff_date", "target_next_recurring_merchant"}
    missing_label_columns = required_label_columns - set(labels.columns)
    if missing_label_columns:
        raise ValueError(f"Labels are missing columns: {sorted(missing_label_columns)}")

    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], utc=True, errors="raise")
    labels["cutoff_date"] = pd.to_datetime(labels["cutoff_date"], utc=True, errors="raise")
    for column in ("description", "type", "direction"):
        transactions[column] = transactions[column].astype("string").str.strip().str.lower()

    transactions = transactions.loc[transactions["direction"].eq("out")].copy()
    transactions = transactions.sort_values(
        ["client_id", "timestamp"], ascending=True, kind="stable"
    ).reset_index(drop=True)

    if labels["client_id"].duplicated().any():
        raise ValueError("Expected exactly one target label row per client_id")
    merged = transactions.merge(
        labels[["client_id", "cutoff_date", "target_next_recurring_merchant"]],
        on="client_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = merged.loc[merged["_merge"].eq("left_only"), "client_id"].nunique()
    if unmatched:
        raise ValueError(f"No training label found for {unmatched} transaction client(s)")
    merged = merged.drop(columns="_merge")
    return transactions, labels, merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        help="Optional direct URL to a dataset ZIP archive to download and extract first.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.url:
        if urlparse(args.url).scheme not in {"http", "https"}:
            parser.error("--url must be an http:// or https:// URL")
        download_and_extract(args.url, args.data_dir)
    transactions, labels, merged = load_training_data(args.data_dir)

    print("Cleaned transactions:")
    transactions.info()
    print("\nMerged transaction and target data (first five rows):")
    print(merged.head())
    print("\nTarget class counts:")
    print(labels["target_next_recurring_merchant"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
