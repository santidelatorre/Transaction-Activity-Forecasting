"""Train-only text and description-as-merchant features for UBS experiments."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN

_DATE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
_IDENTIFIER = re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]{6,}\b")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_NOISE = re.compile(r"[^a-z0-9_]+")


def normalize_description(value: object) -> str:
    """Keep merchant words while masking unstable dates, references and numbers."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = _DATE.sub(" date_token ", text)
    text = _IDENTIFIER.sub(" id_token ", text)
    text = _NUMBER.sub(" num_token ", text)
    return " ".join(_NOISE.sub(" ", text).split())


def client_documents(transactions: pd.DataFrame, client_ids: pd.Index) -> pd.Series:
    """Aggregate normalized history per client, retaining repeated terms."""
    descriptions = transactions["description"].map(normalize_description)
    documents = descriptions.groupby(transactions["client_id"], sort=False).agg(" ".join)
    return documents.reindex(client_ids, fill_value="")


@dataclass
class MerchantFeatureBuilder:
    """Description proxy frequencies and diversity, fitted only on train clients.

    The challenge has no merchant ID. Class associations count clients, not
    transactions. Training rows remove their own contribution from those counts.
    """

    counts_: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False)
    client_counts_: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False)
    class_counts_: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    train_presence_: dict[str, set[str]] = field(default_factory=dict, init=False)
    train_labels_: pd.Series = field(default_factory=lambda: pd.Series(dtype=str), init=False)

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> MerchantFeatureBuilder:
        target = labels.set_index("client_id")[TARGET_COLUMN]
        if not target.index.is_unique or set(target.index) != set(transactions["client_id"]):
            raise ValueError("Train labels and transaction clients must match uniquely")
        normalized = transactions["description"].map(normalize_description)
        self.counts_ = normalized.value_counts()
        pairs = pd.DataFrame({"client_id": transactions["client_id"], "merchant": normalized})
        pairs = pairs.drop_duplicates()
        self.client_counts_ = pairs["merchant"].value_counts()
        pairs[TARGET_COLUMN] = pairs["client_id"].map(target)
        self.class_counts_ = pd.crosstab(pairs["merchant"], pairs[TARGET_COLUMN]).reindex(
            columns=LABELS, fill_value=0
        )
        self.train_presence_ = pairs.groupby("client_id")["merchant"].agg(set).to_dict()
        self.train_labels_ = target
        return self

    def transform(self, transactions: pd.DataFrame, *, training: bool = False) -> pd.DataFrame:
        if self.class_counts_.empty:
            raise RuntimeError("MerchantFeatureBuilder.fit must precede transform")
        frame = transactions[["client_id", "description"]].copy()
        frame["merchant"] = frame["description"].map(normalize_description)
        rows: dict[str, dict[str, float]] = {}
        for client_id, group in frame.groupby("client_id", sort=True):
            counts = group["merchant"].value_counts()
            n = float(counts.sum())
            shares = counts / n
            unique = counts.index.tolist()
            global_counts = self.counts_.reindex(unique, fill_value=0).astype(float)
            client_counts = self.client_counts_.reindex(unique, fill_value=0).astype(float)
            class_counts = self.class_counts_.reindex(unique, fill_value=0).astype(float)
            if training:
                if client_id not in self.train_labels_:
                    raise ValueError("Training transform contains an unknown client")
                own = self.train_presence_[client_id]
                client_counts.loc[list(own)] -= 1
                class_counts.loc[list(own), self.train_labels_[client_id]] -= 1
                global_counts -= counts.astype(float)
            support = class_counts.sum(axis=1)
            posterior = (class_counts + 1.0).div(support + len(LABELS), axis=0)
            raw = group["description"].fillna("").astype(str)
            values = {
                "merchant_unique": float(len(unique)),
                "merchant_unique_share": float(len(unique) / n),
                "merchant_top_share": float(shares.max()),
                "merchant_entropy": float(-(shares * np.log(shares)).sum()),
                "merchant_repeated_share": float(counts[counts.ge(2)].sum() / n),
                "merchant_train_frequency_mean": float(np.log1p(global_counts).mean()),
                "merchant_train_client_frequency_max": float(np.log1p(client_counts).max()),
                "merchant_unseen_share": float(counts[client_counts.eq(0)].sum() / n),
                "merchant_raw_length_mean": float(raw.str.len().mean()),
                "merchant_raw_tokens_mean": float(raw.str.split().str.len().mean()),
                "merchant_raw_digit_share": float(raw.str.contains(r"\d").mean()),
            }
            for label in LABELS:
                values[f"merchant_class_{label}_mean"] = float(posterior[label].mean())
                values[f"merchant_class_{label}_max"] = float(posterior[label].max())
            rows[str(client_id)] = values
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("client_id").fillna(0.0)
