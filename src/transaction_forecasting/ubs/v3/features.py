"""Cross-fitted family identity and payment recurrence, with strict client isolation.

The lift idea is attributed to Santiago c040ec7 and Javier c7075bb. Unlike a
stream label, a client-target association is only evidence about a description.
No training row is transformed with a map containing that client's label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN

FAMILIES = tuple(label for label in LABELS if label != "none")


def validate_history(transactions: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF) -> None:
    if transactions.empty:
        raise ValueError("History must be nonempty")
    if transactions[["client_id", "description", "timestamp"]].isna().any().any():
        raise ValueError("Missing stream key or timestamp")
    if transactions.timestamp.ge(cutoff).any():
        raise ValueError("History must be strictly before cutoff")


class FamilyMap:
    """Positive-family association fitted on distinct client presence only."""

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame):
        validate_history(transactions)
        target = labels.set_index("client_id")[TARGET_COLUMN]
        if not target.index.is_unique or set(target.index) != set(transactions.client_id):
            raise ValueError("Unique labels must exactly match history clients")
        if not target.isin(LABELS).all():
            raise ValueError("Invalid family")
        self.fit_clients_ = set(target.index)
        presence = transactions[["client_id", "description"]].drop_duplicates()
        presence = presence.join(target, on="client_id")
        counts = pd.crosstab(presence.description, presence[TARGET_COLUMN]).reindex(
            columns=LABELS, fill_value=0
        )
        support = counts.sum(axis=1)
        class_sizes = target.value_counts().reindex(LABELS, fill_value=0)
        lift = pd.DataFrame(index=counts.index)
        for family in FAMILIES:
            inside = (counts[family] + 1) / (class_sizes[family] + 2)
            outside = (support - counts[family] + 1) / (len(target) - class_sizes[family] + 2)
            lift[family] = np.log(inside / outside) if class_sizes[family] else 0.0
        best = lift.idxmax(axis=1)
        eligible = support.ge(5) & lift.max(axis=1).ge(np.log(1.5))
        self.mapping_ = best.where(eligible, "unknown")
        self.audit_ = pd.DataFrame(
            {"family": self.mapping_, "clients": support, "log_lift": lift.max(axis=1)}
        )
        return self

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        validate_history(transactions)
        if self.fit_clients_.intersection(transactions.client_id):
            raise ValueError("Family features require clients excluded from mapping fit")
        result = transactions.copy()
        result["family"] = result.description.map(self.mapping_).fillna("unknown")
        return result


def payment_streams(transactions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Summarize outbound card payments; currency remains part of every stream key."""
    validate_history(transactions)
    frame = transactions.loc[
        transactions.direction.eq("out") & transactions.type.eq("card_payment")
    ].sort_values([*keys, "timestamp"], kind="stable")
    frame = frame.drop_duplicates([*keys, "timestamp"])
    frame = frame.assign(gap=frame.groupby(keys).timestamp.diff().dt.total_seconds() / 86400)
    grouped = frame.groupby(keys, sort=True)
    streams = grouped.agg(
        count=("timestamp", "size"),
        last=("timestamp", "max"),
        period=("gap", "median"),
        gap_std=("gap", "std"),
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
    ).reset_index()
    streams["recency"] = (CUTOFF - streams["last"]).dt.total_seconds() / 86400
    streams["amount_cv"] = streams.amount_std.fillna(0) / streams.amount_mean.abs().clip(1e-6)
    streams["regularity"] = 1 / (1 + streams.gap_std.fillna(0) / streams.period.clip(1))
    streams["due"] = (
        np.ceil(streams.recency / streams.period.clip(1)) * streams.period.clip(1) - streams.recency
    )
    streams["stable"] = (streams["count"].ge(2) & streams.amount_cv.le(0.05)).astype(float)
    streams["recent"] = streams.recency.le(30).astype(float)
    streams["score"] = (
        np.log1p(streams["count"])
        * streams.regularity.fillna(0)
        * np.exp(-streams.recency / streams.period.clip(7))
        / (1 + streams.amount_cv)
    )
    return streams


def _aggregate(streams: pd.DataFrame, clients: pd.Index, prefix: str) -> pd.DataFrame:
    grouped = streams.loc[streams["count"].ge(2)].groupby("client_id")
    result = grouped.agg(
        streams=("count", "size"),
        events=("count", "sum"),
        max_events=("count", "max"),
        recency_min=("recency", "min"),
        recency_mean=("recency", "mean"),
        period_min=("period", "min"),
        period_mean=("period", "mean"),
        amount_cv_min=("amount_cv", "min"),
        regularity_max=("regularity", "max"),
        due_min=("due", "min"),
        stable=("stable", "sum"),
        recent=("recent", "sum"),
        score=("score", "max"),
    ).reindex(clients)
    return result.fillna(0).add_prefix(prefix)


def recurrence_features(transactions: pd.DataFrame) -> pd.DataFrame:
    clients = pd.Index(sorted(transactions.client_id.unique()), name="client_id")
    streams = payment_streams(transactions, ["client_id", "description", "currency"])
    return _aggregate(streams, clients, "payment_")


def family_features(mapped: pd.DataFrame) -> pd.DataFrame:
    """Counts plus per-family temporal evidence; no hard winner/none rule."""
    clients = pd.Index(sorted(mapped.client_id.unique()), name="client_id")
    outgoing = mapped.loc[mapped.direction.eq("out") & mapped.type.eq("card_payment")]
    total = outgoing.groupby("client_id").size().reindex(clients, fill_value=0).clip(1)
    exact = payment_streams(mapped, ["client_id", "description", "currency", "family"])
    union = (
        payment_streams(
            mapped.loc[mapped.family.isin(FAMILIES)], ["client_id", "family", "currency"]
        )
        if mapped.family.isin(FAMILIES).any()
        else exact.iloc[:0]
    )
    blocks = []
    for family in FAMILIES:
        rows = outgoing.loc[outgoing.family.eq(family)].groupby("client_id")
        identity = pd.DataFrame(index=clients)
        identity[f"identity_{family}_count"] = rows.size().reindex(clients, fill_value=0)
        identity[f"identity_{family}_share"] = identity.iloc[:, 0] / total
        identity[f"identity_{family}_aliases"] = rows.description.nunique().reindex(
            clients, fill_value=0
        )
        blocks.extend(
            [
                identity,
                _aggregate(exact.loc[exact.family.eq(family)], clients, f"family_{family}_"),
                _aggregate(union.loc[union.family.eq(family)], clients, f"union_{family}_"),
            ]
        )
    return pd.concat(blocks, axis=1)


def cross_fitted_family_features(transactions, labels, folds=5):
    """Every row's supervised mapping is trained without that client's label."""
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    splitter = StratifiedKFold(folds, shuffle=True, random_state=42)
    blocks = []
    for fit_positions, hold_positions in splitter.split(target.index, target):
        fit_ids, hold_ids = target.index[fit_positions], target.index[hold_positions]
        fit = transactions.loc[transactions.client_id.isin(fit_ids)]
        hold = transactions.loc[transactions.client_id.isin(hold_ids)]
        mapper = FamilyMap().fit(fit, labels.loc[labels.client_id.isin(fit_ids)])
        blocks.append(family_features(mapper.transform(hold)))
    result = pd.concat(blocks).reindex(target.index)
    if result.isna().any().any() or not result.index.is_unique:
        raise RuntimeError("Incomplete or duplicated cross-fitted features")
    return result
