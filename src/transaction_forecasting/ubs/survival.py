"""Causal stream recurrence and discrete event-time probabilities (no NONE gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from transaction_forecasting.ubs.data import CUTOFF, LABELS
from transaction_forecasting.ubs.v3.family_text import learn_associations
from transaction_forecasting.ubs.v3.features import FAMILIES, payment_streams

KEYS = ["client_id", "description", "currency"]
SEASONAL = [
    "mirror_q1",
    "anniversary_distance",
    "same_month_last_year",
    "annual_phase",
    "historical_q1_count",
]
CATEGORICAL = ["mcc", "type", "currency"]
EXCLUDE = KEYS + ["cutoff", "last", "time_to_next_event", "event", "bucket", "weight"]


def stream_features(transactions, cutoff=CUTOFF, seasonal=False):
    """Require strictly historical input; reuse baseline stream keys and eligibility."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or cutoff > CUTOFF:
        raise ValueError("Timezone-aware cutoff at/before official cutoff required")
    if transactions.timestamp.ge(cutoff).any():
        raise ValueError("Future rows in features")
    streams = payment_streams(transactions, KEYS)
    frame = transactions.loc[
        transactions.direction.eq("out") & transactions.type.eq("card_payment")
    ]
    rows = []
    for key, group in frame.groupby(KEYS, sort=True):
        group = group.sort_values("timestamp", kind="stable")
        times = pd.DatetimeIndex(group.timestamp.drop_duplicates())
        gaps = np.diff(times.as_unit("ns").asi8) / 86400e9
        typical = float(np.median(gaps)) if len(gaps) else np.nan
        age = (cutoff - times[-1]).total_seconds() / 86400
        amount = group.amount.abs().to_numpy()
        row = dict(zip(KEYS, key, strict=True))
        row.update(
            cutoff=cutoff,
            recency=age,
            gap_mad=np.median(abs(gaps - typical)) if len(gaps) else np.nan,
            last_gap=gaps[-1] if len(gaps) else np.nan,
            last_gap_ratio=gaps[-1] / typical if len(gaps) else np.nan,
            missed_cycles=age / typical if len(gaps) else np.nan,
            observed_cycles=len(gaps),
            amount_median=np.median(amount),
            amount_drift=(amount[-1] - amount[0]) / max(np.median(amount), 1e-6),
            recent_amount_ratio=np.median(amount[-3:]) / max(np.median(amount), 1e-6),
            mcc=str(group.mcc.mode().iloc[0]),
            type=str(group.type.iloc[0]),
        )
        for name, phase in (("weekday", times.dayofweek / 7), ("monthday", times.day / 31)):
            row[name + "_stability"] = abs(np.exp(2j * np.pi * np.asarray(phase)).mean())
        for name, cycle in (("weekly", 7), ("monthly", 30), ("annual", 365)):
            row[name + "_score"] = (
                np.exp(-np.median(abs(gaps - cycle)) / (cycle * 0.2)) if len(gaps) else 0.0
            )
        if seasonal:
            previous = times[times.year == cutoff.year - 1]
            anniversary = times + pd.DateOffset(years=1)
            row.update(
                mirror_q1=int(((previous.month >= 1) & (previous.month <= 3)).any()),
                anniversary_distance=float(np.min(abs((anniversary - cutoff).days))),
                same_month_last_year=int((previous.month == cutoff.month).any()),
                annual_phase=np.cos(2 * np.pi * (cutoff.dayofyear - times[-1].dayofyear) / 365.25),
                historical_q1_count=int((times.month <= 3).sum()),
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=KEYS + ["cutoff"])
    # The baseline builder uses official-cutoff recency/due/score: discard those.
    streams = streams.drop(columns=["recency", "due", "score", "recent"])
    return streams.merge(pd.DataFrame(rows), on=KEYS, validate="one_to_one")


def pseudo_cutoffs(transactions, cutoffs, observation_end=CUTOFF, seasonal=False):
    """[cutoff, cutoff+90d) targets; complete follow-up only, each client total weight 1."""
    end = pd.Timestamp(observation_end)
    cutoffs = sorted(set(pd.Timestamp(c) for c in cutoffs))
    if not cutoffs or any(c + pd.Timedelta(days=90) > end for c in cutoffs):
        raise ValueError("Every cutoff needs a fully observed 90-day horizon")
    future = transactions.loc[
        transactions.direction.eq("out")
        & transactions.type.eq("card_payment")
        & transactions.timestamp.lt(end)
    ]
    blocks = []
    for cutoff in cutoffs:
        history = transactions.loc[transactions.timestamp.lt(cutoff)]
        if history.empty:
            continue
        features = stream_features(history, cutoff, seasonal)
        if features.empty:
            continue
        next_times = future.loc[future.timestamp.ge(cutoff)].groupby(KEYS).timestamp.min()
        features = features.join(next_times.rename("next"), on=KEYS)
        features["time_to_next_event"] = (features.pop("next") - cutoff).dt.total_seconds() / 86400
        features["event"] = features.time_to_next_event.lt(90).astype(int)
        # Half-open intervals avoid ambiguity at exact boundaries: [0,30), [30,60), [60,90).
        features["bucket"] = np.where(
            features.event, np.minimum(features.time_to_next_event.fillna(90).floordiv(30), 2), 3
        ).astype(int)
        blocks.append(features)
    if not blocks:
        raise ValueError("No observable historical streams")
    result = pd.concat(blocks, ignore_index=True)
    size = result.groupby(["client_id", "cutoff"]).client_id.transform("size")
    snapshots = result.groupby("client_id").cutoff.transform("nunique")
    result["weight"] = 1 / (size * snapshots)
    return result


class RecurrenceModel:
    """Regularized logistic recurrence or four-bin event-time model with fixed class order."""

    def __init__(self, kind="hazard", seasonal=False):
        if kind not in ("recurrence", "hazard"):
            raise ValueError("Unknown model")
        self.kind, self.seasonal = kind, seasonal

    def fit(self, snapshots):
        self.clients_ = set(snapshots.client_id)
        self.columns_ = [
            c for c in snapshots if c not in EXCLUDE and (self.seasonal or c not in SEASONAL)
        ] + ["currency"]
        numeric = [c for c in self.columns_ if c not in CATEGORICAL]
        transform = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(
                        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
                    ),
                    numeric,
                ),
                ("context", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ]
        )
        y = snapshots["event" if self.kind == "recurrence" else "bucket"]
        estimator = (
            LogisticRegression(C=1, max_iter=500, random_state=42)
            if y.nunique() > 1
            else DummyClassifier(strategy="prior")
        )
        self.transform_ = transform.fit(snapshots[self.columns_])
        weights = snapshots.weight.to_numpy()
        self.model_ = estimator.fit(
            transform.transform(snapshots[self.columns_]), y, sample_weight=weights / weights.mean()
        )
        return self

    def predict(self, features):
        if self.clients_.intersection(features.client_id):
            raise ValueError("Prediction clients overlap fit clients")
        raw = self.model_.predict_proba(self.transform_.transform(features[self.columns_]))
        nclasses = 2 if self.kind == "recurrence" else 4
        ordered = np.zeros((len(features), nclasses))
        ordered[:, self.model_.classes_.astype(int)] = raw
        if self.kind == "hazard":
            return ordered
        # Without timing information, distribute recurrence mass uniformly over the horizon.
        return np.column_stack([ordered[:, 1] / 3] * 3 + [ordered[:, 0]])


class FamilyEvidence:
    """Existing TRAIN-only soft description lift; evidence is not event ground truth."""

    def fit(self, history, target):
        self.clients_ = set(target.index)
        self.table_ = learn_associations(history, target).scores.clip(lower=0)
        return self

    def transform(self, streams):
        if self.clients_.intersection(streams.client_id):
            raise ValueError("Family evidence requires disjoint clients")
        lift = self.table_.reindex(streams.description).fillna(0).to_numpy()
        strength = lift.sum(axis=1)
        evidence = lift / np.maximum(strength[:, None], 1e-12)
        confidence = 1 - np.exp(-strength)
        return evidence, confidence


def median_gap_probabilities(streams):
    """Uncalibrated median-gap control, with overdue survival decay."""
    period = streams.period.fillna(365).clip(lower=1).to_numpy()
    age = streams.recency.to_numpy()
    active = (streams["count"].to_numpy() >= 2) * np.exp(-np.maximum(age - period, 0) / period)
    wait = np.maximum(period - age, 0)
    p = np.zeros((len(streams), 4))
    bucket = np.minimum((wait / 30).astype(int), 3)
    p[np.arange(len(p)), bucket] = active
    p[:, 3] += 1 - active
    return p


def aggregate(streams, probabilities, evidence, confidence, clients, mode="competing"):
    """Independent stream survival; symmetric within-bin competition, never earliest-wins.

    Direct recurrence already includes inactivity: P_active=1 in the primary model.
    The support ablation tests multiplying by an explicitly uncalibrated active proxy.
    """
    if mode not in ("competing", "support", "pooled"):
        raise ValueError("Unknown aggregation")
    result = pd.DataFrame(0.0, index=pd.Index(clients, name="client_id"), columns=LABELS)
    result["none"] = 1.0
    active = streams["count"].to_numpy() / (streams["count"].to_numpy() + 2)
    mass = probabilities[:, :3] * confidence[:, None]
    if mode == "support":
        mass *= active[:, None]
    for client, positions in streams.groupby("client_id", sort=True).indices.items():
        p, family = mass[positions], evidence[positions]
        if mode == "pooled":
            p = p.sum(axis=1, keepdims=True)
        survival, scores = 1.0, np.zeros(len(FAMILIES))
        for bucket in range(p.shape[1]):
            conditional = p[:, bucket] / np.maximum(1 - p[:, :bucket].sum(axis=1), 1e-12)
            any_event = 1 - np.prod(1 - conditional)
            weights = conditional / max(conditional.sum(), 1e-12)
            scores += survival * any_event * (weights @ family)
            survival *= 1 - any_event
        result.loc[client, list(FAMILIES)] = scores
        result.loc[client, "none"] = survival
    return result
