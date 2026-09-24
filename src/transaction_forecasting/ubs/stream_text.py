"""Leakage-safe stream-level text family scorer (Javier-style, positive families).

Fits char TF-IDF + logistic regression on one weakly labelled stream per positive
train client.  ``none`` is never a stream label.  Inference returns per-client
max probabilities over recurring streams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.stream_oracle import POSITIVE_LABELS
from transaction_forecasting.ubs.text_v2 import normalize_description


def _mode(series: pd.Series) -> str:
    modes = series.astype(str).mode()
    return modes.iloc[0] if not modes.empty else ""


def build_text_streams(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per (client, normalized description) with recurrence evidence."""
    frame = transactions.copy()
    frame["description"] = frame["description"].map(normalize_description)
    frame = frame.sort_values(["client_id", "description", "timestamp"], kind="stable")
    keys = ["client_id", "description"]
    grouped = frame.groupby(keys, sort=False)
    streams = grouped.agg(
        appearances=("timestamp", "size"),
        unique_event_count=("timestamp", "nunique"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        amount_abs_mean=("amount", lambda values: float(values.abs().mean())),
        amount_abs_std=("amount", lambda values: float(values.abs().std(ddof=0))),
        mcc=("mcc", _mode),
        type=("type", _mode),
        direction=("direction", _mode),
    ).reset_index()
    unique_times = frame.drop_duplicates(keys + ["timestamp"])
    unique_times["gap_days"] = (
        unique_times.groupby(keys, sort=False)["timestamp"].diff().dt.total_seconds().div(86_400)
    )
    gaps = unique_times.groupby(keys, sort=False)["gap_days"].agg(
        interval_median="median",
        interval_mean="mean",
        interval_std=lambda values: values.std(ddof=0),
    )
    streams = streams.join(gaps, on=keys)
    streams["days_since_last"] = (CUTOFF - streams["last_timestamp"]).dt.total_seconds().div(86_400)
    streams["interval_cv"] = streams["interval_std"].div(streams["interval_mean"].clip(lower=1.0))
    closeness = [
        np.exp(-(streams["interval_median"] - cycle).abs() / (0.2 * cycle))
        for cycle in (7.0, 14.0, 30.0, 31.0, 90.0)
    ]
    streams["periodicity_closeness"] = pd.concat(closeness, axis=1).max(axis=1).fillna(0.0)
    streams["amount_cv"] = streams["amount_abs_std"].div(
        streams["amount_abs_mean"].clip(lower=1e-6)
    )
    streams["amount_stability"] = 1.0 / (1.0 + streams["amount_cv"].fillna(10.0))
    return streams.replace([np.inf, -np.inf], np.nan)


def recurring_or_fallback(streams: pd.DataFrame) -> pd.DataFrame:
    repeated = streams.loc[streams["unique_event_count"].ge(2)].copy()
    missing = pd.Index(streams["client_id"].unique()).difference(repeated["client_id"].unique())
    if len(missing):
        repeated = pd.concat(
            [repeated, streams.loc[streams["client_id"].isin(missing)]], ignore_index=True
        )
    return repeated


@dataclass
class AssociationTable:
    counts: pd.DataFrame
    class_sizes: pd.Series
    n_clients: int
    scores: pd.DataFrame


def learn_associations(
    streams: pd.DataFrame, targets: pd.Series, *, min_support: int = 2
) -> AssociationTable:
    present = streams[["client_id", "description"]].drop_duplicates()
    present[TARGET_COLUMN] = present["client_id"].map(targets)
    if present[TARGET_COLUMN].isna().any():
        raise ValueError("Association fit contains clients without labels")
    counts = pd.crosstab(present["description"], present[TARGET_COLUMN]).reindex(
        columns=LABELS, fill_value=0
    )
    class_sizes = targets.value_counts().reindex(LABELS, fill_value=0).astype(float)
    total = counts.sum(axis=1).astype(float)
    score_columns = {}
    for label in POSITIVE_LABELS:
        inside = (counts[label] + 0.5) / (class_sizes[label] + 1.0)
        outside = (total - counts[label] + 0.5) / (len(targets) - class_sizes[label] + 1.0)
        score_columns[label] = np.log(inside / outside)
    scores = pd.DataFrame(score_columns).where(total.ge(min_support), 0.0)
    return AssociationTable(counts, class_sizes, len(targets), scores)


def select_weak_candidates(
    streams: pd.DataFrame, targets: pd.Series, associations: AssociationTable
) -> pd.DataFrame:
    positive_targets = targets.loc[targets.isin(POSITIVE_LABELS)]
    candidates = recurring_or_fallback(
        streams.loc[streams["client_id"].isin(positive_targets.index)]
    ).copy()
    candidate_labels = candidates["client_id"].map(positive_targets)
    supports = associations.counts.sum(axis=1)
    lift_values = []
    support_values = []
    for description, label in zip(candidates["description"], candidate_labels, strict=True):
        label_count = float(associations.counts.at[description, label]) - 1.0
        support = float(supports.at[description]) - 1.0
        class_size = float(associations.class_sizes[label]) - 1.0
        inside = (label_count + 0.5) / (class_size + 1.0)
        outside_count = support - label_count
        outside_clients = (associations.n_clients - 1.0) - class_size
        outside = (outside_count + 0.5) / (outside_clients + 1.0)
        lift_values.append(math.log(max(inside, 1e-12) / max(outside, 1e-12)))
        support_values.append(support)
    candidates["weak_label"] = candidate_labels.to_numpy()
    candidates["loo_log_lift"] = lift_values
    candidates["loo_support"] = support_values
    candidates["candidate_score"] = (
        candidates["loo_log_lift"]
        + 0.15 * np.log1p(candidates["appearances"])
        + 0.20 * candidates["periodicity_closeness"]
        + 0.10 * candidates["amount_stability"]
        - 0.002 * candidates["days_since_last"].clip(upper=365)
    )
    eligible = candidates["loo_support"].ge(2) & candidates["loo_log_lift"].ge(math.log(1.25))
    candidates["eligible"] = eligible
    chosen = []
    for _, group in candidates.groupby("client_id", sort=False):
        pool = group.loc[group["eligible"]]
        if pool.empty:
            pool = group
        chosen.append(pool["candidate_score"].idxmax())
    return candidates.loc[chosen].reset_index(drop=True)


def _make_char_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=12_000,
                                sublinear_tf=True,
                            ),
                            "description",
                        )
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def aggregate_stream_probabilities(
    streams: pd.DataFrame, probabilities: np.ndarray, classes: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_frame = pd.DataFrame(probabilities, columns=classes, index=streams.index)
    probability_frame["client_id"] = streams["client_id"].to_numpy()
    client_scores = probability_frame.groupby("client_id", sort=False).max()
    client_scores = client_scores.reindex(columns=list(POSITIVE_LABELS), fill_value=0.0)
    predictions = client_scores.idxmax(axis=1)
    details = []
    for client_id, predicted in predictions.items():
        client_rows = streams.loc[streams["client_id"].eq(client_id)]
        winning_index = probability_frame.loc[client_rows.index, predicted].idxmax()
        row = streams.loc[winning_index]
        details.append(
            {
                "client_id": client_id,
                "prediction": predicted,
                "confidence": float(probability_frame.at[winning_index, predicted]),
                "winning_description": row["description"],
                "winning_appearances": int(row["appearances"]),
            }
        )
    return client_scores, pd.DataFrame(details).set_index("client_id")


@dataclass
class StreamTextFamilyModel:
    """Positive-family text scorer fitted on train-only weak stream candidates."""

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> StreamTextFamilyModel:
        target = labels.set_index("client_id")[TARGET_COLUMN]
        streams = build_text_streams(transactions)
        self.associations_ = learn_associations(streams, target)
        weak = select_weak_candidates(streams, target, self.associations_)
        self.model_ = _make_char_model()
        self.model_.fit(weak, weak["weak_label"])
        self.known_ = set(self.associations_.scores.index)
        self.fit_clients_ = set(target.index.astype(str))
        return self

    def predict_scores(self, transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit StreamTextFamilyModel before predict")
        clients = set(transactions["client_id"].astype(str))
        if clients.intersection(self.fit_clients_):
            raise ValueError("Prediction clients must be disjoint from fit clients")
        streams = recurring_or_fallback(build_text_streams(transactions)).reset_index(drop=True)
        if streams.empty:
            empty = pd.DataFrame(
                0.0, index=pd.Index([], name="client_id"), columns=list(POSITIVE_LABELS)
            )
            return empty, pd.DataFrame()
        probabilities = self.model_.predict_proba(streams)
        classes = self.model_.named_steps["classifier"].classes_
        scores, details = aggregate_stream_probabilities(streams, probabilities, classes)
        details["winning_known"] = details["winning_description"].isin(self.known_)
        return scores, details
