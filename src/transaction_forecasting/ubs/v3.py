"""V3 stream-candidate model aiming above V2 toward the Santiago oracle ceiling.

Uses exact (client, description) streams with a train-only family map. Emits
per-family due-stream features that CatBoost can combine with V2 history
features. Also exposes a confident single-family override over V2.

Stream geometry never uses labels. The family map is fit only on train labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.stream_oracle import (
    POSITIVE_LABELS,
    TrainOnlyFamilyMapper,
    build_streams,
)
from transaction_forecasting.ubs.v2 import HistoryFeatureBuilder, IntegratedV2Model, make_model


def apply_family_map(streams: pd.DataFrame, mapper: TrainOnlyFamilyMapper) -> pd.DataFrame:
    """Apply a fitted description→family map without the disjoint-client guard.

    The guard remains for external audits. Training feature construction needs the
    same map on fit clients; validation still uses a mapper fit only on train.
    """
    result = streams.copy()
    result["mapped_family"] = result["description"].map(mapper.mapping_)
    return result


def _stream_quality(streams: pd.DataFrame) -> pd.DataFrame:
    """Attach unsupervised quality scores to mapped candidate streams."""
    frame = streams.copy()
    gap = frame["gap_median_days"].astype(float).clip(lower=1.0)
    days_to_next = (frame["projected_next_date"] - CUTOFF).dt.total_seconds().div(86400)
    frame["days_to_next"] = days_to_next
    gap_cv = frame["gap_std_days"].astype(float) / gap
    amount_cv = frame["amount_cv"].astype(float)
    frame["regularity"] = 1.0 / (1.0 + gap_cv.fillna(5.0))
    overdue = (-days_to_next).clip(lower=0.0)
    frame["due_weight"] = (
        np.log1p(frame["appearances"].astype(float))
        * frame["regularity"]
        * np.exp(-overdue / 30.0)
        * np.exp(-days_to_next.clip(lower=0.0) / 45.0)
        / (1.0 + amount_cv.fillna(1.0))
    )
    return frame


def client_stream_features(
    transactions: pd.DataFrame,
    mapper: TrainOnlyFamilyMapper,
    client_ids: pd.Index | None = None,
    *,
    streams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per client with due-stream evidence per positive family."""
    clients = (
        pd.Index(sorted(transactions["client_id"].astype(str).unique()), name="client_id")
        if client_ids is None
        else pd.Index(client_ids.astype(str), name="client_id")
    )
    if streams is None:
        streams = build_streams(transactions)
    mapped = apply_family_map(streams, mapper)
    due = mapped[mapped["is_candidate"] & mapped["mapped_family"].notna()].copy()
    features = pd.DataFrame(index=clients)
    features["stream_candidate_count"] = (
        mapped[mapped["is_candidate"]].groupby("client_id").size().reindex(clients, fill_value=0)
    )
    features["stream_mapped_candidate_count"] = (
        due.groupby("client_id").size().reindex(clients, fill_value=0)
    )
    empty_cols = []
    for family in POSITIVE_LABELS:
        empty_cols.extend(
            [
                f"due_{family}_count",
                f"due_{family}_max_weight",
                f"due_{family}_best_lift",
                f"due_{family}_min_days",
                f"due_{family}_max_regularity",
            ]
        )
    for column in empty_cols:
        features[column] = 0.0
    for family in POSITIVE_LABELS:
        features[f"due_{family}_min_days"] = 999.0
    features["stream_unique_families"] = 0.0
    features["stream_top_weight"] = 0.0
    features["stream_top_family_index"] = -1.0
    features["stream_margin"] = 0.0
    features["stream_single_family"] = 0.0
    if due.empty:
        return features.astype(float)

    due = _stream_quality(due)
    lift_map = mapper.mapping_table_.set_index("description")["winning_lift"]
    due["winning_lift"] = due["description"].map(lift_map).fillna(0.0)

    for family in POSITIVE_LABELS:
        subset = due[due["mapped_family"].eq(family)]
        if subset.empty:
            features[f"due_{family}_min_days"] = 999.0
            continue
        grouped = subset.groupby("client_id")
        features[f"due_{family}_count"] = grouped.size().reindex(clients, fill_value=0)
        features[f"due_{family}_max_weight"] = (
            grouped["due_weight"].max().reindex(clients, fill_value=0.0)
        )
        features[f"due_{family}_best_lift"] = (
            grouped["winning_lift"].max().reindex(clients, fill_value=0.0)
        )
        features[f"due_{family}_min_days"] = (
            grouped["days_to_next"].min().reindex(clients, fill_value=999.0)
        )
        features[f"due_{family}_max_regularity"] = (
            grouped["regularity"].max().reindex(clients, fill_value=0.0)
        )

    family_weights = features[[f"due_{family}_max_weight" for family in POSITIVE_LABELS]]
    features["stream_unique_families"] = (family_weights.gt(0)).sum(axis=1).astype(float)
    features["stream_top_weight"] = family_weights.max(axis=1)
    top_idx = family_weights.to_numpy().argmax(axis=1)
    top_idx = np.where(features["stream_top_weight"].to_numpy() > 0, top_idx, -1)
    features["stream_top_family_index"] = top_idx.astype(float)
    sorted_w = np.sort(family_weights.to_numpy(), axis=1)
    features["stream_margin"] = sorted_w[:, -1] - sorted_w[:, -2]
    features["stream_single_family"] = features["stream_unique_families"].eq(1).astype(float)
    return features.astype(float).fillna(0.0)


@dataclass
class StreamV3Model:
    """History CatBoost + due-stream features, blended with frozen V2."""

    v2_blend: float = 0.85
    override_weight: float = 2.0
    override_margin: float = 0.05
    override_min_weight: float = 1.0
    seed: int = 42

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> StreamV3Model:
        target = labels.set_index("client_id")[TARGET_COLUMN]
        self.fit_clients_ = set(target.index.astype(str))
        self.history_ = HistoryFeatureBuilder().fit(transactions)
        train_streams = build_streams(transactions)
        self.mapper_ = TrainOnlyFamilyMapper().fit(train_streams, labels)
        history = self.history_.transform(transactions)
        stream = client_stream_features(
            transactions, self.mapper_, history.index, streams=train_streams
        )
        matrix = history.join(stream)
        self.feature_names_ = matrix.columns.tolist()
        self.model_ = make_model()
        self.model_.iterations = 450
        self.model_.depth = 5
        self.model_.seed = self.seed
        self.model_.fit(matrix, target.reindex(matrix.index))
        self.v2_ = IntegratedV2Model().fit(transactions, labels)
        return self

    def predict_components(self, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit StreamV3Model before predict")
        clients = set(transactions["client_id"].astype(str))
        if clients.intersection(self.fit_clients_):
            raise ValueError("Prediction clients must be disjoint from fit clients")
        history = self.history_.transform(transactions)
        streams = build_streams(transactions)
        stream = client_stream_features(transactions, self.mapper_, history.index, streams=streams)
        matrix = history.join(stream).reindex(columns=self.feature_names_).fillna(0.0)
        stream_model = pd.DataFrame(
            self.model_.predict_proba(matrix), index=matrix.index, columns=LABELS
        )
        v2 = self.v2_.predict_components(transactions)["blend"].reindex(matrix.index)
        blended = self.v2_blend * v2.to_numpy() + (1.0 - self.v2_blend) * stream_model.to_numpy()
        blended = pd.DataFrame(blended, index=matrix.index, columns=LABELS)

        single = stream["stream_single_family"].gt(0.5) & stream["stream_margin"].ge(
            self.override_margin
        )
        top_weight = stream["stream_top_weight"]
        top_idx = stream["stream_top_family_index"].astype(int)
        eligible = single & top_weight.ge(self.override_min_weight)
        for client in matrix.index[eligible]:
            family = POSITIVE_LABELS[int(top_idx.loc[client])]
            boost = float(top_weight.loc[client]) * self.override_weight
            blended.loc[client, family] = float(blended.loc[client, family]) + boost
        if eligible.any():
            values = blended.loc[eligible].to_numpy(dtype=float)
            values = np.clip(values, 1e-12, None)
            values = values / values.sum(axis=1, keepdims=True)
            blended.loc[eligible] = values
        return {
            "stream_model": stream_model,
            "v2": v2,
            "blend": blended,
            "stream_features": stream,
        }

    def predict(self, transactions: pd.DataFrame) -> pd.Series:
        return self.predict_components(transactions)["blend"].idxmax(axis=1)
