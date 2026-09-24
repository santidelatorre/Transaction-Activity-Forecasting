"""StreamV3+ : history+due CatBoost enriched with stream-text family scores."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.stream_oracle import (
    POSITIVE_LABELS,
    TrainOnlyFamilyMapper,
    build_streams,
)
from transaction_forecasting.ubs.stream_text import (
    StreamTextFamilyModel,
    aggregate_stream_probabilities,
    build_text_streams,
    recurring_or_fallback,
)
from transaction_forecasting.ubs.v2 import HistoryFeatureBuilder, IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3 import client_stream_features


@dataclass
class StreamV3PlusModel:
    """V2 blend + CatBoost(history + due-stream + text-family scores)."""

    v2_blend: float = 0.65
    seed: int = 42
    text_override_threshold: float | None = 0.85
    text_override_families: tuple[str, ...] = ("music", "streaming")

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> StreamV3PlusModel:
        target = labels.set_index("client_id")[TARGET_COLUMN]
        self.fit_clients_ = set(target.index.astype(str))
        self.history_ = HistoryFeatureBuilder().fit(transactions)
        train_streams = build_streams(transactions)
        self.mapper_ = TrainOnlyFamilyMapper().fit(train_streams, labels)
        self.text_ = StreamTextFamilyModel().fit(transactions, labels)
        self.v2_ = IntegratedV2Model().fit(transactions, labels)

        history = self.history_.transform(transactions)
        stream = client_stream_features(
            transactions, self.mapper_, history.index, streams=train_streams
        )
        text_feat = self._in_sample_text_features(transactions, history.index)
        matrix = history.join(stream).join(text_feat)
        self.feature_names_ = matrix.columns.tolist()
        self.model_ = make_model()
        self.model_.iterations = 600
        self.model_.depth = 6
        self.model_.learning_rate = 0.04
        self.model_.seed = self.seed
        self.model_.fit(matrix, target.reindex(matrix.index))
        return self

    def _in_sample_text_features(
        self, transactions: pd.DataFrame, clients: pd.Index
    ) -> pd.DataFrame:
        streams = recurring_or_fallback(build_text_streams(transactions)).reset_index(drop=True)
        features = pd.DataFrame(0.0, index=clients, columns=[f"text_{f}" for f in POSITIVE_LABELS])
        features["text_confidence"] = 0.0
        features["text_top_index"] = -1.0
        if streams.empty:
            return features
        probs = self.text_.model_.predict_proba(streams)
        classes = self.text_.model_.named_steps["classifier"].classes_
        scores, details = aggregate_stream_probabilities(streams, probs, classes)
        for family in POSITIVE_LABELS:
            if family in scores.columns:
                features[f"text_{family}"] = scores[family].reindex(clients).fillna(0.0)
        features["text_confidence"] = details["confidence"].reindex(clients).fillna(0.0)
        top = scores.reindex(clients).fillna(0.0)
        features["text_top_index"] = top.to_numpy().argmax(axis=1).astype(float)
        features.loc[top.max(axis=1).le(0), "text_top_index"] = -1.0
        return features

    def _predict_text_features(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        scores, details = self.text_.predict_scores(transactions)
        features = pd.DataFrame(0.0, index=clients, columns=[f"text_{f}" for f in POSITIVE_LABELS])
        features["text_confidence"] = 0.0
        features["text_top_index"] = -1.0
        scores = scores.reindex(clients).fillna(0.0)
        for family in POSITIVE_LABELS:
            if family in scores.columns:
                features[f"text_{family}"] = scores[family]
        details = details.reindex(clients)
        features["text_confidence"] = details["confidence"].fillna(0.0)
        features["text_top_index"] = scores.to_numpy().argmax(axis=1).astype(float)
        features.loc[scores.max(axis=1).le(0), "text_top_index"] = -1.0
        self._last_text_details_ = details
        return features

    def predict_components(self, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit StreamV3PlusModel before predict")
        clients = set(transactions["client_id"].astype(str))
        if clients.intersection(self.fit_clients_):
            raise ValueError("Prediction clients must be disjoint from fit clients")
        history = self.history_.transform(transactions)
        streams = build_streams(transactions)
        stream = client_stream_features(transactions, self.mapper_, history.index, streams=streams)
        text_feat = self._predict_text_features(transactions, history.index)
        matrix = (
            history.join(stream).join(text_feat).reindex(columns=self.feature_names_).fillna(0.0)
        )
        stream_model = pd.DataFrame(
            self.model_.predict_proba(matrix), index=matrix.index, columns=LABELS
        )
        v2 = self.v2_.predict_components(transactions)["blend"].reindex(matrix.index)
        blended = self.v2_blend * v2.to_numpy() + (1.0 - self.v2_blend) * stream_model.to_numpy()
        blended = pd.DataFrame(blended, index=matrix.index, columns=LABELS)
        return {
            "stream_model": stream_model,
            "v2": v2,
            "blend": blended,
            "stream_features": stream,
            "text_features": text_feat,
        }

    def predict(self, transactions: pd.DataFrame) -> pd.Series:
        components = self.predict_components(transactions)
        pred = components["blend"].idxmax(axis=1)
        details = getattr(self, "_last_text_details_", None)
        if details is None or self.text_override_threshold is None:
            return pred
        conf = details["confidence"].reindex(pred.index).fillna(0.0)
        text_pred = details["prediction"].reindex(pred.index).fillna("none")
        eligible = (
            conf.ge(self.text_override_threshold)
            & text_pred.isin(self.text_override_families)
            & pred.ne("none")
        )
        out = pred.copy()
        out.loc[eligible] = text_pred.loc[eligible]
        return out
