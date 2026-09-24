"""Leakage-safe V2.1 candidates built on top of the frozen V2 architecture."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.features import ClientFeatureBuilder
from transaction_forecasting.ubs.models import RecurrenceHeuristic
from transaction_forecasting.ubs.temporal_features import apply_temporal_blocks, temporal_streams
from transaction_forecasting.ubs.text_v2 import MerchantFeatureBuilder
from transaction_forecasting.ubs.v2 import HistoryFeatureBuilder, make_model


def _family_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Select the fixed V1 family block and reject an incomplete projection."""
    result = features.loc[:, features.columns.str.startswith("family_")]
    expected = len(LABELS) * 9
    if result.shape[1] != expected:
        raise ValueError(f"Expected {expected} family features, got {result.shape[1]}")
    return result


def cross_fitted_family_features(
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Build target-derived family features without using a client's own label.

    Each held-out client is transformed by a description mapping fitted on the
    other client folds. This block is for model fitting only. Inference uses one
    mapping fitted on all labelled training clients.
    """
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    if not target.index.is_unique or set(target.index) != set(transactions["client_id"]):
        raise ValueError("Training clients and unique labels must match")
    if n_splits < 2:
        raise ValueError("Cross-fitting requires at least two folds")
    result: pd.DataFrame | None = None
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_positions, held_positions in splitter.split(target.index, target):
        fit_ids = target.index[fit_positions]
        held_ids = target.index[held_positions]
        fit_transactions = transactions.loc[transactions["client_id"].isin(fit_ids)]
        held_transactions = transactions.loc[transactions["client_id"].isin(held_ids)]
        fit_labels = labels.loc[labels["client_id"].isin(fit_ids)]
        mapping = ClientFeatureBuilder().fit(fit_transactions, fit_labels)
        held_features = _family_columns(mapping.transform(held_transactions)).reindex(held_ids)
        if result is None:
            result = pd.DataFrame(index=target.index, columns=held_features.columns, dtype=float)
        result.loc[held_ids, held_features.columns] = held_features
    if result is None or result.isna().any().any():
        raise RuntimeError("Cross-fitted family feature coverage is incomplete")
    return result.astype(float)


class CrossFittedFamilyV21Model:
    """V2 CatBoost plus cross-fitted V1 family evidence and the V2 heuristic."""

    def __init__(self, *, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> CrossFittedFamilyV21Model:
        target = labels.set_index("client_id")[TARGET_COLUMN]
        if set(target) != set(LABELS):
            raise ValueError("Training must contain all eight official classes")
        self.fit_clients_ = set(target.index)
        self.history_ = HistoryFeatureBuilder().fit(transactions)
        history = self.history_.transform(transactions)
        family = cross_fitted_family_features(
            transactions,
            labels,
            n_splits=self.n_splits,
            seed=self.seed,
        ).reindex(history.index)
        matrix = history.join(family)
        if not matrix.index.is_unique or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("V2.1 training matrix must be unique and finite")
        self.feature_names_ = matrix.columns.tolist()
        self.model_ = make_model().fit(matrix, target.reindex(matrix.index))
        self.mapping_ = ClientFeatureBuilder().fit(transactions, labels)
        return self

    def predict_components(self, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the V2.1 model before prediction")
        if self.fit_clients_.intersection(transactions["client_id"]):
            raise ValueError("Prediction clients must be excluded from all fitted state")
        history = self.history_.transform(transactions)
        mapped = self.mapping_.transform(transactions)
        matrix = history.join(_family_columns(mapped)).reindex(columns=self.feature_names_)
        family_probabilities = self.model_.predict_proba(matrix)
        temporal = apply_temporal_blocks(
            mapped,
            temporal_streams(transactions),
            self.mapping_.description_lift_,
            ("periodicity",),
        )
        heuristic = RecurrenceHeuristic(none_bias=-1.0, temperature=1.0).predict_proba(temporal)
        return {
            name: pd.DataFrame(values, index=matrix.index, columns=LABELS)
            for name, values in (
                ("family_model", family_probabilities),
                ("blend", 0.75 * family_probabilities + 0.25 * heuristic),
            )
        }

    def predict(self, transactions: pd.DataFrame) -> pd.Series:
        return self.predict_components(transactions)["blend"].idxmax(axis=1)


class MerchantHistoryV21Model:
    """V2 history CatBoost augmented by normalized leave-one-client-out text."""

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> MerchantHistoryV21Model:
        target = labels.set_index("client_id")[TARGET_COLUMN]
        if not target.index.is_unique or set(target.index) != set(transactions["client_id"]):
            raise ValueError("Training clients and unique labels must match")
        self.fit_clients_ = set(target.index)
        self.history_ = HistoryFeatureBuilder().fit(transactions)
        history = self.history_.transform(transactions)
        self.merchant_ = MerchantFeatureBuilder().fit(transactions, labels)
        merchant = self.merchant_.transform(transactions, training=True).reindex(history.index)
        matrix = history.join(merchant)
        if not matrix.index.is_unique or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("Merchant-history training matrix must be unique and finite")
        self.feature_names_ = matrix.columns.tolist()
        self.model_ = make_model().fit(matrix, target.reindex(matrix.index))
        return self

    def predict_proba(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the merchant-history model before prediction")
        if self.fit_clients_.intersection(transactions["client_id"]):
            raise ValueError("Prediction clients must be excluded from all fitted state")
        history = self.history_.transform(transactions)
        merchant = self.merchant_.transform(transactions).reindex(history.index)
        matrix = history.join(merchant).reindex(columns=self.feature_names_)
        return pd.DataFrame(self.model_.predict_proba(matrix), index=matrix.index, columns=LABELS)

    def predict(self, transactions: pd.DataFrame) -> pd.Series:
        return self.predict_proba(transactions).idxmax(axis=1)
