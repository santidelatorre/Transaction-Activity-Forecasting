"""V2 building blocks with no target-derived training features.

The numeric history aggregates reuse V1 definitions. Family columns are omitted
entirely: their in-sample description mappings can encode the training label.
V1 remains unchanged and runnable through its original entry point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import CUTOFF, LABELS
from transaction_forecasting.ubs.features import ClientFeatureBuilder
from transaction_forecasting.ubs.models import CatBoostClientModel


class HistoryFeatureBuilder(ClientFeatureBuilder):
    """Fit categorical vocabulary on train histories, without accepting labels."""

    def fit(self, transactions: pd.DataFrame) -> HistoryFeatureBuilder:
        if transactions.timestamp.isna().any() or transactions.timestamp.ge(CUTOFF).any():
            raise ValueError("Training history must be strictly before cutoff")
        self.categorical_levels_ = {
            column: sorted(transactions[column].astype(str).unique().tolist())
            for column in ("mcc", "type", "currency", "direction")
        }
        self.description_lift_ = pd.DataFrame(columns=LABELS, dtype=float)
        self.fitted_ = True
        return self

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if transactions.timestamp.isna().any():
            raise ValueError("Missing transaction timestamp")
        features = super().transform(transactions)
        return features.loc[:, ~features.columns.str.startswith("family_")]


def make_model() -> CatBoostClientModel:
    """Esteban's fixed small CatBoost recipe; no search on validation."""
    return CatBoostClientModel(balanced=True, seed=42, iterations=300, depth=4, learning_rate=0.05)


def calibrate_probabilities(probabilities, temperature=2.0, none_bias=-1.5):
    """Adapt Esteban 358d604's frozen logit correction, without validation tuning.

    Negative none bias reduces none predictions. This transform is a decision
    correction; no claim of calibrated confidence is made.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError("Expected eight probability columns in official label order")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Probability rows must sum to one")
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(none_bias):
        raise ValueError("Invalid calibration parameters")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temperature
    logits[:, LABELS.index("none")] += none_bias
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)
