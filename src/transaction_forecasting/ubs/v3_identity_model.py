"""Astra-style cross-fitted family-identity CatBoost (V3-A), without package clash.

Port of integration/v3-discovery ``ubs.v3.model`` arm A / ensemble pieces into a
single module that coexists with our ``ubs.v3`` StreamV3Model file.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.v2 import IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3_identity_features import (
    FamilyMap,
    cross_fitted_family_features,
    family_features,
    validate_history,
)


def _identity_only(family: pd.DataFrame) -> pd.DataFrame:
    return family.filter(regex="^identity_")


@dataclass
class IdentityV3Model:
    """History + cross-fitted identity_* family features, 75/25 with V2 heuristic."""

    seed: int = 42

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> IdentityV3Model:
        validate_history(transactions)
        self.v2_ = IntegratedV2Model().fit(transactions, labels)
        self.mapper_ = FamilyMap().fit(transactions, labels)
        history = self.v2_.history_.transform(transactions)
        family = _identity_only(cross_fitted_family_features(transactions, labels))
        matrix = history.join(family)
        target = labels.set_index("client_id")[TARGET_COLUMN].reindex(matrix.index)
        self.feature_names_ = matrix.columns.tolist()
        self.model_ = make_model()
        self.model_.seed = self.seed
        self.model_.fit(matrix, target)
        self.fit_clients_ = set(target.index.astype(str))
        return self

    def predict_components(self, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        validate_history(transactions)
        clients = set(transactions["client_id"].astype(str))
        if clients.intersection(self.fit_clients_):
            raise ValueError("Prediction clients must be disjoint from fit clients")
        v2 = self.v2_.predict_components(transactions)
        history = self.v2_.history_.transform(transactions)
        family = _identity_only(family_features(self.mapper_.transform(transactions)))
        matrix = history.join(family).reindex(columns=self.feature_names_).fillna(0.0)
        raw = pd.DataFrame(self.model_.predict_proba(matrix), index=matrix.index, columns=LABELS)
        heuristic = (v2["blend"] - 0.75 * v2["history"]) / 0.25
        identity = 0.75 * raw + 0.25 * heuristic
        ensemble = 0.5 * identity + 0.5 * v2["blend"]
        return {
            "v2": v2["blend"],
            "history": v2["history"],
            "identity_raw": raw,
            "identity": identity,
            "ensemble_v2_identity": ensemble,
        }

    def predict(self, transactions: pd.DataFrame) -> pd.Series:
        return self.predict_components(transactions)["identity"].idxmax(axis=1)
