"""Fixed factorial family-aware candidate, independent of the frozen V2 code."""

from __future__ import annotations

import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.v2 import IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3.features import (
    FamilyMap,
    cross_fitted_family_features,
    family_features,
    recurrence_features,
    validate_history,
)


def arm_matrix(history, family, recurrence, arm):
    if arm not in ("A", "B", "AB"):
        raise ValueError("Unknown factorial arm")
    blocks = [history]
    if arm in ("A", "AB"):
        blocks.append(family.filter(regex="^identity_") if arm == "A" else family)
    if arm in ("B", "AB"):
        blocks.append(recurrence)
    return pd.concat(blocks, axis=1).reindex(history.index)


class V3Model:
    """V3-A (family identity) is the promoted competitive baseline.

    ``predict()`` uses exclusively A: V2 history plus family identity, with the
    frozen 75% numeric model / 25% periodicity heuristic blend and no recurrence
    block B. B, AB/full and ensemble remain available for historical research
    and diagnostics through ``predict_components()``.
    """

    def fit(self, transactions, labels):
        validate_history(transactions)
        self.v2_ = IntegratedV2Model().fit(transactions, labels)
        self.mapper_ = FamilyMap().fit(transactions, labels)
        history = self.v2_.history_.transform(transactions)
        family = cross_fitted_family_features(transactions, labels)
        recurrence = recurrence_features(transactions)
        target = labels.set_index("client_id")[TARGET_COLUMN].reindex(history.index)
        self.models_ = {
            arm: make_model().fit(arm_matrix(history, family, recurrence, arm), target)
            for arm in ("A", "B", "AB")
        }
        return self

    def predict_components(self, transactions):
        validate_history(transactions)
        # V2 and mapper both reject clients seen in training.
        v2 = self.v2_.predict_components(transactions)
        history = self.v2_.history_.transform(transactions)
        family = family_features(self.mapper_.transform(transactions))
        recurrence = recurrence_features(transactions)
        heuristic = (v2["blend"] - 0.75 * v2["history"]) / 0.25
        results = {"V2": v2["blend"], "history_control": v2["history"]}
        for arm, model in self.models_.items():
            raw = pd.DataFrame(
                model.predict_proba(arm_matrix(history, family, recurrence, arm)),
                index=history.index,
                columns=LABELS,
            )
            results[arm] = 0.75 * raw + 0.25 * heuristic
        results["full"] = results["AB"].copy()
        results["ensemble"] = 0.5 * results["full"] + 0.5 * results["V2"]
        return results

    def predict(self, transactions):
        return self.predict_components(transactions)["A"].idxmax(axis=1)
