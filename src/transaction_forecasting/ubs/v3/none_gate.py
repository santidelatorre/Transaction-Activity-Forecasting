"""Leakage-safe identity model and compact binary gate utilities.

The gate only decides between a positive subscription and ``none``.  When it
chooses positive, the family is always the highest-probability positive family
from the unchanged V3-A identity model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.v2 import IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3.features import (
    FAMILIES,
    FamilyMap,
    cross_fitted_family_features,
    family_features,
    recurrence_features,
    validate_history,
)
from transaction_forecasting.ubs.v3.model import arm_matrix

PROBABILITY_FEATURES = (
    "a_none_probability",
    "a_positive_probability",
    "a_positive_margin",
    "a_top_probability",
    "a_top_margin",
    "a_entropy",
    "v2_none_probability",
    "v2_positive_probability",
    "v2_positive_margin",
    "v2_top_margin",
    "v2_entropy",
    "none_probability_delta",
    "predictions_agree",
    "positive_family_agree",
)

COMPACT_EVIDENCE_FEATURES = (
    "transaction_count",
    "outgoing_card_count",
    "recurring_description_count",
    "recurring_event_share",
    "mapped_event_share",
    "mapped_family_count",
    "mapped_max_family_share",
    "predicted_family_mapped_share",
    "history_days",
    "recency_days",
    "payment_streams",
    "payment_regularity_max",
    "payment_score",
)


class IdentityV3Model:
    """Exact V3-A identity arm, without fitting the unused B/AB arms."""

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> IdentityV3Model:
        validate_history(transactions)
        self.v2_ = IntegratedV2Model().fit(transactions, labels)
        self.mapper_ = FamilyMap().fit(transactions, labels)
        history = self.v2_.history_.transform(transactions)
        family = cross_fitted_family_features(transactions, labels)
        target = labels.set_index("client_id")[TARGET_COLUMN].reindex(history.index)
        self.model_ = make_model().fit(arm_matrix(history, family, None, "A"), target)
        return self

    def predict_components(
        self, transactions: pd.DataFrame
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        validate_history(transactions)
        v2 = self.v2_.predict_components(transactions)
        history = self.v2_.history_.transform(transactions)
        mapped = self.mapper_.transform(transactions)
        family = family_features(mapped)
        raw = pd.DataFrame(
            self.model_.predict_proba(arm_matrix(history, family, None, "A")),
            index=history.index,
            columns=LABELS,
        )
        heuristic = (v2["blend"] - 0.75 * v2["history"]) / 0.25
        return {"V2": v2["blend"], "A": 0.75 * raw + 0.25 * heuristic}, mapped


def _probability_summary(probabilities: pd.DataFrame, prefix: str) -> pd.DataFrame:
    values = probabilities.to_numpy(dtype=float)
    if probabilities.columns.tolist() != list(LABELS):
        raise ValueError("Probabilities must use official label order")
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError("Invalid probability matrix")
    positive = probabilities.loc[:, FAMILIES]
    sorted_values = np.sort(values, axis=1)
    result = pd.DataFrame(index=probabilities.index)
    result[f"{prefix}_none_probability"] = probabilities["none"]
    result[f"{prefix}_positive_probability"] = positive.max(axis=1)
    result[f"{prefix}_positive_margin"] = positive.max(axis=1) - probabilities["none"]
    result[f"{prefix}_top_probability"] = values.max(axis=1)
    result[f"{prefix}_top_margin"] = sorted_values[:, -1] - sorted_values[:, -2]
    result[f"{prefix}_entropy"] = -(values * np.log(values.clip(1e-12))).sum(axis=1) / np.log(
        len(LABELS)
    )
    return result


def _behavior_features(
    transactions: pd.DataFrame, mapped: pd.DataFrame, positive_family: pd.Series
) -> pd.DataFrame:
    clients = positive_family.index
    result = pd.DataFrame(index=clients)
    grouped = transactions.groupby("client_id")
    result["transaction_count"] = grouped.size().reindex(clients, fill_value=0)
    result["description_count"] = grouped.description.nunique().reindex(clients, fill_value=0)
    result["history_days"] = (
        grouped.timestamp.max().sub(grouped.timestamp.min()).dt.total_seconds().div(86400)
    ).reindex(clients, fill_value=0)
    result["recency_days"] = (
        (CUTOFF - grouped.timestamp.max()).dt.total_seconds().div(86400)
    ).reindex(clients, fill_value=0)

    outgoing = mapped.loc[mapped.direction.eq("out") & mapped.type.eq("card_payment")].copy()
    outgoing_count = outgoing.groupby("client_id").size().reindex(clients, fill_value=0)
    result["outgoing_card_count"] = outgoing_count
    description_counts = outgoing.groupby(["client_id", "description"]).size()
    repeated = description_counts.loc[description_counts.ge(2)]
    result["recurring_description_count"] = (
        repeated.groupby(level="client_id").size().reindex(clients, fill_value=0)
    )
    recurring_events = repeated.groupby(level="client_id").sum().reindex(clients, fill_value=0)
    result["recurring_event_share"] = recurring_events.div(outgoing_count.clip(lower=1))

    positive = outgoing.loc[outgoing.family.isin(FAMILIES)]
    mapped_count = positive.groupby("client_id").size().reindex(clients, fill_value=0)
    result["mapped_event_share"] = mapped_count.div(outgoing_count.clip(lower=1))
    by_family = positive.groupby(["client_id", "family"]).size()
    result["mapped_family_count"] = (
        by_family.groupby(level="client_id").size().reindex(clients, fill_value=0)
    )
    result["mapped_max_family_share"] = by_family.groupby(level="client_id").max().reindex(
        clients, fill_value=0
    ) / outgoing_count.clip(lower=1)
    predicted_keys = pd.MultiIndex.from_arrays(
        [clients, positive_family.reindex(clients)], names=["client_id", "family"]
    )
    result["predicted_family_mapped_count"] = by_family.reindex(
        predicted_keys, fill_value=0
    ).to_numpy()
    result["predicted_family_mapped_share"] = result["predicted_family_mapped_count"].div(
        outgoing_count.clip(lower=1)
    )
    aliases = positive.groupby(["client_id", "family"]).description.nunique()
    result["predicted_family_aliases"] = aliases.reindex(predicted_keys, fill_value=0).to_numpy()
    return result


def build_gate_features(
    a_probabilities: pd.DataFrame,
    v2_probabilities: pd.DataFrame,
    transactions: pd.DataFrame,
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact inference-time features with no target argument."""
    validate_history(transactions)
    if not a_probabilities.index.equals(v2_probabilities.index):
        raise ValueError("Base probability rows must align")
    positive_family = a_probabilities.loc[:, FAMILIES].idxmax(axis=1)
    a = _probability_summary(a_probabilities, "a")
    v2 = _probability_summary(v2_probabilities, "v2")
    features = pd.concat([a, v2], axis=1)
    features["none_probability_delta"] = a_probabilities["none"] - v2_probabilities["none"]
    features["predictions_agree"] = (
        a_probabilities.idxmax(axis=1) == v2_probabilities.idxmax(axis=1)
    ).astype(float)
    features["positive_family_agree"] = (
        positive_family == v2_probabilities.loc[:, FAMILIES].idxmax(axis=1)
    ).astype(float)
    behavior = _behavior_features(transactions, mapped, positive_family)
    recurrence = recurrence_features(transactions)
    result = pd.concat([features, behavior, recurrence], axis=1).reindex(a_probabilities.index)
    if result.isna().any().any() or not result.index.is_unique:
        raise RuntimeError("Gate features are incomplete or duplicated")
    return result.astype(float)


def gate_feature_names(feature_set: str) -> list[str]:
    """Return a predefined compact feature set."""
    if feature_set == "probability":
        return list(PROBABILITY_FEATURES)
    if feature_set == "compact":
        return [*PROBABILITY_FEATURES, *COMPACT_EVIDENCE_FEATURES]
    if feature_set == "all":
        return []
    raise ValueError(f"Unknown gate feature set: {feature_set}")


def make_gate_estimator(kind: str, c: float = 1.0):
    """Create one of the small, frozen gate candidates."""
    if kind == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=c, max_iter=1000, random_state=42, solver="lbfgs"),
        )
    if kind == "tree":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=2,
                min_samples_leaf=30,
                random_state=42,
            ),
        )
    raise ValueError(f"Unknown gate estimator: {kind}")


def apply_none_gate(
    positive_probability: pd.Series,
    positive_family: pd.Series,
    threshold: float,
) -> pd.Series:
    """Combine a binary gate with V3-A's positive-family prediction."""
    if not positive_probability.index.equals(positive_family.index):
        raise ValueError("Gate scores and family predictions must align")
    if not 0 <= threshold <= 1 or not positive_probability.between(0, 1).all():
        raise ValueError("Invalid gate probability or threshold")
    return positive_family.where(positive_probability.ge(threshold), "none")
