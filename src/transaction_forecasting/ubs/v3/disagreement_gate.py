"""Leakage-controlled per-client routing between the frozen V2 and V3-A.

The gate sees only inference-time probability summaries.  Agreement cases keep
the shared prediction.  Learned gates are fitted only on decisive disagreement
cases, where exactly one of V2 and V3-A is correct.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.v2 import IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3.features import (
    FamilyMap,
    cross_fitted_family_features,
    family_features,
    validate_history,
)
from transaction_forecasting.ubs.v3.model import arm_matrix

ModelFactory = Callable[[], ClassifierMixin]


class V2V3AComponents:
    """Fit only the two frozen components needed by this routing experiment."""

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> V2V3AComponents:
        validate_history(transactions)
        self.v2_ = IntegratedV2Model().fit(transactions, labels)
        self.mapper_ = FamilyMap().fit(transactions, labels)
        history = self.v2_.history_.transform(transactions)
        family = cross_fitted_family_features(transactions, labels)
        target = labels.set_index("client_id")[TARGET_COLUMN].reindex(history.index)
        self.v3a_ = make_model().fit(
            arm_matrix(history, family, pd.DataFrame(index=history.index), "A"), target
        )
        return self

    def predict_probabilities(
        self, transactions: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        validate_history(transactions)
        v2 = self.v2_.predict_components(transactions)
        history = self.v2_.history_.transform(transactions)
        family = family_features(self.mapper_.transform(transactions))
        raw_a = pd.DataFrame(
            self.v3a_.predict_proba(
                arm_matrix(history, family, pd.DataFrame(index=history.index), "A")
            ),
            index=history.index,
            columns=LABELS,
        )
        heuristic = (v2["blend"] - 0.75 * v2["history"]) / 0.25
        v3a = 0.75 * raw_a + 0.25 * heuristic
        _validate_probabilities(v2["blend"], v3a)
        return v2["blend"], v3a


def _validate_probabilities(v2: pd.DataFrame, v3a: pd.DataFrame) -> None:
    for name, frame in (("V2", v2), ("V3-A", v3a)):
        if list(frame.columns) != list(LABELS):
            raise ValueError(f"{name} probabilities must use the official label order")
        if not frame.index.is_unique:
            raise ValueError(f"{name} client IDs must be unique")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} probabilities are invalid")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{name} probability rows must sum to one")
    if not v2.index.equals(v3a.index):
        raise ValueError("V2 and V3-A clients must have identical order")


def _margin(frame: pd.DataFrame) -> np.ndarray:
    ordered = np.sort(frame.to_numpy(dtype=float), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _entropy(frame: pd.DataFrame) -> np.ndarray:
    values = np.clip(frame.to_numpy(dtype=float), 1e-12, 1.0)
    return -(values * np.log(values)).sum(axis=1) / np.log(len(LABELS))


def gate_features(v2: pd.DataFrame, v3a: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed, probability-only gate feature set available at inference."""
    _validate_probabilities(v2, v3a)
    features = pd.concat(
        [v2.add_prefix("v2_probability_"), v3a.add_prefix("v3a_probability_")], axis=1
    )
    v2_max = v2.max(axis=1)
    v3a_max = v3a.max(axis=1)
    features["v2_max_probability"] = v2_max
    features["v3a_max_probability"] = v3a_max
    features["max_probability_difference_v3a_minus_v2"] = v3a_max - v2_max
    features["v2_margin"] = _margin(v2)
    features["v3a_margin"] = _margin(v3a)
    features["margin_difference_v3a_minus_v2"] = features.v3a_margin - features.v2_margin
    features["v2_entropy"] = _entropy(v2)
    features["v3a_entropy"] = _entropy(v3a)
    features["entropy_difference_v3a_minus_v2"] = features.v3a_entropy - features.v2_entropy
    features["v2_none_probability"] = v2["none"]
    features["v3a_none_probability"] = v3a["none"]
    return features


def fixed_confidence_route(v2: pd.DataFrame, v3a: pd.DataFrame) -> pd.Series:
    """Choose the more confident argmax on disagreements, with no tuned threshold."""
    _validate_probabilities(v2, v3a)
    v2_prediction = v2.idxmax(axis=1)
    v3a_prediction = v3a.idxmax(axis=1)
    choose_a = v3a.max(axis=1).ge(v2.max(axis=1))
    result = v2_prediction.where(~choose_a, v3a_prediction)
    return result.where(v2_prediction.ne(v3a_prediction), v2_prediction)


def gate_model_factories() -> dict[str, ModelFactory]:
    """Return the small, frozen gate shortlist selected only with TRAIN OOF."""
    return {
        "logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
        "shallow_tree": lambda: DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        ),
        "small_gradient_boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=75,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def disagreement_masks(
    target: pd.Series, v2_prediction: pd.Series, v3a_prediction: pd.Series
) -> pd.DataFrame:
    """Describe disagreement outcomes without inventing labels for both-wrong rows."""
    if not target.index.equals(v2_prediction.index) or not target.index.equals(
        v3a_prediction.index
    ):
        raise ValueError("Target and predictions must have identical client order")
    disagree = v2_prediction.ne(v3a_prediction)
    v2_correct = v2_prediction.eq(target)
    v3a_correct = v3a_prediction.eq(target)
    return pd.DataFrame(
        {
            "disagree": disagree,
            "v2_only": disagree & v2_correct & ~v3a_correct,
            "v3a_only": disagree & v3a_correct & ~v2_correct,
            "both_wrong": disagree & ~v2_correct & ~v3a_correct,
        },
        index=target.index,
    )


def _fit_gate(
    factory: ModelFactory,
    features: pd.DataFrame,
    target: pd.Series,
    v2_prediction: pd.Series,
    v3a_prediction: pd.Series,
) -> ClassifierMixin:
    masks = disagreement_masks(target, v2_prediction, v3a_prediction)
    decisive = masks.v2_only | masks.v3a_only
    binary_target = masks.loc[decisive, "v3a_only"].astype(int)
    if len(binary_target) < 2 or binary_target.nunique() != 2:
        raise ValueError("Gate fit requires decisive disagreements from both systems")
    model = factory()
    model.fit(features.loc[decisive], binary_target)
    return model


def route_with_model(
    model: ClassifierMixin,
    features: pd.DataFrame,
    v2_prediction: pd.Series,
    v3a_prediction: pd.Series,
) -> pd.Series:
    """Apply a binary choose-V3-A gate only where base predictions disagree."""
    if not features.index.equals(v2_prediction.index) or not features.index.equals(
        v3a_prediction.index
    ):
        raise ValueError("Gate inputs must have identical client order")
    result = v3a_prediction.copy()
    disagree = v2_prediction.ne(v3a_prediction)
    if disagree.any():
        choose_a = np.asarray(model.predict(features.loc[disagree]), dtype=int).astype(bool)
        selected = v2_prediction.loc[disagree].where(~choose_a, v3a_prediction.loc[disagree])
        result.loc[disagree] = selected
    return result


def cross_fitted_gate_predictions(
    target: pd.Series,
    v2: pd.DataFrame,
    v3a: pd.DataFrame,
    factory: ModelFactory,
    *,
    folds: int = 5,
    seed: int = 314159,
) -> tuple[pd.Series, pd.Series]:
    """Estimate a gate with a second client-level cross-fitting layer.

    The base inputs must already be OOF probabilities.  The returned gate fold
    records which additional holdout produced every routed prediction.
    """
    _validate_probabilities(v2, v3a)
    if not target.index.equals(v2.index) or not target.isin(LABELS).all():
        raise ValueError("Target must align to probabilities and use official labels")
    features = gate_features(v2, v3a)
    v2_prediction = v2.idxmax(axis=1)
    v3a_prediction = v3a.idxmax(axis=1)
    output = pd.Series(index=target.index, dtype=object, name="prediction")
    gate_fold = pd.Series(index=target.index, dtype="int64", name="gate_fold")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for fold, (fit_positions, hold_positions) in enumerate(
        splitter.split(target.index, target), start=1
    ):
        fit_ids = target.index[fit_positions]
        hold_ids = target.index[hold_positions]
        model = _fit_gate(
            factory,
            features.loc[fit_ids],
            target.loc[fit_ids],
            v2_prediction.loc[fit_ids],
            v3a_prediction.loc[fit_ids],
        )
        output.loc[hold_ids] = route_with_model(
            model,
            features.loc[hold_ids],
            v2_prediction.loc[hold_ids],
            v3a_prediction.loc[hold_ids],
        )
        gate_fold.loc[hold_ids] = fold
    if output.isna().any() or not output.isin(LABELS).all():
        raise RuntimeError("Cross-fitted gate did not produce one valid prediction per client")
    return output, gate_fold


def fit_final_gate(
    target: pd.Series,
    v2: pd.DataFrame,
    v3a: pd.DataFrame,
    factory: ModelFactory,
) -> ClassifierMixin:
    """Fit a frozen gate on decisive TRAIN OOF rows for later unseen inference."""
    features = gate_features(v2, v3a)
    return _fit_gate(factory, features, target, v2.idxmax(axis=1), v3a.idxmax(axis=1))


def routing_diagnostics(
    target: pd.Series,
    v2_prediction: pd.Series,
    v3a_prediction: pd.Series,
    routed_prediction: pd.Series,
) -> dict[str, object]:
    """Return overall and actual-class routing decomposition."""
    if not target.index.equals(routed_prediction.index):
        raise ValueError("Routed predictions must align to target")
    masks = disagreement_masks(target, v2_prediction, v3a_prediction)

    def summarize(index: pd.Index) -> dict[str, int | float]:
        local = masks.loc[index]
        disagreements = int(local.disagree.sum())
        changed = v3a_prediction.loc[index].ne(routed_prediction.loc[index]) & local.disagree
        correct_change = changed & routed_prediction.loc[index].eq(target.loc[index])
        incorrect_change = changed & v3a_prediction.loc[index].eq(target.loc[index])
        return {
            "clients": int(len(index)),
            "disagreements": disagreements,
            "v2_correct_only": int(local.v2_only.sum()),
            "v3a_correct_only": int(local.v3a_only.sum()),
            "both_wrong": int(local.both_wrong.sum()),
            "routing_accuracy_on_disagreements": (
                float(
                    routed_prediction.loc[index][local.disagree]
                    .eq(target.loc[index][local.disagree])
                    .mean()
                )
                if disagreements
                else 0.0
            ),
            "routed_to_v2": int(
                routed_prediction.loc[index][local.disagree]
                .eq(v2_prediction.loc[index][local.disagree])
                .sum()
            ),
            "routed_to_v3a": int(
                routed_prediction.loc[index][local.disagree]
                .eq(v3a_prediction.loc[index][local.disagree])
                .sum()
            ),
            "changed_vs_v3a": int(changed.sum()),
            "correct_changes": int(correct_change.sum()),
            "incorrect_changes": int(incorrect_change.sum()),
            "both_wrong_changes": int((changed & local.both_wrong).sum()),
        }

    return {
        "overall": summarize(target.index),
        "by_actual_class": {label: summarize(target.index[target.eq(label)]) for label in LABELS},
    }
