"""Small TRAIN-selected gym routing experiment; independent of V2/V3 predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.evaluation import evaluate_predictions

# Frozen search space for this experiment, not official challenge thresholds.
THRESHOLDS = tuple(round(value / 1000, 3) for value in range(250, 751, 25))


def aligned_probabilities(v2, identity):
    """Reject invalid probabilities/IDs; align both axes before making decisions."""
    for frame in (v2, identity):
        if frame.empty or not frame.index.is_unique or frame.index.hasnans:
            raise ValueError("Probability client IDs must be nonempty, unique and nonnull")
        if not frame.columns.is_unique or set(frame.columns) != set(LABELS):
            raise ValueError("Expected exactly eight official probability columns")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Invalid probability values")
        if not np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=0):
            raise ValueError("Probability rows must sum to one")
    if set(v2.index) != set(identity.index):
        raise ValueError("Probability client sets must match exactly")
    return v2.loc[:, LABELS], identity.reindex(index=v2.index, columns=LABELS)


def route_gym(v2, identity, threshold):
    """Route only confident V2-only gym disagreements dominating A's top score.

    None means the explicit no-op control. No labels enter this function.
    """
    v2, identity = aligned_probabilities(v2, identity)
    prediction = identity.idxmax(axis=1)
    if threshold is None:
        return prediction
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and inside [0, 1]")
    override = (
        v2.idxmax(axis=1).eq("gym")
        & prediction.ne("gym")
        & v2.gym.ge(threshold)
        & v2.gym.ge(identity.max(axis=1))
    )
    return prediction.mask(override, "gym")


def select_gate(target, v2, identity):
    """Select on TRAIN OOF only, with no-op/fewer overrides breaking exact ties."""
    v2, identity = aligned_probabilities(v2, identity)
    control = identity.idxmax(axis=1)
    search = []
    for threshold in (None, *THRESHOLDS):
        prediction = route_gym(v2, identity, threshold)
        score = evaluate_predictions(target, prediction)
        search.append(
            {
                "threshold": threshold,
                "macro_f1": score["macro_f1"],
                "accuracy": score["accuracy"],
                "overrides": int(prediction.ne(control).sum()),
            }
        )
    selected = max(
        search,
        key=lambda row: (
            row["macro_f1"],
            -row["overrides"],
            row["threshold"] is None,
            row["threshold"] or 0,
        ),
    )
    return selected["threshold"], search


def override_analysis(target, control, candidate):
    evaluate_predictions(target, control)
    evaluate_predictions(target, candidate)
    control, candidate = control.reindex(target.index), candidate.reindex(target.index)
    changed = candidate.ne(control)
    correct = changed & candidate.eq(target)
    harmed = changed & control.eq(target) & candidate.ne(target)
    return {
        "affected_clients": int(changed.sum()),
        "to_gym": int((changed & candidate.eq("gym")).sum()),
        "away_from_gym": int((changed & control.eq("gym")).sum()),
        "correct_overrides": int(correct.sum()),
        "incorrect_overrides": int((changed & ~correct).sum()),
        "previously_correct_now_wrong": int(harmed.sum()),
        "wrong_to_wrong": int((changed & ~correct & ~harmed).sum()),
        "true_classes_of_incorrect_overrides": target[changed & ~correct].value_counts().to_dict(),
        "classes_losing_correct_predictions": target[harmed].value_counts().to_dict(),
    }


def cross_validate_gate(target, v2, identity, folds):
    """Leave one cached OOF fold out of gate selection.

    This is gate-level cross-validation, NOT fully nested base-model CV: base
    fits generating other folds' OOF scores may have seen this fold's labels.
    """
    v2, identity = aligned_probabilities(v2, identity)
    if not folds.index.is_unique or set(folds.index) != set(target.index):
        raise ValueError("Fold IDs must cover target clients exactly")
    folds = folds.reindex(target.index)
    if folds.isna().any() or folds.nunique() < 2:
        raise ValueError("Need at least two complete folds")
    evaluate_predictions(target, identity.idxmax(axis=1))
    candidate = pd.Series(index=target.index, dtype=object)
    records = []
    for fold in sorted(folds.unique()):
        hold = folds.index[folds.eq(fold)]
        fit = folds.index[folds.ne(fold)]
        threshold, _ = select_gate(target.loc[fit], v2.loc[fit], identity.loc[fit])
        prediction = route_gym(v2.loc[hold], identity.loc[hold], threshold)
        candidate.loc[hold] = prediction
        control = identity.loc[hold].idxmax(axis=1)
        records.append(
            {
                "fold": int(fold),
                "threshold": threshold,
                "V2": evaluate_predictions(target.loc[hold], v2.loc[hold].idxmax(axis=1)),
                "A": evaluate_predictions(target.loc[hold], control),
                "candidate": evaluate_predictions(target.loc[hold], prediction),
                "overrides": override_analysis(target.loc[hold], control, prediction),
            }
        )
    return candidate, records
