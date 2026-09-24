"""Small TRAIN-only decision correction; this does not calibrate confidence.

Cross-fitting here isolates the calibration layer on fixed base-model OOF
probabilities. It is not a fully nested refit of the base classifier.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.evaluation import evaluate_predictions

BOUND = 0.25
PENALTY = 0.02
MIN_GAIN = 0.001
GRID = tuple(
    product(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (0.0, -0.075, 0.075, -0.15, 0.15),
        (0.0, -0.075, 0.075, -0.15, 0.15),
    )
)


def validate_probabilities(probabilities: pd.DataFrame) -> None:
    """Require exactly one finite eight-class probability row per client."""
    if not isinstance(probabilities, pd.DataFrame) or list(probabilities.columns) != list(LABELS):
        raise ValueError("Expected eight probability columns in official label order")
    if probabilities.empty or not probabilities.index.is_unique or probabilities.index.hasnans:
        raise ValueError("Expected nonempty unique client IDs")
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8):
        raise ValueError("Probability rows must sum to one")


def _validate_target(probabilities, target):
    validate_probabilities(probabilities)
    if not target.index.equals(probabilities.index) or not target.index.is_unique:
        raise ValueError("Target must align exactly with unique probability clients")
    if not target.isin(LABELS).all():
        raise ValueError("Invalid target classes")


def apply_offsets(probabilities: pd.DataFrame, offsets) -> pd.DataFrame:
    """Normalize p[c] * exp(b[c]); preserve exact zeros and exact identity."""
    validate_probabilities(probabilities)
    bias = np.asarray(offsets, dtype=float)
    if bias.shape != (len(LABELS),) or not np.isfinite(bias).all():
        raise ValueError("Expected eight finite class offsets")
    if (np.abs(bias) > BOUND + 1e-12).any():
        raise ValueError("Class offsets exceed the identity bound")
    if not bias.any():
        return probabilities.copy()
    values = probabilities.to_numpy() * np.exp(bias)
    return pd.DataFrame(
        values / values.sum(axis=1, keepdims=True),
        index=probabilities.index,
        columns=probabilities.columns,
    )


def frequency_direction(probabilities, target):
    """Smoothed true/hard-predicted log frequency ratio, centered and bounded."""
    _validate_target(probabilities, target)
    actual = target.value_counts().reindex(LABELS, fill_value=0).to_numpy()
    predicted = probabilities.idxmax(axis=1).value_counts().reindex(LABELS, fill_value=0).to_numpy()
    direction = np.log((actual + 5.0) / (predicted + 5.0))
    direction -= direction.mean()
    return direction / max(1.0, np.abs(direction).max() / BOUND)


def class_offsets(direction, parameters):
    """Three coefficients: frequency, gym-minus-none, music-minus-streaming."""
    alpha, gym_none, music_streaming = parameters
    bias = alpha * np.asarray(direction, dtype=float).copy()
    bias[LABELS.index("gym")] += gym_none
    bias[LABELS.index("none")] -= gym_none
    bias[LABELS.index("music")] += music_streaming
    bias[LABELS.index("streaming")] -= music_streaming
    return np.clip(bias, -BOUND, BOUND)


def _f1(target, prediction):
    report = evaluate_predictions(target, prediction)
    return np.array([report["per_class"][label]["f1-score"] for label in LABELS])


def select_calibration(probabilities, target, folds):
    """Select coefficients by held-out calibration folds, never resubstitution.

    The data-derived direction is recomputed without each inner holdout. Reject
    candidates losing >.02 F1 in any class, >.005 in more than two classes, or
    failing to preserve/improve at least four classes. Ties prefer identity.
    """
    _validate_target(probabilities, target)
    if not folds.index.equals(target.index) or folds.isna().any() or folds.nunique() < 2:
        raise ValueError("Need at least two aligned client folds")
    partitions = []
    for fold in sorted(folds.unique()):
        hold = folds.eq(fold)
        direction = frequency_direction(probabilities.loc[~hold], target.loc[~hold])
        partitions.append((hold, direction))
    raw_f1 = _f1(target, probabilities.idxmax(axis=1))
    records = []
    for parameters in GRID:
        prediction = pd.Series(index=target.index, dtype=object)
        penalties = []
        for hold, direction in partitions:
            bias = class_offsets(direction, parameters)
            prediction.loc[hold] = apply_offsets(probabilities.loc[hold], bias).idxmax(axis=1)
            penalties.append(float(np.mean((bias / BOUND) ** 2)))
        f1 = _f1(target, prediction)
        delta = f1 - raw_f1
        eligible = bool(
            delta.min() >= -0.02 - 1e-12
            and (delta < -0.005 - 1e-12).sum() <= 2
            and (delta >= -1e-12).sum() >= 4
        )
        regularization = PENALTY * float(np.mean(penalties))
        records.append(
            {
                "parameters": list(parameters),
                "inner_macro_f1": float(f1.mean()),
                "regularization": regularization,
                "objective": float(f1.mean()) - regularization,
                "eligible": eligible,
                "per_class_delta": dict(zip(LABELS, delta.tolist(), strict=True)),
            }
        )
    # Identity is first; strict improvement and deterministic enumeration break ties.
    best = records[0]
    for record in records[1:]:
        if record["eligible"] and record["objective"] > best["objective"] + 1e-12:
            best = record
    if best["objective"] < records[0]["objective"] + MIN_GAIN:
        best = records[0]
    direction = frequency_direction(probabilities, target)
    bias = class_offsets(direction, best["parameters"])
    return {
        "parameters": best["parameters"],
        "direction": direction.tolist(),
        "offsets": bias.tolist(),
        "weights": np.exp(bias).tolist(),
        "selection": best,
        "candidates": records,
    }


def cross_fit_calibration(probabilities, target, folds):
    """Evaluate the complete selection procedure on untouched outer clients."""
    _validate_target(probabilities, target)
    if not folds.index.equals(target.index) or folds.isna().any() or folds.nunique() < 3:
        raise ValueError("Need at least three aligned outer client folds")
    calibrated = probabilities.copy()
    fitted = []
    for fold in sorted(folds.unique()):
        hold = folds.eq(fold)
        selected = select_calibration(probabilities.loc[~hold], target.loc[~hold], folds.loc[~hold])
        calibrated.loc[hold] = apply_offsets(probabilities.loc[hold], selected["offsets"])
        fitted.append(
            {
                "fold": int(fold),
                "fit_clients": int((~hold).sum()),
                "hold_clients": int(hold.sum()),
                **selected,
            }
        )
    return calibrated, fitted
