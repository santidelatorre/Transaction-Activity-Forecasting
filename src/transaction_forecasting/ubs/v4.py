"""V4 synthesis retains the frozen V3-A predictor, fitted on TRAIN only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS, PREDICTION_COLUMN, validate_submission
from transaction_forecasting.ubs.v3.model import V3Model

BASELINE_SHA = "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"
FINAL_MODEL = "V3-A"
MODEL_VERSION = "v4-synthesis-v3a-train-only-1"
RECIPE = {
    "model": FINAL_MODEL,
    "version": MODEL_VERSION,
    "baseline_sha": BASELINE_SHA,
    "candidate": "A",
    "fit_scope": "TRAIN",
    "numeric_weight": 0.75,
    "heuristic_weight": 0.25,
    "seed": 42,
    "cutoff": "2026-01-01",
    "horizon_days": 90,
    "class_order": list(LABELS),
    "valid_labels_used_for_parameter_fit": False,
    "promotion_context": "Previously reused VALID informs rejection; no new tuning",
    "validation_independent": False,
}


def fit_final_model(train, labels):
    """The caller supplies official TRAIN only; no alternative is selected here."""
    return V3Model().fit(train, labels)


def predict_probabilities(model, history):
    scores = model.predict_components(history)["A"]
    values = scores.to_numpy()
    if (
        not scores.index.is_unique
        or set(scores.index) != set(history.client_id)
        or list(scores.columns) != list(LABELS)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1)
    ):
        raise ValueError("Invalid final predictor probabilities")
    return scores


def submission_frame(scores, sample, history):
    if len(sample) != 1000 or sample.client_id.nunique() != 1000:
        raise ValueError("Submission requires exactly 1000 unique sample clients")
    if not scores.index.is_unique or set(scores.index) != set(sample.client_id):
        raise ValueError("Submission probability IDs differ from sample")
    result = sample[["client_id"]].copy()
    result[PREDICTION_COLUMN] = result.client_id.map(scores.idxmax(axis=1))
    validate_submission(result, sample, history)
    return result


def assert_disjoint(train, inference):
    if set(train.client_id) & set(inference.client_id):
        raise ValueError("Inference clients overlap TRAIN")


def target_series(labels):
    from transaction_forecasting.ubs.data import TARGET_COLUMN

    result = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    if not result.index.is_unique or not result.isin(LABELS).all():
        raise ValueError("Invalid client target")
    return pd.Series(result)
