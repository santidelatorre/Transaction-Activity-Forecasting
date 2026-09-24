"""Macro-F1 evaluation and compact error analysis for UBS V1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transaction_forecasting.evaluation.official import classification_metrics
from transaction_forecasting.ubs.data import LABELS


def evaluate_predictions(
    target: pd.Series, prediction: pd.Series | np.ndarray
) -> dict[str, object]:
    """Return official fixed-class metrics, aligning indexed predictions by client."""
    if not target.index.is_unique:
        raise ValueError("Target client IDs must be unique")
    if isinstance(prediction, pd.Series):
        if not prediction.index.is_unique:
            raise ValueError("Prediction client IDs must be unique")
        missing = target.index.difference(prediction.index)
        unexpected = prediction.index.difference(target.index)
        if len(missing) or len(unexpected):
            raise ValueError(
                f"Prediction client mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
            )
        aligned_prediction = prediction.reindex(target.index)
    else:
        aligned_prediction = pd.Series(np.asarray(prediction))
        if len(aligned_prediction) != len(target):
            raise ValueError("Target and prediction lengths must match")

    # Preserve the UBS runner's output keys while sharing the official metric core.
    report = classification_metrics(target, aligned_prediction)
    per_class = report["per_class"]
    return {
        "macro_f1": report["macro_f1"],
        "accuracy": report["accuracy"],
        "per_class": {
            label: {
                "precision": per_class[label]["precision"],
                "recall": per_class[label]["recall"],
                "f1-score": per_class[label]["f1"],
            }
            for label in LABELS
        },
        "confusion_matrix": [
            [report["confusion_true_by_predicted"][actual][guess] for guess in LABELS]
            for actual in LABELS
        ],
        "prediction_distribution": {label: per_class[label]["predicted_count"] for label in LABELS},
    }


def build_error_table(
    client_ids: pd.Index,
    target: pd.Series,
    model_prediction: np.ndarray,
    heuristic_prediction: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Expose false positives/negatives and disagreement wins client by client."""
    aligned_target = target.reindex(client_ids).to_numpy()
    confidence = probabilities.max(axis=1)
    table = pd.DataFrame(
        {
            "client_id": client_ids,
            "actual": aligned_target,
            "model_prediction": model_prediction,
            "heuristic_prediction": heuristic_prediction,
            "model_confidence": confidence,
        }
    )
    table["model_correct"] = table["actual"].eq(table["model_prediction"])
    table["heuristic_correct"] = table["actual"].eq(table["heuristic_prediction"])
    table["heuristic_only_win"] = table["heuristic_correct"] & ~table["model_correct"]
    table["model_only_win"] = table["model_correct"] & ~table["heuristic_correct"]
    return table
