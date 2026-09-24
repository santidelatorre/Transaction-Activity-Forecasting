"""Macro-F1 evaluation and compact error analysis for UBS V1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from transaction_forecasting.ubs.data import LABELS


def evaluate_predictions(
    target: pd.Series, prediction: pd.Series | np.ndarray
) -> dict[str, object]:
    """Return the official fixed-class metrics, aligning indexed predictions by client."""
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
        aligned_prediction = prediction.reindex(target.index).to_numpy()
    else:
        aligned_prediction = np.asarray(prediction)
        if len(aligned_prediction) != len(target):
            raise ValueError("Target and prediction lengths must match")
    unknown_target = set(target).difference(LABELS)
    unknown_prediction = set(aligned_prediction).difference(LABELS)
    if unknown_target or unknown_prediction:
        raise ValueError(
            f"Unknown labels: target={sorted(unknown_target)}, "
            f"prediction={sorted(unknown_prediction)}"
        )
    report = classification_report(
        target,
        aligned_prediction,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                target,
                aligned_prediction,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(target, aligned_prediction)),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score")
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(target, aligned_prediction, labels=LABELS).tolist(),
        "prediction_distribution": pd.Series(aligned_prediction)
        .value_counts()
        .reindex(LABELS, fill_value=0)
        .astype(int)
        .to_dict(),
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
