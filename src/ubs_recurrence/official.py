"""UBS 2026 client-level scoring and submission checks; no model assumptions.

Source: UBS-AG/Swiss-AI-Weeks, hackathons/2026/challenge.md.
The fixed eight-class macro average uses zero for undefined class F1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LABELS = (
    "cloud",
    "gym",
    "insurance",
    "mobile",
    "music",
    "software",
    "streaming",
    "none",
)
TARGET = "target_next_recurring_merchant"
PREDICTION = "predicted_next_recurring_merchant"
CUTOFF_DATE = "2026-01-01"
HORIZON_DAYS = 90
SUBMISSION_COLUMNS = ("client_id", PREDICTION)


def _check_ids(frame: pd.DataFrame, name: str) -> None:
    if "client_id" not in frame.columns:
        raise ValueError(f"{name}: missing client_id")
    if frame.empty:
        raise ValueError(f"{name}: no clients")
    ids = frame["client_id"]
    if not ids.map(
        lambda v: isinstance(v, str) and bool(v.strip()) and v == v.strip()
    ).all():
        raise ValueError(
            f"{name}: client IDs must be nonempty strings without outer whitespace"
        )
    if ids.duplicated().any():
        raise ValueError(f"{name}: duplicate client IDs")


def _check_classes(frame: pd.DataFrame, column: str, name: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"{name}: missing {column}")
    if not frame[column].isin(LABELS).all():
        raise ValueError(f"{name}: invalid labels; expected one of {LABELS}")


def validate_submission(
    predictions: pd.DataFrame, sample: pd.DataFrame
) -> pd.DataFrame:
    """Reject missing/extra/duplicate clients and return a copy in sample order.

    The two column names and their order follow the published sample contract.
    The sample's prediction values are placeholders, never ground truth.
    """
    for name, frame in (("predictions", predictions), ("sample", sample)):
        if tuple(frame.columns) != SUBMISSION_COLUMNS:
            raise ValueError(f"{name}: expected columns {SUBMISSION_COLUMNS}")
        _check_ids(frame, name)
    _check_classes(predictions, PREDICTION, "predictions")
    expected = set(sample["client_id"])
    actual = set(predictions["client_id"])
    if actual != expected:
        raise ValueError(
            f"Client set mismatch: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected"
        )
    return predictions.set_index("client_id").loc[sample["client_id"]].reset_index()


def score_predictions(
    labels: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, object]:
    """Score every labelled client exactly once, aligning by ID rather than row.

    No inner join or missing-prediction fallback: incomplete coverage is an error.
    This is local validation, not a claim about the hidden-test leaderboard.
    """
    if not labels.columns.is_unique:
        raise ValueError("labels: duplicate column names")
    _check_ids(labels, "labels")
    _check_classes(labels, TARGET, "labels")
    if "cutoff_date" not in labels or not labels["cutoff_date"].eq(CUTOFF_DATE).all():
        raise ValueError(f"labels: cutoff_date must be {CUTOFF_DATE}")
    sample = labels[["client_id"]].assign(**{PREDICTION: "none"})
    aligned = validate_submission(predictions, sample)
    return classification_metrics(labels[TARGET], aligned[PREDICTION])


def classification_metrics(
    target: pd.Series, prediction: pd.Series
) -> dict[str, object]:
    """Shared eight-class metric core for aligned vectors (not an ID join).

    Public CSV scoring checks coverage and aligns IDs before calling this core.
    UBS model callers supply vectors in matching client order.
    """
    truth = pd.Series(target).reset_index(drop=True)
    predicted = pd.Series(prediction).reset_index(drop=True)
    if len(truth) == 0 or len(truth) != len(predicted):
        raise ValueError("Target and prediction must have the same nonzero length")
    if not truth.isin(LABELS).all() or not predicted.isin(LABELS).all():
        raise ValueError("Target or prediction contains invalid labels")
    per_class = {}
    for label in LABELS:
        actual_mask = truth.eq(label)
        predicted_mask = predicted.eq(label)
        tp = int((actual_mask & predicted_mask).sum())
        fp = int((~actual_mask & predicted_mask).sum())
        fn = int((actual_mask & ~predicted_mask).sum())
        per_class[label] = {
            "support": int(actual_mask.sum()),
            "predicted_count": int(predicted_mask.sum()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    confusion = {
        actual: {
            guess: int((truth.eq(actual) & predicted.eq(guess)).sum())
            for guess in LABELS
        }
        for actual in LABELS
    }
    return {
        "clients": len(truth),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS),
        "accuracy": float(truth.eq(predicted).mean()),
        "label_order": list(LABELS),
        "undefined_class_f1": 0.0,
        "per_class": per_class,
        "confusion_true_by_predicted": confusion,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    # Preserve literal 'none' and string IDs; never infer numeric client IDs.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser(
        "score", help="Score validation predictions against labels"
    )
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    check = commands.add_parser(
        "validate", help="Check test submission against sample IDs"
    )
    check.add_argument("--sample", type=Path, required=True)
    check.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        predictions = _read_csv(args.predictions)
        if args.command == "score":
            result = score_predictions(_read_csv(args.labels), predictions)
        else:
            validated = validate_submission(predictions, _read_csv(args.sample))
            result = {"valid": True, "clients": len(validated)}
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
