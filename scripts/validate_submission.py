"""Validate the final UBS submission and print an upload-ready quality report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ubs_recurrence.official import LABELS, PREDICTION, SUBMISSION_COLUMNS
from ubs_recurrence.official import validate_submission as validate_contract


def inspect_submission(
    submission_path: Path, sample_path: Path
) -> tuple[dict[str, object], bool]:
    """Return explicit contract checks without masking malformed CSV values."""
    submission = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)

    submission_ids = (
        submission["client_id"] if "client_id" in submission else pd.Series(dtype=str)
    )
    sample_ids = sample["client_id"] if "client_id" in sample else pd.Series(dtype=str)
    predictions = (
        submission[PREDICTION] if PREDICTION in submission else pd.Series(dtype=str)
    )
    missing = set(sample_ids).difference(submission_ids)
    extra = set(submission_ids).difference(sample_ids)
    invalid = predictions[~predictions.isin(LABELS)]

    checks = {
        "rows": len(submission),
        "unique_client_id": int(submission_ids.nunique()),
        "expected_client_id": len(sample),
        "missing_ids": len(missing),
        "extra_ids": len(extra),
        "duplicate_ids": int(submission_ids.duplicated().sum()),
        "invalid_labels": len(invalid),
        "nan_predictions": int(predictions.isna().sum()),
        "blank_predictions": int(predictions.eq("").sum()),
        "column_schema": tuple(submission.columns) == SUBMISSION_COLUMNS,
        "client_alignment": submission_ids.equals(sample_ids),
        "distribution": {label: int(predictions.eq(label).sum()) for label in LABELS},
    }
    try:
        validate_contract(submission, sample)
        contract_valid = True
    except ValueError:
        contract_valid = False
    valid = bool(
        contract_valid
        and checks["rows"] == checks["expected_client_id"]
        and checks["client_alignment"]
        and checks["blank_predictions"] == 0
    )
    return checks, valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("outputs/predictions/submission_stream_identity_clean.csv"),
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/raw/ubs_2026/sample_submission.csv"),
    )
    args = parser.parse_args(argv)
    try:
        checks, valid = inspect_submission(args.submission, args.sample)
    except (OSError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Submission validation failed: {exc}\n")

    def status(condition: bool) -> str:
        return "PASS" if condition else "FAIL"

    print(f"Submission: {args.submission.as_posix()}")
    print(f"Rows: {checks['rows']}")
    print(f"Unique client_id: {checks['unique_client_id']}")
    print(f"Expected client_id: {checks['expected_client_id']}")
    print(f"Missing IDs: {checks['missing_ids']}")
    print(f"Extra IDs: {checks['extra_ids']}")
    print(f"Duplicate IDs: {checks['duplicate_ids']}")
    print(f"Invalid labels: {checks['invalid_labels']}")
    print(f"NaN predictions: {checks['nan_predictions']}")
    print(f"Blank predictions: {checks['blank_predictions']}")
    print(f"Column schema: {status(bool(checks['column_schema']))}")
    print(f"Client alignment: {status(bool(checks['client_alignment']))}")
    print(f"Overall submission validation: {status(valid)}")
    print("Prediction distribution:")
    for label, count in checks["distribution"].items():
        print(f"{label}: {count}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
