"""Format model predictions into the hackathon submission schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ID_COLUMN = "client_id"
PREDICTION_COLUMN = "predicted_next_recurring_merchant"
OUTPUT_COLUMNS = [ID_COLUMN, PREDICTION_COLUMN]
ALLOWED_LABELS = {
    "cloud",
    "gym",
    "insurance",
    "mobile",
    "music",
    "software",
    "streaming",
    "none",
}


def format_predictions(
    sample_submission: pd.DataFrame, raw_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Align predictions to the sample IDs and replace invalid/missing labels with ``none``.

    If raw predictions contain a client more than once, the first row is used.
    The sample submission must contain unique client IDs because the output must
    contain exactly one row per client.
    """
    for frame, label in (
        (sample_submission, "sample_submission"),
        (raw_predictions, "raw_predictions"),
    ):
        missing = {ID_COLUMN, PREDICTION_COLUMN} - set(frame.columns)
        if label == "sample_submission":
            missing.discard(PREDICTION_COLUMN)
        if missing:
            raise ValueError(f"{label} is missing required column(s): {sorted(missing)}")

    if sample_submission[ID_COLUMN].isna().any():
        raise ValueError("sample_submission contains a missing client_id")
    if sample_submission[ID_COLUMN].duplicated().any():
        raise ValueError("sample_submission must contain unique client_id values")

    predictions = raw_predictions[[ID_COLUMN, PREDICTION_COLUMN]].drop_duplicates(
        subset=ID_COLUMN, keep="first"
    )
    result = sample_submission[[ID_COLUMN]].merge(
        predictions, on=ID_COLUMN, how="left", sort=False, validate="one_to_one"
    )
    result[PREDICTION_COLUMN] = result[PREDICTION_COLUMN].where(
        result[PREDICTION_COLUMN].isin(ALLOWED_LABELS), "none"
    )
    return result[OUTPUT_COLUMNS]


def validate_submission(submission_path: str | Path, sample_submission: pd.DataFrame) -> None:
    """Assert that the exported CSV satisfies the submission contract."""
    result = pd.read_csv(submission_path)
    assert list(result.columns) == OUTPUT_COLUMNS, "Incorrect submission columns"
    assert len(result) == len(sample_submission), "Submission row count does not match sample"
    assert (
        result[ID_COLUMN].tolist() == sample_submission[ID_COLUMN].tolist()
    ), "Submission client IDs or order do not match sample_submission"
    assert not result.isna().any().any(), "Submission contains missing values"
    assert (
        result[PREDICTION_COLUMN].isin(ALLOWED_LABELS).all()
    ), "Submission contains labels outside the allowed list"
    print(f"Validation successful: {len(result)} rows, schema and labels are valid.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("my_raw_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("final_submission.csv"))
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    predictions = pd.read_csv(args.predictions)
    final = format_predictions(sample, predictions)
    final.to_csv(args.output, index=False)
    validate_submission(args.output, sample)


if __name__ == "__main__":
    main()
