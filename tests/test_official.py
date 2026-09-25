from io import StringIO

import pandas as pd
import pytest

from ubs_recurrence.official import (
    CUTOFF_DATE,
    LABELS,
    PREDICTION,
    TARGET,
    main,
    score_predictions,
    validate_submission,
)


def example() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(8)],
            "cutoff_date": CUTOFF_DATE,
            TARGET: LABELS,
        }
    )
    predictions = labels[["client_id", TARGET]].rename(columns={TARGET: PREDICTION})
    return labels, predictions


def test_perfect_score_aligns_by_id_and_uses_all_classes() -> None:
    labels, predictions = example()
    result = score_predictions(labels, predictions.iloc[::-1])
    assert result["macro_f1"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["label_order"] == list(LABELS)


def test_absent_classes_have_fixed_zero_f1() -> None:
    labels, predictions = example()
    assert score_predictions(labels.iloc[:1], predictions.iloc[:1])["macro_f1"] == 1 / 8


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "duplicate", "label", "columns", "id"]
)
def test_invalid_submission_is_rejected(kind: str) -> None:
    _, predictions = example()
    sample = predictions.copy()
    if kind == "missing":
        predictions = predictions.iloc[:-1]
    elif kind == "extra":
        predictions.loc[8] = ["unexpected", "none"]
    elif kind == "duplicate":
        predictions.loc[1, "client_id"] = "c0"
    elif kind == "label":
        predictions.loc[0, PREDICTION] = "mortgage"
    elif kind == "columns":
        predictions["confidence"] = 0.8
    else:
        predictions.loc[0, "client_id"] = None
    with pytest.raises(ValueError):
        validate_submission(predictions, sample)


def test_sample_order_and_inputs_are_preserved() -> None:
    _, predictions = example()
    sample = predictions.iloc[::-1].copy()
    sample[PREDICTION] = ""
    original = predictions.copy(deep=True)
    result = validate_submission(predictions, sample)
    assert list(result["client_id"]) == list(sample["client_id"])
    pd.testing.assert_frame_equal(predictions, original)


def test_wrong_cutoff_and_duplicate_labels_are_rejected() -> None:
    labels, predictions = example()
    labels.loc[0, "cutoff_date"] = "2025-01-01"
    with pytest.raises(ValueError, match="cutoff_date"):
        score_predictions(labels, predictions)
    labels.loc[0, "cutoff_date"] = CUTOFF_DATE
    labels.loc[1, "client_id"] = "c0"
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions(labels, predictions)


def test_observed_official_csv_line_endings() -> None:
    csv_text = (
        "client_id,cutoff_date,target_next_recurring_merchant\r\r\n"
        "fixture-a,2026-01-01,cloud\r\r\n"
        "fixture-b,2026-01-01,none\r\r\n"
    )
    labels = pd.read_csv(StringIO(csv_text), dtype=str, keep_default_na=False)
    predictions = pd.DataFrame(
        {"client_id": ["fixture-b", "fixture-a"], PREDICTION: ["none", "cloud"]}
    )
    result = score_predictions(labels, predictions)
    assert result["clients"] == 2
    assert result["macro_f1"] == 2 / 8


def test_official_cli_validates_csv(tmp_path, capsys) -> None:
    labels, predictions = example()
    truth_path, prediction_path = tmp_path / "labels.csv", tmp_path / "predictions.csv"
    labels.to_csv(truth_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    assert (
        main(
            [
                "score",
                "--labels",
                str(truth_path),
                "--predictions",
                str(prediction_path),
            ]
        )
        == 0
    )
    assert '"macro_f1": 1.0' in capsys.readouterr().out
    assert (
        main(
            [
                "validate",
                "--sample",
                str(prediction_path),
                "--predictions",
                str(prediction_path),
            ]
        )
        == 0
    )
    assert '"valid": true' in capsys.readouterr().out
