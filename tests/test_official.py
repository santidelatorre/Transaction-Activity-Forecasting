import pandas as pd
import pytest

from transaction_forecasting.evaluation.official import (
    CUTOFF_DATE,
    LABELS,
    PREDICTION,
    TARGET,
    main,
    score_predictions,
    validate_submission,
)


def example():
    labels = pd.DataFrame(
        {"client_id": [f"c{i}" for i in range(8)], "cutoff_date": CUTOFF_DATE, TARGET: LABELS}
    )
    predictions = labels[["client_id", TARGET]].rename(columns={TARGET: PREDICTION})
    return labels, predictions


def test_perfect_score_aligns_by_id_not_position():
    labels, predictions = example()
    labels.index = range(10, 18)
    result = score_predictions(labels, predictions.iloc[::-1])
    assert result["macro_f1"] == 1.0
    assert result["accuracy"] == 1.0


def test_hand_calculated_error_includes_none_and_all_eight_classes():
    labels, predictions = example()
    predictions.loc[0, PREDICTION] = "none"
    result = score_predictions(labels, predictions)
    # Six perfect classes, cloud=0, none: TP=1, FP=1, FN=0 => F1=2/3.
    assert result["macro_f1"] == pytest.approx((6 + 2 / 3) / 8)
    assert result["accuracy"] == 7 / 8
    assert result["per_class"]["none"]["precision"] == 0.5
    assert result["confusion_true_by_predicted"]["cloud"]["none"] == 1


def test_absent_classes_use_explicit_fixed_label_zero_convention():
    labels, predictions = example()
    assert score_predictions(labels.iloc[:1], predictions.iloc[:1])["macro_f1"] == 1 / 8


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "label", "columns", "id"])
def test_invalid_submission_rejected(kind):
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


def test_missing_prediction_is_not_silently_dropped_from_evaluation():
    labels, predictions = example()
    with pytest.raises(ValueError, match="missing"):
        score_predictions(labels, predictions.iloc[1:])


def test_sample_order_and_inputs_preserved():
    _, predictions = example()
    sample = predictions.iloc[::-1].copy()
    sample[PREDICTION] = ""  # Sample placeholders are not labels.
    original = predictions.copy(deep=True)
    result = validate_submission(predictions, sample)
    assert list(result["client_id"]) == list(sample["client_id"])
    pd.testing.assert_frame_equal(predictions, original)


def test_wrong_cutoff_and_duplicate_labels_rejected():
    labels, predictions = example()
    labels.loc[0, "cutoff_date"] = "2025-01-01"
    with pytest.raises(ValueError, match="cutoff_date"):
        score_predictions(labels, predictions)
    labels.loc[0, "cutoff_date"] = CUTOFF_DATE
    labels.loc[1, "client_id"] = "c0"
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions(labels, predictions)


def test_cli_reads_literal_none_and_checks_submission(tmp_path, capsys):
    labels, predictions = example()
    truth_path, pred_path = tmp_path / "labels.csv", tmp_path / "pred.csv"
    labels.to_csv(truth_path, index=False)
    predictions.to_csv(pred_path, index=False)
    assert main(["score", "--labels", str(truth_path), "--predictions", str(pred_path)]) == 0
    assert '"macro_f1": 1.0' in capsys.readouterr().out
    assert main(["validate", "--sample", str(pred_path), "--predictions", str(pred_path)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    predictions.loc[0, PREDICTION] = "INVALID"
    predictions.to_csv(pred_path, index=False)
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--sample", str(pred_path), "--predictions", str(pred_path)])
    assert exc.value.code == 2
