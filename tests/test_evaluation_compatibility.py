"""Keep UBS runner interfaces compatible with the shared official checks."""

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.evaluation.official import (
    LABELS,
    PREDICTION,
    validate_submission,
)
from transaction_forecasting.ubs.data import validate_submission as validate_ubs_submission
from transaction_forecasting.ubs.evaluation import evaluate_predictions


def test_ubs_metrics_preserve_runner_keys_and_hand_calculated_values():
    target = pd.Series(LABELS, index=[f"c{i}" for i in range(8)])
    prediction = np.array(["none", *LABELS[1:]])
    metrics = evaluate_predictions(target, prediction)
    assert metrics["macro_f1"] == pytest.approx((6 + 2 / 3) / 8)
    assert metrics["accuracy"] == 7 / 8
    assert metrics["per_class"]["none"] == {
        "precision": 0.5,
        "recall": 1.0,
        "f1-score": 2 / 3,
    }
    assert metrics["confusion_matrix"][0][7] == 1
    assert metrics["confusion_matrix"][7][7] == 1
    assert metrics["prediction_distribution"]["none"] == 2
    assert metrics["prediction_distribution"]["cloud"] == 0


def test_ubs_absent_classes_and_invalid_vector_inputs():
    metrics = evaluate_predictions(pd.Series(["none"]), np.array(["none"]))
    assert metrics["macro_f1"] == 1 / 8
    assert metrics["per_class"]["cloud"]["f1-score"] == 0
    for truth, prediction in [([], []), (["none"], []), (["none"], ["invalid"])]:
        with pytest.raises(ValueError):
            evaluate_predictions(pd.Series(truth), np.array(prediction))


def test_shared_submission_checks_keep_distinct_order_and_return_contracts():
    sample = pd.DataFrame({"client_id": ["a", "b"], PREDICTION: ["none", "none"]})
    transactions = pd.DataFrame({"client_id": ["a", "a", "b"]})
    reversed_predictions = sample.iloc[::-1].reset_index(drop=True)
    original = reversed_predictions.copy(deep=True)
    normalized = validate_submission(reversed_predictions, sample)
    pd.testing.assert_frame_equal(normalized, sample)
    pd.testing.assert_frame_equal(reversed_predictions, original)
    with pytest.raises(ValueError, match="order"):
        validate_ubs_submission(reversed_predictions, sample, transactions)
    assert validate_ubs_submission(normalized, sample, transactions) is None
    with pytest.raises(ValueError, match="test"):
        validate_ubs_submission(normalized, sample, transactions.iloc[:1])


@pytest.mark.parametrize("invalid", ["duplicate", "blank", "unknown", "missing"])
def test_both_submission_entrypoints_reject_invalid_predictions(invalid):
    sample = pd.DataFrame({"client_id": ["a", "b"], PREDICTION: ["cloud", "none"]})
    predictions = sample.copy()
    if invalid == "duplicate":
        predictions.loc[1, "client_id"] = "a"
    elif invalid == "blank":
        predictions.loc[0, "client_id"] = ""
    elif invalid == "unknown":
        predictions.loc[0, PREDICTION] = "unknown"
    else:
        predictions = predictions.iloc[:1]
    with pytest.raises(ValueError):
        validate_submission(predictions, sample)
    with pytest.raises(ValueError):
        validate_ubs_submission(predictions, sample, sample[["client_id"]])
