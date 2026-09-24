import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.v3.disagreement_gate import (
    cross_fitted_gate_predictions,
    fixed_confidence_route,
    gate_features,
    gate_model_factories,
    routing_diagnostics,
)


def probabilities(predictions, confidence):
    values = np.full((len(predictions), len(LABELS)), (1 - confidence) / (len(LABELS) - 1))
    for row, label in enumerate(predictions):
        values[row, LABELS.index(label)] = confidence
    return pd.DataFrame(values, index=predictions.index, columns=LABELS)


def test_probability_contract_and_fixed_confidence_route():
    index = pd.Index(["a", "b", "c"], name="client_id")
    v2_prediction = pd.Series(["cloud", "gym", "music"], index=index)
    v3a_prediction = pd.Series(["cloud", "mobile", "streaming"], index=index)
    v2 = probabilities(v2_prediction, 0.8)
    v3a = probabilities(v3a_prediction, 0.7)
    v3a.loc["c"] = probabilities(pd.Series(["streaming"], index=["c"]), 0.9).iloc[0]

    result = fixed_confidence_route(v2, v3a)

    assert result.tolist() == ["cloud", "gym", "streaming"]
    features = gate_features(v2, v3a)
    assert features.shape == (3, 27)
    assert not features.isna().any().any()
    with pytest.raises(ValueError, match="official label order"):
        gate_features(v2.iloc[:, ::-1], v3a)


def test_learned_gate_has_a_second_complete_crossfit_layer():
    clients, truth, v2_guess, v3a_guess, v2_confidence, v3a_confidence = [], [], [], [], [], []
    for label_number, label in enumerate(LABELS):
        wrong = LABELS[(label_number + 1) % len(LABELS)]
        for number in range(10):
            clients.append(f"{label}_{number}")
            truth.append(label)
            if number % 2 == 0:
                v2_guess.append(label)
                v3a_guess.append(wrong)
                v2_confidence.append(0.90)
                v3a_confidence.append(0.65)
            else:
                v2_guess.append(wrong)
                v3a_guess.append(label)
                v2_confidence.append(0.65)
                v3a_confidence.append(0.90)
    index = pd.Index(clients, name="client_id")
    target = pd.Series(truth, index=index)
    v2 = pd.concat(
        [
            probabilities(pd.Series([guess], index=[client]), confidence)
            for client, guess, confidence in zip(index, v2_guess, v2_confidence, strict=True)
        ]
    )
    v3a = pd.concat(
        [
            probabilities(pd.Series([guess], index=[client]), confidence)
            for client, guess, confidence in zip(index, v3a_guess, v3a_confidence, strict=True)
        ]
    )

    prediction, fold = cross_fitted_gate_predictions(
        target, v2, v3a, gate_model_factories()["logistic"]
    )

    assert prediction.index.equals(target.index)
    assert prediction.eq(target).all()
    assert fold.value_counts().sort_index().tolist() == [16] * 5


def test_routing_diagnostics_counts_decisive_and_both_wrong_changes():
    index = pd.Index(["v2", "v3a", "wrong", "agree"], name="client_id")
    target = pd.Series(["cloud", "gym", "insurance", "mobile"], index=index)
    v2 = pd.Series(["cloud", "music", "cloud", "mobile"], index=index)
    v3a = pd.Series(["software", "gym", "streaming", "mobile"], index=index)
    routed = pd.Series(["cloud", "music", "cloud", "mobile"], index=index)

    result = routing_diagnostics(target, v2, v3a, routed)["overall"]

    assert result["disagreements"] == 3
    assert result["v2_correct_only"] == 1
    assert result["v3a_correct_only"] == 1
    assert result["both_wrong"] == 1
    assert result["routed_to_v2"] == 3
    assert result["changed_vs_v3a"] == 3
    assert result["correct_changes"] == 1
    assert result["incorrect_changes"] == 1
    assert result["both_wrong_changes"] == 1
    assert result["routing_accuracy_on_disagreements"] == pytest.approx(1 / 3)
