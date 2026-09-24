import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.v3.none_gate import (
    PROBABILITY_FEATURES,
    apply_none_gate,
    build_gate_features,
    gate_feature_names,
    make_gate_estimator,
)


def histories():
    rows = []
    for client, family in (("a", "music"), ("b", "unknown"), ("c", "streaming")):
        for month in (10, 11, 12):
            rows.append(
                {
                    "client_id": client,
                    "description": f"merchant {client}",
                    "timestamp": pd.Timestamp(f"2025-{month}-01", tz="UTC"),
                    "amount": 10.0,
                    "direction": "out",
                    "type": "card_payment",
                    "currency": "chf",
                    "mcc": "5812",
                    "fee": 0.0,
                    "family": family,
                }
            )
    return pd.DataFrame(rows)


def probabilities():
    values = np.full((3, len(LABELS)), 0.05)
    values[0, LABELS.index("music")] = 0.50
    values[0, LABELS.index("none")] = 0.20
    values[1, LABELS.index("cloud")] = 0.15
    values[1, LABELS.index("none")] = 0.55
    values[2, LABELS.index("streaming")] = 0.45
    values[2, LABELS.index("none")] = 0.25
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=pd.Index(["a", "b", "c"], name="client_id"), columns=LABELS)


def test_gate_features_are_target_free_finite_and_include_compact_evidence():
    mapped = histories()
    transactions = mapped.drop(columns="family")
    a = probabilities()
    v2 = probabilities().iloc[::-1].set_axis(a.index)
    features = build_gate_features(a, v2, transactions, mapped)
    assert features.index.equals(a.index)
    assert features.columns.is_unique
    assert np.isfinite(features.to_numpy()).all()
    assert "client_id" not in features and "target_next_recurring_merchant" not in features
    assert features.loc["a", "mapped_event_share"] == 1
    assert features.loc["b", "mapped_event_share"] == 0
    assert features.loc["a", "recurring_description_count"] == 1
    assert set(gate_feature_names("probability")) == set(PROBABILITY_FEATURES)
    assert set(gate_feature_names("compact")).issubset(features.columns)
    with pytest.raises(ValueError, match="Unknown"):
        gate_feature_names("huge")


def test_gate_preserves_v3_positive_family_and_only_changes_none_decision():
    scores = pd.Series([0.8, 0.2, 0.5], index=["a", "b", "c"])
    family = pd.Series(["music", "cloud", "streaming"], index=scores.index)
    prediction = apply_none_gate(scores, family, 0.5)
    assert prediction.tolist() == ["music", "none", "streaming"]
    with pytest.raises(ValueError, match="align"):
        apply_none_gate(scores, family.iloc[::-1], 0.5)
    with pytest.raises(ValueError, match="Invalid"):
        apply_none_gate(scores, family, 1.1)


@pytest.mark.parametrize("kind", ["logistic", "tree"])
def test_small_gate_estimators_return_binary_probabilities(kind):
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(80, 5)))
    target = pd.Series(np.tile([0, 1], 40))
    model = make_gate_estimator(kind, c=0.1).fit(features, target)
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (80, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    with pytest.raises(ValueError, match="Unknown"):
        make_gate_estimator("large")
