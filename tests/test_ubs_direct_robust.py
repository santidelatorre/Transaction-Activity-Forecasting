"""Isolation and interface checks for the direct-description experiment."""

import numpy as np
import pandas as pd
from scripts.experiments.v4_direct_robust_javi import make_folds

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.direct_robust import (
    DirectFeatures,
    LocalCorruptor,
    client_histories,
    normalized_view_weights,
    ordered_probabilities,
)


def histories():
    frame = pd.DataFrame(
        {
            "client_id": ["secret-a", "secret-a", "secret-b", "secret-b"],
            "description": ["alpha gym", "alpha gym", "beta music", "beta music"],
            "mcc": ["1", "1", "2", "2"],
        }
    )
    return client_histories(frame, pd.Index(["secret-a", "secret-b"]))


def test_vocabulary_and_idf_are_fit_only():
    fit, hold = histories()
    features = DirectFeatures().fit([fit])
    assert "desc:beta music" not in features.names("R0")
    assert "desc_mcc:beta music|2" not in features.names("R3")
    assert features.transform([hold], "R0").nnz == 0
    before = features.char_.idf_.copy()
    features.transform([hold], "R2")
    np.testing.assert_array_equal(before, features.char_.idf_)


def test_corruption_is_deterministic_and_clean_identical():
    source = histories()
    corruptor = LocalCorruptor().fit(source)
    clean = corruptor.transform(source, "clean")
    for original, view in zip(source, clean, strict=True):
        pd.testing.assert_frame_equal(original, view)
    a = corruptor.transform(source, "severe")
    b = corruptor.transform(source, "severe")
    for left, right in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(left, right)


def test_client_id_never_enters_feature_names():
    features = DirectFeatures().fit(histories())
    for representation in ("R0", "R1", "R2", "R3", "R4", "R5"):
        assert all("secret-" not in name for name in features.names(representation))


def test_each_client_has_unit_total_weight():
    clients = ["a", "a", "a", "a", "b", "b"]
    weights = normalized_view_weights(clients)
    assert np.isclose(weights[:4].sum(), 1)
    assert np.isclose(weights[4:].sum(), 1)


def test_probability_class_order():
    class Dummy:
        classes_ = np.array(["none", "cloud"])

        def predict_proba(self, matrix):
            return np.tile([0.7, 0.3], (matrix.shape[0], 1))

    matrix = DirectFeatures().fit(histories()).transform(histories(), "R0")
    result = ordered_probabilities(Dummy(), matrix)
    assert result.shape == (2, len(LABELS))
    np.testing.assert_allclose(result[:, LABELS.index("none")], 0.7)
    np.testing.assert_allclose(result[:, LABELS.index("cloud")], 0.3)
    assert result[:, LABELS.index("music")].sum() == 0


def test_client_folds_are_deterministic_and_disjoint():
    target = np.repeat(np.asarray(LABELS), 10)
    clients = np.array([f"client-{number}" for number in range(len(target))])
    labelled = pd.Series(target, index=clients)
    one = make_folds(labelled)
    two = make_folds(labelled)
    for (fit_a, hold_a), (fit_b, hold_b) in zip(one, two, strict=True):
        np.testing.assert_array_equal(fit_a, fit_b)
        np.testing.assert_array_equal(hold_a, hold_b)
        assert not set(clients[fit_a]).intersection(clients[hold_a])
