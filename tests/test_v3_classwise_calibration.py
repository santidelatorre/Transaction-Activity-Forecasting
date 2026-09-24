import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.v3.calibration import (
    BOUND,
    apply_offsets,
    cross_fit_calibration,
    select_calibration,
    validate_probabilities,
)


def sample():
    rng = np.random.default_rng(123)
    ids = pd.Index([f"client_{i}" for i in range(96)], name="client_id")
    probabilities = pd.DataFrame(rng.dirichlet(np.ones(8), 96), index=ids, columns=LABELS)
    target = pd.Series(np.tile(LABELS, 12), index=ids)
    folds = pd.Series(np.repeat([1, 2, 3], 32), index=ids)
    return probabilities, target, folds


def test_probability_shape_order_normalization_and_valid_classes():
    probabilities, _, _ = sample()
    adjusted = apply_offsets(probabilities, np.linspace(-BOUND, BOUND, 8))
    assert adjusted.shape == probabilities.shape
    assert adjusted.index.equals(probabilities.index)
    assert list(adjusted.columns) == list(LABELS)
    assert set(adjusted.idxmax(axis=1)).issubset(LABELS)
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)
    for bad in (
        probabilities.iloc[:, :7],
        probabilities.iloc[:, ::-1],
        probabilities * 0.5,
        probabilities.assign(none=np.nan),
        probabilities.assign(none=-1),
        probabilities.assign(none=2),
        pd.concat([probabilities, probabilities.iloc[:1]]),
    ):
        with pytest.raises(ValueError):
            validate_probabilities(bad)
    for bias in (np.zeros(7), [np.nan] * 8, [BOUND + 0.01] * 8):
        with pytest.raises(ValueError):
            apply_offsets(probabilities, bias)


def test_stable_exact_identity_zeros_and_ties():
    probabilities, _, _ = sample()
    probabilities.iloc[0] = [1, 0, 0, 0, 0, 0, 0, 0]
    probabilities.iloc[1] = np.full(8, 0.125)
    adjusted = apply_offsets(probabilities, np.zeros(8))
    pd.testing.assert_frame_equal(adjusted, probabilities, check_exact=True)
    pd.testing.assert_series_equal(adjusted.idxmax(axis=1), probabilities.idxmax(axis=1))
    assert adjusted is not probabilities
    weighted = apply_offsets(probabilities, np.linspace(-BOUND, BOUND, 8))
    assert weighted.iloc[0, 1:].eq(0).all()
    expected = (probabilities.to_numpy() * np.exp(np.linspace(-BOUND, BOUND, 8))).argmax(axis=1)
    np.testing.assert_array_equal(weighted.to_numpy().argmax(axis=1), expected)


def test_deterministic_selection_and_invalid_targets():
    probabilities, target, folds = sample()
    first = select_calibration(probabilities, target, folds)
    assert first == select_calibration(probabilities, target, folds)
    assert np.abs(first["offsets"]).max() <= BOUND
    with pytest.raises(ValueError, match="Invalid target classes"):
        select_calibration(probabilities, target.where(target != "gym", "invalid"), folds)
    with pytest.raises(ValueError, match="align"):
        select_calibration(probabilities, target.iloc[::-1], folds)


def test_outer_labels_cannot_change_own_calibration(monkeypatch):
    from transaction_forecasting.ubs.v3 import calibration

    monkeypatch.setattr(calibration, "GRID", ((0, 0, 0), (0.5, 0.075, -0.075)))
    probabilities, target, folds = sample()
    first, fitted = cross_fit_calibration(probabilities, target, folds)
    altered = target.copy()
    altered.loc[folds.eq(1)] = "none"
    second, altered_fitted = cross_fit_calibration(probabilities, altered, folds)
    assert fitted[0] == altered_fitted[0]
    pd.testing.assert_frame_equal(first.loc[folds.eq(1)], second.loc[folds.eq(1)])
    assert set(first.idxmax(axis=1)).issubset(LABELS)
    with pytest.raises(ValueError, match="outer client folds"):
        cross_fit_calibration(probabilities, target, folds * 0)


def test_identity_selected_when_raw_is_perfect():
    probabilities, target, folds = sample()
    probabilities.loc[:, :] = 0
    for client, label in target.items():
        probabilities.at[client, label] = 1
    selected = select_calibration(probabilities, target, folds)
    assert selected["parameters"] == [0, 0, 0]
    assert selected["offsets"] == [0] * 8
