from __future__ import annotations

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.models import (
    LightGBMClientModel,
    TemperatureCalibrator,
    XGBoostClientModel,
    blend_probabilities,
    predict_from_probabilities,
)


def _toy_features(n_clients: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    index = [f"C{i:03d}" for i in range(n_clients)]
    features = pd.DataFrame(
        {
            "n_transactions": rng.integers(10, 100, size=n_clients),
            "best_recurrence_score": rng.random(n_clients),
            "family_cloud_recurrence_score": rng.random(n_clients),
            "family_none_recurrence_score": rng.random(n_clients),
        },
        index=index,
    )
    labels = pd.Series(
        [LABELS[i % len(LABELS)] for i in range(n_clients)],
        index=index,
        name="target_next_recurring_merchant",
    )
    return features, labels


def test_lightgbm_and_xgboost_fit_predict_shape() -> None:
    features, labels = _toy_features()
    lightgbm = LightGBMClientModel(
        balanced=True, n_estimators=20, num_leaves=8, learning_rate=0.1
    ).fit(features, labels)
    xgboost = XGBoostClientModel(
        balanced=True, n_estimators=20, max_depth=3, learning_rate=0.1
    ).fit(features, labels)
    for model in (lightgbm, xgboost):
        probabilities = model.predict_proba(features)
        prediction = model.predict(features)
        assert probabilities.shape == (len(features), len(LABELS))
        assert prediction.shape == (len(features),)
        assert set(prediction).issubset(set(LABELS))


def test_temperature_calibrator_and_blend_preserve_probability_simplex() -> None:
    features, labels = _toy_features(24)
    model = LightGBMClientModel(n_estimators=15, num_leaves=8).fit(features, labels)
    raw = model.predict_proba(features)
    calibrator = TemperatureCalibrator().tune(
        raw,
        labels,
        temperatures=(0.75, 1.0, 1.5),
        none_biases=(-0.5, 0.0, 0.5),
    )
    calibrated = calibrator.apply(raw)
    blended = blend_probabilities([raw, calibrated], [0.4, 0.6])
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.allclose(blended.sum(axis=1), 1.0)
    assert predict_from_probabilities(calibrated).shape == (len(features),)
