import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.features import ClientFeatureBuilder
from transaction_forecasting.ubs.v2 import HistoryFeatureBuilder, calibrate_probabilities


def histories():
    rows = []
    for number, label in enumerate(LABELS):
        for copy in range(2):
            for date in ("2025-10-01", "2025-11-01", "2025-12-01"):
                rows.append(
                    {
                        "client_id": f"{number}_{copy}",
                        "timestamp": pd.Timestamp(date, tz="UTC"),
                        "amount": float(10 + number),
                        "currency": "chf",
                        "description": label,
                        "direction": "out",
                        "fee": 0.0,
                        "mcc": str(number),
                        "type": "card",
                    }
                )
    return pd.DataFrame(rows)


def test_history_projection_is_identical_for_different_training_labels():
    transactions = histories()
    labels = pd.DataFrame(
        {"client_id": transactions.client_id.unique(), TARGET_COLUMN: np.repeat(LABELS, 2)}
    )
    safe = HistoryFeatureBuilder().fit(transactions).transform(transactions)
    for targets in (
        labels,
        labels.assign(**{TARGET_COLUMN: labels[TARGET_COLUMN].iloc[::-1].to_numpy()}),
    ):
        full = ClientFeatureBuilder().fit(transactions, targets).transform(transactions)
        pd.testing.assert_frame_equal(safe, full[safe.columns])
    assert not any(c.startswith("family_") for c in safe)
    assert "client_id" not in safe and TARGET_COLUMN not in safe


def test_v2_fits_categories_only_on_train_and_rejects_future():
    transactions = histories()
    builder = HistoryFeatureBuilder().fit(transactions)
    modified = transactions.assign(mcc="unseen", client_id="new")
    transformed = builder.transform(modified)
    assert "mcc_unseen_count" not in transformed
    assert "unseen" not in builder.categorical_levels_["mcc"]
    for bad in (CUTOFF, pd.NaT):
        with pytest.raises(ValueError):
            builder.transform(transactions.assign(timestamp=bad))
        with pytest.raises(ValueError):
            HistoryFeatureBuilder().fit(transactions.assign(timestamp=bad))


def test_calibration_direction_label_order_and_invalid_inputs():
    raw = np.full((2, 8), 0.1)
    raw[:, LABELS.index("none")] = 0.3
    adjusted = calibrate_probabilities(raw)
    assert np.allclose(adjusted.sum(axis=1), 1)
    assert (adjusted[:, LABELS.index("none")] < raw[:, LABELS.index("none")]).all()
    np.testing.assert_allclose(calibrate_probabilities(raw, 1, 0), raw)
    for bad in (raw[:, :7], raw * 2, np.full((2, 8), np.nan)):
        with pytest.raises(ValueError):
            calibrate_probabilities(bad)
    with pytest.raises(ValueError):
        calibrate_probabilities(raw, temperature=0)
