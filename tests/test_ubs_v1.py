from __future__ import annotations

import pandas as pd
import pytest

from transaction_forecasting.ubs.data import (
    PREDICTION_COLUMN,
    validate_submission,
)
from transaction_forecasting.ubs.features import ClientFeatureBuilder, build_recurrence_streams


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["A", "A", "A", "B", "B"],
            "timestamp": pd.to_datetime(
                ["2025-10-01", "2025-11-01", "2025-12-01", "2025-11-15", "2025-12-15"],
                utc=True,
            ),
            "amount": [10.0, 10.0, 10.0, 20.0, 20.0],
            "currency": ["chf"] * 5,
            "description": ["cloud access"] * 3 + ["member plan"] * 2,
            "direction": ["out"] * 5,
            "fee": [0.0] * 5,
            "mcc": ["5734"] * 5,
            "type": ["card_payment"] * 5,
        }
    )


def test_recurrence_stream_captures_monthly_regular_amount() -> None:
    streams = build_recurrence_streams(_transactions())
    stream = streams.loc[streams["client_id"].eq("A")].iloc[0]
    assert stream["appearances"] == 3
    assert 30 <= stream["median_interval_days"] <= 31
    assert stream["amount_cv"] == 0
    assert stream["regularity"] > 0.95


def test_feature_builder_returns_one_row_per_client_without_labels() -> None:
    labels = pd.DataFrame(
        {
            "client_id": ["A", "B"],
            "target_next_recurring_merchant": ["cloud", "none"],
        }
    )
    builder = ClientFeatureBuilder().fit(_transactions(), labels)
    features = builder.transform(_transactions())
    assert features.index.tolist() == ["A", "B"]
    assert "target_next_recurring_merchant" not in features
    assert features.loc["A", "periodicity_monthly_count"] == 1
    assert "family_cloud_recurrence_score" in features
    assert features.loc["A", "best_recurrence_score"] > 0


def test_submission_validation_requires_sample_order() -> None:
    sample = pd.DataFrame({"client_id": ["A", "B"], PREDICTION_COLUMN: ["none", "none"]})
    test_transactions = pd.DataFrame({"client_id": ["A", "B"]})
    valid = sample.copy()
    validate_submission(valid, sample, test_transactions)
    with pytest.raises(ValueError, match="order"):
        validate_submission(valid.iloc[::-1].reset_index(drop=True), sample, test_transactions)
