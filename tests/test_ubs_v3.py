from __future__ import annotations

import pandas as pd

from transaction_forecasting.ubs.stream_oracle import TrainOnlyFamilyMapper, build_streams
from transaction_forecasting.ubs.v3 import apply_family_map, client_stream_features


def _tx() -> pd.DataFrame:
    rows = []
    for month in (7, 8, 9, 10, 11, 12):
        rows.append(
            {
                "client_id": "A",
                "timestamp": pd.Timestamp(f"2025-{month:02d}-05T10:00:00Z"),
                "amount": 9.99,
                "currency": "chf",
                "direction": "out",
                "type": "card_payment",
                "mcc": "5815",
                "description": "audio streaming",
                "fee": 0.0,
            }
        )
    rows.append(
        {
            "client_id": "B",
            "timestamp": pd.Timestamp("2025-09-01T10:00:00Z"),
            "amount": 20.0,
            "currency": "chf",
            "direction": "out",
            "type": "card_payment",
            "mcc": "5411",
            "description": "grocery store",
            "fee": 0.0,
        }
    )
    return pd.DataFrame(rows)


def test_client_stream_features_shape() -> None:
    train = _tx()
    labels = pd.DataFrame(
        {
            "client_id": ["A", "B"],
            "cutoff_date": "2026-01-01",
            "target_next_recurring_merchant": ["music", "none"],
        }
    )
    streams = build_streams(train)
    mapper = TrainOnlyFamilyMapper(min_support=1, min_lift=1.01).fit(streams, labels)
    features = client_stream_features(train, mapper, streams=streams)
    assert features.shape[0] == 2
    assert "stream_candidate_count" in features.columns
    mapped = apply_family_map(streams, mapper)
    assert "mapped_family" in mapped.columns
