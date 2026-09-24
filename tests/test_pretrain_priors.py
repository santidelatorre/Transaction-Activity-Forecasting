from __future__ import annotations

import pandas as pd

from transaction_forecasting.ubs.pretrain_priors import (
    client_prior_features,
    fit_description_priors,
)


def _transactions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # Recurrent monthly cloud-like description for client A.
    for month in (7, 8, 9, 10, 11, 12):
        rows.append(
            {
                "client_id": "A",
                "timestamp": pd.Timestamp(f"2025-{month:02d}-05T10:00:00Z"),
                "amount": 12.0,
                "currency": "chf",
                "direction": "out",
                "type": "card_payment",
                "mcc": "5734",
                "description": "cloud access",
                "fee": 0.0,
            }
        )
    # One-off description for client B.
    rows.append(
        {
            "client_id": "B",
            "timestamp": pd.Timestamp("2025-09-01T10:00:00Z"),
            "amount": 50.0,
            "currency": "chf",
            "direction": "out",
            "type": "card_payment",
            "mcc": "5411",
            "description": "grocery store",
            "fee": 0.0,
        }
    )
    # Second client with same recurrent description (population evidence).
    for month in (8, 9, 10, 11, 12):
        rows.append(
            {
                "client_id": "C",
                "timestamp": pd.Timestamp(f"2025-{month:02d}-06T10:00:00Z"),
                "amount": 12.5,
                "currency": "chf",
                "direction": "out",
                "type": "card_payment",
                "mcc": "5734",
                "description": "cloud access",
                "fee": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_description_priors_mark_recurrent_descriptions() -> None:
    priors = fit_description_priors(_transactions(), alpha=2.0)
    assert "cloud access" in priors.table.index
    assert float(priors.table.loc["cloud access", "p_recurrent"]) > float(
        priors.table.loc["grocery store", "p_recurrent"]
    )
    assert priors.table.loc["cloud access", "typical_period_days"] > 0


def test_client_prior_features_are_one_row_per_client() -> None:
    frame = _transactions()
    priors = fit_description_priors(frame, alpha=2.0)
    features = client_prior_features(frame, priors)
    assert features.index.tolist() == ["A", "B", "C"]
    assert features.columns.str.startswith("pretrain_").all()
    assert features.loc["A", "pretrain_stream_count"] >= 1
