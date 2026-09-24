"""Focused checks for optional UBS client-history features."""

import pandas as pd
import pytest

from transaction_forecasting.features.v2 import build_client_v2_features


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["a", "a", "a", "b"],
            "timestamp": ["2025-10-01", "2025-11-01", "2025-12-01", "2025-12-30"],
            "amount": [10.0, 10.0, 11.0, 999.0],
            "currency": ["chf", "chf", "chf", "usd"],
            "description": ["gym", "gym", "gym", "other"],
        }
    )


def test_features_align_to_requested_clients_and_preserve_empty_history() -> None:
    result = build_client_v2_features(_history(), ["b", "missing", "a"])
    assert result.index.tolist() == ["b", "missing", "a"]
    assert result.loc["a", "v2_history_count"] == 3
    assert result.loc["missing", "v2_history_count"] == 0
    assert result.loc["missing", "v2_sparse_history"] == 1
    assert result.loc["a", "v2_stream_monthly_count"] == 1
    assert result.loc["a", "v2_description_top_share"] == 1
    assert result.loc["a", "v2_amount_median"] == 10


def test_future_transaction_rejected_even_for_unrequested_client() -> None:
    future = _history().copy()
    future.loc[len(future)] = ["z", "2026-01-01", 1.0, "chf", "x"]
    with pytest.raises(ValueError, match="cutoff"):
        build_client_v2_features(future, ["a"])


def test_groups_are_opt_in_and_unrelated_clients_do_not_change_features() -> None:
    base = _history()
    selected = build_client_v2_features(base, ["a"], groups=("frequency",))
    extra = pd.DataFrame(
        [
            {
                "client_id": "c",
                "timestamp": "2025-12-31",
                "amount": 10**9,
                "currency": "usd",
                "description": "gym",
            }
        ]
    )
    extended = build_client_v2_features(
        pd.concat([base, extra], ignore_index=True), ["a"], groups=("frequency",)
    )
    pd.testing.assert_frame_equal(selected, extended)
    assert selected.columns.tolist() == [
        "v2_count_7d",
        "v2_count_14d",
        "v2_count_30d",
        "v2_count_60d",
        "v2_count_90d",
        "v2_count_180d",
        "v2_recent_30d_share",
    ]


def test_amounts_do_not_mix_currencies() -> None:
    frame = _history()
    frame.loc[len(frame)] = ["a", "2025-12-15", 9000.0, "usd", "other"]
    features = build_client_v2_features(frame, ["a"], groups=("amount",))
    assert features.loc["a", "v2_amount_currency_count"] == 1
    assert features.loc["a", "v2_amount_median"] == 9000
