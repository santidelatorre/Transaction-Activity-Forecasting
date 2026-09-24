import pandas as pd
import pytest

from transaction_forecasting.ubs.data import CUTOFF, TARGET_COLUMN
from transaction_forecasting.ubs.stream_oracle import (
    TrainOnlyFamilyMapper,
    build_streams,
    project_next_date,
)


def transactions(client="client-a", description="cloud access"):
    return pd.DataFrame(
        {
            "client_id": [client] * 3,
            "description": [description] * 3,
            "timestamp": pd.to_datetime(["2025-10-01", "2025-11-01", "2025-12-01"], utc=True),
            "amount": [10.0, 11.0, 12.0],
            "mcc": ["5734"] * 3,
            "type": ["card_payment"] * 3,
            "direction": ["out"] * 3,
            "currency": ["chf"] * 3,
        }
    )


def test_stream_cutoff_is_strict():
    with pytest.raises(ValueError, match="at or after cutoff"):
        build_streams(transactions().assign(timestamp=CUTOFF))


def test_gap_statistics_are_ordered_and_exact():
    stream = build_streams(transactions()).iloc[0]
    assert stream["appearances"] == 3
    assert stream["gaps_days"] == (31.0, 30.0)
    assert stream["gap_median_days"] == 30.5
    assert stream["gap_robust_mean_days"] == 30.5
    assert stream["penultimate_timestamp"] == pd.Timestamp("2025-11-01", tz="UTC")


def test_projection_advances_over_overdue_cycles():
    projected = project_next_date(pd.Timestamp("2025-10-01", tz="UTC"), 30.0)
    assert projected == pd.Timestamp("2026-01-29", tz="UTC")
    assert projected >= CUTOFF


def test_family_mapping_is_train_only_and_rejects_fit_clients_at_transform():
    train = pd.concat(
        [
            transactions("train-1"),
            transactions("train-2"),
            transactions("train-3", "phone contract"),
        ],
        ignore_index=True,
    )
    labels = pd.DataFrame(
        {
            "client_id": ["train-1", "train-2", "train-3"],
            TARGET_COLUMN: ["cloud", "cloud", "mobile"],
        }
    )
    mapper = TrainOnlyFamilyMapper().fit(build_streams(train), labels)
    assert mapper.mapping_["cloud access"] == "cloud"
    valid_streams = build_streams(transactions("valid-1"))
    mapped = mapper.transform(valid_streams)
    assert mapped["mapped_family"].dropna().unique().tolist() == ["cloud"]
    assert mapper.fit_clients_ == {"train-1", "train-2", "train-3"}
    with pytest.raises(ValueError, match="disjoint"):
        mapper.transform(build_streams(transactions("train-1")))
