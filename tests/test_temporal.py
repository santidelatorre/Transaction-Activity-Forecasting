import pandas as pd
import pytest

from transaction_forecasting.data.contracts import mock_transactions
from transaction_forecasting.evaluation.temporal import temporal_split, train_validation_test_split


def test_temporal_split_keeps_test_after_train() -> None:
    train, test = temporal_split(mock_transactions(), test_fraction=0.3)
    assert train["timestamp"].max() < test["timestamp"].min()


def test_equal_timestamps_never_cross_boundary():
    frame = pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02", "2025-01-02"]})
    train, test = temporal_split(frame, test_fraction=1 / 3)
    assert len(train) == 1
    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()


def test_no_strict_split_and_missing_timestamp_rejected():
    for timestamps in (["2025-01-01"] * 3, ["2025-01-01", None]):
        with pytest.raises(ValueError):
            temporal_split(pd.DataFrame({"timestamp": timestamps}))


def test_three_way_split_keeps_all_partitions_in_time_order() -> None:
    train, validation, test = train_validation_test_split(
        mock_transactions(), time_column="timestamp", validation_fraction=0.2, test_fraction=0.2
    )
    assert train["timestamp"].max() < validation["timestamp"].min() < test["timestamp"].min()
