from transaction_forecasting.data.contracts import mock_transactions
from transaction_forecasting.evaluation.temporal import temporal_split, train_validation_test_split


def test_temporal_split_keeps_test_after_train() -> None:
    train, test = temporal_split(mock_transactions(), test_fraction=0.3)
    assert train["timestamp"].max() < test["timestamp"].min()


def test_three_way_split_keeps_all_partitions_in_time_order() -> None:
    train, validation, test = train_validation_test_split(
        mock_transactions(), time_column="timestamp", validation_fraction=0.2, test_fraction=0.2
    )
    assert train["timestamp"].max() < validation["timestamp"].min() < test["timestamp"].min()
