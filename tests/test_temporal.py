from transaction_forecasting.data.contracts import mock_transactions
from transaction_forecasting.evaluation.temporal import temporal_split


def test_temporal_split_keeps_test_after_train() -> None:
    train, test = temporal_split(mock_transactions(), test_fraction=0.3)
    assert train["timestamp"].max() < test["timestamp"].min()
