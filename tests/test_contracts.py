import pandas as pd
import pytest

from transaction_forecasting.data.contracts import mock_transactions, validate_transactions


def test_mock_matches_canonical_contract() -> None:
    frame = mock_transactions()
    assert frame["timestamp"].dtype == pd.DatetimeTZDtype(tz="UTC")
    assert len(frame) == 7


def test_missing_required_column_is_explicit() -> None:
    with pytest.raises(ValueError, match="transaction_id"):
        validate_transactions(pd.DataFrame({"customer_id": ["c-1"]}))
