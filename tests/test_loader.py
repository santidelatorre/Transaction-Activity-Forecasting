import pandas as pd
import pytest

from transaction_forecasting.config import DatasetConfig
from transaction_forecasting.data.loader import load_dataset, validate_frame


def test_csv_loader_applies_declared_contract(tmp_path) -> None:
    source = tmp_path / "events.csv"
    pd.DataFrame({"when": ["2026-01-01"], "value": [1.5]}).to_csv(source, index=False)
    frame = load_dataset(
        DatasetConfig(
            path=source,
            format="csv",
            required_columns=("when", "value"),
            column_types={"when": "datetime", "value": "numeric"},
        )
    )
    assert str(frame["when"].dtype) == "datetime64[ns, UTC]"


def test_validation_reports_missing_required_column() -> None:
    with pytest.raises(ValueError, match="required columns: amount"):
        validate_frame(pd.DataFrame({"id": [1]}), required_columns=("id", "amount"))
