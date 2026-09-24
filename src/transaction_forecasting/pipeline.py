"""Small end-to-end path kept runnable before the challenge dataset arrives."""

from __future__ import annotations

import pandas as pd

from transaction_forecasting.data.contracts import mock_transactions
from transaction_forecasting.models.baseline import predict_next


def run_smoke_pipeline() -> tuple[pd.DataFrame, dict[str, object]]:
    """Run mock data through validation, prediction, and a minimal evaluation."""
    transactions = mock_transactions()
    predictions = predict_next(transactions)
    metrics = {
        "rows_in": len(transactions),
        "customers_predicted": int(predictions["customer_id"].nunique()),
        "model": "statistical-baseline",
        "status": "smoke-pass",
    }
    return predictions, metrics
