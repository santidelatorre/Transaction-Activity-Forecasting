"""Deterministic recurrence baseline and common prediction output."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from transaction_forecasting.data.contracts import validate_transactions


@dataclass(frozen=True)
class Prediction:
    customer_id: str
    entity: str
    expected_date: pd.Timestamp
    expected_amount: float
    confidence: float
    explanation: str
    model_name: str = "statistical-baseline"
    model_version: str = "0.1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def predict_next(frame: pd.DataFrame) -> pd.DataFrame:
    """Predict one next recurring activity per customer using robust rules."""
    transactions = validate_transactions(frame)
    predictions: list[Prediction] = []
    for customer_id, customer_rows in transactions.groupby("customer_id", sort=True):
        merchant_summary = (
            customer_rows.groupby("merchant", as_index=False)
            .agg(
                transaction_count=("transaction_id", "count"), latest_timestamp=("timestamp", "max")
            )
            .sort_values(["transaction_count", "latest_timestamp"], ascending=[False, False])
        )
        merchant = merchant_summary.iloc[0]["merchant"]
        latest = (
            customer_rows[customer_rows["merchant"] == merchant].sort_values("timestamp").iloc[-1]
        )
        merchant_rows = customer_rows[customer_rows["merchant"] == merchant]
        dates = merchant_rows["timestamp"].sort_values()
        intervals = dates.diff().dropna().dt.total_seconds().div(86400)
        interval = float(intervals.median()) if not intervals.empty else 30.0
        amount = float(merchant_rows["amount"].median())
        regularity = 1.0 if len(intervals) >= 2 and intervals.std(ddof=0) <= 3 else 0.6
        predictions.append(
            Prediction(
                customer_id=str(customer_id),
                entity=str(latest["merchant"]),
                expected_date=latest["timestamp"] + pd.Timedelta(days=interval),
                expected_amount=amount,
                confidence=regularity,
                explanation=(
                    f"Recurring {latest['merchant']} activity; "
                    f"typical interval: {interval:.0f} days"
                ),
            )
        )
    return pd.DataFrame([prediction.to_dict() for prediction in predictions])
