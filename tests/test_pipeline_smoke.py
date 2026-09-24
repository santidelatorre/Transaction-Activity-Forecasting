from transaction_forecasting.pipeline import run_smoke_pipeline


def test_smoke_pipeline_returns_business_output() -> None:
    predictions, metrics = run_smoke_pipeline()
    assert set(("customer_id", "entity", "expected_date", "confidence", "explanation")).issubset(
        predictions.columns
    )
    assert metrics["status"] == "smoke-pass"
