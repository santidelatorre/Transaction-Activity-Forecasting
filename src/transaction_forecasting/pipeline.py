"""Configurable, leakage-safe pipeline with a runnable synthetic mode."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from transaction_forecasting.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    SplitConfig,
    load_config,
)
from transaction_forecasting.data.contracts import mock_transactions
from transaction_forecasting.data.loader import load_dataset
from transaction_forecasting.evaluation.experiments import save_experiment
from transaction_forecasting.evaluation.metrics import regression_metrics
from transaction_forecasting.evaluation.temporal import train_validation_test_split
from transaction_forecasting.features.base import FeatureTransformer, IdentityTransformer
from transaction_forecasting.models.base import ForecastModel
from transaction_forecasting.models.baseline import predict_next
from transaction_forecasting.models.mean import MeanRegressor
from transaction_forecasting.preprocessing.cleaning import clean_frame
from transaction_forecasting.utils.logging import configure_logging
from transaction_forecasting.utils.random import set_random_seed

LOGGER = logging.getLogger(__name__)


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


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    """Load configured real data and execute the generic pipeline."""
    config.validate_for_run()
    return run_frame(load_dataset(config.dataset), config)


def run_frame(
    frame: pd.DataFrame,
    config: PipelineConfig,
    *,
    transformer: FeatureTransformer | None = None,
    model: ForecastModel | None = None,
) -> dict[str, object]:
    """Run a frame through clean, split, train, evaluate, and persist stages."""
    set_random_seed(config.random_seed)
    cleaned = clean_frame(frame, drop_duplicate_rows=config.drop_duplicate_rows)
    train, validation, test = train_validation_test_split(
        cleaned,
        time_column=config.split.time_column,
        validation_fraction=config.split.validation_fraction,
        test_fraction=config.split.test_fraction,
        embargo_rows=config.split.embargo_rows,
    )
    target_column = config.model.target_column
    if target_column not in cleaned.columns:
        raise ValueError(f"Missing configured target column: {target_column}")
    transformer = transformer or IdentityTransformer()
    transformer.fit(train)
    train_features = transformer.transform(train).drop(columns=[target_column])
    validation_features = transformer.transform(validation).drop(columns=[target_column])
    test_features = transformer.transform(test).drop(columns=[target_column])
    fitted_model = model or _build_model(config.model)
    fitted_model.fit(train_features, train[target_column])
    validation_metrics = regression_metrics(
        validation[target_column], fitted_model.predict(validation_features)
    )
    test_metrics = regression_metrics(test[target_column], fitted_model.predict(test_features))
    result: dict[str, object] = {
        "model": fitted_model.name,
        "seed": config.random_seed,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "validation": validation_metrics,
        "test": test_metrics,
    }
    save_experiment(config.experiment.results_path, result)
    LOGGER.info("Finished %s: test MAE %.4f", fitted_model.name, test_metrics["mae"])
    return result


def synthetic_config(
    results_path: str | Path = "outputs/metrics/experiments.jsonl",
) -> PipelineConfig:
    """Return an isolated test configuration; its fields are not dataset assumptions."""
    return PipelineConfig(
        dataset=DatasetConfig(required_columns=("event_time", "target")),
        split=SplitConfig(time_column="event_time", validation_fraction=0.2, test_fraction=0.2),
        model=ModelConfig(target_column="target"),
        experiment=ExperimentConfig(results_path=Path(results_path)),
    )


def synthetic_frame() -> pd.DataFrame:
    """Create a deterministic generic frame to smoke-test all pipeline stages."""
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC"),
            "signal": list(range(20)),
            "target": [float(value % 5) for value in range(20)],
        }
    )


def _build_model(config: ModelConfig) -> ForecastModel:
    if config.baseline == "mean_regressor":
        return MeanRegressor()
    raise ValueError(f"Unsupported configured baseline: {config.baseline}")


def main() -> None:
    """Run the dataset-configured pipeline or its generic synthetic smoke path."""
    parser = argparse.ArgumentParser(description="Run the transaction forecasting pipeline")
    parser.add_argument("--config", default="configs/default.toml", help="Path to TOML config")
    parser.add_argument(
        "--synthetic", action="store_true", help="Run the generic synthetic example"
    )
    parser.add_argument("--results-path", help="Override experiment JSONL output path")
    arguments = parser.parse_args()
    configure_logging()
    if arguments.synthetic:
        output_path = arguments.results_path or "outputs/metrics/experiments.jsonl"
        result = run_frame(synthetic_frame(), synthetic_config(output_path))
    else:
        config = load_config(arguments.config)
        result = run_pipeline(config)
    print(result)


if __name__ == "__main__":
    main()
