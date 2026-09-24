"""Small common metrics surface; expand after the official metric is known."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return MAE and RMSE with aligned indices."""
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")
    if aligned_actual.empty:
        raise ValueError("Metrics require at least one aligned observation")
    errors = pd.to_numeric(aligned_actual, errors="raise") - pd.to_numeric(
        aligned_predicted, errors="raise"
    )
    return {"mae": float(errors.abs().mean()), "rmse": float(np.sqrt((errors**2).mean()))}


def compare_models(
    results: list[dict[str, object]], metric: str = "mae"
) -> list[dict[str, object]]:
    """Sort experiment summaries by an ascending loss metric."""
    if not results:
        raise ValueError(f"Each result must contain numeric metric '{metric}'")

    def metric_value(result: dict[str, object]) -> float:
        value = result.get(metric)
        if not isinstance(value, int | float):
            raise ValueError(f"Each result must contain numeric metric '{metric}'")
        return float(value)

    return sorted(results, key=metric_value)
