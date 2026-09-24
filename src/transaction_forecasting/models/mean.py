"""Simple deterministic baseline usable with any numeric synthetic target."""

from __future__ import annotations

import pandas as pd


class MeanRegressor:
    """Predict the training-target mean; replace only after real target definition."""

    name = "mean_regressor"

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> MeanRegressor:
        del features
        numeric_target = pd.to_numeric(target, errors="raise")
        if numeric_target.empty:
            raise ValueError("Cannot fit MeanRegressor with an empty target")
        self._mean = float(numeric_target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self._mean is None:
            raise RuntimeError("MeanRegressor must be fitted before prediction")
        return pd.Series(self._mean, index=features.index, dtype=float, name="prediction")
