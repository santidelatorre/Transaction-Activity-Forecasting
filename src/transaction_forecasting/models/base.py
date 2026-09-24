"""Common model interface for comparable forecasting candidates."""

from __future__ import annotations

from typing import Protocol, Self

import pandas as pd


class ForecastModel(Protocol):
    name: str

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self: ...

    def predict(self, features: pd.DataFrame) -> pd.Series: ...
