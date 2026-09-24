"""Leakage-safe feature transformation interface."""

from __future__ import annotations

from typing import Protocol, Self

import pandas as pd


class FeatureTransformer(Protocol):
    """Fit on train data only, then transform each split independently."""

    def fit(self, frame: pd.DataFrame) -> Self: ...

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...


class IdentityTransformer:
    """Placeholder transformer until dataset-specific features are agreed."""

    def fit(self, frame: pd.DataFrame) -> Self:
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()
