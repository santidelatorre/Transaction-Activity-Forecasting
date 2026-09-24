"""Dataset-neutral cleaning helpers."""

from __future__ import annotations

import pandas as pd


def clean_frame(frame: pd.DataFrame, *, drop_duplicate_rows: bool = True) -> pd.DataFrame:
    """Return a copy with optional exact duplicate removal and no imputation rules."""
    result = frame.copy()
    if drop_duplicate_rows:
        result = result.drop_duplicates().reset_index(drop=True)
    return result
