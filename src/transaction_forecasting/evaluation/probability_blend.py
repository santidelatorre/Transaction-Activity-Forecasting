"""Strict ID/label alignment for a scalar probability blend; no fitting or labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transaction_forecasting.evaluation.official import LABELS


def validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and within [0, 1]")


def align_probabilities(frame: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    """Validate without renormalizing, filling missing values, or changing probabilities."""
    if (
        not clients.is_unique
        or clients.hasnans
        or not frame.index.is_unique
        or frame.index.hasnans
        or "" in clients
        or "" in frame.index
        or set(frame.index) != set(clients)
    ):
        raise ValueError("Probability client IDs must be unique and match exactly")
    if not frame.columns.is_unique or set(frame.columns) != set(LABELS):
        raise ValueError("Expected exactly the eight official probability columns")
    aligned = frame.loc[clients, list(LABELS)]
    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-10):
        raise ValueError("Probability rows must sum to one")
    return aligned


def blend_probabilities(v2: pd.DataFrame, identity: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Return (1-alpha)*V2 + alpha*A in V2 client order and official class order."""
    validate_alpha(alpha)
    left = align_probabilities(v2, v2.index)
    right = align_probabilities(identity, left.index)
    if alpha == 0:
        return left.copy()
    if alpha == 1:
        return right.copy()
    return (1 - alpha) * left + alpha * right


def select_alpha(scores: dict[float, float], tolerance: float = 0.0001) -> float:
    """Predeclared near-tie rule: pure model, then closest to 50/50, then lower alpha."""
    if not scores or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Expected scores and a finite nonnegative tie tolerance")
    for alpha, score in scores.items():
        validate_alpha(alpha)
        if not np.isfinite(score):
            raise ValueError("Scores must be finite")
    best = max(scores.values())
    tied = [alpha for alpha, score in scores.items() if best - score <= tolerance]
    return min(tied, key=lambda alpha: (alpha not in (0, 1), abs(alpha - 0.5), alpha))
