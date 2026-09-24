"""Temporal leakage, calendar boundaries, censoring, and ranking ties."""

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.next_date import (
    DAY,
    calendar_date,
    forecast,
    ranking_metrics,
    ranking_rows,
    temporal_metrics,
)


def days(*values):
    return pd.to_datetime(list(values), utc=True).astype("int64").to_numpy() // DAY


def test_calendar_end_of_month_and_quarterly():
    history = days("2024-11-30", "2024-12-31", "2025-01-31")
    cutoff = days("2025-02-01")[0]
    assert calendar_date(history, cutoff, 1) == days("2025-02-28")[0]
    assert calendar_date(history, cutoff, 3) == days("2025-04-30")[0]
    history = days("2024-11-30", "2024-12-30", "2025-01-30")
    assert calendar_date(history, cutoff, 1) == days("2025-02-28")[0]
    assert calendar_date(history, days("2025-03-01")[0], 1) == days("2025-03-30")[0]


def test_cutoff_is_strict_and_overdue_activity_is_separate():
    with pytest.raises(ValueError, match="strictly before"):
        forecast(np.array([1, 8, 15]), 15)
    predictions, occurs = forecast(np.array([1, 8, 15]), 100)
    assert predictions["median"] == 100
    assert predictions["automatic_active"] >= 100
    assert occurs["automatic"]
    assert not occurs["automatic_active"]


def test_targets_do_not_change_predictions():
    from scripts.experiments.ginestar_next_date import build_cases

    frame = pd.DataFrame(
        {
            "client_id": ["a"] * 4,
            "description": ["x"] * 4,
            "timestamp": pd.to_datetime(
                ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-05"], utc=True
            ),
        }
    )
    left, _ = build_cases(frame, ("2025-04-01",))
    frame.loc[3, "timestamp"] = pd.Timestamp("2025-05-20", tz="UTC")
    right, _ = build_cases(frame, ("2025-04-01",))
    assert left.predicted.equals(right.predicted)
    assert left.occurs.equals(right.occurs)
    assert not left.actual.equals(right.actual)


def test_censoring_and_uniform_tie_credit():
    frame = pd.DataFrame(
        {
            "cutoff": ["x"] * 3,
            "client_id": ["a"] * 3,
            "predicted": [10.0] * 3,
            "actual": [10.0, 20.0, np.nan],
            "occurs": [True] * 3,
            "periodicity": ["monthly"] * 3,
        }
    )
    summary = temporal_metrics(frame)
    assert summary["mae"] == 5
    assert summary["recall_90d"] == 1
    assert summary["precision_90d"] == 2 / 3
    rank = ranking_metrics(ranking_rows(frame))
    assert rank["top1"] == pytest.approx(1 / 3)
    assert rank["top2"] == pytest.approx(2 / 3)
    assert rank["chosen_censored"] == pytest.approx(1 / 3)


def test_simultaneous_true_events_and_none():
    frame = pd.DataFrame(
        {
            "cutoff": ["x"] * 2,
            "client_id": ["a"] * 2,
            "predicted": [10.0, 11.0],
            "actual": [10.0, 10.0],
            "occurs": [True] * 2,
            "periodicity": ["weekly"] * 2,
        }
    )
    assert ranking_metrics(ranking_rows(frame))["top1"] == 1
    frame["actual"] = np.nan
    frame["occurs"] = False
    result = ranking_metrics(ranking_rows(frame))
    assert result["none_recall"] == 1
    assert result["none_precision"] == 1
