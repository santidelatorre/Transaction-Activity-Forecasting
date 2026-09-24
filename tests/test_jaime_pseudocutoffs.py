"""Leakage and split regression tests for Jaime's pseudo-cutoff experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/experiments/jaime_pseudocutoffs.py"
SPEC = importlib.util.spec_from_file_location("jaime_pseudocutoffs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def transactions(dates, descriptions):
    return pd.DataFrame(
        {
            "client_id": "a",
            "timestamp": pd.to_datetime(dates, utc=True),
            "description": descriptions,
            "amount": range(10, 10 + len(dates)),
            "mcc": "1",
            "type": "card",
            "direction": "out",
        }
    )


def test_future_changes_labels_but_not_candidate_features():
    cutoff = pd.Timestamp("2025-06-01", tz="UTC")
    history = transactions(
        ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
        ["rent", "rent", "rent", "rent"],
    )
    early = transactions(["2025-06-10"], ["rent"])
    absent = transactions(["2025-06-10"], ["new stream"])
    config = MODULE.Config(horizon_days=90)
    first = pd.DataFrame(
        MODULE.candidate_rows_at_cutoff("a", history, cutoff, config, future=early)
    )
    second = pd.DataFrame(
        MODULE.candidate_rows_at_cutoff("a", history, cutoff, config, future=absent)
    )
    pd.testing.assert_frame_equal(first[MODULE.MODEL_FEATURES], second[MODULE.MODEL_FEATURES])
    assert first.loc[first.stream.eq("rent"), "is_winner"].item() == 1
    assert second.loc[second.stream.eq(MODULE.NONE_STREAM), "is_winner"].item() == 1


def test_events_after_horizon_are_never_accepted_as_labels():
    cutoff = pd.Timestamp("2025-06-01", tz="UTC")
    history = transactions(["2025-01-01", "2025-02-01"], ["rent", "rent"])
    too_late = transactions(["2025-09-01"], ["rent"])
    try:
        MODULE.candidate_rows_at_cutoff(
            "a", history, cutoff, MODULE.Config(horizon_days=90), future=too_late
        )
    except ValueError as error:
        assert "forecast window" in str(error)
    else:
        raise AssertionError("An event outside the horizon was accepted")


def test_client_folds_keep_every_pseudocutoff_together():
    rows = []
    for client in range(10):
        for cutoff in range(3):
            rows.append({"client_id": f"c{client}", "pseudo_id": f"c{client}-{cutoff}"})
    frame = pd.DataFrame(rows)
    seen = set()
    for fit, heldout in MODULE.client_folds(frame, folds=5, seed=42):
        fit_clients = set(frame.loc[fit, "client_id"])
        heldout_clients = set(frame.loc[heldout, "client_id"])
        assert not fit_clients & heldout_clients
        assert not seen & heldout_clients
        seen |= heldout_clients
    assert seen == set(frame.client_id)


def test_rule_rolls_over_overdue_stream_without_using_future():
    assert MODULE._next_due_days(recency=40, median_gap=30) == 20
    assert MODULE._next_due_days(recency=30, median_gap=30) == 0
