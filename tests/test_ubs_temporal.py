"""Regression tests for temporal evidence and client-isolated evaluation."""

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.features import ClientFeatureBuilder
from transaction_forecasting.ubs.temporal_experiment import (
    _calibrate,
    client_partitions,
    historical_date_diagnostics,
)
from transaction_forecasting.ubs.temporal_features import (
    apply_temporal_blocks,
    temporal_streams,
)


def events(dates):
    return pd.DataFrame(
        {
            "client_id": "a",
            "description": "opaque",
            "timestamp": pd.to_datetime(dates, utc=True),
        }
    )


def test_intervals_and_duplicate_timestamps():
    frame = events(["2025-12-01", "2025-12-08", "2025-12-15", "2025-12-15"])
    result = temporal_streams(frame).iloc[0]
    assert result.transaction_count == 4
    assert result.unique_event_count == 3
    assert result.duplicate_timestamp_count == 1
    assert result.number_of_intervals == 2
    for statistic in ("mean", "median", "min", "max"):
        assert result[f"interval_{statistic}"] == 7
    assert result.interval_std == result.interval_mad == result.interval_cv == 0
    assert result.cycle_7_closeness == 1
    assert result.weekday_concentration == pytest.approx(1)
    assert result.regularity_confidence == 0.5
    assert result.overdue_days == 10
    pd.testing.assert_frame_equal(temporal_streams(frame), temporal_streams(frame.iloc[::-1]))


@pytest.mark.parametrize("dates", [["2025-12-01"], ["2025-12-01", "2025-12-08"]])
def test_sparse_history_does_not_claim_regular(dates):
    result = temporal_streams(events(dates)).iloc[0]
    assert np.isnan(result.interval_std)
    assert np.isnan(result.interval_mad)
    assert np.isnan(result.cycle_7_closeness)
    assert result.regularity_confidence == 0
    if len(dates) == 1:
        assert pd.isna(result.expected_next_date)
        assert result.horizon_evidence == 0
        assert not result.expected_date_known


def test_month_end_calendar_and_horizon_boundary():
    frame = events(["2025-10-31", "2025-11-30", "2025-12-31"])
    result = temporal_streams(frame).iloc[0]
    assert result.calendar_month_used
    assert result.expected_next_date == pd.Timestamp("2026-01-31", tz="UTC")
    assert result.days_until_expected_next == 30
    assert result.expected_next_within_horizon
    # The future observation interval is [cutoff, cutoff + horizon).
    short = temporal_streams(frame, horizon=30).iloc[0]
    assert not short.expected_next_within_horizon
    assert short.horizon_evidence == 0


@pytest.mark.parametrize("bad", ["2026-01-01", "2026-02-01", None])
def test_future_and_missing_timestamps_rejected(bad):
    with pytest.raises(ValueError):
        temporal_streams(events(["2025-12-01", bad]))


def test_irregular_intervals_and_activity_denominators():
    result = temporal_streams(events(["2025-01-01", "2025-01-02", "2025-12-31"])).iloc[0]
    assert result.interval_median == 182
    assert result.interval_mad == 181
    assert result.interval_cv > 0.9
    assert 0 < result.regularity_confidence < 0.3
    assert result.events_90d == 1
    singleton = temporal_streams(events(["2025-12-31T23:59:59Z"])).iloc[0]
    assert np.isfinite(singleton.recent_to_history_rate)


def test_disabled_blocks_are_exact_v1_and_no_mutation():
    streams = temporal_streams(events(["2025-12-01", "2025-12-08", "2025-12-15"]))
    base = pd.DataFrame(
        {f"family_{label}_recurrence_score": [2.0, 3.0] for label in LABELS}, index=["a", "b"]
    )
    original = base.copy(deep=True)
    mapping = pd.DataFrame(1.0, index=["opaque"], columns=LABELS)
    saved_mapping = mapping.copy(deep=True)
    pd.testing.assert_frame_equal(apply_temporal_blocks(base, streams, mapping), base)
    result = apply_temporal_blocks(base, streams, mapping, ("horizon",))
    assert result.loc["a", "family_cloud_recurrence_score"] < 2
    assert result.loc["a", "family_none_recurrence_score"] == 2
    pd.testing.assert_series_equal(result.loc["b"], base.loc["b"])
    pd.testing.assert_frame_equal(base, original)
    pd.testing.assert_frame_equal(mapping, saved_mapping)
    with pytest.raises(ValueError):
        apply_temporal_blocks(base, streams, mapping, ("text",))


def test_client_folds_are_disjoint_and_cover_exactly_once():
    target = pd.Series(np.repeat(LABELS, 20), index=[f"id{i}" for i in range(160)])
    held = []
    for fit, calibration, outer, test in client_partitions(target, 5, 42, 0.2):
        assert not set(fit) & set(calibration)
        assert set(fit) | set(calibration) == set(outer)
        assert not set(outer) & set(test)
        assert set(target.loc[fit]) == set(LABELS)
        held.extend(test)
    assert len(held) == len(set(held)) == len(target)


def test_historical_labels_use_only_the_future_window():
    frame = events(["2025-01-01", "2025-01-08", "2025-01-15", "2025-02-01", "2025-06-01"])
    cutoff = pd.Timestamp("2025-02-01", tz="UTC")
    result = historical_date_diagnostics(frame, cutoff, 30)
    assert result["future_observed_streams"] == 1  # Event exactly at cutoff is a target.
    assert result["date_error_pairs"] == 1
    assert result["date_errors_conditional_on_observation"]["v1_median_date"]["mae_days"] == 10
    changed = pd.concat([frame, events(["2025-08-01"])])
    assert historical_date_diagnostics(changed, cutoff, 30) == result
    pd.testing.assert_frame_equal(
        temporal_streams(frame.loc[frame.timestamp < cutoff], cutoff),
        temporal_streams(changed.loc[changed.timestamp < cutoff], cutoff),
    )
    with pytest.raises(ValueError, match="fully observed"):
        historical_date_diagnostics(frame, pd.Timestamp("2025-12-01", tz="UTC"), 90)


def test_interval_units_do_not_depend_on_pandas_timestamp_resolution():
    frame = events(["2025-12-01", "2025-12-08", "2025-12-15"])
    expected = temporal_streams(frame)
    frame["timestamp"] = frame.timestamp.dt.as_unit("us")
    pd.testing.assert_frame_equal(temporal_streams(frame), expected)


def test_calibration_mapping_excludes_calibration_and_heldout_labels(monkeypatch):
    rows, labels, fit, calibration = [], [], [], []
    target_column = "target_next_recurring_merchant"
    for label in LABELS:
        for number in range(4):
            client = f"{label}_{number}"
            frame = events(["2025-10-01", "2025-11-01", "2025-12-01"])
            frame["client_id"], frame["description"] = client, label
            frame = frame.assign(
                amount=10.0, fee=0.0, currency="chf", direction="out", mcc="1", type="card"
            )
            rows.append(frame)
            labels.append({"client_id": client, target_column: label})
            if number < 2:
                fit.append(client)
            elif number == 2:
                calibration.append(client)
    transactions, labels = pd.concat(rows), pd.DataFrame(labels)
    streams = temporal_streams(transactions)
    original_fit = ClientFeatureBuilder.fit
    learned = []

    def checked_fit(self, fit_transactions, fit_labels):
        assert set(fit_transactions.client_id) == set(fit)
        assert set(fit_labels.client_id) == set(fit)
        result = original_fit(self, fit_transactions, fit_labels)
        learned.append(self.description_lift_.copy())
        return result

    monkeypatch.setattr(ClientFeatureBuilder, "fit", checked_fit)
    first = _calibrate(
        transactions, labels, pd.Index(fit), pd.Index(calibration), streams, {"v1": []}
    )
    # Changing outer-held-out labels must not change calibration or learned mapping.
    labels.loc[~labels.client_id.isin(fit + calibration), target_column] = "none"
    second = _calibrate(
        transactions, labels, pd.Index(fit), pd.Index(calibration), streams, {"v1": []}
    )
    pd.testing.assert_frame_equal(learned[0], learned[1])
    assert first["v1"] == second["v1"]


def test_no_observed_future_is_serializable_without_fake_date_errors():
    import json

    result = historical_date_diagnostics(
        events(["2025-01-01"]), pd.Timestamp("2025-02-01", tz="UTC"), 30
    )
    assert result["date_error_pairs"] == 0
    assert result["date_errors_conditional_on_observation"]["v1_median_date"]["mae_days"] is None
    json.dumps(result, allow_nan=False)
