"""Research contracts: no future labels, transparent clocks, honest ablation."""

import numpy as np
import pandas as pd
import pytest

from ubs_recurrence.research_time import feature_groups, research_matrix


def candidates():
    names = ["weekly", "monthly", "stale", "missing", "annual", "irregular", "due_now", "single"]
    index = pd.MultiIndex.from_tuples([(name, "streaming") for name in names], names=["client_id", "family"])
    return pd.DataFrame({
        "family_index": 6,
        "is_none": 0,
        "amount0_count": [6, 6, 6, -999, 2, 6, 6, 1],
        "amount0_gap_median": [7, 30, 30, -999, 365, 30, 30, 30],
        "amount0_last_age": [6, 28, 121, -999, 10, 28, 30, 28],
        "amount0_gap_cv": [0.01, 0.03, 0.01, -999, 0.0, 0.8, 0.0, 0.0],
    }, index=index)


def test_weekly_and_monthly_use_relative_observed_cycle():
    x = research_matrix(candidates(), "cycle_state")
    weekly = x.loc[("weekly", "streaming")]
    monthly = x.loc[("monthly", "streaming")]
    assert weekly.amount0_cycle_supported == 1
    assert weekly.amount0_cycle_elapsed == pytest.approx(6 / 7)
    assert weekly.amount0_cycle_due_delay == pytest.approx(1 / 7)
    assert monthly.amount0_cycle_elapsed == pytest.approx(28 / 30)
    assert monthly.amount0_cycle_due_delay == pytest.approx(2 / 30)
    assert weekly.amount0_cycle_active == monthly.amount0_cycle_active == 1
    assert weekly.amount0_cycle_due_within_90_active == monthly.amount0_cycle_due_within_90_active == 1


def test_stale_stream_is_not_resurrected_by_modulo_and_due_now_is_not_missed():
    x = research_matrix(candidates(), "cycle_state")
    stale = x.loc[("stale", "streaming")]
    assert stale.amount0_cycle_elapsed == pytest.approx(121 / 30)
    assert stale.amount0_cycle_overdue == pytest.approx(121 / 30 - 1)
    assert stale.amount0_cycle_missed == 4
    assert stale.amount0_cycle_due_delay == 0
    assert stale.amount0_cycle_active == stale.amount0_cycle_due_within_90_active == 0
    due = x.loc[("due_now", "streaming")]
    assert due.amount0_cycle_missed == due.amount0_cycle_overdue == due.amount0_cycle_due_delay == 0
    assert due.amount0_cycle_active == 1


def test_missing_and_short_histories_have_explicit_unavailability():
    x = research_matrix(candidates(), "cycle_state")
    for name in ["missing", "single"]:
        row = x.loc[(name, "streaming")]
        assert row.amount0_cycle_available == row.amount0_cycle_active == 0
        assert row.amount0_cycle_elapsed == row.amount0_cycle_due_delay == -999
    assert (x.amount1_cycle_available == 0).all()
    assert (x.broad0_cycle_elapsed == -999).all()
    assert np.isfinite(x.to_numpy()).all()
    absent = candidates().assign(amount0_last_age=np.nan)
    assert (research_matrix(absent, "cycle_state").amount0_cycle_available == 0).all()


def test_annual_supported_but_outside_horizon_and_irregular_not_active():
    x = research_matrix(candidates(), "cycle_state")
    annual = x.loc[("annual", "streaming")]
    assert annual.amount0_cycle_supported == annual.amount0_cycle_active == 1
    assert annual.amount0_cycle_due_within_90_active == 0
    irregular = x.loc[("irregular", "streaming")]
    assert irregular.amount0_cycle_available == 1
    assert irregular.amount0_cycle_supported == irregular.amount0_cycle_active == 0


@pytest.mark.parametrize("period", [7, 14, 28, 30, 31, 60, 90, 365])
def test_supported_calendar_periods_include_weekly_and_annual(period):
    frame = candidates().iloc[:1].assign(amount0_gap_median=period, amount0_last_age=period / 2)
    row = research_matrix(frame, "cycle_state").iloc[0]
    assert row.amount0_cycle_fit_error == 0
    assert row.amount0_cycle_supported == 1
    assert row.amount0_cycle_elapsed == 0.5


def test_feature_groups_and_ablation_remove_derived_time_without_other_counts():
    temporal = [
        "amount0_last_age", "client_first_age", "amount1_span", "broad0_gap_mad",
        "amount0_next_active_global_mean", "broad0_active_rank", "amount0_period_rounded",
        "amount0_phase_strength_30", "amount0_linear_residual", "amount0_linear_next",
        "broad0_overdue_ratio", "client_count_90", "amount0_count_30",
        "amount0_refund_after_last", "amount0_refund_last_age", "broad0_refund_count90",
        "amount0_refund_count30", "amount0_hour_std", "amount0_weekend_fraction",
        "amount0_amount_change", "broad0_last_fee", "own_template_mcc_90_sum",
        "global_none__any_90_diversity", "amount0_dom_std", "amount1_dow_std",
        "global__month_count", "amount0_clock_strength", "amount0_cycle_available",
    ]
    keep = ["amount0_count", "client_total_count", "amount0_count_rank", "stream_count", "amount0_refund_count", "amount0_refund_ratio", "amount0_own_template_count", "amount0_own_mcc", "amount0_amount_cv", "amount0_fee_mean", "family_index", "is_none", "client_currency_chf"]
    columns = temporal + keep
    grouped = feature_groups(columns)
    flattened = sum(grouped.values(), [])
    assert len(flattened) == len(set(flattened)) == len(columns)
    assert set(flattened) == set(columns)
    assert grouped["temporal"] == temporal
    assert "amount0_count" in grouped["context"]
    frame = pd.DataFrame(np.ones((2, len(columns))), columns=columns)
    assert research_matrix(frame, "no_temporal").columns.tolist() == keep
    no_refund = research_matrix(frame, "no_refund")
    assert not any("refund" in col for col in no_refund)
    assert "amount0_last_age" in no_refund


def test_schema_index_and_inputs_preserved_for_all_variants():
    frame = candidates().iloc[::-1]
    before = frame.copy(deep=True)
    for variant in ["control", "no_temporal", "no_refund", "cycle_state"]:
        result = research_matrix(frame, variant)
        assert result.index.equals(frame.index)
        assert result is not frame
        result.iloc[0, 0] = 12345
        pd.testing.assert_frame_equal(frame, before)
    full = research_matrix(frame, "cycle_state")
    missing = research_matrix(frame * np.nan, "cycle_state")
    assert full.columns.equals(missing.columns)
    assert len(full.columns) == len(frame.columns) + 27
    pd.testing.assert_frame_equal(full[frame.columns], frame)


def test_labels_duplicate_columns_and_unknown_variants_are_rejected():
    with pytest.raises(ValueError, match="Labels"):
        research_matrix(candidates().assign(target_next_recurring_merchant="streaming"), "control")
    with pytest.raises(ValueError, match="unique"):
        research_matrix(pd.DataFrame([[1, 2]], columns=["a", "a"]), "control")
    with pytest.raises(ValueError, match="Unknown"):
        research_matrix(candidates(), "bad")
    with pytest.raises(ValueError, match="already"):
        research_matrix(research_matrix(candidates(), "cycle_state"), "cycle_state")
