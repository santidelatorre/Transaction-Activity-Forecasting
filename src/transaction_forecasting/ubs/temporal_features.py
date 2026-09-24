"""Cutoff-safe time summaries of the existing (client, description) streams.

Descriptions are opaque stream identifiers, not new text features. Missing interval
statistics remain NaN; two events do not establish regularity. Evidence is a bounded
heuristic, never a calibrated probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import CUTOFF, LABELS

CYCLES = (7, 14, 28, 30, 31, 90, 180, 365)
WINDOWS = (30, 60, 90, 180)
BLOCKS = ("intervals", "periodicity", "activity", "horizon")


def temporal_streams(
    transactions: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF, horizon: int = 90
) -> pd.DataFrame:
    """Summarize unique event times; reject future input instead of silently using it."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or horizon <= 0:
        raise ValueError("A timezone-aware cutoff and positive horizon are required")
    frame = transactions[["client_id", "description", "timestamp"]].copy()
    if frame.isna().any().any():
        raise ValueError("Null stream keys or timestamps")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].isna().any() or frame["timestamp"].ge(cutoff).any():
        raise ValueError("Transactions must be strictly before cutoff")
    rows = []
    for (client, description), group in frame.groupby(["client_id", "description"], sort=True):
        times = pd.DatetimeIndex(group["timestamp"].drop_duplicates().sort_values())
        gaps = np.diff(times.as_unit("ns").asi8) / 86_400_000_000_000
        n, k = len(times), len(gaps)
        age = (cutoff - times[-1]).total_seconds() / 86400
        exposure = (cutoff - times[0]).total_seconds() / 86400
        median = float(np.median(gaps)) if k else np.nan
        mean = float(np.mean(gaps)) if k else np.nan
        std = float(np.std(gaps)) if k >= 2 else np.nan
        mad = float(np.median(np.abs(gaps - median))) if k >= 2 else np.nan
        cv = std / mean if k >= 2 else np.nan
        support = k / (k + 2)
        confidence = support / (1 + cv) if k >= 2 else 0.0
        row = {
            "client_id": client,
            "description": description,
            "transaction_count": len(group),
            "unique_event_count": n,
            "duplicate_timestamp_count": len(group) - n,
            "number_of_intervals": k,
            "days_since_last": age,
            "days_since_first": exposure,
            "active_span_days": exposure - age,
            "interval_mean": mean,
            "interval_median": median,
            "interval_min": float(gaps.min()) if k else np.nan,
            "interval_max": float(gaps.max()) if k else np.nan,
            "interval_std": std,
            "interval_mad": mad,
            "interval_cv": cv,
            "interval_relative_mad": mad / median if k >= 2 else np.nan,
            "last_interval": float(gaps[-1]) if k else np.nan,
            "last_interval_relative_deviation": abs(gaps[-1] - median) / median if k else np.nan,
            "support": support,
            "regularity_confidence": confidence,
        }
        for cycle in CYCLES:
            row[f"cycle_{cycle}_closeness"] = (
                float(np.exp(-np.median(abs(gaps - cycle)) / (0.15 * cycle))) if k >= 2 else np.nan
            )
        for name, phase in (("weekday", times.dayofweek / 7), ("monthday", (times.day - 1) / 31)):
            row[f"{name}_concentration"] = (
                float(abs(np.exp(2j * np.pi * np.asarray(phase)).mean())) if n >= 3 else np.nan
            )
        for window in WINDOWS:
            count = int((times >= cutoff - pd.Timedelta(days=window)).sum())
            row[f"events_{window}d"] = count
            row[f"share_{window}d"] = count / n
        historical_rate = n / max(exposure, 1)
        recent_rate = row["events_90d"] / max(min(exposure, 90), 1)
        row["recent_to_history_rate"] = recent_rate / historical_rate
        expected = times[-1] + pd.Timedelta(days=median) if k else pd.NaT
        # Calendar months vary in length; only supported monthly streams use this rule.
        monthly = k >= 2 and 27 <= median <= 32 and row["monthday_concentration"] >= 0.95
        if monthly:
            expected = (
                (times[-1] + pd.offsets.MonthEnd(1))
                if times.is_month_end.all()
                else (times[-1] + pd.DateOffset(months=1))
            )
        wait = (expected - cutoff).total_seconds() / 86400 if k else np.nan
        row.update(
            {
                "calendar_month_used": bool(monthly),
                "expected_next_date": expected,
                "days_until_expected_next": wait,
                "overdue_days": max(-wait, 0) if k else np.nan,
                "expected_next_within_horizon": bool(0 <= wait < horizon) if k else False,
                "expected_date_known": bool(k),
                "horizon_evidence": (
                    support * np.exp(-max(-wait, 0) / max(median, 1))
                    if k and wait < horizon
                    else 0.0
                ),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["client_id", "description"])
        )
    result = pd.DataFrame(rows).set_index(["client_id", "description"])
    result["expected_next_date"] = pd.to_datetime(result["expected_next_date"], utc=True)
    return result


def apply_temporal_blocks(
    baseline: pd.DataFrame,
    streams: pd.DataFrame,
    description_lift: pd.DataFrame,
    blocks: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Reweight V1 recurrence scores only; mappings and other V1 inputs stay fixed.

    Each block yields a factor in [0.5, 1.5], aggregated with the existing positive
    family lift. Their geometric mean avoids multiplying four penalties. Unknown
    descriptions have no family evidence and leave V1 unchanged. None retains V1's
    score; its bias is calibrated on the same inner clients for every variant.
    """
    if set(blocks).difference(BLOCKS) or len(set(blocks)) != len(blocks):
        raise ValueError("Unknown or duplicate temporal blocks")
    if not baseline.index.is_unique or not streams.index.is_unique:
        raise ValueError("Duplicate clients or streams")
    result = baseline.copy()
    if not blocks or streams.empty:
        return result
    factors = {
        "intervals": 0.5 + streams["regularity_confidence"],
        "periodicity": 0.5
        + streams[[f"cycle_{c}_closeness" for c in CYCLES]].max(axis=1).fillna(0)
        * streams["support"],
        "activity": 0.5 + streams["recent_to_history_rate"].clip(0, 2) / 2,
        "horizon": 0.5 + streams["horizon_evidence"],
    }
    factor = np.exp(np.mean([np.log(factors[name]) for name in blocks], axis=0))
    descriptions = streams.index.get_level_values("description")
    for label in LABELS:
        if label == "none":
            continue
        weights = description_lift[label].reindex(descriptions).fillna(0).to_numpy()
        # Match the repeated-stream eligibility of V1; singleton summaries remain diagnostic.
        weights = weights * streams["unique_event_count"].ge(2).to_numpy()
        weighted = pd.DataFrame(
            {"weight": weights, "weighted": weights * factor}, index=streams.index
        )
        sums = weighted.groupby(level="client_id").sum()
        scale = sums["weighted"].div(sums["weight"].replace(0, np.nan))
        column = f"family_{label}_recurrence_score"
        result[column] = result[column] * scale.reindex(result.index).fillna(1)
    return result
