"""Opt-in client-level UBS history features; no labels or fitted state."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

GROUPS = frozenset({"recency", "frequency", "intervals", "amount", "description"})
REQUIRED = frozenset({"client_id", "timestamp", "amount", "currency", "description"})


def _client_features(
    rows: pd.DataFrame, cutoff: pd.Timestamp, groups: frozenset[str], windows: tuple[int, ...]
) -> dict[str, float]:
    """Summarize one client's already filtered, chronological pre-cutoff history."""
    result: dict[str, float] = {}
    n = len(rows)
    times = rows["timestamp"]
    if "recency" in groups:
        result["v2_history_count"] = float(n)
        result["v2_sparse_history"] = float(n < 3)
        result["v2_days_since_last"] = (
            float((cutoff - times.iloc[-1]).total_seconds() / 86400) if n else np.nan
        )
        result["v2_history_span_days"] = (
            float((times.iloc[-1] - times.iloc[0]).total_seconds() / 86400) if n else np.nan
        )
    if "frequency" in groups:
        for days in windows:
            result[f"v2_count_{days}d"] = float((times >= cutoff - pd.Timedelta(days=days)).sum())
        if 30 in windows:
            recent = int((times >= cutoff - pd.Timedelta(days=30)).sum())
            result["v2_recent_30d_share"] = float(recent / n) if n else np.nan
    if "intervals" in groups:
        gaps = times.diff().dt.total_seconds().div(86400).dropna().to_numpy(dtype=float)
        positive = gaps[gaps > 0]
        result["v2_gap_median_days"] = float(np.median(positive)) if len(positive) else np.nan
        result["v2_gap_std_days"] = float(np.std(positive)) if len(positive) >= 2 else np.nan
        result["v2_weekly_gap_share"] = (
            float(np.mean((positive >= 5) & (positive <= 9))) if len(positive) else np.nan
        )
        result["v2_monthly_gap_share"] = (
            float(np.mean((positive >= 27) & (positive <= 32))) if len(positive) else np.nan
        )
    if "amount" in groups:
        # Use one currency at a time; the most recent currency is an explicit choice.
        same = rows.loc[rows["currency"].eq(rows["currency"].iloc[-1])] if n else rows
        amounts = same["amount"].to_numpy(dtype=float)
        result["v2_amount_currency_count"] = float(len(amounts))
        result["v2_amount_median"] = float(np.median(amounts)) if len(amounts) else np.nan
        result["v2_amount_mad"] = (
            float(np.median(np.abs(amounts - np.median(amounts)))) if len(amounts) else np.nan
        )
        result["v2_amount_cv"] = (
            float(np.std(amounts) / abs(np.mean(amounts)))
            if len(amounts) >= 2 and not np.isclose(np.mean(amounts), 0)
            else np.nan
        )
        result["v2_recent_3_amount_delta"] = (
            float(np.median(amounts[-3:]) - np.median(amounts[:-3]))
            if len(amounts) >= 4
            else np.nan
        )
    if "description" in groups:
        counts = rows["description"].astype(str).value_counts()
        result["v2_description_unique"] = float(len(counts))
        result["v2_description_top_share"] = float(counts.iloc[0] / n) if n else np.nan
        result["v2_description_repeated_count"] = float(counts.ge(2).sum())
        if n:
            probabilities = counts.to_numpy(dtype=float) / n
            result["v2_description_entropy"] = float(-np.sum(probabilities * np.log(probabilities)))
        else:
            result["v2_description_entropy"] = np.nan
        streams = rows.groupby(["description", "currency"], sort=False)
        interval_medians: list[float] = []
        amount_cvs: list[float] = []
        for _, stream in streams:
            if len(stream) < 2:
                continue
            gaps = stream["timestamp"].diff().dt.total_seconds().div(86400).dropna().to_numpy()
            if len(gaps):
                interval_medians.append(float(np.median(gaps)))
            values = stream["amount"].to_numpy(dtype=float)
            if np.isclose(np.mean(values), 0):
                continue
            amount_cvs.append(float(np.std(values) / abs(np.mean(values))))
        result["v2_stream_monthly_count"] = float(sum(27 <= gap <= 32 for gap in interval_medians))
        result["v2_stream_weekly_count"] = float(sum(5 <= gap <= 9 for gap in interval_medians))
        result["v2_stream_stable_amount_count"] = float(sum(cv <= 0.05 for cv in amount_cvs))
    return result


def build_client_v2_features(
    transactions: pd.DataFrame,
    client_ids: Iterable[str],
    *,
    cutoff: str | pd.Timestamp = "2026-01-01",
    groups: Iterable[str] = GROUPS,
    windows: tuple[int, ...] = (7, 14, 30, 60, 90, 180),
) -> pd.DataFrame:
    """Build optional UBS numeric features indexed by requested ``client_id``.

    All transactions must predate cutoff. IDs may include clients with no history.
    Output can be joined to the existing UBS V1 client feature table by index.
    """
    selected = frozenset(groups)
    if not selected or selected - GROUPS:
        raise ValueError(f"Select valid feature groups from {sorted(GROUPS)}")
    if any(day <= 0 for day in windows) or len(set(windows)) != len(windows):
        raise ValueError("Windows must contain distinct positive day counts")
    missing = REQUIRED - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing UBS transaction columns: {sorted(missing)}")
    if transactions[list(REQUIRED)].isna().any().any():
        raise ValueError("Required UBS transaction columns cannot contain nulls")
    ids = pd.Index(client_ids, name="client_id")
    if ids.has_duplicates:
        raise ValueError("client_ids must be unique")
    instant = pd.Timestamp(cutoff)
    instant = instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")
    work = transactions.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="raise", utc=True)
    work["amount"] = pd.to_numeric(work["amount"], errors="raise")
    if work["timestamp"].ge(instant).any():
        raise ValueError("Transactions at or after cutoff would leak future information")
    work = work.sort_values(["client_id", "timestamp"], kind="stable")
    by_client = {client: rows for client, rows in work.groupby("client_id", sort=False)}
    empty = work.iloc[:0]
    values = [
        _client_features(by_client.get(client, empty), instant, selected, windows) for client in ids
    ]
    return pd.DataFrame(values, index=ids).astype(float)
