"""Date-only stream forecasts; no labels, amounts, or client identity as predictors."""

from __future__ import annotations

import calendar
import math

import numpy as np
import pandas as pd

METHODS = (
    "last_gap",
    "mean",
    "median",
    "trimmed_mean",
    "ewma",
    "median_phase",
    "weekly",
    "fortnightly",
    "monthly",
    "quarterly",
    "automatic",
    "automatic_active",
)
DAY = 86_400_000_000_000


def eligible(days: np.ndarray) -> bool:
    """Three distinct days, >=14-day span, median gap >=3 days; no future filters."""
    return len(days) >= 3 and days[-1] - days[0] >= 14 and np.median(np.diff(days)) >= 3


def periodicity(days: np.ndarray) -> str:
    gaps = np.diff(days)
    period = float(np.median(gaps))
    if np.median(abs(gaps - period)) / period > 0.35:
        return "irregular"
    cycles = {"weekly": 7, "fortnightly": 14, "monthly": 30.4375, "quarterly": 91.3125}
    name = min(cycles, key=lambda key: abs(period / cycles[key] - 1))
    return name if abs(period / cycles[name] - 1) <= 0.25 else "other"


def phase_date(days: np.ndarray, cutoff: int, period: float) -> float:
    """Robust lattice anchored at last event; allow missing historical cycles."""
    phase = np.median(days - np.rint((days - days[-1]) / period) * period)
    return float(phase + max(1, math.ceil((cutoff - phase) / period)) * period)


def calendar_date(days: np.ndarray, cutoff: int, months: int) -> float:
    dates = pd.to_datetime(days, unit="D", utc=True)
    day = int(np.rint(np.median(dates.day)))
    end_of_month = float(np.mean(dates.is_month_end)) >= 0.6
    last = dates[-1]
    month_index = last.year * 12 + last.month - 1
    for step in range(1, 240):
        year, month0 = divmod(month_index + step * months, 12)
        month = month0 + 1
        last_day = calendar.monthrange(year, month)[1]
        candidate = (
            pd.Timestamp(
                year=year,
                month=month,
                day=last_day if end_of_month else min(day, last_day),
                tz="UTC",
            ).value
            // DAY
        )
        if candidate >= cutoff:
            return float(candidate)
    raise ValueError("Calendar projection exceeds supported range")


def forecast(days: np.ndarray, cutoff: int) -> tuple[dict[str, float], dict[str, bool]]:
    """Input must be sorted unique integer UTC days strictly before cutoff.

    Raw gap forecasts overdue at cutoff predict immediately. Phase/calendar models
    project the next lattice occurrence. Active variant abstains after 2.5 periods.
    Dates remain available even when occurrence is vetoed, avoiding MAE selection bias.
    """
    days = np.asarray(days, dtype=np.int64)
    if len(days) < 3 or np.any(np.diff(days) <= 0) or days[-1] >= cutoff:
        raise ValueError("Need three sorted distinct days strictly before cutoff")
    gaps = np.diff(days).astype(float)
    median = float(np.median(gaps))
    ordered = np.sort(gaps)
    trim = int(len(gaps) * 0.2)
    robust = ordered[trim : len(gaps) - trim] if trim else ordered
    ewma = gaps[0]
    for gap in gaps[1:]:
        ewma = 0.4 * gap + 0.6 * ewma
    periods = {
        "last_gap": gaps[-1],
        "mean": gaps.mean(),
        "median": median,
        "trimmed_mean": robust.mean(),
        "ewma": ewma,
    }
    predictions = {name: float(max(cutoff, days[-1] + gap)) for name, gap in periods.items()}
    predictions.update(
        {
            "median_phase": phase_date(days, cutoff, median),
            "weekly": phase_date(days, cutoff, 7),
            "fortnightly": phase_date(days, cutoff, 14),
            "monthly": calendar_date(days, cutoff, 1),
            "quarterly": calendar_date(days, cutoff, 3),
        }
    )
    kind = periodicity(days)
    predictions["automatic"] = predictions.get(kind, predictions["median_phase"])
    predictions["automatic_active"] = predictions["automatic"]
    occurrence = {name: value < cutoff + 90 for name, value in predictions.items()}
    occurrence["automatic_active"] &= cutoff - days[-1] <= 2.5 * median
    return predictions, occurrence


def temporal_metrics(frame: pd.DataFrame) -> dict:
    observed = frame.actual.notna()
    errors = abs(frame.loc[observed, "predicted"] - frame.loc[observed, "actual"])
    tp = int((observed & frame.occurs).sum())
    return {
        "n": len(frame),
        "events_90d": int(observed.sum()),
        "mae": float(errors.mean()),
        "median_ae": float(errors.median()),
        **{f"within_{n}d": float(errors.le(n).mean()) for n in (3, 7, 14)},
        "recall_90d": tp / observed.sum() if observed.any() else np.nan,
        "precision_90d": tp / frame.occurs.sum() if frame.occurs.any() else np.nan,
    }


def tie_coverage(ranked: pd.DataFrame, truth: pd.Series, k: int) -> float:
    """Expected coverage under uniform random order within equal forecast dates."""
    slots = min(k, len(ranked))
    miss = 1.0
    for _, group in ranked.groupby("rank_date", sort=True):
        take = min(slots, len(group))
        good = int(truth.loc[group.index].sum())
        miss *= (
            math.comb(len(group) - good, take) / math.comb(len(group), take)
            if len(group) - good >= take
            else 0.0
        )
        slots -= take
        if not slots:
            break
    return 1 - miss


def ranking_rows(frame: pd.DataFrame, min_candidates: int = 2) -> pd.DataFrame:
    rows = []
    for (cutoff, client), group in frame.groupby(["cutoff", "client_id"], sort=False):
        if len(group) < min_candidates:
            continue
        first = group.actual.min()
        predicted_none = not group.occurs.any()
        row = {
            "cutoff": cutoff,
            "client_id": client,
            "candidates": len(group),
            "actual_none": pd.isna(first),
            "predicted_none": predicted_none,
            "top1": np.nan,
            "top2": np.nan,
            "chosen_mae": np.nan,
            "chosen_censored": np.nan,
            "random_top1": np.nan,
            "random_top2": np.nan,
            "periodicity": "none",
        }
        if pd.notna(first):
            truth = group.actual.eq(first)
            ranked = group.assign(rank_date=group.predicted.where(group.occurs, np.inf))
            chosen = ranked.loc[ranked.rank_date.eq(ranked.rank_date.min())]
            kinds = group.loc[truth, "periodicity"].unique()
            row.update(
                top1=0.0 if predicted_none else tie_coverage(ranked, truth, 1),
                top2=0.0 if predicted_none else tie_coverage(ranked, truth, 2),
                chosen_mae=float(abs(chosen.predicted - chosen.actual).mean()),
                chosen_censored=float(chosen.actual.isna().mean()),
                periodicity=kinds[0] if len(kinds) == 1 else "mixed",
                random_top1=float(truth.mean()),
                random_top2=tie_coverage(group.assign(rank_date=0), truth, 2),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_metrics(rows: pd.DataFrame) -> dict:
    none = rows.actual_none
    predicted = rows.predicted_none
    tp = int((none & predicted).sum())
    return {
        "n_clients_cutoffs": len(rows),
        "ranking_cases": int((~none).sum()),
        "top1": float(rows.top1.mean()),
        "top2": float(rows.top2.mean()),
        "chosen_mae": float(rows.chosen_mae.mean()),
        "chosen_censored": float(rows.chosen_censored.mean()),
        "random_top1": float(rows.random_top1.mean()),
        "random_top2": float(rows.random_top2.mean()),
        "none_cases": int(none.sum()),
        "none_predicted": int(predicted.sum()),
        "none_recall": tp / none.sum() if none.any() else np.nan,
        "none_precision": tp / predicted.sum() if predicted.any() else np.nan,
        "none_specificity": float((~predicted[~none]).mean()),
        "none_accuracy": float(none.eq(predicted).mean()),
    }
