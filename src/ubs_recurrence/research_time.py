"""Isolated, label-free representations for the stream-time research experiment.

These helpers never change the production feature builders. In particular,
``no_temporal`` is a *downstream column ablation*: candidate streams were already
selected using recency, so it cannot remove every influence of transaction time.
Feature groups describe columns, not independent or causal information sources.
"""

import re

import numpy as np
import pandas as pd


PERIODS = np.array([7.0, 14.0, 28.0, 30.0, 31.0, 60.0, 90.0, 365.0])
PREFIXES = ("amount0", "amount1", "broad0")
MISSING = -999.0
GROUPS = ("temporal", "refund", "identity", "amount", "context")
_TIME_TOKEN = re.compile(
    r"(?:^|_)(?:age|span|gap|next|active|period|phase|overdue|recency|recent|"
    r"timestamp|hour|weekend|dom|dow|month|weekday|clock|cycle)(?:_|$)"
)
_TIME_WINDOW = re.compile(r"(?:^|_)count_?(?:30|60|90|180)(?:_|$)")
_TEMPLATE_WINDOW = re.compile(r"(?:^|_)(?:any|mcc)_90(?:_|$)")
_FORBIDDEN = {"label", "target", "y", "target_next_recurring_merchant", "etiqueta_cliente", "client_id"}


def _is_temporal(column):
    name = column.lower()
    return bool(
        _TIME_TOKEN.search(name)
        or _TIME_WINDOW.search(name)
        or _TEMPLATE_WINDOW.search(name)
        or "linear_residual" in name
        or "amount_change" in name
        or "refund_after_last" in name
        or "last_fee" in name
    )


def feature_groups(columns):
    """Return an exhaustive, disjoint partition, preserving input column order.

    Temporal precedence includes time-dependent refunds, interactions, ranks and
    client aggregates. Static counts remain available. Consequently the refund
    group contains only non-temporal refund columns; ``no_refund`` separately
    removes *all* refund columns, including those in the temporal group.
    Unrecognised columns fall into context and remain visible in the manifest.
    """
    result = {name: [] for name in GROUPS}
    columns = list(columns)
    if len(columns) != len(set(columns)):
        raise ValueError("Research features require unique column names")
    for column in columns:
        if not isinstance(column, str):
            raise ValueError("Research features require string column names")
        name = column.lower()
        if _is_temporal(name):
            group = "temporal"
        elif "refund" in name:
            group = "refund"
        elif any(token in name for token in ("semantic", "mcc", "template", "description", "generic_fraction")) or name in {"family_index", "is_none"}:
            group = "identity"
        else:
            # "amount0" is a stream slot, not proof that its contents are amounts.
            statistic = re.sub(r"^(?:amount|broad)\d+_", "", name)
            group = "amount" if any(token in statistic for token in ("amount", "price", "fee")) else "context"
        result[group].append(column)
    return result


def _numeric(frame, column):
    if column not in frame:
        return np.full(len(frame), np.nan)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.where((values == MISSING) | ~np.isfinite(values), np.nan, values)


def _cycle_features(frame):
    """27 transparent candidate-state features from historical observations.

    Each slot gets nine features. Availability requires at least two payments,
    a positive median gap and a nonnegative age. Calendar support requires a
    median gap within 20% of one of PERIODS and gap CV <= 0.35, when CV exists.
    With only two payments this is weak evidence, not proof of recurrence.

    The continuous clock uses the observed median, not a rounded calendar.
    Elapsed cycles duplicate the existing overdue_ratio for gaps >= one day;
    this experiment tests a representation and reliability gates, not new data.
    """
    extra = {}
    for prefix in PREFIXES:
        count = _numeric(frame, prefix + "_count")
        gap = _numeric(frame, prefix + "_gap_median")
        age = _numeric(frame, prefix + "_last_age")
        available = np.isfinite(count) & (count >= 2) & np.isfinite(gap) & (gap > 0) & np.isfinite(age) & (age >= 0)
        safe_gap = np.where(available, gap, 1.0)
        closest = PERIODS[np.abs(safe_gap[:, None] - PERIODS[None, :]).argmin(axis=1)]
        error = np.abs(safe_gap - closest) / closest
        supported = available & (error <= 0.20)
        if prefix + "_gap_cv" in frame:
            cv = _numeric(frame, prefix + "_gap_cv")
            supported &= np.isfinite(cv) & (cv >= 0) & (cv <= 0.35)
        elapsed = np.where(available, age / safe_gap, 0.0)
        delay = np.maximum(1.0 - elapsed, 0.0)
        overdue = np.maximum(elapsed - 1.0, 0.0)
        # Predicted due dates strictly before the cutoff; an event exactly due
        # now has not yet been missed. This is a diagnostic, not an observed fact.
        missed = np.maximum(np.ceil(elapsed) - 1.0, 0.0)
        active = supported & (elapsed <= 1.35)
        due = active & (np.maximum(safe_gap - age, 0.0) <= 90.0)
        values = {
            "available": available.astype(float),
            "supported": supported.astype(float),
            "fit_error": np.where(available, error, MISSING),
            "elapsed": np.where(available, elapsed, MISSING),
            "due_delay": np.where(available, delay, MISSING),
            "overdue": np.where(available, overdue, MISSING),
            "missed": np.where(available, missed, MISSING),
            "active": active.astype(float),
            "due_within_90_active": due.astype(float),
        }
        extra.update({f"{prefix}_cycle_{name}": value for name, value in values.items()})
    return pd.DataFrame(extra, index=frame.index)


def research_matrix(frame, variant):
    """Return a fresh candidate matrix for one predeclared research variant.

    Variants: control, no_temporal, no_refund, cycle_state. Row order and index
    are preserved. No labels are accepted; features use history only. The cycle
    schema always adds 27 columns, even for slots absent from this partition.
    No prediction is wrapped with modulo: stale streams stay inactive.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Research features require a pandas DataFrame")
    groups = feature_groups(frame.columns)
    forbidden = {c for c in frame.columns if c.lower() in _FORBIDDEN}
    if forbidden:
        raise ValueError(f"Labels and identifiers must not be feature columns: {sorted(forbidden)}")
    if variant == "control":
        return frame.copy(deep=True)
    if variant == "no_temporal":
        return frame.drop(columns=groups["temporal"]).copy(deep=True)
    if variant == "no_refund":
        return frame.drop(columns=[c for c in frame.columns if "refund" in c.lower()]).copy(deep=True)
    if variant == "cycle_state":
        extra = _cycle_features(frame)
        if set(extra.columns) & set(frame.columns):
            raise ValueError("Cycle-state features have already been added")
        return pd.concat([frame.copy(deep=True), extra], axis=1)
    raise ValueError(f"Unknown research variant: {variant}")
