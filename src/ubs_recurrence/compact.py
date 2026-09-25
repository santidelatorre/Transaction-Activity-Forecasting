"""Compact recurrence physics and global evidence, avoiding large template banks."""

import numpy as np
import pandas as pd

from .data import CUTOFF


def global_features(df, ids):
    d = df.copy()
    d["age"] = (CUTOFF - d.timestamp).dt.total_seconds() / 86400
    g = d.groupby("client_id")
    x = g.agg(
        total_count=("amount", "size"),
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
        first_age=("age", "max"),
        last_age=("age", "min"),
    )
    for col in ["type", "direction", "currency", "mcc"]:
        counts = pd.crosstab(d.client_id, d[col])
        counts.columns = [col + "_" + str(v) for v in counts]
        x = pd.concat([x, counts], axis=1)
    for window in [30, 60, 90, 180]:
        x[f"count_{window}"] = d[d.age <= window].groupby("client_id").size()
    return x.reindex(ids).fillna(0)


def compact_features(ranking, global_x):
    ids = ranking.index.get_level_values(0).unique()
    x = ranking
    cols = ["is_none", "family_index", "stream_count"]
    stats = [
        "count",
        "amount_mean",
        "amount_median",
        "amount_cv",
        "amount_mad",
        "amount_change",
        "first_age",
        "last_age",
        "span",
        "gap_median",
        "gap_mean",
        "gap_std",
        "gap_mad",
        "gap_cv",
        "gap_last",
        "gap_recent",
        "next_median",
        "next_recent",
        "linear_period",
        "linear_residual",
        "linear_next",
        "overdue_ratio",
        "count_30",
        "count_60",
        "count_90",
        "count_180",
        "own_semantic",
        "own_mcc",
        "own_template_diversity",
        "own_template_count",
        "other_semantic",
        "price_support",
        "normalized_price",
        "description_top_fraction",
        "generic_fraction",
    ]
    for p in ["amount0", "broad0", "amount1"]:
        cols.extend(p + "_" + s for s in stats if p + "_" + s in x)
    keep = x[[c for c in cols if c in x]].copy()
    extra = {}
    for p in ["amount0", "broad0", "amount1"]:
        count = x[p + "_count"].to_numpy()
        median = x[p + "_gap_median"].to_numpy()
        age = x[p + "_last_age"].to_numpy()
        observed = (count >= 2) & (median >= 5) & (median <= 120)
        period = np.array([14, 28, 30, 60, 90])[
            np.abs(median[:, None] - np.array([14, 28, 30, 60, 90])[None, :]).argmin(
                axis=1
            )
        ]
        active = observed & (age < period * 1.35)
        extra[p + "_active"] = active.astype(int)
        extra[p + "_next_active"] = np.where(
            active, np.mod(x[p + "_next_median"].to_numpy(), period), 999
        )
        extra[p + "_period_rounded"] = np.where(observed, period, -999)
        for c in [
            "own_semantic",
            "own_mcc",
            "own_template_diversity",
            "own_template_count",
        ]:
            if p + "_" + c in x:
                extra[p + "_" + c + "_per_event"] = np.where(
                    count > 0,
                    np.maximum(x[p + "_" + c].to_numpy(), 0) / np.maximum(count, 1),
                    0,
                )
    keep = pd.concat([keep, pd.DataFrame(extra, index=x.index)], axis=1)
    additional = {}
    for col in [
        "amount0_next_active",
        "amount0_active",
        "amount0_count",
        "amount0_last_age",
        "amount0_amount_cv",
        "amount0_own_semantic",
        "broad0_active",
        "broad0_count",
        "broad0_next_active",
    ]:
        if col not in keep:
            continue
        v = keep[col].replace(-999, np.nan)
        g = v.groupby(level=0, sort=False)
        for name, op in [("minimum", "min"), ("maximum", "max"), ("mean", "mean")]:
            additional[col + "_global_" + name] = g.transform(op)
        additional[col + "_rank"] = g.rank(method="min")
    for col in global_x:
        additional["client_" + col] = np.repeat(
            global_x.reindex(ids)[col].to_numpy(), 8
        )
    return pd.concat([keep, pd.DataFrame(additional, index=x.index)], axis=1).fillna(
        -999
    )
