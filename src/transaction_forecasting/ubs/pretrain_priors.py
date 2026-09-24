"""Unsupervised description priors from population transaction histories.

Priors are fitted without labels. Callers may pass train-only histories or
train + unlabeled pretrain. Validation/test labels must never enter fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import CUTOFF
from transaction_forecasting.ubs.features import PERIODICITY_BINS, build_recurrence_streams

PRIOR_FEATURE_PREFIX = "pretrain_"


@dataclass
class DescriptionPriors:
    """Frozen population statistics keyed by normalized description."""

    table: pd.DataFrame
    global_defaults: dict[str, float]
    source_clients: int
    source_transactions: int
    includes_unlabeled: bool

    def lookup(self, descriptions: pd.Series) -> pd.DataFrame:
        """Return prior rows aligned to descriptions, filling unknowns with globals."""
        aligned = self.table.reindex(descriptions.astype(str))
        for column, default in self.global_defaults.items():
            if column in aligned.columns:
                aligned[column] = aligned[column].fillna(default)
        aligned.index = descriptions.index
        return aligned


def _period_bin(median_days: float) -> str:
    for name, (lower, upper) in PERIODICITY_BINS.items():
        if lower <= median_days < upper:
            return name
    return "other"


def fit_description_priors(
    transactions: pd.DataFrame,
    *,
    alpha: float = 5.0,
    includes_unlabeled: bool = False,
) -> DescriptionPriors:
    """Estimate smoothed description-level recurrence and merchant priors."""
    if transactions.empty:
        raise ValueError("Cannot fit priors on an empty transaction frame")
    if transactions["timestamp"].isna().any() or transactions["timestamp"].ge(CUTOFF).any():
        raise ValueError("Prior fitting requires pre-cutoff timestamps only")
    if alpha <= 0:
        raise ValueError("Smoothing alpha must be positive")

    ordered = transactions.sort_values(["client_id", "description", "timestamp"], kind="stable")
    pair_counts = (
        ordered.groupby(["client_id", "description"], sort=False)
        .size()
        .rename("appearances")
        .reset_index()
    )
    clients_per_desc = pair_counts.groupby("description")["client_id"].nunique()
    recurrent_pairs = pair_counts[pair_counts["appearances"].ge(2)]
    recurrent_clients = recurrent_pairs.groupby("description")["client_id"].nunique()

    streams = build_recurrence_streams(ordered)
    if streams.empty:
        stream_stats = pd.DataFrame(
            columns=[
                "typical_period_days",
                "gap_cv_prior",
                "amount_cv_prior",
                "regularity_prior",
                "direction_out_share_prior",
                "stream_support",
                "p_weekly",
                "p_biweekly",
                "p_monthly",
                "p_quarterly",
                "p_annual",
            ]
        )
    else:
        streams = streams.copy()
        streams["period_bin"] = streams["median_interval_days"].map(_period_bin)
        period_rates = (
            streams.pivot_table(
                index="description",
                columns="period_bin",
                values="client_id",
                aggfunc="count",
                fill_value=0,
            )
            .reindex(columns=["weekly", "biweekly", "monthly", "quarterly", "annual"], fill_value=0)
            .astype(float)
        )
        period_rates = period_rates.div(period_rates.sum(axis=1).clip(lower=1.0), axis=0)
        period_rates = period_rates.rename(
            columns={
                "weekly": "p_weekly",
                "biweekly": "p_biweekly",
                "monthly": "p_monthly",
                "quarterly": "p_quarterly",
                "annual": "p_annual",
            }
        )
        stream_stats = streams.groupby("description", sort=False).agg(
            typical_period_days=("median_interval_days", "median"),
            gap_cv_prior=("interval_cv", "median"),
            amount_cv_prior=("amount_cv", "median"),
            regularity_prior=("regularity", "median"),
            direction_out_share_prior=("direction_out_share", "median"),
            stream_support=("client_id", "size"),
        )
        stream_stats = stream_stats.join(period_rates, how="left")

    # Modal MCC / type / currency among all events for the description.
    def _mode_or_empty(series: pd.Series) -> str:
        if series.empty:
            return ""
        counts = series.value_counts()
        return str(counts.index[0])

    event_stats = ordered.groupby("description", sort=False).agg(
        n_transactions=("timestamp", "size"),
        amount_mean_prior=("amount", "median"),
        fee_positive_rate=("fee", lambda values: float(pd.Series(values).gt(0).mean())),
        typical_mcc=("mcc", _mode_or_empty),
        typical_type=("type", _mode_or_empty),
        typical_currency=("currency", _mode_or_empty),
        weekend_share_prior=(
            "timestamp",
            lambda values: float(
                pd.to_datetime(pd.Series(values), utc=True).dt.dayofweek.ge(5).mean()
            ),
        ),
    )

    table = pd.DataFrame({"n_clients": clients_per_desc}).join(
        pd.DataFrame({"n_recurrent_clients": recurrent_clients}), how="left"
    )
    table["n_recurrent_clients"] = table["n_recurrent_clients"].fillna(0.0)
    table = table.join(event_stats, how="left").join(stream_stats, how="left")
    table = table.fillna(
        {
            "n_transactions": 0.0,
            "stream_support": 0.0,
            "typical_period_days": np.nan,
            "gap_cv_prior": np.nan,
            "amount_cv_prior": np.nan,
            "regularity_prior": np.nan,
            "direction_out_share_prior": np.nan,
            "p_weekly": 0.0,
            "p_biweekly": 0.0,
            "p_monthly": 0.0,
            "p_quarterly": 0.0,
            "p_annual": 0.0,
            "weekend_share_prior": 0.0,
            "fee_positive_rate": 0.0,
            "amount_mean_prior": 0.0,
        }
    )

    global_rate = float(table["n_recurrent_clients"].sum() / max(table["n_clients"].sum(), 1.0))
    table["p_recurrent"] = (table["n_recurrent_clients"] + alpha * global_rate) / (
        table["n_clients"] + alpha
    )
    # Shrink sparse stream statistics toward population medians.
    global_defaults = {
        "p_recurrent": global_rate,
        "typical_period_days": float(
            table["typical_period_days"].median(skipna=True)
            if table["typical_period_days"].notna().any()
            else 30.0
        ),
        "gap_cv_prior": float(
            table["gap_cv_prior"].median(skipna=True)
            if table["gap_cv_prior"].notna().any()
            else 1.0
        ),
        "amount_cv_prior": float(
            table["amount_cv_prior"].median(skipna=True)
            if table["amount_cv_prior"].notna().any()
            else 0.5
        ),
        "regularity_prior": float(
            table["regularity_prior"].median(skipna=True)
            if table["regularity_prior"].notna().any()
            else 0.5
        ),
        "direction_out_share_prior": float(
            table["direction_out_share_prior"].median(skipna=True)
            if table["direction_out_share_prior"].notna().any()
            else 0.5
        ),
        "p_weekly": float(table["p_weekly"].mean()),
        "p_biweekly": float(table["p_biweekly"].mean()),
        "p_monthly": float(table["p_monthly"].mean()),
        "p_quarterly": float(table["p_quarterly"].mean()),
        "p_annual": float(table["p_annual"].mean()),
        "weekend_share_prior": float(table["weekend_share_prior"].mean()),
        "fee_positive_rate": float(table["fee_positive_rate"].mean()),
        "amount_mean_prior": float(table["amount_mean_prior"].median()),
        "stream_support": 0.0,
        "n_clients": 0.0,
        "n_transactions": 0.0,
        "n_recurrent_clients": 0.0,
    }
    for column, default in global_defaults.items():
        if column in table.columns:
            table[column] = table[column].fillna(default)

    # Strength of evidence for short-history regularization.
    table["log_support"] = np.log1p(table["n_clients"])
    table["rare_description"] = (table["n_clients"] < alpha).astype(float)
    return DescriptionPriors(
        table=table.sort_index(),
        global_defaults=global_defaults,
        source_clients=int(ordered["client_id"].nunique()),
        source_transactions=int(len(ordered)),
        includes_unlabeled=includes_unlabeled,
    )


def client_prior_features(
    transactions: pd.DataFrame,
    priors: DescriptionPriors,
    client_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """Aggregate description priors to one row per client without using labels."""
    clients = (
        pd.Index(sorted(transactions["client_id"].unique()), name="client_id")
        if client_ids is None
        else client_ids
    )
    ordered = transactions.sort_values(["client_id", "timestamp"], kind="stable").copy()
    lookup = priors.lookup(ordered["description"])
    ordered = ordered.join(lookup.reset_index(drop=True))

    # Event-level aggregates.
    grouped = ordered.groupby("client_id", sort=False)
    features = pd.DataFrame(index=clients)
    for column in (
        "p_recurrent",
        "typical_period_days",
        "gap_cv_prior",
        "amount_cv_prior",
        "regularity_prior",
        "direction_out_share_prior",
        "p_weekly",
        "p_biweekly",
        "p_monthly",
        "p_quarterly",
        "p_annual",
        "weekend_share_prior",
        "log_support",
        "rare_description",
    ):
        features[f"{PRIOR_FEATURE_PREFIX}mean_{column}"] = grouped[column].mean()
        features[f"{PRIOR_FEATURE_PREFIX}max_{column}"] = grouped[column].max()

    # Stream-level: attach priors to recurring (client, description) streams.
    streams = build_recurrence_streams(ordered)
    if streams.empty:
        for column in (
            "p_recurrent",
            "typical_period_days",
            "gap_cv_prior",
            "regularity_prior",
            "log_support",
        ):
            features[f"{PRIOR_FEATURE_PREFIX}stream_mean_{column}"] = 0.0
            features[f"{PRIOR_FEATURE_PREFIX}stream_max_{column}"] = 0.0
        features[f"{PRIOR_FEATURE_PREFIX}stream_count"] = 0.0
        features[f"{PRIOR_FEATURE_PREFIX}short_history_stream_count"] = 0.0
        features[f"{PRIOR_FEATURE_PREFIX}unseen_desc_share"] = 0.0
    else:
        stream_priors = priors.lookup(streams["description"])
        streams = streams.reset_index(drop=True).join(stream_priors.reset_index(drop=True))
        stream_grouped = streams.groupby("client_id", sort=False)
        features[f"{PRIOR_FEATURE_PREFIX}stream_count"] = stream_grouped.size().reindex(
            clients, fill_value=0
        )
        features[f"{PRIOR_FEATURE_PREFIX}short_history_stream_count"] = (
            streams[streams["appearances"].le(3)]
            .groupby("client_id")
            .size()
            .reindex(clients, fill_value=0)
        )
        for column in (
            "p_recurrent",
            "typical_period_days",
            "gap_cv_prior",
            "regularity_prior",
            "log_support",
        ):
            features[f"{PRIOR_FEATURE_PREFIX}stream_mean_{column}"] = stream_grouped[column].mean()
            features[f"{PRIOR_FEATURE_PREFIX}stream_max_{column}"] = stream_grouped[column].max()
        # Regularize short streams toward population period.
        streams["period_residual"] = (
            streams["median_interval_days"] - streams["typical_period_days"]
        ).abs() / streams["typical_period_days"].clip(lower=1.0)
        features[f"{PRIOR_FEATURE_PREFIX}stream_mean_period_residual"] = stream_grouped[
            "period_residual"
        ].mean()

    # Unseen / rare description exposure on the client.
    known = ordered["description"].astype(str).isin(priors.table.index)
    features[f"{PRIOR_FEATURE_PREFIX}unseen_desc_share"] = (
        (~known).groupby(ordered["client_id"]).mean()
    )
    features[f"{PRIOR_FEATURE_PREFIX}mean_rare_description"] = grouped["rare_description"].mean()

    features = features.reindex(clients).replace([np.inf, -np.inf], np.nan)
    return features.fillna(0.0).astype(float)
