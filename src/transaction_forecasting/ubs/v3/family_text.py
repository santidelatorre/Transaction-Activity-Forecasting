"""Selected family-mining routines from Javier c7075bb, with guarded inference.

Weak candidate selection conditions on the training target, but removes the
client's contribution from association counts. This is weak supervision,
not event ground truth. Only disjoint clients may be predicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.text_v2 import normalize_description
from transaction_forecasting.ubs.v3.features import validate_history

POSITIVE_LABELS = tuple(label for label in LABELS if label != "none")


@dataclass(frozen=True)
class AssociationTable:
    """Client-presence counts learned from a fit partition only."""

    counts: pd.DataFrame
    class_sizes: pd.Series
    n_clients: int
    scores: pd.DataFrame


def _mode(series: pd.Series) -> str:
    modes = series.astype(str).mode()
    return str(modes.iloc[0]) if len(modes) else "unknown"


def build_streams(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per client and normalized description with recurrence evidence."""
    validate_history(transactions)
    frame = transactions.copy()
    frame["description"] = frame["description"].map(normalize_description)
    frame = frame.sort_values(["client_id", "description", "timestamp"], kind="stable")
    keys = ["client_id", "description"]
    grouped = frame.groupby(keys, sort=False)
    streams = grouped.agg(
        appearances=("timestamp", "size"),
        unique_event_count=("timestamp", "nunique"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        amount_abs_mean=("amount", lambda values: float(values.abs().mean())),
        amount_abs_std=("amount", lambda values: float(values.abs().std(ddof=0))),
        mcc=("mcc", _mode),
        type=("type", _mode),
        direction=("direction", _mode),
    ).reset_index()

    unique_times = frame.drop_duplicates(keys + ["timestamp"])
    unique_times["gap_days"] = (
        unique_times.groupby(keys, sort=False)["timestamp"].diff().dt.total_seconds().div(86_400)
    )
    gaps = unique_times.groupby(keys, sort=False)["gap_days"].agg(
        interval_median="median",
        interval_mean="mean",
        interval_std=lambda values: values.std(ddof=0),
    )
    streams = streams.join(gaps, on=keys)
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    streams["days_since_last"] = (cutoff - streams["last_timestamp"]).dt.total_seconds().div(86_400)
    streams["active_span_days"] = (
        (streams["last_timestamp"] - streams["first_timestamp"]).dt.total_seconds().div(86_400)
    )
    streams["interval_cv"] = streams["interval_std"].div(streams["interval_mean"].clip(lower=1.0))
    streams["regularity"] = 1.0 / (1.0 + streams["interval_cv"].fillna(5.0))
    closeness = []
    for cycle in (7.0, 14.0, 30.0, 31.0, 90.0):
        closeness.append(np.exp(-(streams["interval_median"] - cycle).abs() / (0.2 * cycle)))
    streams["periodicity_closeness"] = pd.concat(closeness, axis=1).max(axis=1).fillna(0.0)
    streams["amount_cv"] = streams["amount_abs_std"].div(
        streams["amount_abs_mean"].clip(lower=1e-6)
    )
    streams["amount_stability"] = 1.0 / (1.0 + streams["amount_cv"].fillna(10.0))
    exposure = (cutoff - streams["first_timestamp"]).dt.total_seconds().div(86_400).clip(lower=1.0)
    recent_window = np.minimum(exposure, 90.0)
    recent_count = (
        frame.loc[frame["timestamp"].ge(cutoff - pd.Timedelta(days=90))]
        .groupby(keys)
        .size()
        .reindex(pd.MultiIndex.from_frame(streams[keys]), fill_value=0)
        .to_numpy()
    )
    streams["recent_to_history_rate"] = (recent_count / recent_window) / (
        streams["appearances"] / exposure
    )
    return streams.replace([np.inf, -np.inf], np.nan)


def recurring_or_fallback(streams: pd.DataFrame) -> pd.DataFrame:
    """Keep repeated streams; retain singletons only for clients without recurrence."""
    repeated = streams.loc[streams["unique_event_count"].ge(2)].copy()
    missing = pd.Index(streams["client_id"].unique()).difference(repeated["client_id"].unique())
    if len(missing):
        repeated = pd.concat(
            [repeated, streams.loc[streams["client_id"].isin(missing)]], ignore_index=True
        )
    return repeated


def learn_associations(
    streams: pd.DataFrame, targets: pd.Series, *, min_support: int = 2
) -> AssociationTable:
    """Learn smoothed description lift from unique fit-client presence."""
    present = streams[["client_id", "description"]].drop_duplicates()
    present[TARGET_COLUMN] = present["client_id"].map(targets)
    if present[TARGET_COLUMN].isna().any():
        raise ValueError("Association fit contains clients without labels")
    counts = pd.crosstab(present["description"], present[TARGET_COLUMN]).reindex(
        columns=LABELS, fill_value=0
    )
    class_sizes = targets.value_counts().reindex(LABELS, fill_value=0).astype(float)
    total = counts.sum(axis=1).astype(float)
    score_columns = {}
    for label in POSITIVE_LABELS:
        inside = (counts[label] + 0.5) / (class_sizes[label] + 1.0)
        outside = (total - counts[label] + 0.5) / (len(targets) - class_sizes[label] + 1.0)
        score_columns[label] = np.log(inside / outside)
    scores = pd.DataFrame(score_columns).where(total.ge(min_support), 0.0)
    return AssociationTable(counts, class_sizes, len(targets), scores)


def select_weak_candidates(
    streams: pd.DataFrame, targets: pd.Series, associations: AssociationTable
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Select one weakly labelled stream per positive client with own counts removed."""
    positive_targets = targets.loc[targets.isin(POSITIVE_LABELS)]
    candidates = recurring_or_fallback(
        streams.loc[streams["client_id"].isin(positive_targets.index)]
    ).copy()
    candidate_labels = candidates["client_id"].map(positive_targets)
    supports = associations.counts.sum(axis=1)
    lift_values = []
    support_values = []
    for description, label in zip(candidates["description"], candidate_labels, strict=True):
        label_count = float(associations.counts.at[description, label]) - 1.0
        support = float(supports.at[description]) - 1.0
        class_size = float(associations.class_sizes[label]) - 1.0
        inside = (label_count + 0.5) / (class_size + 1.0)
        outside_count = support - label_count
        outside_clients = (associations.n_clients - 1.0) - class_size
        outside = (outside_count + 0.5) / (outside_clients + 1.0)
        lift_values.append(math.log(max(inside, 1e-12) / max(outside, 1e-12)))
        support_values.append(support)
    candidates["weak_label"] = candidate_labels.to_numpy()
    candidates["loo_log_lift"] = lift_values
    candidates["loo_support"] = support_values
    candidates["candidate_score"] = (
        candidates["loo_log_lift"]
        + 0.15 * np.log1p(candidates["appearances"])
        + 0.20 * candidates["periodicity_closeness"]
        + 0.10 * candidates["amount_stability"]
        - 0.002 * candidates["days_since_last"].clip(upper=365)
    )
    eligible = candidates["loo_support"].ge(2) & candidates["loo_log_lift"].ge(math.log(1.25))
    candidates["eligible"] = eligible
    chosen_rows = []
    fallback_count = 0
    for _, group in candidates.groupby("client_id", sort=False):
        pool = group.loc[group["eligible"]]
        if pool.empty:
            pool = group
            fallback_count += 1
        chosen_rows.append(pool["candidate_score"].idxmax())
    selected = candidates.loc[chosen_rows].reset_index(drop=True)
    diagnostics = {
        "positive_clients": float(len(positive_targets)),
        "selected_candidates": float(len(selected)),
        "fallback_clients": float(fallback_count),
        "eligible_selection_rate": float(selected["eligible"].mean()),
        "median_loo_support": float(selected["loo_support"].median()),
        "median_loo_log_lift": float(selected["loo_log_lift"].median()),
        "median_appearances": float(selected["appearances"].median()),
    }
    return selected, diagnostics


class TextFamilyModel:
    """Javier's frozen char model; no binary none gate and no VALID selection."""

    def fit(self, transactions, labels):
        validate_history(transactions)
        target = labels.set_index("client_id")[TARGET_COLUMN]
        if not target.index.is_unique or set(target.index) != set(transactions.client_id):
            raise ValueError("Unique labels must exactly match history clients")
        self.fit_clients_ = set(target.index)
        streams = build_streams(transactions)
        associations = learn_associations(streams, target)
        selected, self.diagnostics_ = select_weak_candidates(streams, target, associations)
        self.model_ = make_pipeline(
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=12000,
                sublinear_tf=True,
            ),
            LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            ),
        )
        self.model_.fit(selected.description, selected.weak_label)
        return self

    def predict_scores(self, transactions):
        validate_history(transactions)
        if self.fit_clients_.intersection(transactions.client_id):
            raise ValueError("Prediction clients must be excluded from family model fit")
        streams = recurring_or_fallback(build_streams(transactions)).reset_index(drop=True)
        scores = pd.DataFrame(
            self.model_.predict_proba(streams.description),
            columns=self.model_.classes_,
            index=streams.index,
        )
        scores["client_id"] = streams.client_id
        return scores.groupby("client_id").max().reindex(columns=POSITIVE_LABELS)


def focused_correction(v2_prediction, scores):
    """Fixed 0.85 score gate, preserving V2 none; scores are not calibrated."""
    if set(v2_prediction.index) != set(scores.index) or not scores.index.is_unique:
        raise ValueError("Family scores must exactly match prediction clients")
    scores = scores.reindex(v2_prediction.index)
    family = scores.idxmax(axis=1)
    eligible = (
        v2_prediction.ne("none") & scores.max(axis=1).ge(0.85) & family.isin(("music", "streaming"))
    )
    return v2_prediction.where(~eligible, family)
