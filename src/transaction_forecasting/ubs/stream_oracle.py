"""Leakage-safe diagnostics for exact-description recurring streams.

This module is deliberately an audit, not a predictive V3 model.  Stream
geometry never uses labels; the exact-description family map is fitted only on
labelled training clients and can then be applied to disjoint clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.evaluation.official import classification_metrics
from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN

POSITIVE_LABELS = tuple(label for label in LABELS if label != "none")
HORIZON_DAYS = 90
MIN_APPEARANCES = 2
MIN_MAPPING_SUPPORT = 2
MIN_MAPPING_LIFT = 1.5


def _mode(series: pd.Series) -> object:
    modes = series.astype(str).mode()
    return modes.iloc[0] if not modes.empty else ""


def _robust_mean(values: np.ndarray) -> float:
    """Return a 10%-winsorized mean, defined even for short gap sequences."""
    lower, upper = np.quantile(values, [0.1, 0.9])
    return float(np.clip(values, lower, upper).mean())


def _periodicity(gap_days: float) -> str:
    if gap_days < 10:
        return "weekly"
    if gap_days < 18:
        return "biweekly"
    if gap_days < 45:
        return "monthly"
    if gap_days < 110:
        return "quarterly"
    if gap_days <= 400:
        return "annual"
    return "other"


def project_next_date(
    last_timestamp: pd.Timestamp, gap_days: float, cutoff: pd.Timestamp = CUTOFF
) -> pd.Timestamp:
    """Project the first cadence occurrence on or after the cutoff."""
    last = pd.Timestamp(last_timestamp)
    cutoff = pd.Timestamp(cutoff)
    if last.tzinfo is None or cutoff.tzinfo is None:
        raise ValueError("Projection timestamps must be timezone-aware")
    if last >= cutoff or not np.isfinite(gap_days) or gap_days <= 0:
        raise ValueError("Projection needs a pre-cutoff event and a positive finite gap")
    cycles = max(1, ceil((cutoff - last).total_seconds() / 86400 / gap_days))
    return last + pd.to_timedelta(cycles * gap_days, unit="D")


def build_streams(
    transactions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp = CUTOFF,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Summarize every exact ``(client_id, description)`` stream.

    A candidate is intentionally liberal for an upper-bound audit: at least two
    observations and a median-gap projection in the closed 90-day horizon.
    No label, client-id value, validation statistic, or post-cutoff event is used.
    """
    cutoff = pd.Timestamp(cutoff)
    required = {
        "client_id",
        "description",
        "timestamp",
        "amount",
        "mcc",
        "type",
        "direction",
        "currency",
    }
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing stream columns: {sorted(missing)}")
    if cutoff.tzinfo is None or horizon_days <= 0:
        raise ValueError("A timezone-aware cutoff and positive horizon are required")
    frame = transactions.loc[:, sorted(required)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame[["client_id", "description", "timestamp"]].isna().any().any():
        raise ValueError("Null stream keys or timestamps")
    if frame["timestamp"].ge(cutoff).any():
        raise ValueError("Stream input contains transactions at or after cutoff")
    frame = frame.sort_values(["client_id", "description", "timestamp"], kind="stable")

    rows: list[dict[str, object]] = []
    for (client_id, description), group in frame.groupby(["client_id", "description"], sort=True):
        timestamps = pd.DatetimeIndex(group["timestamp"].sort_values())
        gaps = np.diff(timestamps.as_unit("ns").asi8) / 86_400_000_000_000
        amounts = pd.to_numeric(group["amount"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(amounts).all():
            raise ValueError("Stream amounts must be finite numeric values")
        appearances = len(timestamps)
        gap_median = float(np.median(gaps)) if len(gaps) else np.nan
        projected = (
            project_next_date(timestamps[-1], gap_median, cutoff)
            if len(gaps) and gap_median > 0
            else pd.NaT
        )
        horizon_end = cutoff + pd.Timedelta(days=horizon_days)
        is_candidate = bool(
            appearances >= MIN_APPEARANCES
            and pd.notna(projected)
            and cutoff <= projected <= horizon_end
        )
        amount_mean = float(amounts.mean())
        amount_std = float(amounts.std(ddof=0))
        rows.append(
            {
                "client_id": str(client_id),
                "description": str(description),
                "appearances": appearances,
                "timestamps": tuple(timestamp.isoformat() for timestamp in timestamps),
                "last_timestamp": timestamps[-1],
                "penultimate_timestamp": timestamps[-2] if appearances >= 2 else pd.NaT,
                "gaps_days": tuple(float(value) for value in gaps),
                "gap_median_days": gap_median,
                "gap_robust_mean_days": _robust_mean(gaps) if len(gaps) else np.nan,
                "gap_std_days": float(gaps.std(ddof=0)) if len(gaps) else np.nan,
                "gap_mad_days": (
                    float(np.median(np.abs(gaps - gap_median))) if len(gaps) else np.nan
                ),
                "recency_days": (cutoff - timestamps[-1]).total_seconds() / 86400,
                "amount_mean": amount_mean,
                "amount_median": float(np.median(amounts)),
                "amount_std": amount_std,
                "amount_mad": float(np.median(np.abs(amounts - np.median(amounts)))),
                "amount_cv": amount_std / abs(amount_mean) if amount_mean else np.nan,
                "mcc": _mode(group["mcc"]),
                "type": _mode(group["type"]),
                "direction": _mode(group["direction"]),
                "currency": _mode(group["currency"]),
                "approx_periodicity": _periodicity(gap_median) if len(gaps) else "unknown",
                "projected_next_date": projected,
                "is_recurrent": appearances >= MIN_APPEARANCES,
                "is_candidate": is_candidate,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class TrainOnlyFamilyMapper:
    """One-family-per-description mapping learned from training clients only."""

    min_support: int = MIN_MAPPING_SUPPORT
    min_lift: float = MIN_MAPPING_LIFT

    def fit(self, candidate_streams: pd.DataFrame, train_labels: pd.DataFrame):
        required = {"client_id", "description", "is_candidate"}
        if required.difference(candidate_streams.columns):
            raise ValueError("Candidate stream table has an invalid schema")
        if {"client_id", TARGET_COLUMN}.difference(train_labels.columns):
            raise ValueError("Training labels have an invalid schema")
        labels = train_labels[["client_id", TARGET_COLUMN]].copy()
        labels["client_id"] = labels["client_id"].astype(str)
        if labels["client_id"].duplicated().any() or not labels[TARGET_COLUMN].isin(LABELS).all():
            raise ValueError("Training labels must contain unique clients and official classes")
        stream_clients = set(candidate_streams["client_id"].astype(str))
        label_clients = set(labels["client_id"])
        if not stream_clients.issubset(label_clients):
            raise ValueError("Every mapping-fit stream must belong to a labelled train client")

        self.fit_clients_ = label_clients
        due = candidate_streams.loc[
            candidate_streams["is_candidate"], ["client_id", "description"]
        ].drop_duplicates()
        due["client_id"] = due["client_id"].astype(str)
        present = due.merge(labels, on="client_id", how="left", validate="many_to_one")
        total_clients = len(labels)
        class_sizes = labels[TARGET_COLUMN].value_counts().reindex(LABELS, fill_value=0)
        description_support = present.groupby("description")["client_id"].nunique()
        records: list[dict[str, object]] = []
        for description, support in description_support.items():
            subset = present[present["description"].eq(description)]
            counts = subset[TARGET_COLUMN].value_counts()
            lifts: dict[str, float] = {}
            for family in POSITIVE_LABELS:
                inside_count = int(counts.get(family, 0))
                outside_count = int(support - inside_count)
                inside_rate = (inside_count + 1.0) / (float(class_sizes[family]) + 2.0)
                outside_rate = (outside_count + 1.0) / (
                    float(total_clients - class_sizes[family]) + 2.0
                )
                lifts[family] = inside_rate / outside_rate
            family = max(POSITIVE_LABELS, key=lambda label: (lifts[label], -LABELS.index(label)))
            eligible = support >= self.min_support and lifts[family] >= self.min_lift
            records.append(
                {
                    "description": description,
                    "family": family if eligible else pd.NA,
                    "support_clients": int(support),
                    "winning_lift": float(lifts[family]),
                    "eligible": bool(eligible),
                }
            )
        self.mapping_table_ = pd.DataFrame(records)
        if self.mapping_table_.empty:
            self.mapping_ = pd.Series(dtype="object")
        else:
            eligible = self.mapping_table_[self.mapping_table_["eligible"]]
            self.mapping_ = eligible.set_index("description")["family"]
        return self

    def transform(self, streams: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "mapping_"):
            raise RuntimeError("Fit the family mapper before transform")
        clients = set(streams["client_id"].astype(str))
        overlap = clients.intersection(self.fit_clients_)
        if overlap:
            raise ValueError("Mapping evaluation clients must be disjoint from mapping-fit clients")
        result = streams.copy()
        result["mapped_family"] = result["description"].map(self.mapping_)
        return result


def client_candidate_table(streams: pd.DataFrame, client_ids: pd.Index | list[str]) -> pd.DataFrame:
    """Collapse stream evidence to candidate-family sets per client."""
    clients = pd.Index([str(value) for value in client_ids], name="client_id")
    recurrent = streams[streams["is_recurrent"]]
    due = streams[streams["is_candidate"]]
    mapped_due = due[due["mapped_family"].notna()]
    mapped_all = recurrent[recurrent["mapped_family"].notna()]

    def sets(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("client_id")["mapped_family"].agg(
            lambda values: tuple(sorted(set(values), key=LABELS.index))
        )

    table = pd.DataFrame(index=clients)
    table["recurrent_streams"] = (
        recurrent.groupby("client_id").size().reindex(clients, fill_value=0)
    )
    table["candidate_streams"] = due.groupby("client_id").size().reindex(clients, fill_value=0)
    table["mapped_candidate_streams"] = (
        mapped_due.groupby("client_id").size().reindex(clients, fill_value=0)
    )
    table["candidate_families"] = (
        sets(mapped_due)
        .reindex(clients)
        .map(lambda value: value if isinstance(value, tuple) else ())
    )
    table["all_recurrent_families"] = (
        sets(mapped_all)
        .reindex(clients)
        .map(lambda value: value if isinstance(value, tuple) else ())
    )
    return table


def _maximum_macro_f1_predictions(table: pd.DataFrame) -> pd.Series:
    """Find the exact best allowed single-label assignment with a binary MILP.

    Correctly covered positives and true ``none`` clients are fixed to their
    correct label.  Each uncovered positive may select one of its detected
    candidate families or abstain to ``none``.  False-positive counts are
    linearized with ordered binary increments of each class F1 curve.
    """
    prediction = table["target"].copy()
    failures = table[table["target"].ne("none") & ~table["target_in_candidates"]]
    if failures.empty:
        return prediction
    allowed = {
        client: tuple((*families, "none"))
        for client, families in failures["candidate_families"].items()
    }
    assignment_keys = [(client, label) for client, choices in allowed.items() for label in choices]
    failure_count = len(failures)
    increment_keys = [(label, k) for label in LABELS for k in range(1, failure_count + 1)]
    variable_count = len(assignment_keys) + len(increment_keys)
    assignment_position = {key: index for index, key in enumerate(assignment_keys)}
    increment_position = {
        key: len(assignment_keys) + index for index, key in enumerate(increment_keys)
    }

    objective = np.zeros(variable_count)
    for label in LABELS:
        support = int(table["target"].eq(label).sum())
        true_positives = (
            support
            if label == "none"
            else int((table["target"].eq(label) & table["target_in_candidates"]).sum())
        )
        false_negatives = support - true_positives

        def f1(
            false_positives: int,
            tp: int = true_positives,
            fn: int = false_negatives,
        ) -> float:
            denominator = 2 * tp + fn + false_positives
            return 2 * tp / denominator if denominator else 0.0

        for k in range(1, failure_count + 1):
            # scipy minimizes, so use the positive loss of the kth false positive.
            objective[increment_position[(label, k)]] = f1(k - 1) - f1(k)

    rows = len(allowed) + len(LABELS) + len(LABELS) * (failure_count - 1)
    matrix = lil_matrix((rows, variable_count), dtype=float)
    lower = np.full(rows, -np.inf)
    upper = np.full(rows, np.inf)
    row = 0
    for client, choices in allowed.items():
        for label in choices:
            matrix[row, assignment_position[(client, label)]] = 1.0
        lower[row] = upper[row] = 1.0
        row += 1
    for label in LABELS:
        for client, choices in allowed.items():
            if label in choices:
                matrix[row, assignment_position[(client, label)]] = 1.0
        for k in range(1, failure_count + 1):
            matrix[row, increment_position[(label, k)]] = -1.0
        lower[row] = upper[row] = 0.0
        row += 1
    for label in LABELS:
        for k in range(1, failure_count):
            matrix[row, increment_position[(label, k + 1)]] = 1.0
            matrix[row, increment_position[(label, k)]] = -1.0
            upper[row] = 0.0
            row += 1
    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 120},
    )
    if not result.success:
        raise RuntimeError(f"Oracle MILP did not reach an optimum: {result.message}")
    for (client, label), position in assignment_position.items():
        if result.x[position] > 0.5:
            prediction.loc[client] = label
    return prediction


def score_candidate_oracle(
    client_table: pd.DataFrame, labels: pd.DataFrame, *, optimize: bool = True
) -> dict[str, object]:
    """Score candidate coverage and optimistic oracle assignments.

    The exact oracle maximizes official Macro-F1 across allowed candidate
    families plus ``none``.  The secondary abstaining oracle sends every
    uncovered positive to ``none``.  Neither is a deployable prediction rule.
    """
    target = labels.set_index("client_id")[TARGET_COLUMN].copy()
    target.index = target.index.astype(str)
    table = client_table.reindex(target.index).copy()
    if table.isna().any().any():
        raise ValueError("Candidate table does not cover every labelled client")
    table["target"] = target
    table["target_in_candidates"] = [
        actual in families and actual != "none"
        for actual, families in zip(table["target"], table["candidate_families"], strict=True)
    ]
    table["abstaining_oracle_prediction"] = np.where(
        table["target_in_candidates"], table["target"], "none"
    )
    table["oracle_prediction"] = (
        _maximum_macro_f1_predictions(table)
        if optimize
        else table["abstaining_oracle_prediction"].copy()
    )
    metrics = classification_metrics(table["target"], table["oracle_prediction"])
    abstaining_metrics = classification_metrics(
        table["target"], table["abstaining_oracle_prediction"]
    )

    positives = table[table["target"].ne("none")]
    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        subset = table[table["target"].eq(label)]
        hits = int(subset["target_in_candidates"].sum()) if label != "none" else 0
        per_class[label] = {
            "support": len(subset),
            "candidate_family_hits": hits,
            "candidate_family_recall": hits / len(subset)
            if len(subset) and label != "none"
            else 0.0,
            "mean_candidate_streams": float(subset["candidate_streams"].mean()),
            "mean_candidate_families": float(subset["candidate_families"].map(len).mean()),
        }
    return {
        "candidate_family_recall": float(positives["target_in_candidates"].mean()),
        "oracle_metrics": metrics,
        "abstaining_oracle_metrics": abstaining_metrics,
        "per_class": per_class,
        "clients": table,
    }


def failure_breakdown(client_table: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return mutually exclusive positive and none failure categories."""
    positive = client_table[client_table["target"].ne("none")].copy()
    positive["failure"] = "family_mapping_miss"
    positive.loc[positive["recurrent_streams"].eq(0), "failure"] = "no_recurrent_stream"
    positive.loc[
        positive["recurrent_streams"].gt(0) & positive["candidate_streams"].eq(0),
        "failure",
    ] = "no_temporally_due_stream"
    temporal_miss = [
        actual in families and actual not in candidates
        for actual, families, candidates in zip(
            positive["target"],
            positive["all_recurrent_families"],
            positive["candidate_families"],
            strict=True,
        )
    ]
    positive.loc[temporal_miss, "failure"] = "target_family_mapped_but_not_due"
    positive.loc[positive["target_in_candidates"], "failure"] = "target_family_candidate"

    none = client_table[client_table["target"].eq("none")].copy()
    none["category"] = "due_streams_without_family_mapping"
    none.loc[none["recurrent_streams"].eq(0), "category"] = "no_recurrent_stream"
    none.loc[none["recurrent_streams"].gt(0) & none["candidate_streams"].eq(0), "category"] = (
        "recurrent_but_not_due"
    )
    none.loc[none["mapped_candidate_streams"].gt(0), "category"] = (
        "due_mapped_stream_indistinguishable"
    )
    positive_categories = (
        "no_recurrent_stream",
        "no_temporally_due_stream",
        "target_family_mapped_but_not_due",
        "family_mapping_miss",
        "target_family_candidate",
    )
    none_categories = (
        "no_recurrent_stream",
        "recurrent_but_not_due",
        "due_streams_without_family_mapping",
        "due_mapped_stream_indistinguishable",
    )
    by_class = {
        label: positive.loc[positive["target"].eq(label), "failure"]
        .value_counts()
        .reindex(positive_categories, fill_value=0)
        .astype(int)
        .to_dict()
        for label in POSITIVE_LABELS
    }
    hits = positive[positive["target_in_candidates"]]
    return {
        "positive": positive["failure"]
        .value_counts()
        .reindex(positive_categories, fill_value=0)
        .astype(int)
        .to_dict(),
        "positive_by_class": by_class,
        "selection_ambiguity": {
            "target_family_candidate": len(hits),
            "single_candidate_family": int(hits["candidate_families"].map(len).eq(1).sum()),
            "multiple_candidate_families": int(hits["candidate_families"].map(len).gt(1).sum()),
            "multiple_candidate_streams": int(hits["candidate_streams"].gt(1).sum()),
        },
        "none": none["category"]
        .value_counts()
        .reindex(none_categories, fill_value=0)
        .astype(int)
        .to_dict(),
    }


def oof_train_audit(
    train_streams: pd.DataFrame, train_labels: pd.DataFrame, folds: int = 5
) -> dict[str, float]:
    """Measure mapping transfer inside train with client-disjoint stratified folds."""
    labels = train_labels[["client_id", TARGET_COLUMN]].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    outputs = []
    for fit_index, held_index in splitter.split(labels["client_id"], labels[TARGET_COLUMN]):
        fit_labels = labels.iloc[fit_index]
        held_labels = labels.iloc[held_index]
        fit_clients = set(fit_labels["client_id"].astype(str))
        held_clients = set(held_labels["client_id"].astype(str))
        mapper = TrainOnlyFamilyMapper().fit(
            train_streams[train_streams["client_id"].astype(str).isin(fit_clients)], fit_labels
        )
        held_streams = mapper.transform(
            train_streams[train_streams["client_id"].astype(str).isin(held_clients)]
        )
        table = client_candidate_table(held_streams, held_labels["client_id"])
        scored = score_candidate_oracle(table, held_labels, optimize=False)
        outputs.append(scored)
    total_hits = sum(int(output["clients"]["target_in_candidates"].sum()) for output in outputs)
    total_positives = sum(int(output["clients"]["target"].ne("none").sum()) for output in outputs)
    combined = pd.concat([output["clients"] for output in outputs])
    metrics = classification_metrics(combined["target"], combined["oracle_prediction"])
    return {
        "folds": folds,
        "candidate_family_recall": total_hits / total_positives,
        "abstaining_oracle_macro_f1": float(metrics["macro_f1"]),
    }


def evidence_verdict(
    recall: float, oracle_macro_f1: float, per_class: dict[str, dict[str, float | int]]
) -> str:
    """Apply predeclared, validation-independent evidence thresholds."""
    weak_classes = min(
        float(per_class["music"]["candidate_family_recall"]),
        float(per_class["streaming"]["candidate_family_recall"]),
    )
    if recall >= 0.75 and oracle_macro_f1 >= 0.65 and weak_classes >= 0.60:
        return "STRONG EVIDENCE FOR STREAM APPROACH"
    if recall < 0.45 or oracle_macro_f1 < 0.45:
        return "WEAK EVIDENCE FOR STREAM APPROACH"
    return "MIXED EVIDENCE"
