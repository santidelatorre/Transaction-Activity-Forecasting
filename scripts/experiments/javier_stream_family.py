"""Leakage-safe discovery experiment for recurring stream family classification.

The challenge target exists at client level, not stream level.  This script therefore
mines one candidate recurring stream per positive training client with leave-one-client-
out alias statistics.  Every reported TRAIN estimate is out of fold by client.  VALID is
used only after the configuration has been selected from TRAIN OOF results.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from transaction_forecasting.ubs.data import (
    LABELS,
    TARGET_COLUMN,
    load_ubs_data,
)
from transaction_forecasting.ubs.text_v2 import normalize_description

POSITIVE_LABELS = tuple(label for label in LABELS if label != "none")
TEXT_CONFIGS: dict[str, tuple[str, ...]] = {
    "tfidf_word": ("word",),
    "tfidf_char": ("char",),
    "tfidf_word_char": ("word", "char"),
    "text_mcc_type_direction": ("word", "char", "categorical"),
    "text_periodicity": ("word", "char", "periodicity"),
    "text_amount_stability": ("word", "char", "amount"),
    "text_other_signals": ("word", "char", "other"),
    "text_all": ("word", "char", "categorical", "periodicity", "amount", "other"),
}
PERIODICITY_COLUMNS = [
    "interval_median",
    "interval_cv",
    "periodicity_closeness",
    "regularity",
]
AMOUNT_COLUMNS = ["amount_abs_mean", "amount_cv", "amount_stability"]
OTHER_COLUMNS = [
    "appearances",
    "unique_event_count",
    "days_since_last",
    "active_span_days",
    "recent_to_history_rate",
]
CATEGORICAL_COLUMNS = ["mcc", "type", "direction"]


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


def make_stream_model(parts: tuple[str, ...]) -> Pipeline:
    """Build one fixed, regularized linear model for a named ablation."""
    transformers: list[tuple[str, Any, Any]] = []
    if "word" in parts:
        transformers.append(
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True),
                "description",
            )
        )
    if "char" in parts:
        transformers.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=12_000,
                    sublinear_tf=True,
                ),
                "description",
            )
        )
    if "categorical" in parts:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            )
        )
    for name, columns in (
        ("periodicity", PERIODICITY_COLUMNS),
        ("amount", AMOUNT_COLUMNS),
        ("other", OTHER_COLUMNS),
    ):
        if name in parts:
            transformers.append(
                (
                    name,
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler(with_mean=False)),
                        ]
                    ),
                    columns,
                )
            )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def aggregate_stream_probabilities(
    streams: pd.DataFrame, probabilities: np.ndarray, classes: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the strongest stream evidence per family and retain the winning stream."""
    probability_frame = pd.DataFrame(probabilities, columns=classes, index=streams.index)
    probability_frame["client_id"] = streams["client_id"]
    client_scores = probability_frame.groupby("client_id", sort=False).max()
    client_scores = client_scores.reindex(columns=POSITIVE_LABELS, fill_value=0.0)
    predictions = client_scores.idxmax(axis=1)
    details = []
    for client_id, predicted in predictions.items():
        client_rows = streams.loc[streams["client_id"].eq(client_id)]
        winning_index = probability_frame.loc[client_rows.index, predicted].idxmax()
        row = streams.loc[winning_index]
        details.append(
            {
                "client_id": client_id,
                "prediction": predicted,
                "confidence": float(probability_frame.at[winning_index, predicted]),
                "winning_description": row["description"],
                "winning_appearances": int(row["appearances"]),
                "winning_known": False,
            }
        )
    return client_scores, pd.DataFrame(details).set_index("client_id")


def predict_rules(
    streams: pd.DataFrame, associations: AssociationTable
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact train-derived alias rule; unsupported aliases contribute no family evidence."""
    eligible = recurring_or_fallback(streams)
    raw_scores = associations.scores.reindex(eligible["description"]).fillna(0.0)
    # Logistic squashing makes rule confidence bounded but is not calibration.
    probabilities = 1.0 / (1.0 + np.exp(-raw_scores.to_numpy()))
    scores, details = aggregate_stream_probabilities(
        eligible.reset_index(drop=True), probabilities, np.asarray(POSITIVE_LABELS)
    )
    known = set(associations.scores.index)
    details["winning_known"] = details["winning_description"].isin(known)
    return scores, details


def predict_stream_model(
    model: Pipeline, streams: pd.DataFrame, known: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = recurring_or_fallback(streams).reset_index(drop=True)
    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(eligible)
    scores, details = aggregate_stream_probabilities(eligible, probabilities, classifier.classes_)
    details["winning_known"] = details["winning_description"].isin(known)
    return scores, details


def make_none_gate() -> Pipeline:
    """Separate client-level binary model for the none decision."""
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "word",
                            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                            "document",
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=10_000,
                                sublinear_tf=True,
                            ),
                            "document",
                        ),
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler(with_mean=False)),
                                ]
                            ),
                            [
                                "stream_count",
                                "repeated_stream_count",
                                "max_appearances",
                                "max_periodicity",
                                "max_regularity",
                                "min_recency",
                            ],
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def build_none_features(streams: pd.DataFrame, client_ids: pd.Index) -> pd.DataFrame:
    rows = []
    for client_id in client_ids:
        group = streams.loc[streams["client_id"].eq(client_id)]
        repeated = group.loc[group["unique_event_count"].ge(2)]
        document_group = repeated if len(repeated) else group
        rows.append(
            {
                "client_id": client_id,
                "document": " ".join(document_group["description"].astype(str)),
                "stream_count": len(group),
                "repeated_stream_count": len(repeated),
                "max_appearances": group["appearances"].max(),
                "max_periodicity": group["periodicity_closeness"].max(),
                "max_regularity": group["regularity"].max(),
                "min_recency": group["days_since_last"].min(),
            }
        )
    return pd.DataFrame(rows).set_index("client_id")


def family_metrics(target: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    target = target.loc[target.isin(POSITIVE_LABELS)]
    prediction = prediction.reindex(target.index)
    report = classification_report(
        target,
        prediction,
        labels=POSITIVE_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                target,
                prediction,
                labels=POSITIVE_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(target.eq(prediction).mean()),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in POSITIVE_LABELS
        },
        "confusion_matrix": confusion_matrix(target, prediction, labels=POSITIVE_LABELS).tolist(),
    }


def all_class_metrics(target: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    report = classification_report(
        target, prediction.reindex(target.index), labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": float(
            f1_score(
                target,
                prediction.reindex(target.index),
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(target.eq(prediction.reindex(target.index)).mean()),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(target, prediction, labels=LABELS).tolist(),
    }


def alias_table(associations: AssociationTable) -> pd.DataFrame:
    support = associations.counts.sum(axis=1)
    learned_family = associations.scores.idxmax(axis=1)
    rows = []
    for description, family in learned_family.items():
        family_support = int(associations.counts.at[description, family])
        rows.append(
            {
                "description": description,
                "learned_family": family,
                "support": int(support.at[description]),
                "family_support": family_support,
                "confidence": float(
                    (family_support + 1.0) / (support.at[description] + len(LABELS))
                ),
                "log_lift": float(associations.scores.at[description, family]),
            }
        )
    return (
        pd.DataFrame(rows)
        .loc[lambda frame: frame["support"].ge(3) & frame["log_lift"].gt(0)]
        .sort_values(["confidence", "support", "log_lift"], ascending=False)
        .reset_index(drop=True)
    )


def save_confusion(path: Path, matrix: list[list[int]], labels: tuple[str, ...]) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("actual").to_csv(path)


def evidence_analysis(
    target: pd.Series, details: pd.DataFrame, train_aliases: set[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    positive = target.loc[target.isin(POSITIVE_LABELS)]
    joined = details.reindex(positive.index).copy()
    joined["actual"] = positive
    joined["correct"] = joined["prediction"].eq(joined["actual"])
    joined["known"] = joined["winning_description"].isin(train_aliases)
    joined["evidence_band"] = pd.cut(
        joined["winning_appearances"],
        bins=[0, 2, 4, np.inf],
        labels=["2", "3-4", "5+"],
    ).astype(str)
    rows = []
    for band, group in joined.groupby("evidence_band", observed=True):
        rows.append(
            {
                "evidence_band": band,
                "clients": len(group),
                "accuracy": float(group["correct"].mean()),
                "macro_f1": float(
                    f1_score(
                        group["actual"],
                        group["prediction"],
                        labels=POSITIVE_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "mean_confidence": float(group["confidence"].mean()),
            }
        )
    coverage = {
        "winning_known_clients": int(joined["known"].sum()),
        "winning_unseen_clients": int((~joined["known"]).sum()),
        "known_accuracy": float(joined.loc[joined["known"], "correct"].mean()),
        "unseen_accuracy": (
            float(joined.loc[~joined["known"], "correct"].mean())
            if (~joined["known"]).any()
            else None
        ),
    }
    return pd.DataFrame(rows), coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/ubs_2026")
    parser.add_argument("--output-dir", default="outputs/metrics/v3_discovery/javier_stream_family")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-v2-integration", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_ubs_data(args.data_dir)
    train_target = data.train_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    valid_target = data.valid_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    train_streams = build_streams(data.train_transactions)
    valid_streams = build_streams(data.valid_transactions)

    names = ["keyword_alias_rules", *TEXT_CONFIGS]
    oof_predictions = {name: pd.Series(index=train_target.index, dtype=object) for name in names}
    oof_details = {
        name: pd.DataFrame(
            index=train_target.index,
            columns=[
                "prediction",
                "confidence",
                "winning_description",
                "winning_appearances",
                "winning_known",
            ],
        )
        for name in names
    }
    none_oof = pd.Series(index=train_target.index, dtype=bool)
    fold_rows = []
    candidate_diagnostics = []
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    for fold, (fit_position, held_position) in enumerate(
        splitter.split(train_target.index, train_target), start=1
    ):
        fit_ids = train_target.index[fit_position]
        held_ids = train_target.index[held_position]
        if set(fit_ids).intersection(held_ids):
            raise RuntimeError("Client leakage between fit and holdout")
        fit_target = train_target.loc[fit_ids]
        held_target = train_target.loc[held_ids]
        fit_streams = train_streams.loc[train_streams["client_id"].isin(fit_ids)]
        held_streams = train_streams.loc[train_streams["client_id"].isin(held_ids)]
        associations = learn_associations(fit_streams, fit_target)
        candidates, diagnostics = select_weak_candidates(fit_streams, fit_target, associations)
        diagnostics["fold"] = float(fold)
        candidate_diagnostics.append(diagnostics)
        known = set(associations.counts.index)

        rule_scores, rule_details = predict_rules(held_streams, associations)
        oof_predictions["keyword_alias_rules"].loc[held_ids] = rule_scores.idxmax(axis=1).reindex(
            held_ids
        )
        oof_details["keyword_alias_rules"].loc[held_ids] = rule_details.reindex(held_ids)
        fold_rows.append(
            {
                "fold": fold,
                "configuration": "keyword_alias_rules",
                **family_metrics(held_target, rule_scores.idxmax(axis=1)),
            }
        )
        for name, parts in TEXT_CONFIGS.items():
            model = make_stream_model(parts)
            model.fit(candidates, candidates["weak_label"])
            scores, details = predict_stream_model(model, held_streams, known)
            oof_predictions[name].loc[held_ids] = scores.idxmax(axis=1).reindex(held_ids)
            oof_details[name].loc[held_ids] = details.reindex(held_ids)
            fold_rows.append(
                {
                    "fold": fold,
                    "configuration": name,
                    **family_metrics(held_target, scores.idxmax(axis=1)),
                }
            )

        fit_none_features = build_none_features(fit_streams, fit_ids)
        held_none_features = build_none_features(held_streams, held_ids)
        none_gate = make_none_gate()
        none_gate.fit(fit_none_features, fit_target.eq("none"))
        none_oof.loc[held_ids] = none_gate.predict(held_none_features)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "candidates": int(diagnostics["selected_candidates"]),
                    "fallbacks": int(diagnostics["fallback_clients"]),
                }
            ),
            flush=True,
        )

    if any(prediction.isna().any() for prediction in oof_predictions.values()):
        raise RuntimeError("Incomplete OOF family predictions")
    if none_oof.isna().any():
        raise RuntimeError("Incomplete OOF none predictions")
    oof_metrics = {
        name: family_metrics(train_target, prediction)
        for name, prediction in oof_predictions.items()
    }
    best_name = max(oof_metrics, key=lambda name: oof_metrics[name]["macro_f1"])
    combined_oof = oof_predictions[best_name].copy()
    combined_oof.loc[none_oof.astype(bool)] = "none"
    none_binary_report = classification_report(
        train_target.eq("none"), none_oof.astype(bool), output_dict=True, zero_division=0
    )
    oof_all_metrics = all_class_metrics(train_target, combined_oof)

    final_associations = learn_associations(train_streams, train_target)
    final_candidates, final_candidate_diagnostics = select_weak_candidates(
        train_streams, train_target, final_associations
    )
    train_aliases = set(final_associations.counts.index)
    valid_predictions = {}
    valid_details = {}
    rule_scores, rule_details = predict_rules(valid_streams, final_associations)
    valid_predictions["keyword_alias_rules"] = rule_scores.idxmax(axis=1)
    valid_details["keyword_alias_rules"] = rule_details
    for name, parts in TEXT_CONFIGS.items():
        model = make_stream_model(parts)
        model.fit(final_candidates, final_candidates["weak_label"])
        scores, details = predict_stream_model(model, valid_streams, train_aliases)
        valid_predictions[name] = scores.idxmax(axis=1)
        valid_details[name] = details
    valid_metrics = {
        name: family_metrics(valid_target, prediction)
        for name, prediction in valid_predictions.items()
    }

    final_none_gate = make_none_gate()
    train_none_features = build_none_features(train_streams, train_target.index)
    valid_none_features = build_none_features(valid_streams, valid_target.index)
    final_none_gate.fit(train_none_features, train_target.eq("none"))
    valid_none = pd.Series(
        final_none_gate.predict(valid_none_features), index=valid_none_features.index
    )
    valid_combined = valid_predictions[best_name].copy()
    valid_combined.loc[valid_none] = "none"
    valid_all_metrics = all_class_metrics(valid_target, valid_combined)
    valid_none_report = classification_report(
        valid_target.eq("none"), valid_none, output_dict=True, zero_division=0
    )

    aliases = alias_table(final_associations)
    aliases.to_csv(output_dir / "learned_aliases.csv", index=False)
    evidence, winning_coverage = evidence_analysis(
        valid_target, valid_details[best_name], train_aliases
    )
    evidence.to_csv(output_dir / "evidence_bands.csv", index=False)
    valid_recurring = recurring_or_fallback(valid_streams)
    valid_recurring["known"] = valid_recurring["description"].isin(train_aliases)
    coverage = {
        "train_unique_descriptions": len(train_aliases),
        "valid_unique_descriptions": int(valid_recurring["description"].nunique()),
        "valid_known_unique_descriptions": int(
            valid_recurring.loc[valid_recurring["known"], "description"].nunique()
        ),
        "valid_unseen_unique_descriptions": int(
            valid_recurring.loc[~valid_recurring["known"], "description"].nunique()
        ),
        "valid_stream_row_known_share": float(valid_recurring["known"].mean()),
        "valid_transaction_weighted_known_share": float(
            valid_recurring.loc[valid_recurring["known"], "appearances"].sum()
            / valid_recurring["appearances"].sum()
        ),
        **winning_coverage,
    }

    v2_integration = None
    v2_path = Path("outputs/metrics/ubs_v2/validation_predictions.csv")
    if not args.skip_v2_integration and v2_path.exists():
        v2 = pd.read_csv(v2_path, dtype=str).set_index("client_id").iloc[:, 0]
        if set(v2.index) == set(valid_target.index):
            v2 = v2.reindex(valid_target.index)
            details = valid_details[best_name].reindex(valid_target.index)
            # Keep V2's stronger none decision. Weak-label probabilities are ranking scores only.
            high_confidence = details["confidence"].astype(float).ge(0.85)
            positive_override = high_confidence & v2.ne("none")
            focused_override = positive_override & valid_predictions[best_name].reindex(
                valid_target.index
            ).isin(("music", "streaming"))
            integrated_positive = v2.copy()
            integrated_positive.loc[positive_override] = (
                valid_predictions[best_name].reindex(valid_target.index).loc[positive_override]
            )
            integrated_focused = v2.copy()
            integrated_focused.loc[focused_override] = (
                valid_predictions[best_name].reindex(valid_target.index).loc[focused_override]
            )
            v2_integration = {
                "threshold": 0.85,
                "none_policy": "preserve V2 none predictions",
                "v2": all_class_metrics(valid_target, v2),
                "all_positive_families": {
                    "eligible": int(positive_override.sum()),
                    "changed_predictions": int((integrated_positive != v2).sum()),
                    "metrics": all_class_metrics(valid_target, integrated_positive),
                },
                "music_streaming_only": {
                    "eligible": int(focused_override.sum()),
                    "changed_predictions": int((integrated_focused != v2).sum()),
                    "metrics": all_class_metrics(valid_target, integrated_focused),
                },
                "selection_warning": (
                    "Exploratory VALID comparison; do not promote without fresh held-out evidence."
                ),
            }

    comparison_rows = []
    for name in names:
        comparison_rows.append(
            {
                "configuration": name,
                "oof_macro_f1_positive": oof_metrics[name]["macro_f1"],
                "oof_accuracy_positive": oof_metrics[name]["accuracy"],
                "valid_macro_f1_positive": valid_metrics[name]["macro_f1"],
                "valid_accuracy_positive": valid_metrics[name]["accuracy"],
                "selected_from_oof": name == best_name,
            }
        )
    pd.DataFrame(comparison_rows).sort_values("oof_macro_f1_positive", ascending=False).to_csv(
        output_dir / "model_comparison.csv", index=False
    )

    predictions_frame = pd.DataFrame(
        {
            "client_id": valid_target.index,
            "actual": valid_target,
            "family_prediction": valid_predictions[best_name].reindex(valid_target.index),
            "none_gate": valid_none.reindex(valid_target.index),
            "combined_prediction": valid_combined.reindex(valid_target.index),
            "confidence": valid_details[best_name].reindex(valid_target.index)["confidence"],
            "winning_description": valid_details[best_name].reindex(valid_target.index)[
                "winning_description"
            ],
            "winning_appearances": valid_details[best_name].reindex(valid_target.index)[
                "winning_appearances"
            ],
        }
    )
    predictions_frame.to_csv(output_dir / "valid_predictions.csv", index=False)
    save_confusion(
        output_dir / "confusion_positive_oof.csv",
        oof_metrics[best_name]["confusion_matrix"],
        POSITIVE_LABELS,
    )
    save_confusion(
        output_dir / "confusion_positive_valid.csv",
        valid_metrics[best_name]["confusion_matrix"],
        POSITIVE_LABELS,
    )
    save_confusion(
        output_dir / "confusion_all_valid.csv",
        valid_all_metrics["confusion_matrix"],
        LABELS,
    )
    (output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    results = {
        "protocol": {
            "folds": args.folds,
            "seed": args.seed,
            "grouping": "client_id; one label row per client; disjoint outer folds",
            "candidate_selection": (
                "one recurring stream per positive fit client; alias counts exclude the client "
                "being selected; support/lift plus recurrence, periodicity, amount stability, "
                "and recency"
            ),
            "valid_policy": "final fit on TRAIN only; VALID inference and scoring only",
            "positive_labels": POSITIVE_LABELS,
        },
        "best_oof_configuration": best_name,
        "oof_positive": oof_metrics,
        "valid_positive": valid_metrics,
        "none_oof_binary": {
            "precision": float(none_binary_report["True"]["precision"]),
            "recall": float(none_binary_report["True"]["recall"]),
            "f1": float(none_binary_report["True"]["f1-score"]),
            "support": int(none_binary_report["True"]["support"]),
        },
        "none_valid_binary": {
            "precision": float(valid_none_report["True"]["precision"]),
            "recall": float(valid_none_report["True"]["recall"]),
            "f1": float(valid_none_report["True"]["f1-score"]),
            "support": int(valid_none_report["True"]["support"]),
            "predicted_none": int(valid_none.sum()),
        },
        "oof_all_with_none_gate": oof_all_metrics,
        "valid_all_with_none_gate": valid_all_metrics,
        "coverage": coverage,
        "candidate_diagnostics_by_fold": candidate_diagnostics,
        "fold_metrics": fold_rows,
        "final_candidate_diagnostics": final_candidate_diagnostics,
        "evidence_bands": evidence.to_dict(orient="records"),
        "music_to_streaming_valid": int(
            (
                valid_target.eq("music")
                & valid_predictions[best_name].reindex(valid_target.index).eq("streaming")
            ).sum()
        ),
        "streaming_to_music_valid": int(
            (
                valid_target.eq("streaming")
                & valid_predictions[best_name].reindex(valid_target.index).eq("music")
            ).sum()
        ),
        "v2_integration": v2_integration,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "best": best_name,
                "oof_positive_macro_f1": oof_metrics[best_name]["macro_f1"],
                "valid_positive_macro_f1": valid_metrics[best_name]["macro_f1"],
                "valid_all_macro_f1": valid_all_metrics["macro_f1"],
                "music_f1": valid_metrics[best_name]["per_class"]["music"]["f1"],
                "streaming_f1": valid_metrics[best_name]["per_class"]["streaming"]["f1"],
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
