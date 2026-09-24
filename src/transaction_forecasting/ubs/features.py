"""One-row-per-client features tailored to the UBS transaction histories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from transaction_forecasting.evaluation.official import validate_client_ids
from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN, validate_history

RECENT_WINDOWS = (7, 14, 30, 60, 90, 180)
PERIODICITY_BINS = {
    "weekly": (4.0, 10.0),
    "biweekly": (10.0, 18.0),
    "monthly": (18.0, 45.0),
    "quarterly": (45.0, 110.0),
    "annual": (110.0, 400.0),
}


def _safe_cv(values: pd.Series) -> float:
    mean = float(values.mean())
    return float(values.std(ddof=0) / mean) if mean else 0.0


def build_client_documents(transactions: pd.DataFrame, client_ids: pd.Index) -> pd.Series:
    """Create train-fittable text documents while preserving repeated descriptions."""
    validate_history(transactions)
    documents = transactions.groupby("client_id", sort=False)["description"].agg(" ".join)
    return documents.reindex(client_ids, fill_value="")


def build_recurrence_streams(transactions: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated merchant-like descriptions per client."""
    validate_history(transactions)
    keys = ["client_id", "description", "currency", "direction"]
    ordered = transactions.sort_values([*keys, "timestamp"]).copy()
    ordered["interval_days"] = (
        ordered.groupby(keys, sort=False)["timestamp"].diff().dt.total_seconds().div(86400)
    )
    grouped = ordered.groupby(keys, sort=False)
    streams = grouped.agg(
        appearances=("timestamp", "size"),
        last_timestamp=("timestamp", "max"),
        amount_mean=("amount", "mean"),
        amount_median=("amount", "median"),
        mcc_unique=("mcc", "nunique"),
        type_unique=("type", "nunique"),
        direction_out_share=("direction", lambda x: float(x.eq("out").mean())),
    )
    streams["amount_std"] = grouped["amount"].std(ddof=0)
    streams["median_interval_days"] = grouped["interval_days"].median()
    streams["interval_std_days"] = grouped["interval_days"].std(ddof=0)
    streams = streams[streams["appearances"].ge(2)].reset_index()
    streams["days_since_last"] = (CUTOFF - streams["last_timestamp"]).dt.total_seconds().div(86400)
    streams["amount_cv"] = streams["amount_std"] / streams["amount_mean"].clip(lower=1e-9)
    streams["interval_cv"] = streams["interval_std_days"] / streams["median_interval_days"].clip(
        lower=1.0
    )
    next_gap = (streams["days_since_last"] - streams["median_interval_days"]).abs()
    streams["regularity"] = 1.0 / (1.0 + streams["interval_cv"])
    streams["due_score"] = np.exp(-next_gap / streams["median_interval_days"].clip(lower=7.0))
    streams["base_recurrence_score"] = (
        np.log1p(streams["appearances"])
        * streams["regularity"]
        * streams["due_score"]
        / (1.0 + streams["amount_cv"])
    )
    streams["mcc_repeats"] = streams["mcc_unique"].eq(1).astype(int)
    streams["type_repeats"] = streams["type_unique"].eq(1).astype(int)
    return streams.drop(columns=["mcc_unique", "type_unique"])


@dataclass
class ClientFeatureBuilder:
    """Fit train-only description associations, then aggregate each split independently."""

    categorical_levels_: dict[str, list[str]] = field(default_factory=dict, init=False)
    description_lift_: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> ClientFeatureBuilder:
        """Learn only vocabulary/category levels and smoothed family-description lift."""
        validate_history(transactions)
        validate_client_ids(labels, "training labels")
        if not labels[TARGET_COLUMN].isin(LABELS).all():
            raise ValueError("Invalid training target labels")
        label_series = labels.set_index("client_id")[TARGET_COLUMN]
        if set(transactions["client_id"]) != set(label_series.index):
            raise ValueError("Training transactions and labels must contain the same clients")
        self.categorical_levels_ = {
            column: sorted(transactions[column].astype(str).unique().tolist())
            for column in ("mcc", "type", "currency", "direction")
        }
        present = transactions[["client_id", "description"]].drop_duplicates()
        present = present.join(label_series, on="client_id")
        class_sizes = label_series.value_counts().reindex(LABELS, fill_value=0).astype(float)
        lifts: dict[str, pd.Series] = {}
        for label in LABELS:
            if class_sizes[label] == 0:
                # Smoothing must not invent positive evidence for an unseen class.
                lifts[label] = pd.Series(0.0, index=present["description"].unique())
                continue
            in_counts = present.loc[present[TARGET_COLUMN].eq(label), "description"].value_counts()
            out_counts = present.loc[
                ~present[TARGET_COLUMN].eq(label), "description"
            ].value_counts()
            index = in_counts.index.union(out_counts.index)
            inside = (in_counts.reindex(index, fill_value=0) + 1.0) / (class_sizes[label] + 2.0)
            outside_n = float(len(label_series) - class_sizes[label])
            outside = (out_counts.reindex(index, fill_value=0) + 1.0) / (outside_n + 2.0)
            log_lift = np.log(inside / outside).clip(lower=0.0, upper=np.log(10.0))
            lifts[label] = log_lift
        lift_table = pd.DataFrame(lifts).fillna(0.0)
        support = present["description"].value_counts().reindex(lift_table.index, fill_value=0)
        winning_label = lift_table.idxmax(axis=1)
        winning_lift = lift_table.max(axis=1)
        exclusive = pd.DataFrame(0.0, index=lift_table.index, columns=lift_table.columns)
        eligible = winning_lift.ge(np.log(1.5)) & support.ge(2)
        for label in LABELS:
            chosen = eligible & winning_label.eq(label)
            exclusive.loc[chosen, label] = winning_lift[chosen]
        self.description_lift_ = exclusive
        self.fitted_ = True
        return self

    def fit_transform(
        self, transactions: pd.DataFrame, labels: pd.DataFrame, *, n_splits: int = 5, seed: int = 42
    ) -> pd.DataFrame:
        """Cross-fit target-derived train features, retaining full-train state for inference.

        Folds contain whole clients and do not depend on labels. Every family's
        training feature excludes that client's entire fold from its lift table.
        Category levels (unsupervised) are learned from train only.
        """
        self.fit(transactions, labels)
        clients = pd.Index(sorted(labels["client_id"]), name="client_id")
        if len(clients) < 2:
            raise ValueError("Cross-fitting needs at least two training clients")
        folds = KFold(n_splits=min(n_splits, len(clients)), shuffle=True, random_state=seed)
        features = self.transform(transactions)
        family_columns = features.columns[features.columns.str.startswith("family_")]
        for train_positions, held_positions in folds.split(clients):
            train_ids, held_ids = clients[train_positions], clients[held_positions]
            fold_builder = ClientFeatureBuilder().fit(
                transactions[transactions["client_id"].isin(train_ids)],
                labels[labels["client_id"].isin(train_ids)],
            )
            held_streams = build_recurrence_streams(
                transactions[transactions["client_id"].isin(held_ids)]
            )
            held_features = fold_builder._add_recurrence_features(
                pd.DataFrame(index=held_ids), held_streams
            )
            features.loc[held_ids, family_columns] = held_features[family_columns]
        return features

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Return deterministic numeric features indexed by client_id."""
        if not self.fitted_:
            raise RuntimeError("ClientFeatureBuilder.fit must be called before transform")
        validate_history(transactions)
        clients = pd.Index(sorted(transactions["client_id"].unique()), name="client_id")
        features = pd.DataFrame(index=clients)
        ordered = transactions.sort_values(["client_id", "timestamp"], kind="stable").copy()
        grouped = ordered.groupby("client_id", sort=False)

        features["n_transactions"] = grouped.size()
        features["n_out"] = grouped["direction"].apply(lambda x: int(x.eq("out").sum()))
        features["n_in"] = grouped["direction"].apply(lambda x: int(x.eq("in").sum()))
        features["out_share"] = features["n_out"] / features["n_transactions"]
        features["fee_positive_share"] = grouped["fee"].apply(lambda x: float(x.gt(0).mean()))
        for column in ("mcc", "description", "type", "currency"):
            features[f"n_unique_{column}"] = grouped[column].nunique()

        first = grouped["timestamp"].min()
        last = grouped["timestamp"].max()
        features["history_days"] = (last - first).dt.total_seconds().div(86400)
        features["days_since_last_transaction"] = (CUTOFF - last).dt.total_seconds().div(86400)
        features["transactions_per_30d"] = features["n_transactions"] / (
            features["history_days"].clip(lower=1.0) / 30.0
        )
        ordered["age_days"] = (CUTOFF - ordered["timestamp"]).dt.total_seconds().div(86400)
        for window in RECENT_WINDOWS:
            recent = ordered[ordered["age_days"] <= window]
            counts = recent.groupby("client_id").size().reindex(clients, fill_value=0)
            outgoing = (
                recent[recent["direction"].eq("out")]
                .groupby("client_id")
                .size()
                .reindex(clients, fill_value=0)
            )
            features[f"transactions_last_{window}d"] = counts
            features[f"outgoing_last_{window}d"] = outgoing
            features[f"frequency_last_{window}d"] = counts / float(window)

        ordered["day_of_week"] = ordered["timestamp"].dt.dayofweek
        ordered["month"] = ordered["timestamp"].dt.strftime("%Y-%m")
        ordered["is_weekend"] = ordered["day_of_week"].ge(5)
        features["weekend_share"] = grouped["is_weekend"].mean()
        for day in range(7):
            features[f"dow_{day}_share"] = grouped["day_of_week"].apply(
                lambda x, day=day: float(x.eq(day).mean())
            )
        monthly = ordered.groupby(["client_id", "month"]).size().rename("count")
        features["active_months"] = monthly.groupby("client_id").size()
        features["monthly_count_mean"] = monthly.groupby("client_id").mean()
        features["monthly_count_std"] = monthly.groupby("client_id").std(ddof=0)
        features["monthly_count_cv"] = features["monthly_count_std"] / features[
            "monthly_count_mean"
        ].clip(lower=1e-6)
        gaps = grouped["timestamp"].diff().dt.total_seconds().div(86400)
        ordered["event_gap_days"] = gaps
        for statistic in ("mean", "median", "std"):
            features[f"event_gap_{statistic}"] = ordered.groupby("client_id")["event_gap_days"].agg(
                statistic
            )

        extra_features: dict[str, pd.Series] = {}
        for column, levels in self.categorical_levels_.items():
            counts = pd.crosstab(ordered["client_id"], ordered[column].astype(str)).reindex(
                index=clients, columns=levels, fill_value=0
            )
            for level in levels:
                safe_level = str(level).replace(" ", "_")
                extra_features[f"{column}_{safe_level}_count"] = counts[level]
                extra_features[f"{column}_{safe_level}_share"] = (
                    counts[level] / features["n_transactions"]
                )

        for currency in self.categorical_levels_["currency"]:
            currency_rows = ordered[ordered["currency"].eq(currency)]
            for column in ("amount", "fee"):
                currency_grouped = currency_rows.groupby("client_id")[column]
                for statistic in ("sum", "mean", "median", "std", "min", "max"):
                    extra_features[f"{column}_{currency}_{statistic}"] = currency_grouped.agg(
                        statistic
                    ).reindex(clients)

        features = pd.concat([features, pd.DataFrame(extra_features, index=clients)], axis=1)

        streams = build_recurrence_streams(ordered)
        features = self._add_recurrence_features(features, streams)
        return features.replace([np.inf, -np.inf], np.nan).astype(float)

    def _add_recurrence_features(
        self, features: pd.DataFrame, streams: pd.DataFrame
    ) -> pd.DataFrame:
        clients = features.index
        grouped = streams.groupby("client_id", sort=False)
        recurrence_features: dict[str, pd.Series] = {}
        recurrence_features["repeated_description_count"] = grouped.size()
        recurrence_features["repeated_transaction_count"] = grouped["appearances"].sum()
        recurrence_features["max_description_appearances"] = grouped["appearances"].max()
        recurrence_features["mean_description_appearances"] = grouped["appearances"].mean()
        recurrence_features["regular_stream_count"] = grouped["interval_std_days"].apply(
            lambda x: int(x.le(3.0).sum())
        )
        recurrence_features["stable_amount_stream_count"] = grouped["amount_cv"].apply(
            lambda x: int(x.le(0.05).sum())
        )
        recurrence_features["best_recurrence_score"] = grouped["base_recurrence_score"].max()
        recurrence_features["mean_recurrence_score"] = grouped["base_recurrence_score"].mean()
        for column in (
            "median_interval_days",
            "interval_std_days",
            "interval_cv",
            "days_since_last",
            "amount_cv",
            "regularity",
            "due_score",
        ):
            recurrence_features[f"stream_{column}_mean"] = grouped[column].mean()
            recurrence_features[f"stream_{column}_min"] = grouped[column].min()
        for name, (lower, upper) in PERIODICITY_BINS.items():
            periodic = streams["median_interval_days"].ge(lower) & streams[
                "median_interval_days"
            ].lt(upper)
            recurrence_features[f"periodicity_{name}_count"] = (
                streams.assign(match=periodic.astype(int))
                .groupby("client_id")["match"]
                .sum()
                .reindex(clients, fill_value=0)
            )

        lift = self.description_lift_.reindex(streams["description"], fill_value=0.0)
        lift.index = streams.index
        family_features: dict[str, pd.Series] = {}
        for label in LABELS:
            labelled = streams.assign(description_lift=lift[label].to_numpy())
            labelled["weighted_score"] = (
                labelled["base_recurrence_score"] * labelled["description_lift"]
            )
            evidence = labelled[labelled["description_lift"].gt(0)]
            evidence_grouped = evidence.groupby("client_id", sort=False)
            prefix = f"family_{label}"
            family_features[f"{prefix}_occurrences"] = evidence_grouped["appearances"].sum()
            family_features[f"{prefix}_descriptions"] = evidence_grouped.size()
            family_features[f"{prefix}_recency"] = evidence_grouped["days_since_last"].min()
            family_features[f"{prefix}_frequency"] = evidence_grouped["appearances"].max()
            family_features[f"{prefix}_regularity"] = evidence_grouped["regularity"].max()
            family_features[f"{prefix}_amount_similarity"] = 1.0 / (
                1.0 + evidence_grouped["amount_cv"].min()
            )
            family_features[f"{prefix}_description_lift"] = evidence_grouped[
                "description_lift"
            ].max()
            family_features[f"{prefix}_recurrence_score"] = evidence_grouped["weighted_score"].sum()
        recurrence_frame = pd.DataFrame(recurrence_features, index=clients)
        family_frame = pd.DataFrame(family_features, index=clients)
        return pd.concat([features, recurrence_frame, family_frame], axis=1).fillna(0.0)
