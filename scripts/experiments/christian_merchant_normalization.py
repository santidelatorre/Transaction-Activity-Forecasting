"""Conservative unsupervised description normalization discovery experiment.

The alias model is learned only from TRAIN transactions plus the unlabeled
pretrain partition. Official validation is loaded only after the mapping and
the fixed recurrence comparison have been frozen and written to disk.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from transaction_forecasting.ubs.data import (
    CUTOFF,
    LABELS,
    TARGET_COLUMN,
    read_labels,
    read_transactions,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v2 import IntegratedV2Model

SEED = 42
CYCLES = np.asarray([7.0, 14.0, 28.0, 30.0, 31.0, 90.0, 180.0, 365.0])
GENERIC_TOKENS = {
    "access",
    "bill",
    "billing",
    "card",
    "charge",
    "core",
    "digital",
    "merchant",
    "monthly",
    "online",
    "order",
    "pay",
    "payment",
    "plan",
    "plus",
    "premium",
    "purchase",
    "service",
    "subscription",
}
TOKEN_ALIASES = {
    "dgtl": "digital",
    "mth": "monthly",
    "prem": "premium",
    "prod": "productivity",
    "svc": "service",
    "stream": "streaming",
}


def clean_text(value: object) -> str:
    """Lowercase, strip accents and collapse punctuation/whitespace."""
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def tokens(value: object) -> tuple[str, ...]:
    return tuple(clean_text(value).split())


def light_normalize(value: object) -> str:
    normalized = []
    for token in tokens(value):
        token = TOKEN_ALIASES.get(token, token)
        if len(token) > 5 and token.endswith("ing") and token not in {"streaming", "billing"}:
            token = token[:-3]
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized.append(token)
    return " ".join(normalized)


def sorted_tokens(value: object) -> str:
    return " ".join(sorted(tokens(value)))


def informative_tokens(value: object) -> tuple[str, ...]:
    return tuple(token for token in light_normalize(value).split() if token not in GENERIC_TOKENS)


def generic_normalize(value: object) -> str:
    informative = informative_tokens(value)
    return " ".join(informative) if informative else light_normalize(value)


def token_jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = (
        set(light_normalize(left).split()),
        set(light_normalize(right).split()),
    )
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _mode_and_purity(series: pd.Series) -> tuple[str, float]:
    counts = series.astype(str).value_counts()
    return str(counts.index[0]), float(counts.iloc[0] / counts.sum())


def description_profiles(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build unsupervised categorical, amount and cadence evidence by description."""
    frame = transactions.copy()
    frame["description"] = frame["description"].map(clean_text)
    rows: list[dict[str, object]] = []
    sorted_frame = frame.sort_values(["description", "client_id", "timestamp"])
    cadence = (
        sorted_frame.assign(
            gap_days=sorted_frame.groupby(["description", "client_id"])["timestamp"]
            .diff()
            .dt.total_seconds()
            .div(86_400)
        )
        .groupby("description")["gap_days"]
        .median()
    )
    for description, group in frame.groupby("description", sort=True):
        mcc, mcc_purity = _mode_and_purity(group["mcc"])
        tx_type, type_purity = _mode_and_purity(group["type"])
        direction, direction_purity = _mode_and_purity(group["direction"])
        absolute_amount = group["amount"].astype(float).abs()
        rows.append(
            {
                "description": description,
                "transactions": len(group),
                "clients": group["client_id"].nunique(),
                "mcc_mode": mcc,
                "mcc_purity": mcc_purity,
                "type_mode": tx_type,
                "type_purity": type_purity,
                "direction_mode": direction,
                "direction_purity": direction_purity,
                "amount_median": float(absolute_amount.median()),
                "cadence_median": float(cadence.get(description, np.nan)),
            }
        )
    return pd.DataFrame(rows).set_index("description")


def evidence_compatibility(left: pd.Series, right: pd.Series) -> tuple[float, dict[str, object]]:
    """Score non-text agreement, penalizing conflicts in stable categorical fields."""
    categorical_scores = []
    conflicts = []
    for prefix in ("mcc", "type", "direction"):
        stable = min(float(left[f"{prefix}_purity"]), float(right[f"{prefix}_purity"])) >= 0.8
        same = left[f"{prefix}_mode"] == right[f"{prefix}_mode"]
        categorical_scores.append(1.0 if same else (0.55 if not stable else 0.0))
        if stable and not same:
            conflicts.append(prefix)

    left_amount, right_amount = float(left["amount_median"]), float(right["amount_median"])
    amount_distance = abs(math.log1p(left_amount) - math.log1p(right_amount))
    amount_score = math.exp(-amount_distance)
    left_cadence, right_cadence = float(left["cadence_median"]), float(right["cadence_median"])
    if np.isfinite(left_cadence) and np.isfinite(right_cadence):
        cadence_distance = abs(left_cadence - right_cadence) / max(left_cadence, right_cadence, 1)
        cadence_score = math.exp(-3 * cadence_distance)
    else:
        cadence_distance, cadence_score = np.nan, 0.75
    compatibility = float(
        0.2 * categorical_scores[0]
        + 0.15 * categorical_scores[1]
        + 0.15 * categorical_scores[2]
        + 0.3 * amount_score
        + 0.2 * cadence_score
    )
    return compatibility, {
        "compatibility": compatibility,
        "categorical_conflicts": ",".join(conflicts),
        "amount_log_distance": amount_distance,
        "cadence_relative_distance": cadence_distance,
    }


def nearest_candidate_pairs(matrix: sparse.spmatrix, neighbors: int = 15) -> set[tuple[int, int]]:
    count = min(neighbors, matrix.shape[0])
    model = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(matrix)
    _, indices = model.kneighbors(matrix)
    return {
        (min(row, int(other)), max(row, int(other)))
        for row, near in enumerate(indices)
        for other in near[1:]
        if row != int(other)
    }


def similarity_evidence(descriptions: list[str], profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate sparse close-string candidates and attach all merge signals."""
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    char_matrix = char_vectorizer.fit_transform(descriptions)
    word_matrix = word_vectorizer.fit_transform(descriptions)
    candidate_pairs = nearest_candidate_pairs(char_matrix) | nearest_candidate_pairs(word_matrix)

    buckets: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, description in enumerate(descriptions):
        for name, key in (
            ("sorted", sorted_tokens(description)),
            ("light", light_normalize(description)),
            ("generic", generic_normalize(description)),
        ):
            buckets[(name, key)].append(index)
    for members in buckets.values():
        if 1 < len(members) <= 30:
            candidate_pairs.update((min(a, b), max(a, b)) for a, b in combinations(members, 2))

    rows = []
    for left_index, right_index in sorted(candidate_pairs):
        left, right = descriptions[left_index], descriptions[right_index]
        char_similarity = float(char_matrix[left_index].multiply(char_matrix[right_index]).sum())
        tfidf_similarity = float(word_matrix[left_index].multiply(word_matrix[right_index]).sum())
        jaccard = token_jaccard(left, right)
        compatibility, evidence = evidence_compatibility(profiles.loc[left], profiles.loc[right])
        shared_informative = bool(set(informative_tokens(left)) & set(informative_tokens(right)))
        rows.append(
            {
                "left": left,
                "right": right,
                "char_similarity": char_similarity,
                "token_jaccard": jaccard,
                "tfidf_similarity": tfidf_similarity,
                "same_sorted_tokens": sorted_tokens(left) == sorted_tokens(right),
                "same_light_form": light_normalize(left) == light_normalize(right),
                "same_generic_form": generic_normalize(left) == generic_normalize(right),
                "shared_informative_token": shared_informative,
                **evidence,
            }
        )
    return pd.DataFrame(rows)


class ConservativeClusters:
    """Deterministic complete-link clusters to prevent similarity chaining."""

    def __init__(self, descriptions: list[str], frequencies: pd.Series):
        self.members = {description: {description} for description in descriptions}
        self.parent = {description: description for description in descriptions}
        self.frequencies = frequencies

    def root(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def merge(self, left: str, right: str, allowed_pairs: set[frozenset[str]]) -> bool:
        left_root, right_root = self.root(left), self.root(right)
        if left_root == right_root:
            return False
        combined = self.members[left_root] | self.members[right_root]
        if len(combined) > 20:
            return False
        cross_pairs = {
            frozenset((a, b)) for a in self.members[left_root] for b in self.members[right_root]
        }
        if not cross_pairs.issubset(allowed_pairs):
            return False
        representative = max(
            combined,
            key=lambda value: (int(self.frequencies.get(value, 0)), -len(value), value),
        )
        other = right_root if representative in self.members[left_root] else left_root
        keep = left_root if representative in self.members[left_root] else right_root
        self.parent[other] = keep
        self.members[keep] = combined
        del self.members[other]
        return True

    def mapping(self) -> dict[str, str]:
        result = {}
        for members in self.members.values():
            representative = max(
                members,
                key=lambda value: (int(self.frequencies.get(value, 0)), -len(value), value),
            )
            result.update({member: representative for member in members})
        return result


def learned_mapping(
    descriptions: list[str], frequencies: pd.Series, evidence: pd.DataFrame, method: str
) -> dict[str, str]:
    """Cluster pairs accepted by one frozen conservative rule."""
    categorical_ok = evidence["categorical_conflicts"].eq("")
    amount_ok = evidence["amount_log_distance"].le(0.35)
    cadence_ok = evidence["cadence_relative_distance"].isna() | evidence[
        "cadence_relative_distance"
    ].le(0.35)
    compatible = evidence["compatibility"].ge(0.82) & categorical_ok & amount_ok & cadence_ok
    if method == "char_ngram":
        accepted = (
            compatible
            & evidence["char_similarity"].ge(0.92)
            & (evidence["shared_informative_token"] | evidence["char_similarity"].ge(0.97))
        )
    elif method == "token_jaccard":
        accepted = (
            compatible & evidence["token_jaccard"].ge(0.66) & evidence["shared_informative_token"]
        )
    elif method == "tfidf":
        accepted = (
            compatible
            & evidence["tfidf_similarity"].ge(0.84)
            & evidence["shared_informative_token"]
        )
    elif method == "ensemble":
        lexical_votes = (
            evidence["char_similarity"].ge(0.88).astype(int)
            + evidence["token_jaccard"].ge(0.60).astype(int)
            + evidence["tfidf_similarity"].ge(0.80).astype(int)
            + evidence["same_light_form"].astype(int)
            + (evidence["same_generic_form"] & evidence["shared_informative_token"]).astype(int)
        )
        accepted = compatible & evidence["shared_informative_token"] & lexical_votes.ge(2)
    else:
        raise ValueError(f"Unknown learned method: {method}")

    accepted_evidence = evidence.loc[accepted].copy()
    accepted_evidence["rank"] = (
        accepted_evidence["compatibility"]
        + accepted_evidence["char_similarity"]
        + accepted_evidence["token_jaccard"]
        + accepted_evidence["tfidf_similarity"]
    )
    allowed_pairs = {
        frozenset((row.left, row.right)) for row in accepted_evidence.itertuples(index=False)
    }
    clusters = ConservativeClusters(descriptions, frequencies)
    for row in accepted_evidence.sort_values("rank", ascending=False).itertuples(index=False):
        clusters.merge(row.left, row.right, allowed_pairs)
    return clusters.mapping()


def deterministic_mapping(descriptions: list[str], method: str) -> dict[str, str]:
    normalizers = {
        "exact": str,
        "lowercase": lambda value: str(value).lower(),
        "punctuation_whitespace": clean_text,
        "token_sorting": sorted_tokens,
        "generic_tokens": generic_normalize,
        "light_stemming": light_normalize,
    }
    return {description: normalizers[method](description) for description in descriptions}


def apply_mapping(
    transactions: pd.DataFrame, mapping: dict[str, str], fallback=light_normalize
) -> pd.DataFrame:
    transformed = transactions.copy()
    cleaned = transformed["description"].map(clean_text)
    transformed["description"] = cleaned.map(mapping).fillna(cleaned.map(fallback))
    return transformed


def stream_table(transactions: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    frame = apply_mapping(transactions, mapping).sort_values(
        ["client_id", "description", "timestamp"]
    )
    group_columns = ["client_id", "description"]
    frame["gap_days"] = (
        frame.groupby(group_columns)["timestamp"].diff().dt.total_seconds().div(86_400)
    )
    frame["gap_median"] = frame.groupby(group_columns)["gap_days"].transform("median")
    frame["gap_abs_deviation"] = (frame["gap_days"] - frame["gap_median"]).abs()
    grouped = frame.groupby(group_columns, sort=False)
    result = grouped.agg(
        events=("timestamp", "size"),
        unique_events=("timestamp", "nunique"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        gap_median=("gap_days", "median"),
        gap_std=("gap_days", "std"),
        gap_mad=("gap_abs_deviation", "median"),
        amount_mean=("amount", "mean"),
        amount_std=("amount", "std"),
        mcc_values=("mcc", "nunique"),
        type_values=("type", "nunique"),
        direction_values=("direction", "nunique"),
    )
    result["gap_relative_mad"] = result["gap_mad"] / result["gap_median"].replace(0, np.nan)
    result["amount_cv"] = result["amount_std"] / result["amount_mean"].abs().replace(0, np.nan)
    cycle_distance = np.min(
        np.abs(result["gap_median"].fillna(-10).to_numpy()[:, None] - CYCLES[None, :])
        / CYCLES[None, :],
        axis=1,
    )
    result["interpretable_periodicity"] = (
        result["unique_events"].ge(3)
        & (cycle_distance <= 0.2)
        & result["gap_relative_mad"].le(0.25)
    )
    result["stable_mcc_type"] = result["mcc_values"].eq(1) & result["type_values"].eq(1)
    result["candidate"] = (
        result["unique_events"].ge(3)
        & result["interpretable_periodicity"]
        & result["amount_cv"].fillna(0).le(0.20)
        & result["stable_mcc_type"]
    )
    result["days_since_last"] = (CUTOFF - result["last_timestamp"]).dt.total_seconds() / 86_400
    return result


def stream_metrics(
    transactions: pd.DataFrame, mapping: dict[str, str], method: str
) -> tuple[dict[str, object], pd.DataFrame]:
    streams = stream_table(transactions, mapping)
    candidates = streams[streams["candidate"]]
    partition_descriptions = set(transactions["description"].map(clean_text))
    description_count = len(partition_descriptions)
    normalized_count = len(
        {
            mapping.get(description, light_normalize(description))
            for description in partition_descriptions
        }
    )
    metrics: dict[str, object] = {
        "method": method,
        "original_descriptions": int(description_count),
        "normalized_groups": int(normalized_count),
        "description_merges": int(description_count - normalized_count),
        "streams": int(len(streams)),
        "mean_appearances": float(streams["events"].mean()),
        "streams_ge_2_pct": float(100 * streams["unique_events"].ge(2).mean()),
        "streams_ge_3_pct": float(100 * streams["unique_events"].ge(3).mean()),
        "streams_ge_4_pct": float(100 * streams["unique_events"].ge(4).mean()),
        "median_gap_mad": float(streams.loc[streams["unique_events"].ge(3), "gap_mad"].median()),
        "median_gap_std": float(streams.loc[streams["unique_events"].ge(3), "gap_std"].median()),
        "median_gap_relative_mad": float(
            streams.loc[streams["unique_events"].ge(3), "gap_relative_mad"].median()
        ),
        "median_amount_cv": float(
            streams.loc[streams["unique_events"].ge(2), "amount_cv"].median()
        ),
        "interpretable_periodicity_pct": float(100 * streams["interpretable_periodicity"].mean()),
        "stable_mcc_type_pct": float(100 * streams["stable_mcc_type"].mean()),
        "candidate_streams": int(len(candidates)),
        "candidate_client_coverage": int(candidates.index.get_level_values("client_id").nunique()),
        "candidate_client_coverage_pct": float(
            100
            * candidates.index.get_level_values("client_id").nunique()
            / transactions["client_id"].nunique()
        ),
    }
    return metrics, streams


def fit_recurrence_labels(
    transactions: pd.DataFrame, labels: pd.DataFrame, mapping: dict[str, str]
) -> pd.DataFrame:
    transformed = apply_mapping(transactions, mapping)
    client_keys = transformed[["client_id", "description"]].drop_duplicates()
    joined = client_keys.merge(
        labels[["client_id", TARGET_COLUMN]], on="client_id", validate="many_to_one"
    )
    counts = pd.crosstab(joined["description"], joined[TARGET_COLUMN]).reindex(
        columns=LABELS, fill_value=0
    )
    non_none = counts.drop(columns="none")
    result = pd.DataFrame(
        {
            "label": non_none.idxmax(axis=1),
            "label_clients": non_none.max(axis=1),
            "all_clients": counts.sum(axis=1),
        }
    )
    result["purity"] = result["label_clients"] / result["all_clients"]
    return result


def fixed_recurrence_predict(
    transactions: pd.DataFrame, mapping: dict[str, str], label_map: pd.DataFrame
) -> pd.Series:
    streams = stream_table(transactions, mapping).reset_index()
    streams = streams.join(label_map, on="description")
    support = (streams["unique_events"] / 4).clip(upper=1)
    regularity = np.where(
        streams["unique_events"].ge(3),
        1 / (1 + streams["gap_relative_mad"].fillna(2)),
        0.4,
    )
    amount = 1 / (1 + streams["amount_cv"].abs().fillna(1))
    recency = np.exp(-streams["days_since_last"].clip(lower=0) / 90)
    streams["score"] = (
        0.35 * support + 0.30 * regularity + 0.20 * amount + 0.15 * recency
    ) * streams["purity"].fillna(0)
    eligible = streams[
        streams["label"].notna()
        & streams["unique_events"].ge(2)
        & streams["label_clients"].ge(2)
        & streams["purity"].ge(0.35)
        & streams["score"].ge(0.45)
    ]
    best = eligible.sort_values(
        ["client_id", "score", "unique_events", "description"],
        ascending=[True, False, False, True],
    ).drop_duplicates("client_id")
    clients = pd.Index(transactions["client_id"].unique(), name="client_id")
    prediction = pd.Series("none", index=clients, name="prediction")
    prediction.update(best.set_index("client_id")["label"])
    return prediction


def predictive_comparison(
    train_transactions: pd.DataFrame,
    train_labels: pd.DataFrame,
    mappings: dict[str, dict[str, str]],
) -> dict[str, object]:
    fit_ids, holdout_ids = train_test_split(
        train_labels["client_id"],
        test_size=0.25,
        random_state=SEED,
        stratify=train_labels[TARGET_COLUMN],
    )
    fit_labels = train_labels[train_labels["client_id"].isin(fit_ids)]
    holdout_labels = train_labels[train_labels["client_id"].isin(holdout_ids)]
    fit_tx = train_transactions[train_transactions["client_id"].isin(fit_ids)]
    holdout_tx = train_transactions[train_transactions["client_id"].isin(holdout_ids)]
    target = holdout_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}
    for method in ("exact", "ensemble"):
        label_map = fit_recurrence_labels(fit_tx, fit_labels, mappings[method])
        prediction = fixed_recurrence_predict(holdout_tx, mappings[method], label_map)
        results[method] = evaluate_predictions(target, prediction)
    results["delta_macro_f1"] = float(results["ensemble"]["macro_f1"]) - float(
        results["exact"]["macro_f1"]
    )
    results["fit_clients"] = len(fit_ids)
    results["holdout_clients"] = len(holdout_ids)
    return results


def cluster_examples(
    mapping: dict[str, str], profiles: pd.DataFrame, evidence: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for description, group in mapping.items():
        groups[group].append(description)
    pair_lookup = evidence.set_index(["left", "right"])
    rows = []
    for group, members in groups.items():
        if len(members) < 2:
            continue
        pair_rows = []
        for left, right in combinations(sorted(members), 2):
            key = (left, right) if (left, right) in pair_lookup.index else (right, left)
            if key in pair_lookup.index:
                pair_rows.append(pair_lookup.loc[key])
        compatibility = (
            min(float(row["compatibility"]) for row in pair_rows) if pair_rows else np.nan
        )
        mcc_modes = profiles.loc[members, "mcc_mode"].nunique()
        type_modes = profiles.loc[members, "type_mode"].nunique()
        suspicious = (
            len(members) >= 6
            or mcc_modes > 1
            or type_modes > 1
            or not any(informative_tokens(member) for member in members)
            or (np.isfinite(compatibility) and compatibility < 0.88)
        )
        rows.append(
            {
                "normalized_group": group,
                "descriptions": " | ".join(sorted(members)),
                "group_size": len(members),
                "transactions": int(profiles.loc[members, "transactions"].sum()),
                "minimum_compatibility": compatibility,
                "mcc_modes": int(mcc_modes),
                "type_modes": int(type_modes),
                "suspicious": bool(suspicious),
            }
        )
    clusters = pd.DataFrame(rows)
    if clusters.empty:
        return clusters, clusters
    good = clusters[~clusters["suspicious"]].sort_values(
        ["transactions", "minimum_compatibility"], ascending=False
    )
    dangerous = clusters[clusters["suspicious"]].sort_values(
        ["group_size", "transactions"], ascending=False
    )
    return good.head(30), dangerous.head(30)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def run_v2_validation(
    train_transactions: pd.DataFrame,
    train_labels: pd.DataFrame,
    valid_transactions: pd.DataFrame,
    valid_labels: pd.DataFrame,
    mappings: dict[str, dict[str, str]],
) -> dict[str, object]:
    target = valid_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}
    for method in ("exact", "ensemble"):
        train = apply_mapping(train_transactions, mappings[method])
        valid = apply_mapping(valid_transactions, mappings[method])
        prediction = IntegratedV2Model().fit(train, train_labels).predict(valid)
        results[method] = evaluate_predictions(target, prediction)
    results["delta_macro_f1"] = float(results["ensemble"]["macro_f1"]) - float(
        results["exact"]["macro_f1"]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics/v3_discovery/christian_merchant_normalization"),
    )
    parser.add_argument("--skip-v2", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discovery data only: no validation/test partition is opened above this boundary.
    train_transactions = read_transactions(args.data_dir / "train_transactions.jsonl")
    train_labels = read_labels(args.data_dir / "train_labels.csv")
    pretrain_transactions = read_transactions(
        args.data_dir / "unlabeled_pretrain_transactions.jsonl"
    )
    discovery = pd.concat([train_transactions, pretrain_transactions], ignore_index=True)
    discovery["description"] = discovery["description"].map(clean_text)
    frequencies = discovery["description"].value_counts()
    descriptions = sorted(frequencies.index.tolist())
    profiles = description_profiles(discovery)
    evidence = similarity_evidence(descriptions, profiles)

    mappings = {
        method: deterministic_mapping(descriptions, method)
        for method in (
            "exact",
            "lowercase",
            "punctuation_whitespace",
            "token_sorting",
            "generic_tokens",
            "light_stemming",
        )
    }
    for method in ("char_ngram", "token_jaccard", "tfidf", "ensemble"):
        mappings[method] = learned_mapping(descriptions, frequencies, evidence, method)

    intrinsic_rows = []
    for method, mapping in mappings.items():
        metrics, _ = stream_metrics(train_transactions, mapping, method)
        intrinsic_rows.append(metrics)
    intrinsic = pd.DataFrame(intrinsic_rows)
    predictive = predictive_comparison(train_transactions, train_labels, mappings)
    good, dangerous = cluster_examples(mappings["ensemble"], profiles, evidence)

    mapping_frame = pd.DataFrame(
        {
            "description": descriptions,
            "normalized_description": [mappings["ensemble"][value] for value in descriptions],
        }
    )
    frozen = {
        "seed": SEED,
        "learning_partitions": ["train_transactions", "unlabeled_pretrain_transactions"],
        "validation_used_for_learning": False,
        "generic_tokens": sorted(GENERIC_TOKENS),
        "token_aliases": TOKEN_ALIASES,
        "ensemble_rule": (
            "complete-link; compatibility>=0.82; no stable categorical conflict; "
            "amount log distance<=0.35; cadence relative distance<=0.35; "
            "informative token plus >=2 lexical votes"
        ),
        "recurrence_thresholds": {
            "minimum_events": 2,
            "minimum_mapping_clients": 2,
            "minimum_mapping_purity": 0.35,
            "minimum_score": 0.45,
        },
    }
    write_json(args.output_dir / "frozen_configuration.json", frozen)
    intrinsic.to_csv(args.output_dir / "intrinsic_comparison.csv", index=False)
    mapping_frame.to_csv(args.output_dir / "description_mapping.csv", index=False)
    evidence.sort_values("compatibility", ascending=False).to_csv(
        args.output_dir / "candidate_pair_evidence.csv", index=False
    )
    good.to_csv(args.output_dir / "good_merges.csv", index=False)
    dangerous.to_csv(args.output_dir / "dangerous_merges.csv", index=False)
    write_json(args.output_dir / "internal_predictive_comparison.json", predictive)

    # Frozen boundary: validation is loaded exactly once, after all discovery artifacts exist.
    valid_transactions = read_transactions(args.data_dir / "valid_transactions.jsonl")
    valid_labels = read_labels(args.data_dir / "valid_labels.csv")
    label_maps = {
        method: fit_recurrence_labels(train_transactions, train_labels, mappings[method])
        for method in ("exact", "ensemble")
    }
    target = valid_labels.set_index("client_id")[TARGET_COLUMN]
    official_recurrence = {}
    for method in ("exact", "ensemble"):
        prediction = fixed_recurrence_predict(
            valid_transactions, mappings[method], label_maps[method]
        )
        official_recurrence[method] = evaluate_predictions(target, prediction)
    official_recurrence["delta_macro_f1"] = float(
        official_recurrence["ensemble"]["macro_f1"]
    ) - float(official_recurrence["exact"]["macro_f1"])
    write_json(args.output_dir / "official_recurrence_comparison.json", official_recurrence)

    v2_results = None
    v2_path = args.output_dir / "official_v2_comparison.json"
    if not args.skip_v2:
        v2_results = run_v2_validation(
            train_transactions,
            train_labels,
            valid_transactions,
            valid_labels,
            mappings,
        )
        write_json(v2_path, v2_results)
    elif v2_path.exists():
        v2_results = json.loads(v2_path.read_text(encoding="utf-8"))

    exact_row = intrinsic.set_index("method").loc["exact"]
    ensemble_row = intrinsic.set_index("method").loc["ensemble"]
    summary = {
        "discovery_transactions": len(discovery),
        "discovery_clients": discovery["client_id"].nunique(),
        "original_descriptions": len(descriptions),
        "normalized_groups": len(set(mappings["ensemble"].values())),
        "merged_descriptions": len(descriptions) - len(set(mappings["ensemble"].values())),
        "train_stream_change": int(ensemble_row["streams"] - exact_row["streams"]),
        "mean_appearances_delta": float(
            ensemble_row["mean_appearances"] - exact_row["mean_appearances"]
        ),
        "candidate_stream_delta": int(
            ensemble_row["candidate_streams"] - exact_row["candidate_streams"]
        ),
        "candidate_coverage_pp": float(
            ensemble_row["candidate_client_coverage_pct"]
            - exact_row["candidate_client_coverage_pct"]
        ),
        "internal_macro_f1_delta": predictive["delta_macro_f1"],
        "official_recurrence_macro_f1_delta": official_recurrence["delta_macro_f1"],
        "official_v2_macro_f1_delta": None if v2_results is None else v2_results["delta_macro_f1"],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
