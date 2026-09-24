"""Create a reproducible exploratory report for the UBS Swiss AI Weeks 2026 data."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = (
    "cloud",
    "gym",
    "insurance",
    "mobile",
    "music",
    "software",
    "streaming",
    "none",
)
CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
SPECIAL_COLUMNS = ("description", "mcc", "type", "amount", "currency", "direction", "timestamp")


def read_transactions(path: Path) -> pd.DataFrame:
    """Read one official JSONL transaction history with stable timestamp typing."""
    frame = pd.read_json(path, lines=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    return frame


def profile_pretrain(
    path: Path, sample_size: int = 100_000
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Stream exact size/client/cutoff checks and retain a reproducible uniform sample."""
    with path.open(encoding="utf-8") as source:
        row_count = sum(1 for _ in source)
    step = max(1, row_count // sample_size)
    clients: set[str] = set()
    sample: list[dict[str, object]] = []
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    after_cutoff = 0
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            row = json.loads(line)
            clients.add(str(row["client_id"]))
            timestamp = str(row["timestamp"])
            first_timestamp = min(first_timestamp, timestamp) if first_timestamp else timestamp
            last_timestamp = max(last_timestamp, timestamp) if last_timestamp else timestamp
            after_cutoff += timestamp >= "2026-01-01"
            if index % step == 0 and len(sample) < sample_size:
                sample.append(row)
    sample_frame = pd.DataFrame(sample)
    sample_frame["timestamp"] = pd.to_datetime(sample_frame["timestamp"], utc=True, errors="raise")
    return sample_frame, {
        "dataset": "unlabeled pretrain",
        "transactions": row_count,
        "clients": len(clients),
        "transactions/client mean": row_count / max(len(clients), 1),
        "transactions/client median": "not calculated (streamed)",
        "transactions/client p95": "not calculated (streamed)",
        "file MB": file_size_mb(path),
        "memory MB": sample_frame.memory_usage(deep=True).sum() / (1024**2),
        "first timestamp": first_timestamp,
        "last timestamp": last_timestamp,
        "after cutoff": after_cutoff,
    }


def markdown_table(frame: pd.DataFrame, *, limit: int | None = None) -> str:
    """Render a compact Markdown table without optional notebook dependencies."""
    if limit is not None:
        frame = frame.head(limit)
    prepared = frame.copy()
    for column in prepared.columns:
        if pd.api.types.is_float_dtype(prepared[column]):
            prepared[column] = prepared[column].map(lambda value: f"{value:.4f}")
    headers = [str(column).replace("|", "\\|") for column in prepared.columns]
    rows = []
    for row in prepared.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", *rows]
    )


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)


def dataset_summary(name: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    per_client = frame.groupby("client_id").size()
    return {
        "dataset": name,
        "transactions": len(frame),
        "clients": int(frame["client_id"].nunique()),
        "transactions/client mean": float(per_client.mean()),
        "transactions/client median": float(per_client.median()),
        "transactions/client p95": float(per_client.quantile(0.95)),
        "file MB": file_size_mb(path),
        "memory MB": frame.memory_usage(deep=True).sum() / (1024**2),
        "first timestamp": frame["timestamp"].min().isoformat(),
        "last timestamp": frame["timestamp"].max().isoformat(),
        "after cutoff": int((frame["timestamp"] >= CUTOFF).sum()),
    }


def schema_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column]
        top = series.astype(str).value_counts(dropna=False).head(3)
        frequent = "; ".join(f"{value} ({count})" for value, count in top.items())
        row: dict[str, object] = {
            "column": column,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing %": float(series.isna().mean() * 100),
            "unique": int(series.nunique(dropna=True)),
            "top values": frequent,
        }
        if pd.api.types.is_numeric_dtype(series):
            row["min"] = float(series.min())
            row["p01"] = float(series.quantile(0.01))
            row["median"] = float(series.median())
            row["p99"] = float(series.quantile(0.99))
            row["max"] = float(series.max())
        rows.append(row)
    return pd.DataFrame(rows)


def target_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    counts = labels["target_next_recurring_merchant"].value_counts().reindex(LABELS, fill_value=0)
    return pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "percent": (counts.values / len(labels)) * 100,
        }
    )


def recurrence_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure repeated descriptions per client, without inferring merchant families."""
    repeated: list[dict[str, float | int | str]] = []
    for (client_id, description), group in frame.sort_values("timestamp").groupby(
        ["client_id", "description"], sort=False
    ):
        if len(group) < 2:
            continue
        intervals = group["timestamp"].diff().dropna().dt.total_seconds().div(86400)
        amounts = group["amount"].astype(float)
        repeated.append(
            {
                "client_id": str(client_id),
                "description": str(description),
                "appearances": len(group),
                "median_interval_days": float(intervals.median()),
                "interval_std_days": float(intervals.std(ddof=0)),
                "days_since_last": float(
                    (CUTOFF - group["timestamp"].max()).total_seconds() / 86400
                ),
                "amount_unique": int(amounts.nunique()),
                "amount_cv": float(amounts.std(ddof=0) / amounts.mean())
                if amounts.mean() != 0
                else float("nan"),
            }
        )
    groups = pd.DataFrame(repeated)
    if groups.empty:
        return groups, pd.DataFrame()
    groups["periodicity"] = pd.cut(
        groups["median_interval_days"],
        bins=[-np.inf, 10, 18, 45, 110, 400, np.inf],
        labels=[
            "weekly-ish",
            "biweekly-ish",
            "monthly-ish",
            "quarterly-ish",
            "annual-ish",
            "other",
        ],
    )
    client_coverage = groups.groupby("client_id").size()
    summary = pd.DataFrame(
        {
            "repeated description groups": [len(groups)],
            "clients with repeat descriptions": [int(client_coverage.size)],
            "share of clients with repeats %": [
                float(client_coverage.size / frame.client_id.nunique() * 100)
            ],
            "median repeat appearances": [float(groups["appearances"].median())],
            "regular groups (std <= 3d) %": [
                float((groups["interval_std_days"] <= 3).mean() * 100)
            ],
            "same amount groups %": [float((groups["amount_unique"] == 1).mean() * 100)],
            "amount CV <= 5% groups %": [float((groups["amount_cv"] <= 0.05).mean() * 100)],
        }
    )
    periodicity = (
        groups["periodicity"]
        .value_counts(dropna=False)
        .rename_axis("periodicity")
        .reset_index(name="groups")
    )
    return summary, periodicity


def text_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptions = frame["description"].fillna("").astype(str).str.lower()
    tokens = descriptions.str.findall(r"[a-z]{3,}").explode()
    frequent = tokens.value_counts().head(30).rename_axis("term").reset_index(name="count")
    patterns = "|".join(re.escape(label) for label in LABELS if label != "none")
    family_words = descriptions.str.extract(f"({patterns})", expand=False).value_counts()
    direct_terms = family_words.rename_axis("label term in description").reset_index(
        name="transactions"
    )
    return frequent, direct_terms


def label_signal_summary(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Check target-associated text/MCC patterns without fitting a predictive model."""
    joined = transactions.merge(
        labels[["client_id", "target_next_recurring_merchant"]], on="client_id"
    )
    output: list[dict[str, object]] = []
    for label in LABELS:
        label_rows = joined[joined["target_next_recurring_merchant"] == label]
        label_clients = label_rows["client_id"].nunique()
        pattern = re.escape(label)
        direct_client_hits = (
            label_rows.assign(
                hit=label_rows["description"].str.contains(pattern, case=False, na=False)
            )
            .groupby("client_id")["hit"]
            .any()
            .sum()
        )
        top_mcc = label_rows["mcc"].astype(str).value_counts().index[:3].tolist()
        top_description = label_rows["description"].astype(str).value_counts().index[:3].tolist()
        output.append(
            {
                "label": label,
                "clients": label_clients,
                "clients with literal label in description %": direct_client_hits
                / max(label_clients, 1)
                * 100,
                "top MCC": ", ".join(top_mcc),
                "top descriptions": "; ".join(top_description),
            }
        )
    return pd.DataFrame(output)


def high_purity_signals(
    transactions: pd.DataFrame, labels: pd.DataFrame, column: str, minimum_clients: int = 10
) -> pd.DataFrame:
    """Find descriptions or MCCs that concentrate strongly in one target class."""
    joined = transactions.merge(
        labels[["client_id", "target_next_recurring_merchant"]], on="client_id"
    )
    client_level = joined[["client_id", column, "target_next_recurring_merchant"]].drop_duplicates()
    counts = pd.crosstab(
        client_level[column].astype(str), client_level["target_next_recurring_merchant"]
    )
    result = pd.DataFrame(
        {
            column: counts.index,
            "clients": counts.sum(axis=1),
            "dominant label": counts.idxmax(axis=1),
            "dominant label clients": counts.max(axis=1),
        }
    )
    result["purity %"] = result["dominant label clients"] / result["clients"] * 100
    return result[result["clients"] >= minimum_clients].sort_values(
        ["purity %", "clients"], ascending=[False, False]
    )


def description_enrichment(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Show descriptions disproportionately present in each target's histories."""
    joined = (
        transactions[["client_id", "description"]].drop_duplicates().merge(labels, on="client_id")
    )
    rows: list[dict[str, object]] = []
    for label in LABELS:
        in_label = joined[joined["target_next_recurring_merchant"] == label]
        outside_label = joined[joined["target_next_recurring_merchant"] != label]
        inside_rate = (
            in_label.groupby("description")["client_id"].nunique() / in_label["client_id"].nunique()
        )
        outside_rate = (
            outside_label.groupby("description")["client_id"].nunique()
            / outside_label["client_id"].nunique()
        )
        rates = pd.DataFrame({"inside": inside_rate, "outside": outside_rate}).fillna(0)
        rates = rates[rates["inside"] >= 0.1]
        rates["lift"] = (rates["inside"] + 0.001) / (rates["outside"] + 0.001)
        for description, values in (
            rates.sort_values(["lift", "inside"], ascending=False).head(5).iterrows()
        ):
            rows.append(
                {
                    "target": label,
                    "description": description,
                    "target client coverage %": values["inside"] * 100,
                    "other-client coverage %": values["outside"] * 100,
                    "lift": values["lift"],
                }
            )
    return pd.DataFrame(rows)


def label_file_checks(
    labels: pd.DataFrame, transactions: pd.DataFrame, name: str
) -> dict[str, object]:
    return {
        "set": name,
        "label rows": len(labels),
        "unique clients": labels["client_id"].nunique(),
        "duplicate label client IDs": int(labels["client_id"].duplicated().sum()),
        "missing target": int(labels["target_next_recurring_merchant"].isna().sum()),
        "unknown targets": int((~labels["target_next_recurring_merchant"].isin(LABELS)).sum()),
        "cutoff values": ", ".join(sorted(labels["cutoff_date"].astype(str).unique())),
        "labels absent from transactions": int(
            len(set(labels["client_id"]).difference(transactions["client_id"]))
        ),
    }


def anomaly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "check": [
                "exact duplicate transaction rows",
                "duplicate client/timestamp pairs",
                "amount <= 0",
                "fee < 0",
                "fee > 0",
                "timestamps at/after cutoff",
            ],
            "count": [
                int(frame.duplicated().sum()),
                int(frame.duplicated(["client_id", "timestamp"]).sum()),
                int((frame["amount"] <= 0).sum()),
                int((frame["fee"] < 0).sum()),
                int((frame["fee"] > 0).sum()),
                int((frame["timestamp"] >= CUTOFF).sum()),
            ],
        }
    )


def history_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("client_id")["timestamp"]
    history = grouped.agg(first="min", last="max", transactions="size")
    history["history_days"] = (history["last"] - history["first"]).dt.total_seconds() / 86400
    history["transactions_per_30d"] = history["transactions"] / (
        history["history_days"].clip(lower=1) / 30
    )
    return pd.DataFrame(
        {
            "metric": [
                "history days mean",
                "history days median",
                "history days p95",
                "transactions / 30d mean",
                "last event within 30d of cutoff %",
            ],
            "value": [
                history["history_days"].mean(),
                history["history_days"].median(),
                history["history_days"].quantile(0.95),
                history["transactions_per_30d"].mean(),
                ((CUTOFF - history["last"]).dt.days <= 30).mean() * 100,
            ],
        }
    )


def render_report(data_dir: Path, output_path: Path) -> None:
    transaction_paths = {
        "train": data_dir / "train_transactions.jsonl",
        "validation": data_dir / "valid_transactions.jsonl",
        "test": data_dir / "test_transactions.jsonl",
    }
    transactions = {name: read_transactions(path) for name, path in transaction_paths.items()}
    pretrain_path = data_dir / "unlabeled_pretrain_transactions.jsonl"
    pretrain_sample, pretrain_summary = profile_pretrain(pretrain_path)
    labels = {
        "train": pd.read_csv(data_dir / "train_labels.csv"),
        "validation": pd.read_csv(data_dir / "valid_labels.csv"),
    }
    summaries = pd.DataFrame(
        [
            *[
                dataset_summary(name, frame, transaction_paths[name])
                for name, frame in transactions.items()
            ],
            pretrain_summary,
        ]
    )
    overlaps: list[dict[str, object]] = []
    for left, right in combinations(transactions, 2):
        shared = set(transactions[left]["client_id"]).intersection(transactions[right]["client_id"])
        overlaps.append({"sets": f"{left} / {right}", "shared clients": len(shared)})
    target_train = target_distribution(labels["train"]).rename(
        columns={"count": "train count", "percent": "train %"}
    )
    target_valid = target_distribution(labels["validation"]).rename(
        columns={"count": "validation count", "percent": "validation %"}
    )
    target_compare = target_train.merge(target_valid, on="label")
    target_compare["absolute difference pp"] = (
        target_compare["train %"] - target_compare["validation %"]
    ).abs()
    recurrence_train, periodicity_train = recurrence_summary(transactions["train"])
    recurrence_valid, periodicity_valid = recurrence_summary(transactions["validation"])
    vocabulary, direct_terms = text_summary(transactions["train"])
    train_special = schema_summary(transactions["train"]).query("column in @SPECIAL_COLUMNS")
    label_signals = label_signal_summary(transactions["train"], labels["train"])
    description_purity = high_purity_signals(transactions["train"], labels["train"], "description")
    mcc_purity = high_purity_signals(
        transactions["train"], labels["train"], "mcc", minimum_clients=30
    )
    enriched_descriptions = description_enrichment(transactions["train"], labels["train"])
    label_checks = pd.DataFrame(
        [
            label_file_checks(labels["train"], transactions["train"], "train"),
            label_file_checks(labels["validation"], transactions["validation"], "validation"),
        ]
    )
    submission = pd.read_csv(data_dir / "sample_submission.csv")
    submission_check = pd.DataFrame(
        [
            {
                "sample submission rows": len(submission),
                "unique submission clients": submission["client_id"].nunique(),
                "test clients": transactions["test"]["client_id"].nunique(),
                "submission/test ID symmetric difference": len(
                    set(submission["client_id"]).symmetric_difference(
                        transactions["test"]["client_id"]
                    )
                ),
            }
        ]
    )
    comparison_rows: list[dict[str, object]] = []
    for dataset_name, frame in {
        **transactions,
        "unlabeled pretrain (100k sample)": pretrain_sample,
    }.items():
        for column in ("currency", "direction", "type", "mcc"):
            top_value = frame[column].astype(str).value_counts().index[0]
            top_share = frame[column].astype(str).value_counts(normalize=True).iloc[0] * 100
            comparison_rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "top value": top_value,
                    "top share %": top_share,
                    "unique": frame[column].nunique(),
                }
            )
    report = f"""# UBS Swiss AI Weeks 2026 dataset analysis

Generated with `python scripts/analyze_ubs_dataset.py --data-dir {data_dir.as_posix()}`.
All source data remains local and ignored by Git. The challenge is a client-level multiclass prediction: given each client's history before `{CUTOFF.date()}`, predict the merchant family expected to recur during the following 90 days.

## Dataset size and cutoff checks

{markdown_table(summaries)}

### Client-set overlap

{markdown_table(pd.DataFrame(overlaps))}

All `after cutoff` counts must be zero. Any nonzero value invalidates a leakage-safe evaluation.

## Labels and imbalance

{markdown_table(target_compare)}

### Label and submission contract checks

{markdown_table(label_checks)}

{markdown_table(submission_check)}

Macro-F1 weights every class equally. Therefore the smallest classes need explicit validation and prediction coverage; an accuracy-oriented majority-class strategy is unsuitable.

## Transaction schema (train)

{markdown_table(schema_summary(transactions["train"]))}

## Requested transaction variables

{markdown_table(train_special)}

### Train anomalies

{markdown_table(anomaly_summary(transactions["train"]))}

`amount` includes its original currency unit. Do not compare or aggregate monetary magnitude across currencies without an explicit conversion decision.

## Time coverage and client history

{markdown_table(history_summary(transactions["train"]))}

The dataset summary above also reports first/last transaction timestamps per partition and the pre-cutoff check.

## Recurrence in histories

### Train

{markdown_table(recurrence_train)}

{markdown_table(periodicity_train)}

### Validation

{markdown_table(recurrence_valid)}

{markdown_table(periodicity_valid)}

Repeated `(client_id, description)` groups provide the most direct candidate streams. Useful per-stream measurements are appearance count, days since last event, median interval, interval standard deviation, amount stability, direction, MCC and type. The report deliberately treats the interval bins as descriptive approximations rather than hard rules.

## Text and possible direct signals

### Most frequent train description terms

{markdown_table(vocabulary)}

### Literal target-family words observed in train descriptions

{markdown_table(direct_terms)}

### Per-label historical associations in train

{markdown_table(label_signals)}

### Descriptions with high target purity in train (at least 10 clients)

{markdown_table(description_purity, limit=25)}

### MCC target purity in train (at least 30 clients)

{markdown_table(mcc_purity)}

### Descriptions enriched by target in train

{markdown_table(enriched_descriptions)}

Literal label names, merchant-family-like descriptions, target-pure MCCs, or synthetic identifier sequences can produce shortcut signals. The high-purity table makes this concrete: a description can act as a near-direct family indicator. It is valid historical evidence if it remains predictive on the disjoint validation clients, but should be treated as a generator artifact risk rather than assumed to generalize.

## Distribution comparison including unlabeled pretrain

{markdown_table(pd.DataFrame(comparison_rows))}

The unlabeled file is profiled exactly for rows, clients and cutoff dates. Its categorical comparison uses a deterministic 100,000-row uniform-by-file sample to keep this report fast and reproducible. It is appropriate for unsupervised vocabulary, merchant-normalization, or recurrence-statistics learning only if its type, currency, MCC and temporal distributions resemble labelled partitions. It cannot supply target labels and must never be mixed into supervised validation scoring.

## Anomaly and leakage checks to carry into modelling

1. Reject any transaction at or after the cutoff and fit every learned transformation on train clients only.
2. Preserve client disjointness across train, validation and test. The overlap table is the first guardrail.
3. Audit literal target words and highly target-pure descriptions/MCCs. A synthetic generator can make these unusually easy but they may still be legitimate historical evidence; validation determines whether they generalize.
4. Treat `client_id` as an identifier only. Never use its raw or numeric suffix as a model feature without a validation-supported reason.
5. Keep currency with amount, inspect negative/zero/extreme amounts and fees, and avoid global scaling fitted on validation/test.
6. Build features from history only: repeated descriptions/MCCs, amount stability, cadence, recency and direction/type proportions.

## First baseline recommendation

Start with a deterministic client-level recurrence scorer that identifies repeated outgoing description/MCC streams, ranks the family-like candidate by recency, monthly/weekly cadence, count and amount stability, then emits `none` when no stream is convincing. Tune its thresholds exclusively on validation macro-F1. A next step is a regularized multiclass model on the same client-level aggregates plus sparse description TF-IDF features; retain the deterministic candidate features for interpretability.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/UBS_DATASET_ANALYSIS.md"),
        help="Markdown output path",
    )
    arguments = parser.parse_args()
    render_report(arguments.data_dir, arguments.output)


if __name__ == "__main__":
    main()
