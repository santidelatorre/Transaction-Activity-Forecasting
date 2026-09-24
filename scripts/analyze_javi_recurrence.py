"""Measure UBS recurrence by target class without changing the model pipeline.

Run from the repository root with the official ZIP extracted under data/raw/ubs_2026.
Only aggregate statistics are written; raw transactions remain ignored by Git.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median, pstdev

CUTOFF = datetime(2026, 1, 1, tzinfo=UTC)
LABELS = ("cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none")
PERIODS = ("weekly", "biweekly", "monthly", "quarterly", "annual", "other")


def _period(days: float) -> str:
    if days <= 10:
        return "weekly"
    if days <= 18:
        return "biweekly"
    if days <= 45:
        return "monthly"
    if days <= 110:
        return "quarterly"
    if days <= 400:
        return "annual"
    return "other"


def _read_partition(root: Path, prefix: str) -> tuple[dict[str, str], dict[str, list[dict]]]:
    labels: dict[str, str] = {}
    with (root / f"{prefix}_labels.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            client = row["client_id"]
            label = row["target_next_recurring_merchant"]
            if client in labels or label not in LABELS:
                raise ValueError(f"Duplicate client or invalid label: {client}")
            labels[client] = label
    histories: dict[str, list[dict]] = defaultdict(list)
    with (root / f"{prefix}_transactions.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            if timestamp >= CUTOFF:
                raise ValueError(f"Future transaction for {row['client_id']}")
            row["timestamp"] = timestamp
            histories[row["client_id"]].append(row)
    if set(histories) != set(labels):
        raise ValueError(f"Transaction/label client mismatch in {prefix}")
    return labels, histories


def _percent(count: int, total: int) -> float:
    return round(100 * count / total, 2) if total else 0.0


def _learn_candidate_descriptions(
    labels: dict[str, str], histories: dict[str, list[dict]]
) -> dict[str, str]:
    """Assign descriptions to families using train clients only, for EDA.

    This is a fixed descriptive screen, not a transaction-level ground truth.
    Require at least 20 clients with the description, 20% coverage within a
    family, and a smoothed coverage lift of at least 2 over other clients.
    """
    clients_by_description: dict[str, set[str]] = defaultdict(set)
    for client, history in histories.items():
        for description in {str(row["description"]).lower().strip() for row in history}:
            clients_by_description[description].add(client)
    class_sizes = Counter(labels.values())
    selected: dict[str, str] = {}
    for description, clients in clients_by_description.items():
        if len(clients) < 20:
            continue
        counts = Counter(labels[client] for client in clients)
        best: tuple[float, str] | None = None
        for family in LABELS:
            if family == "none":
                continue
            coverage = counts[family] / class_sizes[family]
            outside = (len(clients) - counts[family] + 1) / (len(labels) - class_sizes[family] + 2)
            lift = ((counts[family] + 1) / (class_sizes[family] + 2)) / outside
            if coverage >= 0.2 and lift >= 2:
                candidate = (lift, family)
                if best is None or candidate > best:
                    best = candidate
        if best is not None:
            selected[description] = best[1]
    return selected


def _candidate_stats(history: list[dict], mapping: dict[str, str]) -> dict:
    """Summarize mapped historical series without using this client's label."""
    streams: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(history, key=lambda item: item["timestamp"]):
        description = str(row["description"]).lower().strip()
        if description in mapping:
            streams[description].append(row)
    result = {
        "candidate_streams": 0,
        "recent_candidate_streams": 0,
        "regular_candidate_streams": 0,
        "recent_regular_candidate_streams": 0,
        "candidate_families": set(),
        "recent_candidate_families": set(),
    }
    for description, rows in streams.items():
        if len(rows) < 2:
            continue
        result["candidate_streams"] += 1
        family = mapping[description]
        result["candidate_families"].add(family)
        gaps = [
            (right["timestamp"] - left["timestamp"]).total_seconds() / 86400
            for left, right in zip(rows, rows[1:], strict=False)
        ]
        typical_gap = median(gaps)
        last_age = (CUTOFF - rows[-1]["timestamp"]).total_seconds() / 86400
        recent = typical_gap > 0 and last_age <= 2 * typical_gap
        regular = len(gaps) >= 2 and pstdev(gaps) <= 3
        result["recent_candidate_streams"] += int(recent)
        if recent:
            result["recent_candidate_families"].add(family)
        result["regular_candidate_streams"] += int(regular)
        result["recent_regular_candidate_streams"] += int(recent and regular)
    return result


def _client_stats(history: list[dict]) -> tuple[dict, Counter[str], Counter[str]]:
    ordered = sorted(history, key=lambda row: row["timestamp"])
    first, last = ordered[0]["timestamp"], ordered[-1]["timestamp"]
    streams: dict[str, list[dict]] = defaultdict(list)
    for row in ordered:
        streams[str(row["description"]).lower().strip()].append(row)
    stats = {
        "transactions": len(ordered),
        "history_days": (last - first).total_seconds() / 86400,
        "last_age_days": (CUTOFF - last).total_seconds() / 86400,
        "repeated_streams": 0,
        "regular_streams": 0,
        "regular_monthly_out_streams": 0,
        "same_timestamp_pairs": 0,
        "zero_gap_streams": 0,
        "single_gap_streams": 0,
        "very_long_gap_streams": 0,
    }
    timestamps = Counter(row["timestamp"] for row in ordered)
    stats["same_timestamp_pairs"] = sum(n * (n - 1) // 2 for n in timestamps.values())
    periods: Counter[str] = Counter()
    regular_periods: Counter[str] = Counter()
    for rows in streams.values():
        if len(rows) < 2:
            continue
        stats["repeated_streams"] += 1
        gaps = [
            (right["timestamp"] - left["timestamp"]).total_seconds() / 86400
            for left, right in zip(rows, rows[1:], strict=False)
        ]
        gap_median = median(gaps)
        period = _period(gap_median)
        periods[period] += 1
        if len(gaps) == 1:
            stats["single_gap_streams"] += 1
        if any(gap == 0 for gap in gaps):
            stats["zero_gap_streams"] += 1
        if any(gap > 365 for gap in gaps):
            stats["very_long_gap_streams"] += 1
        # A single gap has zero standard deviation by definition, so require 3 events.
        if len(gaps) >= 2 and pstdev(gaps) <= 3:
            stats["regular_streams"] += 1
            regular_periods[period] += 1
            if period == "monthly" and all(row["direction"] == "out" for row in rows):
                stats["regular_monthly_out_streams"] += 1
    return stats, periods, regular_periods


def analyze(
    labels: dict[str, str],
    histories: dict[str, list[dict]],
    prefix: str,
    candidate_descriptions: dict[str, str],
) -> dict:
    """Return reproducible per-class recurrence and anomaly aggregates."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    periods: Counter[str] = Counter()
    regular_periods: Counter[str] = Counter()
    for client, label in labels.items():
        stats, client_periods, client_regular_periods = _client_stats(histories[client])
        stats.update(_candidate_stats(histories[client], candidate_descriptions))
        by_label[label].append(stats)
        periods.update(client_periods)
        regular_periods.update(client_regular_periods)
    class_rows = []
    for label in (*LABELS, "any_family"):
        rows = (
            [row for family in LABELS if family != "none" for row in by_label[family]]
            if label == "any_family"
            else by_label[label]
        )
        count = len(rows)
        class_rows.append(
            {
                "label": label,
                "clients": count,
                "transactions_median": median(row["transactions"] for row in rows),
                "transactions_min": min(row["transactions"] for row in rows),
                "clients_le_2_tx": sum(row["transactions"] <= 2 for row in rows),
                "clients_le_10_tx": sum(row["transactions"] <= 10 for row in rows),
                "history_le_30d": sum(row["history_days"] <= 30 for row in rows),
                "history_days_min": round(min(row["history_days"] for row in rows), 2),
                "last_age_gt_30d": sum(row["last_age_days"] > 30 for row in rows),
                "repeated_streams_median": median(row["repeated_streams"] for row in rows),
                "with_regular_stream_pct": _percent(
                    sum(row["regular_streams"] > 0 for row in rows), count
                ),
                "with_regular_monthly_out_pct": _percent(
                    sum(row["regular_monthly_out_streams"] > 0 for row in rows), count
                ),
                "with_candidate_stream_pct": _percent(
                    sum(row["candidate_streams"] > 0 for row in rows), count
                ),
                "with_recent_candidate_stream_pct": _percent(
                    sum(row["recent_candidate_streams"] > 0 for row in rows), count
                ),
                "with_regular_candidate_stream_pct": _percent(
                    sum(row["regular_candidate_streams"] > 0 for row in rows), count
                ),
                "with_recent_regular_candidate_stream_pct": _percent(
                    sum(row["recent_regular_candidate_streams"] > 0 for row in rows), count
                ),
                "with_target_family_stream_pct": (
                    None
                    if label in {"none", "any_family"}
                    else _percent(sum(label in row["candidate_families"] for row in rows), count)
                ),
                "with_recent_target_family_stream_pct": (
                    None
                    if label in {"none", "any_family"}
                    else _percent(
                        sum(label in row["recent_candidate_families"] for row in rows), count
                    )
                ),
                "last_age_days_median": round(median(row["last_age_days"] for row in rows), 2),
                "zero_gap_streams": sum(row["zero_gap_streams"] for row in rows),
                "single_gap_streams": sum(row["single_gap_streams"] for row in rows),
                "very_long_gap_streams": sum(row["very_long_gap_streams"] for row in rows),
                "same_timestamp_pairs": sum(row["same_timestamp_pairs"] for row in rows),
            }
        )
    return {
        "partition": prefix,
        "clients": len(labels),
        "transactions": sum(len(history) for history in histories.values()),
        "periods_all_repeated_streams": {name: periods[name] for name in PERIODS},
        "periods_regular_3plus_events": {name: regular_periods[name] for name in PERIODS},
        "classes": class_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/javi_recurrence.json"))
    args = parser.parse_args()
    train_labels, train_histories = _read_partition(args.data_dir, "train")
    valid_labels, valid_histories = _read_partition(args.data_dir, "valid")
    candidate_descriptions = _learn_candidate_descriptions(train_labels, train_histories)
    result = {
        "candidate_descriptions_by_family": dict(Counter(candidate_descriptions.values())),
        "train": analyze(train_labels, train_histories, "train", candidate_descriptions),
        "valid": analyze(valid_labels, valid_histories, "valid", candidate_descriptions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
