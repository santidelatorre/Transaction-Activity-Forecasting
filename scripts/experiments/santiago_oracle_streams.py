"""Run Santiago's train-only exact-description stream oracle audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from transaction_forecasting.ubs.data import read_labels, read_transactions
from transaction_forecasting.ubs.stream_oracle import (
    TrainOnlyFamilyMapper,
    build_streams,
    client_candidate_table,
    evidence_verdict,
    failure_breakdown,
    oof_train_audit,
    score_candidate_oracle,
)

BASE_COMMIT = "5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749"
V2_MACRO_F1 = 0.3915494559105542


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics/v3_discovery/santiago_oracle_streams"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze all detector/mapping work using TRAIN before validation labels are read.
    train_transactions = read_transactions(args.data_dir / "train_transactions.jsonl")
    train_labels = read_labels(args.data_dir / "train_labels.csv")
    valid_transactions = read_transactions(args.data_dir / "valid_transactions.jsonl")
    train_streams = build_streams(train_transactions)
    valid_streams = build_streams(valid_transactions)
    oof = oof_train_audit(train_streams, train_labels)
    mapper = TrainOnlyFamilyMapper().fit(train_streams, train_labels)
    mapped_valid_streams = mapper.transform(valid_streams)
    client_table = client_candidate_table(
        mapped_valid_streams, pd.Index(valid_transactions["client_id"].unique())
    )

    # Validation labels enter only here, after every learned object is frozen.
    valid_labels = read_labels(args.data_dir / "valid_labels.csv")
    scored = score_candidate_oracle(client_table, valid_labels)
    clients = scored.pop("clients")
    breakdown = failure_breakdown(clients)
    due_valid = mapped_valid_streams[mapped_valid_streams["is_candidate"]]
    verdict = evidence_verdict(
        scored["candidate_family_recall"],
        scored["oracle_metrics"]["macro_f1"],
        scored["per_class"],
    )
    summary = {
        "base_commit": BASE_COMMIT,
        "v2_macro_f1": V2_MACRO_F1,
        "stream_definition": "exact (client_id, description), >=2 appearances",
        "candidate_definition": "first median-gap projection on/after cutoff is <=90 days",
        "mapping_definition": "train-only winner positive-family lift >=1.5, support >=2 clients",
        "validation_clients": len(valid_labels),
        "mean_candidate_streams_per_client": float(clients["candidate_streams"].mean()),
        "mean_candidate_families_per_client": float(clients["candidate_families"].map(len).mean()),
        "eligible_train_description_mappings": int(mapper.mapping_table_["eligible"].sum()),
        "mapped_validation_candidate_stream_share": float(
            due_valid["mapped_family"].notna().mean()
        ),
        "oof_train": oof,
        **scored,
        "failure_breakdown": breakdown,
        "verdict": verdict,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    mapped_valid_streams.to_csv(args.output_dir / "validation_streams.csv", index=False)
    clients.reset_index().to_csv(args.output_dir / "validation_client_audit.csv", index=False)
    mapper.mapping_table_.to_csv(args.output_dir / "train_family_mapping.csv", index=False)
    pd.DataFrame(scored["per_class"]).T.rename_axis("class").to_csv(
        args.output_dir / "class_breakdown.csv"
    )
    print(json.dumps(_json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
