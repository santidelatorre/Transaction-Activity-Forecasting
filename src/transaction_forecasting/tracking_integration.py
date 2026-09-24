"""Adapt UBS validation records and read tracking history without creating a DB."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from transaction_forecasting.experiment_tracking import ExperimentLogger

DEFAULT_DATABASE = "outputs/experiments/experiments.sqlite3"


def record_validation_run(
    *,
    repo_root: Path,
    experiments: list[dict[str, Any]],
    details: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    selected_model: str,
) -> list[str]:
    """Record evaluated candidates, before refitting on train + validation.

    Feature entries describe feature sets, not individual transformed columns.
    The complete configuration and candidate notes preserve tuning context.
    """
    logger = ExperimentLogger(repo_root=repo_root)
    run_id = str(uuid.uuid4())
    ids = []
    for row in experiments:
        name = row["model"]
        metrics = details[name]
        ids.append(
            logger.log_experiment(
                description=f"UBS validation candidate: {name}",
                model_name=name,
                model_version="ubs-v1",
                features=[row["feature_set"]],
                hyperparameters={
                    "config": settings,
                    "class_weight": row["class_weight"],
                    "candidate_notes": row["notes"],
                },
                metrics={
                    **metrics,
                    "f1_per_class": {
                        label: values["f1-score"] for label, values in metrics["per_class"].items()
                    },
                    "run_id": run_id,
                    "scope": "validation",
                    "validation_clients": row["validation_clients"],
                },
                compute_seconds=row["train_seconds"] + row["inference_seconds"],
                result="selected" if name == selected_model else "evaluated",
                notes=(
                    f"{row['notes']}; feature-set description; "
                    "compute time excludes shared feature construction and scoring"
                ),
                risk_notes=(
                    "Validation used for tuning and model selection; "
                    "not an independent test or leaderboard score."
                ),
            )
        )
    return ids


def read_experiments(database: Path, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    """Read a bounded history page; a missing DB is an ordinary empty state."""
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("limit must be 1..100 and offset must be non-negative")
    if not database.is_file():
        return {"available": False, "experiments": [], "limit": limit, "offset": offset}
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM experiments ORDER BY recorded_at_utc DESC, experiment_id DESC "
            "LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    records = []
    for row in rows:
        record = dict(row)
        for field in ("features", "hyperparameters", "metrics"):
            record[field] = json.loads(record.pop(f"{field}_json"))
        record["git_dirty"] = None if record["git_dirty"] is None else bool(record["git_dirty"])
        records.append(record)
    return {"available": True, "experiments": records, "limit": limit, "offset": offset}
