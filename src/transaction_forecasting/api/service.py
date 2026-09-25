"""Read and validate artifacts produced by the UBS baseline runner."""

from __future__ import annotations

import json
import os
import sqlite3
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from transaction_forecasting.api import dashboard
from transaction_forecasting.evaluation.official import (
    LABELS,
    PREDICTION,
    validate_submission,
)
from transaction_forecasting.tracking_integration import DEFAULT_DATABASE, read_experiments
from transaction_forecasting.ubs.data import read_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _artifact_root() -> Path:
    return Path(os.environ.get("RECURRING_ARTIFACT_ROOT", str(PROJECT_ROOT))).resolve()


def get_dashboard() -> dict[str, Any]:
    try:
        return dashboard.snapshot(_artifact_root())
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ArtifactUnavailable(
            "The V3-A evaluation artifacts are unavailable or invalid."
        ) from error


def get_experiments(*, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("Invalid experiment page")
    try:
        root = _artifact_root()
        current = get_dashboard()
        imported = current["experiments"]
        # Artifact comparisons precede the paginated SQLite history. No DB is created.
        db_offset = max(0, offset - len(imported))
        selected = imported[offset : offset + limit]
        remaining = limit - len(selected)
        recorded = read_experiments(
            root / DEFAULT_DATABASE, limit=max(1, remaining), offset=db_offset
        )
        db_rows = recorded["experiments"] if remaining else []
        return {
            "available": bool(imported) or recorded["available"],
            "experiments": [*selected, *db_rows],
            "limit": limit,
            "offset": offset,
            "comparison_group": current["comparison_group"],
        }
    except (OSError, sqlite3.Error, ValueError, KeyError) as error:
        raise ArtifactUnavailable("Experiment history is unavailable or invalid.") from error


class ArtifactUnavailable(Exception):
    """An output exists but cannot be safely served by the dashboard."""


@lru_cache(maxsize=1)
def _settings() -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / "ubs_v1.toml").open("rb") as config_file:
        return tomllib.load(config_file)


def _configured_path(section: str, key: str) -> Path:
    configured = Path(_settings()[section][key])
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def _submission() -> pd.DataFrame | None:
    submission_path = _configured_path("outputs", "submission")
    sample_path = _configured_path("data", "directory") / "sample_submission.csv"
    if not submission_path.is_file() or not sample_path.is_file():
        return None

    try:
        predictions = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
        sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
        return validate_submission(predictions, sample)
    except (OSError, ValueError) as error:
        raise ArtifactUnavailable(f"The generated submission is invalid: {error}") from error


def _transactions() -> pd.DataFrame | None:
    transaction_path = _configured_path("data", "directory") / "test_transactions.jsonl"
    if not transaction_path.is_file():
        return None
    stat = transaction_path.stat()
    return _read_transactions_cached(str(transaction_path), stat.st_mtime_ns)


@lru_cache(maxsize=2)
def _read_transactions_cached(path: str, _modified_ns: int) -> pd.DataFrame:
    try:
        return read_transactions(path)
    except (OSError, ValueError) as error:
        raise ArtifactUnavailable(f"Test transaction data is invalid: {error}") from error


def _summary_metrics() -> dict[str, Any] | None:
    summary_path = _configured_path("outputs", "metrics_directory") / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open(encoding="utf-8") as summary_file:
            return json.load(summary_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactUnavailable(f"The generated metrics summary is invalid: {error}") from error


def get_health() -> dict[str, Any]:
    predictions = _submission() is not None
    metrics = _summary_metrics() is not None
    transactions = (_configured_path("data", "directory") / "test_transactions.jsonl").is_file()
    current = get_dashboard()
    return {
        "status": "ready" if predictions or current["available"] else "waiting_for_outputs",
        "predictions_available": predictions,
        "metrics_available": metrics,
        "transactions_available": transactions,
        "dashboard_available": current["available"],
        "dashboard_model_version": dashboard.MODEL_VERSION,
    }


def get_overview(*, page: int, page_size: int, search: str = "") -> dict[str, Any]:
    predictions = _submission()
    if predictions is None:
        return {
            "available": False,
            "message": (
                "No validated submission is available. Run scripts/run_ubs_baseline.py first."
            ),
            "summary": None,
            "distribution": [],
            "clients": [],
            "page": page,
            "page_size": page_size,
            "total_filtered": 0,
            "metrics": _summary_metrics(),
            "transactions_available": False,
        }

    transactions = _transactions()
    if transactions is None:
        history = pd.DataFrame(columns=["client_id", "timestamp"])
    else:
        history = transactions[["client_id", "timestamp"]]

    observed = history.groupby("client_id", sort=False).agg(
        transaction_count=("timestamp", "size"), last_observed=("timestamp", "max")
    )
    clients = predictions[["client_id", PREDICTION]].rename(columns={PREDICTION: "prediction"})
    clients = clients.join(observed, on="client_id")
    if transactions is None:
        clients["transaction_count"] = None
    else:
        clients["transaction_count"] = clients["transaction_count"].fillna(0).astype(int)
    clients["last_observed"] = clients["last_observed"].map(
        lambda value: value.isoformat() if pd.notna(value) else None
    )

    query = search.strip().casefold()
    filtered = clients
    if query:
        mask = clients["client_id"].str.casefold().str.contains(query, regex=False)
        mask |= clients["prediction"].str.casefold().str.contains(query, regex=False)
        filtered = clients[mask]

    offset = (page - 1) * page_size
    records = json.loads(filtered.iloc[offset : offset + page_size].to_json(orient="records"))
    counts = predictions[PREDICTION].value_counts().reindex(LABELS, fill_value=0)
    total_clients = len(predictions)
    with_history = (
        int(clients["transaction_count"].gt(0).sum()) if transactions is not None else None
    )
    family_count = int(predictions[PREDICTION].ne("none").sum())

    return {
        "available": True,
        "message": None,
        "summary": {
            "clients": total_clients,
            "family_count": family_count,
            "distinct_families": int(
                predictions.loc[predictions[PREDICTION].ne("none"), PREDICTION].nunique()
            ),
            "none_count": int(predictions[PREDICTION].eq("none").sum()),
            "with_history": with_history,
            "without_history": total_clients - with_history if with_history is not None else None,
        },
        "distribution": [
            {"label": label, "count": int(count), "share": int(count) / total_clients}
            for label, count in counts.items()
        ],
        "clients": records,
        "page": page,
        "page_size": page_size,
        "total_filtered": len(filtered),
        "metrics": _summary_metrics(),
        "transactions_available": transactions is not None,
    }


def get_client(client_id: str) -> dict[str, Any] | None:
    predictions = _submission()
    if predictions is None:
        return None
    selected = predictions.loc[predictions["client_id"].eq(client_id)]
    if selected.empty:
        return None

    transactions = _transactions()
    if transactions is None:
        history: list[dict[str, Any]] = []
    else:
        client_history = transactions.loc[transactions["client_id"].eq(client_id)].copy()
        client_history["timestamp"] = client_history["timestamp"].map(
            lambda value: value.isoformat()
        )
        history = json.loads(client_history.to_json(orient="records"))

    return {
        "client_id": client_id,
        "prediction": str(selected.iloc[0][PREDICTION]),
        "transactions": history,
        "transactions_available": transactions is not None,
    }


def get_results() -> dict[str, Any]:
    metrics_dir = _configured_path("outputs", "metrics_directory")
    summary = _summary_metrics()
    validation_path = metrics_dir / "validation_metrics.json"
    experiments_path = metrics_dir / "experiments.csv"

    validation = None
    try:
        if validation_path.is_file():
            with validation_path.open(encoding="utf-8") as validation_file:
                validation = json.load(validation_file)

        experiments = []
        if experiments_path.is_file():
            experiments = json.loads(pd.read_csv(experiments_path).to_json(orient="records"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactUnavailable(f"The evaluation artifacts are invalid: {error}") from error

    return {
        "available": summary is not None,
        "summary": summary,
        "labels": list(LABELS),
        "validation_metrics": validation,
        "experiments": experiments,
        "scope": "validation",
    }


def submission_csv() -> str | None:
    predictions = _submission()
    return None if predictions is None else predictions.to_csv(index=False)
