"""Small, dependency-free experiment logging backed by concurrent-safe SQLite."""

from __future__ import annotations

import json
import math
import numbers
import platform
import sqlite3
import subprocess
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    recorded_at_utc TEXT NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER,
    description TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    features_json TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    baseline_macro_f1 REAL,
    delta_vs_baseline REAL,
    compute_seconds REAL,
    result TEXT NOT NULL,
    notes TEXT NOT NULL,
    risk_notes TEXT NOT NULL,
    python_version TEXT NOT NULL
)
"""


class ExperimentLogger:
    """Append experiment records to one SQLite database.

    SQLite serializes concurrent writes from separate processes. Each call opens
    its own connection, enables WAL mode, and waits briefly if another run is
    writing at the same time.

    Args:
        db_path: SQLite file path. Relative paths resolve from ``repo_root``.
            By default records go to ``outputs/experiments/experiments.sqlite3``.
        repo_root: Git working tree root. Defaults to the root of this package.
        timeout_seconds: Maximum time SQLite waits for a concurrent writer.
    """

    def __init__(
        self,
        db_path: str | Path = "outputs/experiments/experiments.sqlite3",
        repo_root: str | Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.repo_root = (
            Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
        )
        db_path = Path(db_path)
        self.db_path = db_path if db_path.is_absolute() else self.repo_root / db_path
        self.timeout_seconds = timeout_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(_SCHEMA)

    def log_experiment(
        self,
        *,
        description: str,
        model_name: str,
        model_version: str,
        features: list[str],
        hyperparameters: dict[str, Any],
        metrics: dict[str, Any],
        result: str = "not_evaluated",
        baseline_macro_f1: float | None = None,
        compute_seconds: float | None = None,
        notes: str = "",
        risk_notes: str = "",
    ) -> str:
        """Validate and save an experiment, returning its unique record ID.

        ``metrics`` must contain ``macro_f1``, ``accuracy``, and a mapping named
        ``f1_per_class``. All score values must be finite and between 0 and 1.
        Input dictionaries must contain JSON-serializable values.
        """
        self._require_text(description, "description")
        self._require_text(model_name, "model_name")
        self._require_text(model_version, "model_version")
        self._require_text(result, "result")
        if not isinstance(features, list) or any(not isinstance(item, str) for item in features):
            raise TypeError("features must be a list of strings")
        if not isinstance(hyperparameters, dict):
            raise TypeError("hyperparameters must be a dictionary")
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary")
        if not isinstance(notes, str):
            raise TypeError("notes must be a string")
        if not isinstance(risk_notes, str):
            raise TypeError("risk_notes must be a string")

        normalized_metrics = self._validate_metrics(metrics)
        normalized_baseline = (
            None
            if baseline_macro_f1 is None
            else self._score(baseline_macro_f1, "baseline_macro_f1")
        )
        normalized_compute_seconds = self._compute_seconds(compute_seconds)
        git_commit, git_dirty = self._git_metadata()
        record = (
            str(uuid.uuid4()),
            datetime.now(UTC).isoformat(),
            git_commit,
            None if git_dirty is None else int(git_dirty),
            description,
            model_name,
            model_version,
            self._json(features),
            self._json(hyperparameters),
            self._json(normalized_metrics),
            normalized_baseline,
            (
                None
                if normalized_baseline is None
                else normalized_metrics["macro_f1"] - normalized_baseline
            ),
            normalized_compute_seconds,
            result,
            notes,
            risk_notes,
            platform.python_version(),
        )

        query = """
        INSERT INTO experiments (
            experiment_id, recorded_at_utc, git_commit, git_dirty, description,
            model_name, model_version, features_json, hyperparameters_json,
            metrics_json, baseline_macro_f1, delta_vs_baseline, compute_seconds,
            result, notes, risk_notes, python_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with closing(self._connect()) as connection, connection:
            connection.execute(query, record)
        return record[0]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.timeout_seconds)
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        return connection

    def _git_metadata(self) -> tuple[str | None, bool | None]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None, None
        return commit or None, bool(status.strip())

    @classmethod
    def _validate_metrics(cls, metrics: dict[str, Any]) -> dict[str, Any]:
        required = {"macro_f1", "accuracy", "f1_per_class"}
        missing = required - metrics.keys()
        if missing:
            raise ValueError(f"metrics is missing required keys: {', '.join(sorted(missing))}")
        per_class = metrics["f1_per_class"]
        if not isinstance(per_class, dict) or not per_class:
            raise ValueError("metrics['f1_per_class'] must be a non-empty dictionary")
        normalized: dict[str, Any] = {
            "macro_f1": cls._score(metrics["macro_f1"], "macro_f1"),
            "accuracy": cls._score(metrics["accuracy"], "accuracy"),
            "f1_per_class": {
                str(label): cls._score(score, f"f1_per_class[{label!r}]")
                for label, score in per_class.items()
            },
        }
        for key, value in metrics.items():
            if key not in required:
                normalized[key] = value
        return normalized

    @staticmethod
    def _score(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a numeric score")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
        return score

    @staticmethod
    def _compute_seconds(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("compute_seconds must be a number or None")
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("compute_seconds must be finite and non-negative")
        return seconds

    @staticmethod
    def _json(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "features, hyperparameters, and metrics must be JSON-serializable"
            ) from error

    @staticmethod
    def _require_text(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
