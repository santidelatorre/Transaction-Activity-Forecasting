"""Tests for the standalone experiment logger."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transaction_forecasting.experiment_tracking import ExperimentLogger


def _log(logger: ExperimentLogger, run_number: int) -> str:
    return logger.log_experiment(
        description=f"Compare run {run_number} against V1",
        model_name="test-model",
        model_version="v1",
        features=["recency", "frequency"],
        hyperparameters={"seed": run_number},
        metrics={
            "macro_f1": 0.3,
            "accuracy": 0.4,
            "f1_per_class": {"cloud": 0.2, "none": 0.5},
        },
        baseline_macro_f1=0.2710243,
        compute_seconds=1.25,
        result="completed",
        notes="synthetic test record",
        risk_notes="none",
    )


def test_logs_reproducibility_metadata_and_baseline_delta(tmp_path: Path) -> None:
    database = tmp_path / "experiments.sqlite3"
    logger = ExperimentLogger(db_path=database, repo_root=tmp_path)

    experiment_id = _log(logger, 1)

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """SELECT experiment_id, git_commit, description, model_name,
                      model_version, features_json, hyperparameters_json,
                      metrics_json, baseline_macro_f1, delta_vs_baseline,
                      compute_seconds, result, notes, risk_notes, python_version
               FROM experiments"""
        ).fetchone()

    assert row is not None
    assert row[0] == experiment_id
    assert row[1] is None  # tmp_path is not a Git working tree
    assert row[2:5] == ("Compare run 1 against V1", "test-model", "v1")
    assert json.loads(row[5]) == ["recency", "frequency"]
    assert json.loads(row[6]) == {"seed": 1}
    assert json.loads(row[7])["f1_per_class"] == {"cloud": 0.2, "none": 0.5}
    assert row[8] == 0.2710243
    assert abs(row[9] - (0.3 - 0.2710243)) < 1e-12
    assert row[10:14] == (1.25, "completed", "synthetic test record", "none")
    assert row[14]


def test_parallel_calls_do_not_lose_records(tmp_path: Path) -> None:
    database = tmp_path / "parallel.sqlite3"
    logger = ExperimentLogger(db_path=database, repo_root=tmp_path, timeout_seconds=10)

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(lambda n: _log(logger, n), range(40)))

    with sqlite3.connect(database) as connection:
        count, distinct_ids = connection.execute(
            "SELECT count(*), count(DISTINCT experiment_id) FROM experiments"
        ).fetchone()

    assert len(set(ids)) == 40
    assert (count, distinct_ids) == (40, 40)
