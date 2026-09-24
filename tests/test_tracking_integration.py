from pathlib import Path

import pytest

from transaction_forecasting.api import service
from transaction_forecasting.tracking_integration import (
    DEFAULT_DATABASE,
    read_experiments,
    record_validation_run,
)


def test_missing_database_is_not_created(tmp_path):
    database = tmp_path / DEFAULT_DATABASE
    assert read_experiments(database)["experiments"] == []
    assert not database.parent.exists()


def test_runner_records_and_api_reads_validation(tmp_path, monkeypatch):
    row = {
        "model": "dummy",
        "feature_set": "none",
        "class_weight": "none",
        "notes": "fixture",
        "train_seconds": 0,
        "inference_seconds": 0.1,
        "validation_clients": 2,
    }
    metrics = {"macro_f1": 0.5, "accuracy": 0.5, "per_class": {"none": {"f1-score": 0.5}}}
    ids = record_validation_run(
        repo_root=tmp_path,
        experiments=[row, {**row, "model": "other"}],
        details={"dummy": metrics, "other": metrics},
        settings={"project": {"random_seed": 42}},
        selected_model="dummy",
    )
    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    history = service.get_experiments()
    assert {item["experiment_id"] for item in history["experiments"]} == set(ids)
    records = {item["model_name"]: item for item in history["experiments"]}
    assert records["dummy"]["metrics"]["f1_per_class"] == {"none": 0.5}
    assert records["dummy"]["result"] == "selected"
    assert records["other"]["result"] == "evaluated"
    assert records["dummy"]["metrics"]["run_id"] == records["other"]["metrics"]["run_id"]
    assert records["dummy"]["compute_seconds"] == 0.1
    assert records["dummy"]["baseline_macro_f1"] is None
    first = service.get_experiments(limit=1)["experiments"]
    second = service.get_experiments(limit=1, offset=1)["experiments"]
    assert first[0]["experiment_id"] != second[0]["experiment_id"]
    assert service.get_experiments(offset=2)["experiments"] == []


def test_corrupt_database_reports_unavailable(tmp_path, monkeypatch):
    database = tmp_path / DEFAULT_DATABASE
    database.parent.mkdir(parents=True)
    database.write_text("invalid database", encoding="utf-8")
    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    with pytest.raises(service.ArtifactUnavailable):
        service.get_experiments()


@pytest.mark.parametrize("limit,offset", [(0, 0), (101, 0), (1, -1)])
def test_invalid_page(limit, offset):
    with pytest.raises(ValueError):
        read_experiments(Path("missing.sqlite3"), limit=limit, offset=offset)
