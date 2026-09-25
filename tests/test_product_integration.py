import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.product.provenance import (
    DEFAULT_BUNDLE,
    ROOT,
    ArtifactUnavailable,
    digest,
    model_lock,
    verify_bundle,
    verify_source,
)


def test_locked_sha_source_and_no_stale_product_config(tmp_path):
    lock = verify_source()
    assert lock["base_sha"] == "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"
    assert lock["candidate"] == "A" and lock["runner"] == "scripts/run_ubs_v3.py"
    path, blob = next(iter(lock["source_blobs"].items()))
    source = tmp_path / path
    source.parent.mkdir(parents=True)
    source.write_bytes((ROOT / path).read_bytes())
    verify_source(tmp_path, {"source_blobs": {path: blob}})
    source.write_text("tampered source", encoding="utf-8")
    with pytest.raises(ArtifactUnavailable, match="source changed"):
        verify_source(tmp_path, {"source_blobs": {path: blob}})
    for directory in (
        ROOT / "src/transaction_forecasting/product",
        ROOT / "src/transaction_forecasting/api",
    ):
        for file in directory.glob("*.py"):
            text = file.read_text(encoding="utf-8")
            assert "configs/ubs_v1.toml" not in text
            assert "submission_v1.csv" not in text


def test_missing_and_wrong_sha_bundles_fail_closed(tmp_path):
    with pytest.raises(ArtifactUnavailable):
        verify_bundle(tmp_path, tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model": {**model_lock(), "base_sha": "incorrect"},
                "fit_scope": "TRAIN+VALID",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactUnavailable, match="locked V3-A"):
        verify_bundle(tmp_path, tmp_path)


def test_api_unknown_missing_artifacts_and_budget():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from test_product_agent import StubAdapter
    from transaction_forecasting.api.main import create_app, get_adapter

    app = create_app()
    app.dependency_overrides[get_adapter] = lambda: StubAdapter()
    with TestClient(app) as client:
        assert client.get("/api/v1/clients/missing/prediction").status_code == 404
        assert client.post("/api/v1/clients/missing/investigate").status_code == 404
        assert client.post("/api/v1/clients/fixture/investigate?max_steps=99").status_code == 422
        assert client.get("/api/v1/health").json()["base_sha"] == "fixture-sha"

    def unavailable():
        raise ArtifactUnavailable("Missing frozen bundle")

    app.dependency_overrides[get_adapter] = unavailable
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 503
        assert client.get("/api/v1/clients/fixture/prediction").status_code == 503


@pytest.mark.skipif(
    os.environ.get("DEMO_REAL_TESTS") != "1", reason="Local challenge data required"
)
def test_real_runner_api_agent_and_submission_parity(tmp_path):
    from fastapi.testclient import TestClient

    from transaction_forecasting.api.main import create_app, get_adapter
    from transaction_forecasting.product.adapter import ModelAdapter
    from transaction_forecasting.ubs.data import PREDICTION_COLUMN

    adapter = ModelAdapter()
    runner_path = ROOT / "scripts/run_ubs_v3.py"
    spec = importlib.util.spec_from_file_location("frozen_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    # Exercise the actual official runner predictions() path, including its A column.
    # Text diagnostics are irrelevant to A and cannot choose its winner.
    class UnusedTextDiagnostics:
        def predict_scores(self, transactions):
            return adapter._scores.drop(columns="none")

    frame = runner.predictions(
        adapter._model, UnusedTextDiagnostics(), adapter._transactions, tmp_path, "test"
    )
    official = adapter._model.predict(adapter._transactions)
    pd.testing.assert_series_equal(frame.A, official, check_names=False)
    stored = adapter._predictions.set_index("client_id")[PREDICTION_COLUMN]
    pd.testing.assert_series_equal(stored.sort_index(), official.sort_index(), check_names=False)
    assert len(stored) == 1000 and stored.index.is_unique
    before = {path.name: digest(path) for path in DEFAULT_BUNDLE.iterdir() if path.is_file()}
    app = create_app()
    app.dependency_overrides[get_adapter] = lambda: adapter
    clients = list(
        dict.fromkeys([case["client_id"] for case in adapter.get_cases()] + list(stored.index[:6]))
    )
    with TestClient(app) as client:
        for client_id in clients:
            response = client.get(f"/api/v1/clients/{client_id}")
            assert response.status_code == 200, response.text
            result = response.json()
            prediction = result["prediction"]
            assert prediction["predicted_family"] == frame.loc[client_id, "A"]
            assert prediction["base_sha"] == model_lock()["base_sha"]
            np.testing.assert_allclose(
                list(prediction["scores"].values()), adapter._scores.loc[client_id], atol=1e-10
            )
            assert result["history"] and prediction["horizon_days"] == 90
            assert all(row["timestamp"] < "2026-01-01" for row in result["history"])
            assert "balance" not in result and "probability" not in prediction
            investigation = client.post(f"/api/v1/clients/{client_id}/investigate").json()
            assert investigation["predicted_family"] == prediction["predicted_family"]
            assert investigation["steps"] <= 8 and not investigation["submission_modified"]
        assert client.get("/api/v1/clients/UNKNOWN").status_code == 404
        if (ROOT / "frontend/dist/index.html").exists():
            assert client.get("/").status_code == 200
    after = {path.name: digest(path) for path in DEFAULT_BUNDLE.iterdir() if path.is_file()}
    assert before == after
