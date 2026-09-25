"""Verify actual runner/bundle/adapter parity without requiring the HTTP extras."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from transaction_forecasting.product.adapter import ModelAdapter, UnknownClient
from transaction_forecasting.product.agent import investigate
from transaction_forecasting.product.provenance import DEFAULT_BUNDLE, ROOT, digest
from transaction_forecasting.ubs.data import LABELS, PREDICTION_COLUMN
from transaction_forecasting.ubs.v4 import predict_probabilities


def main():
    adapter = ModelAdapter()
    final = ROOT / "outputs/metrics/ubs_v4_final"
    scores = pd.read_csv(final / "test_probabilities.csv", index_col="client_id")
    fresh = predict_probabilities(adapter._model, adapter._transactions).reindex(scores.index)
    np.testing.assert_allclose(fresh, scores, atol=1e-12, rtol=0)
    stored = pd.read_csv(final / "submission_v4.csv").set_index("client_id")[PREDICTION_COLUMN]
    assert stored.eq(fresh.idxmax(axis=1).reindex(stored.index)).all()
    before = {p.name: digest(p) for p in DEFAULT_BUNDLE.iterdir() if p.is_file()}
    ids = list(
        dict.fromkeys([c["client_id"] for c in adapter.get_cases()] + list(stored.index[:6]))
    )
    steps = []
    for client in ids:
        prediction = adapter.predict_client(client)
        assert prediction["predicted_family"] == stored.loc[client]
        np.testing.assert_allclose(
            [prediction["scores"][label] for label in LABELS], scores.loc[client], atol=1e-10
        )
        result = investigate(adapter, client)
        assert result["predicted_family"] == prediction["predicted_family"]
        assert not result["submission_modified"] and result["steps"] <= 8
        steps.append(result["steps"])
        assert all(row["timestamp"] < "2026-01-01" for row in adapter.get_client_history(client))
    try:
        adapter.predict_client("UNKNOWN-TEST-CLIENT")
    except UnknownClient:
        pass
    else:
        raise AssertionError("Unknown client accepted")
    after = {p.name: digest(p) for p in DEFAULT_BUNDLE.iterdir() if p.is_file()}
    assert before == after
    result = {
        "level": "VERIFIED",
        "bulk_score_parity_clients": len(stored),
        "adapter_and_agent_clients": len(ids),
        "agent_steps": steps,
        "artifacts_unchanged": True,
        "unknown_client_rejected": True,
        "fit_scope": adapter.get_model_metadata()["fit_scope"],
        "submission_sha256": digest(final / "submission_v4.csv"),
        "api_http_verified": False,
        "frontend_build_verified": False,
        "limitation": "Missing FastAPI/npm dependencies; no HTTP or browser test",
    }
    (ROOT / "outputs/metrics/v4_synthesis/product_parity.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
