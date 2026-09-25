from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from transaction_forecasting.product.adapter import UnknownClient
from transaction_forecasting.product.agent import TOOLS, execute_tool, investigate
from transaction_forecasting.product.evidence import candidate_streams, data_quality


class StubAdapter:
    """Synthetic unit fixture only; never used in product code or demonstration."""

    def __init__(self, **overrides):
        self.prediction = {
            "client_id": "fixture",
            "predicted_family": "music",
            "score": 0.8,
            "margin": 0.6,
            "scores": {"music": 0.8, "cloud": 0.2},
            "top_alternatives": [{"family": "cloud", "score": 0.2}],
            "component_disagreement": False,
            "component_choices": {
                "family_identity_model": "music",
                "periodicity_heuristic": "music",
            },
            "base_sha": "fixture-sha",
            "evidence": {
                "supporting_stream_count": 1,
                "candidate_ambiguity": False,
                "data_quality": {"degraded": False},
            },
        }
        self.prediction.update(overrides)

    def predict_client(self, client_id):
        if client_id != "fixture":
            raise UnknownClient(client_id)
        return deepcopy(self.prediction)

    def get_candidate_streams(self, client_id):
        return [{"associated_family": "music", "strong_recurrence": True, "count": 4}]

    def get_data_quality(self, client_id):
        return deepcopy(self.prediction["evidence"]["data_quality"])

    def get_client_history(self, client_id):
        return [{"description": "unit fixture", "timestamp": "2025-12-01"}]

    def get_model_metadata(self):
        return {"base_sha": "fixture-sha", "model_version": "fixture", "artifact_sha256": "test"}

    def get_global_metrics(self):
        return {"macro_f1": 0.2, "scope": "unit fixture"}


def tools_used(report):
    return [step["tool"] for step in report["trace"]]


def test_clear_prediction_stops_without_fixed_tool_sequence():
    result = investigate(StubAdapter(), "fixture")
    assert tools_used(result) == ["predict_client"]
    assert result["stop_reason"] == "sufficient_evidence"


@pytest.mark.parametrize("signal", ["margin", "quality", "candidate", "disagreement", "missing"])
def test_conditional_investigation_signals(signal):
    adapter = StubAdapter()
    if signal == "margin":
        adapter.prediction["margin"] = 0.01
    if signal == "quality":
        adapter.prediction["evidence"]["data_quality"]["degraded"] = True
    if signal == "candidate":
        adapter.prediction["evidence"]["candidate_ambiguity"] = True
    if signal == "disagreement":
        adapter.prediction["component_disagreement"] = True
    if signal == "missing":
        adapter.prediction["evidence"]["supporting_stream_count"] = 0
    result = investigate(adapter, "fixture")
    chosen = tools_used(result)
    expected = {
        "margin": {"inspect_candidate_streams", "compare_alternatives"},
        "quality": {"inspect_data_quality", "inspect_recurrence"},
        "candidate": {"inspect_candidate_streams", "compare_alternatives"},
        "disagreement": {"compare_alternatives", "inspect_candidate_streams"},
        "missing": {"inspect_history", "inspect_recurrence"},
    }
    assert set(chosen) == expected[signal] | {"predict_client"}
    assert len(chosen) == len(set(chosen)) <= 8
    assert result["stop_reason"] == "evidence_limit"
    assert result["predicted_family"] == adapter.prediction["predicted_family"]


def test_budget_invalid_tools_and_untrusted_reasoner_cannot_change_prediction():
    adapter = StubAdapter(margin=0.01)
    assert investigate(adapter, "fixture", max_steps=1)["stop_reason"] == "tool_budget_exhausted"

    class BadPolicy:
        name = "test reasoning plugin"

        def next_tool(self, observations, available):
            observations["predict_client"]["predicted_family"] = "cloud"
            return "write_submission"

    result = investigate(adapter, "fixture", BadPolicy())
    assert result["stop_reason"] == "invalid_or_repeated_tool"
    assert result["predicted_family"] == "music"
    assert result["trace"][0]["result"]["predicted_family"] == "music"
    assert not result["submission_modified"]
    with pytest.raises(ValueError):
        execute_tool(adapter, "fixture", "write_submission")
    with pytest.raises(ValueError):
        investigate(adapter, "fixture", max_steps=9)


def test_all_eight_tools_available_and_none_is_preserved():
    adapter = StubAdapter(predicted_family="none")
    adapter.prediction["evidence"]["supporting_stream_count"] = 0
    adapter.prediction["scores"]["none"] = 0.8
    for tool in TOOLS:
        assert execute_tool(adapter, "fixture", tool) is not None
    report = investigate(adapter, "fixture")
    assert report["predicted_family"] == "none"
    assert "missing_supporting_recurrence" in report["uncertainty_flags"]


def test_evidence_is_computable_keeps_currencies_and_excludes_refunds():
    history = pd.DataFrame(
        {
            "client_id": ["fixture"] * 4,
            "description": ["service payment"] * 4,
            "currency": ["chf"] * 4,
            "direction": ["out"] * 4,
            "type": ["card_payment"] * 4,
            "timestamp": pd.to_datetime(
                ["2025-09-01", "2025-10-01", "2025-10-31", "2025-11-30"], utc=True
            ),
            "amount": [10.0] * 4,
        }
    )
    mapper = SimpleNamespace(mapping_=pd.Series({"service payment": "unknown"}))
    mapper.transform = lambda frame: frame.assign(family=frame.description.map(mapper.mapping_))
    combined = pd.concat(
        [
            history,
            history.assign(currency="eur", amount=20),
            history.assign(direction="in", type="refund", amount=1000),
        ]
    )
    streams = candidate_streams(combined, mapper)
    assert len(streams) == 2
    assert {s["amount_mean"] for s in streams} == {10, 20}
    assert all(s["median_gap_days"] == 30 and s["count"] == 4 for s in streams)
    assert all(s["associated_family"] == "unknown" for s in streams)
    assert all(s["strong_recurrence"] for s in streams)
    assert not {"due", "balance", "future_amount", "merchant_identity"}.intersection(streams[0])
    quality = data_quality(combined, mapper)
    assert quality["generic_share"] == quality["unknown_identity_share"] == 1
    assert data_quality(history.assign(type="transfer"), mapper)["generic_share"] is None
