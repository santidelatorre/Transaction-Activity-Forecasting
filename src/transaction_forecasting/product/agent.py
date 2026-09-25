"""Bounded tool decisions. A reasoning backend can implement DecisionPolicy."""

from __future__ import annotations

from copy import deepcopy
from typing import Protocol

TOOLS = (
    "predict_client",
    "inspect_history",
    "inspect_data_quality",
    "inspect_candidate_streams",
    "inspect_recurrence",
    "compare_alternatives",
    "show_model_metadata",
    "show_global_metrics",
)


class DecisionPolicy(Protocol):
    """Return an available tool name or None to stop; no free-text evidence accepted.

    A future LLM backend gets only this state and allowlist, never file writers or
    a submission handle. The executor enforces bounds and generates conclusions.
    """

    name: str

    def next_tool(self, observations: dict, available: tuple[str, ...]) -> str | None: ...


class ConditionalPolicy:
    name = "deterministic conditional policy"

    def next_tool(self, observations: dict, available: tuple[str, ...]) -> str | None:
        if "predict_client" not in observations:
            return "predict_client"
        prediction = observations["predict_client"]
        evidence = prediction["evidence"]
        needs = []
        if evidence["data_quality"]["degraded"]:
            needs += ["inspect_data_quality", "inspect_recurrence"]
        if evidence["supporting_stream_count"] == 0:
            needs += ["inspect_history", "inspect_recurrence"]
        if prediction["margin"] < 0.15 or evidence["candidate_ambiguity"]:
            needs += ["inspect_candidate_streams", "compare_alternatives"]
        if prediction["component_disagreement"]:
            needs += ["compare_alternatives", "inspect_candidate_streams"]
        if "inspect_candidate_streams" in observations:
            if not any(s["strong_recurrence"] for s in observations["inspect_candidate_streams"]):
                needs += ["inspect_recurrence"]
        if prediction["score"] < 0.45:
            needs += ["inspect_history"]
        return next((tool for tool in needs if tool in available), None)


def execute_tool(adapter, client_id: str, tool: str):
    """Read-only allowlist shared by deterministic and future reasoning backends."""
    if tool == "predict_client":
        return adapter.predict_client(client_id)
    if tool == "inspect_history":
        return adapter.get_client_history(client_id)
    if tool == "inspect_data_quality":
        return adapter.get_data_quality(client_id)
    if tool == "inspect_candidate_streams":
        return adapter.get_candidate_streams(client_id)
    if tool == "inspect_recurrence":
        streams = adapter.get_candidate_streams(client_id)
        return {
            "streams": streams,
            "strong_streams": sum(s["strong_recurrence"] for s in streams),
            "scope": "Observed outbound card payments, grouped by description and currency",
        }
    if tool == "compare_alternatives":
        prediction = adapter.predict_client(client_id)
        streams = adapter.get_candidate_streams(client_id)
        families = [prediction["predicted_family"], prediction["top_alternatives"][0]["family"]]
        return {
            "alternatives": [
                {
                    "family": family,
                    "score": prediction["scores"][family],
                    "associated_streams": [s for s in streams if s["associated_family"] == family],
                }
                for family in families
            ],
            "component_choices": prediction["component_choices"],
            "warning": "Historical associations are not verified merchant identities",
        }
    if tool == "show_model_metadata":
        return adapter.get_model_metadata()
    if tool == "show_global_metrics":
        return adapter.get_global_metrics()
    raise ValueError("Unknown or forbidden tool")


def investigate(adapter, client_id: str, policy: DecisionPolicy | None = None, max_steps=8) -> dict:
    if not 1 <= max_steps <= len(TOOLS):
        raise ValueError("Tool budget must be between 1 and 8")
    policy = policy or ConditionalPolicy()
    observations, trace = {}, []
    # Prediction anchors the explanation; it is not a ground-truth future label.
    frozen_prediction = adapter.predict_client(client_id)
    stop_reason = "tool_budget_exhausted"
    for step in range(max_steps):
        available = tuple(tool for tool in TOOLS if tool not in observations)
        tool = (
            "predict_client" if step == 0 else policy.next_tool(deepcopy(observations), available)
        )
        if tool is None:
            stop_reason = (
                "sufficient_evidence" if not risk_flags(frozen_prediction) else "evidence_limit"
            )
            break
        if tool not in available:
            stop_reason = "invalid_or_repeated_tool"
            break
        result = execute_tool(adapter, client_id, tool)
        observations[tool] = result
        trace.append({"step": step + 1, "tool": tool, "result": deepcopy(result)})
    flags = risk_flags(frozen_prediction)
    uncertain = bool(flags) or stop_reason != "sufficient_evidence"
    if frozen_prediction["predicted_family"] == "none":
        conclusion = (
            "No recurring family is predicted within 90 days; "
            "review the history with the client if a commitment is expected."
        )
    elif uncertain:
        conclusion = "Review the observed streams and confirm the commitment with the client."
    else:
        conclusion = "A recurring commitment has historical support; confirm it with the client."
    return {
        "backend": policy.name,
        "client_id": client_id,
        "predicted_family": frozen_prediction["predicted_family"],
        "base_sha": frozen_prediction["base_sha"],
        "trace": trace,
        "steps": len(trace),
        "max_steps": max_steps,
        "stop_reason": stop_reason,
        "uncertainty_flags": flags,
        "conclusion": conclusion,
        "submission_modified": False,
        "scope": "Investigation does not change the frozen prediction or resolve missing evidence.",
    }


def risk_flags(prediction: dict) -> list[str]:
    evidence = prediction["evidence"]
    checks = {
        "small_prediction_margin": prediction["margin"] < 0.15,
        "low_model_score": prediction["score"] < 0.45,
        "identity_quality_degraded": evidence["data_quality"]["degraded"],
        "multiple_candidate_families": evidence["candidate_ambiguity"],
        "components_disagree": prediction["component_disagreement"],
        "missing_supporting_recurrence": evidence["supporting_stream_count"] == 0,
    }
    return [name for name, present in checks.items() if present]
