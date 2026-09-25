"""Read-only presentation of recorded V1–V4 evidence and the frozen V3-A arm.

No fitting, label loading or prediction rewriting occurs in this module.
Historical report order is explicitly different from experiment chronology.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

BASE_SHA = "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"
MODEL_VERSION = "V3-A"
EVIDENCE_PATH = Path(__file__).resolve().parents[3] / "reports/dashboard_evidence.json"
ARMS = {
    "V2": ("V2 baseline", "History and periodicity"),
    "history_control": ("History only", "History-feature control"),
    "A": ("V3-A · family identity", "Cross-fitted family identity features"),
    "B": ("V3-B · recurrence", "Additional periodicity features"),
    "full": ("V3 full", "Identity and periodicity"),
    "ensemble": ("V3 / V2 blend", "Fixed 50/50 blend"),
    "focused": ("Focused correction", "Fixed 0.85 threshold; no VALID search"),
}
VERSION_ARMS = {"A": "V3-A", "B": "V3-B", "full": "V3 full", "ensemble": "V3/V2 blend"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise ValueError("Metrics must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Metrics must be finite and between 0 and 1")
    return float(value)


def _client_count(metrics: dict[str, Any]) -> int:
    matrix = metrics.get("confusion_matrix", [])
    if len(matrix) != 8 or any(len(row) != 8 for row in matrix):
        raise ValueError("Expected the official eight-class confusion matrix")
    counts = [value for row in matrix for value in row]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts):
        raise ValueError("Invalid confusion-matrix counts")
    if sum(counts) == 0:
        raise ValueError("Validation cohort is empty")
    return sum(counts)


def _macro_recall(metrics: dict[str, Any]) -> float:
    """Compute class-balanced recall from the official true-by-predicted matrix."""
    _client_count(metrics)
    matrix = metrics["confusion_matrix"]
    recalls = [row[index] / sum(row) if sum(row) else 0.0 for index, row in enumerate(matrix)]
    return sum(recalls) / len(recalls)


def _evidence() -> dict[str, Any]:
    evidence = _json(EVIDENCE_PATH)
    if evidence["metric"] != "official fixed-eight-class macro_f1":
        raise ValueError("Unexpected dashboard evidence metric")
    if evidence["v4_decision"]["base_sha"] != BASE_SHA:
        raise ValueError("V4 decision does not reference the frozen baseline")
    rows = [*evidence["version_results"], *evidence["v4_experiments"]]
    mainline = evidence.get("mainline_v4", {})
    rows.extend(mainline.get("official_valid", []))
    rows.extend(mainline.get("ablations", []))
    for row in rows:
        for key in ("macro_f1", "train_oof", "valid"):
            if row.get(key) is not None:
                _score(row[key])
    return evidence


def _history(
    evidence: dict[str, Any],
    measured: dict[str, Any] | None,
    digest: str | None,
    provenance: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    history = [
        dict(row, kind="measured", is_subversion=False) for row in evidence["version_results"]
    ]
    # A matching V2 control, cohort size and input fingerprints establish the
    # shared official VALID protocol. Otherwise the V3 line is separated.
    v2_control = measured.get("V2") if measured else None
    fingerprints = (provenance or {}).get("fingerprints", {})
    comparable = (
        v2_control is not None
        and abs(_score(v2_control["macro_f1"]) - history[-1]["macro_f1"]) < 1e-6
        and _client_count(v2_control) == history[-1]["clients"]
        and all(
            fingerprints.get(path) == expected_sha
            for path, expected_sha in evidence["version_source"]["valid_fingerprints"].items()
        )
    )
    for arm, label in VERSION_ARMS.items():
        result = measured.get(arm) if measured else None
        history.append(
            {
                "id": f"v3-{arm.lower()}",
                "version": "V3",
                "label": label,
                "macro_f1": _score(result["macro_f1"]) if result else None,
                "scope": "VALID",
                "clients": _client_count(result) if result else None,
                "protocol": (
                    "official-eight-class-valid-1000"
                    if comparable and result and _client_count(result) == history[-1]["clients"]
                    else "v3-valid-unmatched-control"
                ),
                "source": f"outputs/metrics/ubs_v3/valid_results.json#{arm}",
                "report_sha256": digest,
                "evidence": "measured; reused VALID" if result else "artifact unavailable",
                "kind": "measured" if result else "unavailable",
                "is_subversion": True,
                "selected": arm == "A",
            }
        )
    history.append(dict(evidence["v4_decision"], kind="decision", is_subversion=False))
    return history


def snapshot(root: Path) -> dict[str, Any]:
    evidence = _evidence()
    directory = root / "outputs/metrics/ubs_v3"
    report_path = directory / "valid_results.json"
    empty = {
        "available": False,
        "model_version": MODEL_VERSION,
        "base_sha": BASE_SHA,
        "horizon_days": 90,
        "cutoff": "2026-01-01",
        "metrics": None,
        "importance": [],
        "experiments": [],
        "comparison_group": None,
        "scope": "validation",
        "version_history": _history(evidence, None, None, None),
        "v4_experiments": evidence["v4_experiments"],
        "v4_diagnostics": evidence["diagnostics"],
        "mainline_v4": evidence.get("mainline_v4", {}),
        "evidence_sources": {
            "versions": evidence["version_source"],
            "v4": evidence["v4_source"],
        },
    }
    if not report_path.is_file():
        return empty

    report = _json(report_path)
    measured = report["metrics"]
    selected = measured["A"]
    baseline = _score(measured["V2"]["macro_f1"]) if "V2" in measured else None
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    group = f"v3-valid-{digest[:16]}"
    provenance_path = directory / "valid_provenance.json"
    provenance = _json(provenance_path) if provenance_path.is_file() else {}
    rows = []
    # Preserve source report order; never manufacture timestamps or tuning events.
    for arm, metrics in measured.items():
        if arm not in ARMS:
            continue
        score = _score(metrics["macro_f1"])
        title, milestone = ARMS[arm]
        rows.append(
            {
                "experiment_id": f"{group}-{arm}",
                "sequence": len(rows) + 1,
                "candidate": arm,
                "model_name": title,
                "model_version": MODEL_VERSION if arm == "A" else arm,
                "recorded_at_utc": None,
                "git_commit": provenance.get("commit"),
                "result": "selected" if arm == "A" else "evaluated",
                "milestone": milestone,
                "metrics": {
                    "macro_f1": score,
                    "accuracy": _score(metrics["accuracy"]),
                    "validation_clients": _client_count(metrics),
                    "scope": "validation",
                    "run_id": group,
                },
                "delta_vs_baseline": score - baseline if baseline is not None else None,
                "source": "valid_results.json",
                "order_kind": "report",
                "notes": (
                    "Historical comparison on reused VALID; source-report order, not chronology."
                ),
            }
        )

    importance = []
    importance_path = directory / "importance_A.csv"
    if importance_path.is_file():
        frame = pd.read_csv(importance_path, usecols=["feature", "importance"])
        if frame["feature"].isna().any() or frame["feature"].duplicated().any():
            raise ValueError("Invalid feature names in importance_A.csv")
        values = pd.to_numeric(frame["importance"], errors="raise")
        if not values.map(math.isfinite).all() or values.lt(0).any():
            raise ValueError("Invalid measured feature importance")
        frame["importance"] = values
        importance = frame.sort_values("importance", ascending=False).head(12).to_dict("records")

    return {
        **empty,
        "available": True,
        "metrics": {
            "macro_f1": _score(selected["macro_f1"]),
            "accuracy": _score(selected["accuracy"]),
            "macro_recall": _macro_recall(selected),
            "validation_clients": _client_count(selected),
            "baseline_macro_f1": baseline,
            "delta_vs_baseline": (
                _score(selected["macro_f1"]) - baseline if baseline is not None else None
            ),
        },
        "importance": importance,
        "version_history": _history(evidence, measured, digest, provenance),
        "importance_scope": "CatBoost arm A, fitted on TRAIN for VALID evaluation",
        "importance_method": "CatBoost PredictionValuesChange",
        "experiments": rows,
        "comparison_group": group,
        "validation_independent": False,
        "evaluation_sha": provenance.get("commit"),
        "report_sha256": digest,
        "ensemble_weights": {"catboost": 0.75, "recurrence_heuristic": 0.25},
    }
