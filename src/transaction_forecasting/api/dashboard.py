"""Read-only presentation of the promoted V3-A arm and its evaluation artifacts.

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
ARMS = {
    "V2": ("V2 baseline", "History and recurrence"),
    "history_control": ("History only", "Control: history features only"),
    "A": ("V3-A · family identity", "Family identity features"),
    "B": ("V3-B · recurrence", "Additional recurrence features"),
    "full": ("Full V3", "Identity and recurrence features"),
    "ensemble": ("V3 / V2 blend", "50/50 blend"),
    "focused": ("Targeted correction", "Family-level correction"),
}


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


def snapshot(root: Path) -> dict[str, Any]:
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
                "notes": "Historical comparison on reused VALID; shown in report order.",
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
            "validation_clients": _client_count(selected),
            "baseline_macro_f1": baseline,
            "delta_vs_baseline": (
                _score(selected["macro_f1"]) - baseline if baseline is not None else None
            ),
        },
        "importance": importance,
        "importance_scope": "CatBoost arm A, trained on TRAIN and evaluated on VALID",
        "importance_method": "CatBoost PredictionValuesChange",
        "experiments": rows,
        "comparison_group": group,
        "validation_independent": False,
        "evaluation_sha": provenance.get("commit"),
        "report_sha256": digest,
        "ensemble_weights": {"catboost": 0.75, "recurrence_heuristic": 0.25},
    }
