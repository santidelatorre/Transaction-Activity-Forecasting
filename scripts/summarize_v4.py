"""Write aggregate V4 decision evidence; never copy client-level artifacts to Git."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels
from transaction_forecasting.ubs.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/metrics/v4_synthesis"
FINAL = ROOT / "outputs/metrics/ubs_v4_final"
RANKER = ROOT / "outputs/metrics/v4_soft_candidate_santiago"
REPORTS = ROOT / "reports"
BASE_SHA = "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def compact_evidence(name, value):
    """Keep pooled metrics and fold F1, without duplicating large run logs."""
    if value is None:
        return None
    if name == "ginestar":
        return {
            "scorecards": value["scorecards"],
            "fold_macro_f1": [
                {
                    "fold": r["fold"],
                    "model": r["model"],
                    "view": r["view"],
                    "macro_f1": r["metrics"]["macro_f1"],
                }
                for r in value["fold_results"]
            ],
            "runtime_seconds": value["runtime_seconds"],
            "baseline_reproduction": value["baseline_reproduction"],
            "external_shift": value["external_shift"],
            "provenance": value["provenance"],
        }
    if name == "laura":
        return {
            "metrics": value["metrics"],
            "fold_macro_f1": [
                {k: v["macro_f1"] for k, v in fold.items()} for fold in value["folds"]
            ],
            "snapshots": value["snapshots"],
            "clients": value["clients"],
            "cutoffs": value["cutoffs"],
            "valid_labels_read": False,
        }
    if name == "santiago":
        replay = value["replay"]
        return {
            "rescored": value["rescored"],
            "fingerprints": replay["fingerprints"],
            "runner_hash_mismatch": replay["runner_hash_mismatch"],
            "regenerated_fold_1_features_max_delta": replay[
                "regenerated_fold_1_features_max_delta"
            ],
            "folds": [
                {k: r[k] for k in ("fold", "macro_f1", "max_probability_delta")}
                for r in replay["folds"]
            ],
            "equal_average": replay["equal_average"],
        }
    return value


def rescore(path, target):
    frame = pd.read_csv(path, index_col="client_id")
    if list(frame.columns) != list(LABELS) or not frame.index.is_unique:
        raise ValueError("Unexpected probability schema")
    import numpy as np

    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
        raise ValueError("Invalid probability values")
    return evaluate_predictions(target, frame.idxmax(axis=1))


def main():
    inventory = read(OUT / "branch_inventory.json")
    target = read_labels(ROOT / "data/raw/ubs_2026/train_labels.csv").set_index("client_id")[
        TARGET_COLUMN
    ]
    valid_target = read_labels(ROOT / "data/raw/ubs_2026/valid_labels.csv").set_index("client_id")[
        TARGET_COLUMN
    ]
    baseline = rescore(RANKER / "train_oof_baseline_probabilities.csv", target)
    valid = read(FINAL / "valid_results.json")
    if valid:
        valid = valid["metrics"]
        assert abs(valid["macro_f1"] - 0.4241110977365116) < 1e-12
    ranker_scores = {
        "oof": {
            name: rescore(RANKER / f"train_oof_{name}_probabilities.csv", target)
            for name in ("baseline", "raw_family", "linear", "catboost")
        },
        "event_thinning": {
            name: rescore(RANKER / f"train_oof_corrupt_{name}_probabilities.csv", target)
            for name in ("baseline", "raw_family", "linear", "catboost")
        },
        "valid_reused": {
            name: rescore(RANKER / f"valid_{name}_probabilities.csv", valid_target)
            for name in ("baseline", "raw_family", "catboost")
        },
    }
    verdicts = {
        "ginestar": "STRONG",
        "javi": "NEGATIVE",
        "christian": "MIXED",
        "santiago": "MIXED",
        "laura": "NEGATIVE",
        "esteban": "NEGATIVE",
        "jaime": "PROMISING",
    }
    summaries = {
        "javi": read(REPORTS / "handoff/v4_direct_robust_javi_summary.json"),
        "christian": read(REPORTS / "handoff/v4_merchant_intelligence_christian_summary.json"),
        "santiago": read(REPORTS / "handoff/v4_soft_candidate_santiago_summary.json"),
    }
    reproduced = {
        "ginestar": read(OUT / "stress/summary.json"),
        "javi": read(OUT / "direct_reproduction.json"),
        "christian": None,
        "santiago": {"replay": read(OUT / "ranker_reproduction.json"), "rescored": ranker_scores},
        "laura": read(OUT / "survival_reproduction.json"),
        "esteban": read(OUT / "svd_reproduction.json"),
        "jaime": read(OUT / "product_parity.json"),
    }
    tests = read(OUT / "branch_tests.json")
    experiments = {
        name: {
            **record,
            "verdict": verdicts[name],
            "reported_aggregate": summaries.get(name)
            if name != "santiago"
            else {
                "source": "reports/handoff/v4_soft_candidate_santiago_summary.json",
                "note": "Use independently rescored aggregates below; original summary preserved",
            },
            "verified": compact_evidence(name, reproduced[name]),
            "specific_tests": tests.get(name),
            "peak_ram_bytes": None,
            "gpu": None,
        }
        for name, record in inventory.items()
    }
    submission = FINAL / "submission_v4.csv"
    prediction_counts = {
        "train_oof": baseline["prediction_distribution"],
        "valid": valid["prediction_distribution"] if valid else None,
        "test": None,
    }
    if submission.exists():
        from transaction_forecasting.ubs.data import (
            PREDICTION_COLUMN,
            read_transactions,
            validate_submission,
        )

        frame = pd.read_csv(submission, dtype=str)
        sample = pd.read_csv(ROOT / "data/raw/ubs_2026/sample_submission.csv", dtype=str)
        history = read_transactions(ROOT / "data/raw/ubs_2026/test_transactions.jsonl")
        validate_submission(frame, sample, history)
        assert (
            len(frame) == 1000
            and frame.client_id.nunique() == 1000
            and not frame.isna().any().any()
        )
        prediction_counts["test"] = (
            frame[PREDICTION_COLUMN].value_counts().reindex(LABELS, fill_value=0).to_dict()
        )
    limitations = [
        "Cached branch heads only; fetch blocked by read-only .git and remote network unavailable.",
        "VALID reused repeatedly; reported comparisons are not independent holdout evidence.",
        "TRAIN-only final fit; challenge policy did not clearly authorize VALID-label refit.",
        "No TEST labels, hidden-test score or measured financial benefit.",
        "Christian full experiment not rerun; local OOF/index absent.",
        "Santiago runner hash differs; model sources match, one evidence fold regenerated.",
        "Javi corruption depends on PYTHONHASHSEED; severe result differs from author report.",
        "Merchant/SVD encoder is transductive; SVD auxiliary gap probe splits events.",
        "Synthetic corruption is a simulation, one seed; no claim to match the UBS generator.",
        "No nested meta-stack justified; equal average rejected TRAIN-only.",
        "FastAPI/npm unavailable; HTTP/API frontend build and visual QA not verified here.",
        "Global formatter crashes on inaccessible temporaries; explicit source check passes.",
        "pre-commit hooks unavailable: stale cache then fresh GitHub fetch failure.",
        "Commit/push blocked; working tree remains on integration/v4-synthesis.",
    ]
    runtime = {
        name: read(OUT / f"{name}_runtime.json") for name in ("ranker", "direct", "svd", "survival")
    }
    runtime.update(
        final_valid=read(FINAL / "valid_runtime.json"),
        final_submission=read(FINAL / "submission_runtime.json"),
        stress_seconds=reproduced["ginestar"]["runtime_seconds"]
        if reproduced["ginestar"]
        else None,
    )
    decision = {
        "baseline_sha": BASE_SHA,
        "integration_branch": "integration/v4-synthesis",
        "branch_heads": {name: record["head"] for name, record in inventory.items()},
        "remote_freshness_verified": False,
        "experiments": experiments,
        "final_model": "V3-A",
        "model_version": "v4-synthesis-v3a-train-only-1",
        "fit_scope": "TRAIN",
        "selection_reason": (
            "Retain frozen baseline: no convincing transferable V4 gain; equal average "
            "underperforms ranker on TRAIN clean/stress. Previously reused VALID informs "
            "promotion rejection, never parameter fitting."
        ),
        "rejected_alternatives": [
            "direct robust NB",
            "merchant B/C",
            "soft ranker",
            "equal average",
            "survival and hybrids",
            "SVD/denoising",
        ],
        "train_oof_macro_f1": baseline["macro_f1"],
        "valid_macro_f1": valid["macro_f1"] if valid else None,
        "delta_vs_baseline": {"train_oof": 0.0, "valid": 0.0 if valid else None},
        "per_class_f1": {
            "train_oof": {k: v["f1-score"] for k, v in baseline["per_class"].items()},
            "valid": {k: v["f1-score"] for k, v in valid["per_class"].items()} if valid else None,
        },
        "prediction_counts": prediction_counts,
        "metrics": {"train_oof": baseline, "valid": valid},
        "complementarity": read(OUT / "complementarity.json"),
        "runtime": runtime,
        "tests": read(OUT / "quality_checks.json"),
        "submission_path": str(submission.relative_to(ROOT)) if submission.exists() else None,
        "submission_sha256": digest(submission),
        "submission_uploaded": False,
        "model_sha256": digest(FINAL / "model.joblib"),
        "source_and_training_provenance": read(FINAL / "frozen_recipe.json"),
        "commit_sha": None,
        "push_completed": False,
        "limitations": limitations,
    }
    (REPORTS / "v4_final_decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    appendix = [
        "## Verification appendix (generated from local evidence)",
        "",
        f"Final VALID: **{decision['valid_macro_f1']}**; TRAIN OOF: **{baseline['macro_f1']}**.",
        f"Submission: `{decision['submission_path']}`; SHA-256 `{decision['submission_sha256']}`.",
        "Validated: 1,000 rows/unique IDs, exact schema/order/classes and no NaNs."
        if submission.exists()
        else "Submission not yet available.",
        "",
    ]
    if reproduced["ginestar"]:
        cards = reproduced["ginestar"]["scorecards"]["A"]
        appendix += [
            "**VERIFIED complete common stress reproduction:**",
            "",
            "| View | Macro-F1 | Drop vs clean |",
            "|---|---:|---:|",
        ]
        appendix += [
            f"| {name} | {cards[name]['macro_f1']:.12f} | {cards[name]['drop_absolute']:.12f} |"
            for name in ("clean", "mild", "medium", "severe")
        ]
        appendix += [
            "",
            f"Runtime: {runtime['stress_seconds']:.1f} s, CPU; no peak RAM measurement.",
        ]
    else:
        appendix += ["Common stress reproduction is incomplete; section 5 remains REPORTED."]
    appendix += [
        "",
        "Quality checks and product parity:",
        "",
        "```json",
        json.dumps({"tests": decision["tests"], "product": reproduced["jaime"]}, indent=2),
        "```",
        "",
        "Final commit SHA: **null**; push: **not completed**. "
        "Read-only `.git` and unavailable remote prevent publication.",
    ]
    report_path = REPORTS / "v4_synthesis.md"
    report = report_path.read_text(encoding="utf-8").split("<!-- VERIFIED_APPENDIX -->")[0]
    report_path.write_text(
        report + "<!-- VERIFIED_APPENDIX -->\n\n" + "\n".join(appendix) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: decision[k]
                for k in (
                    "final_model",
                    "train_oof_macro_f1",
                    "valid_macro_f1",
                    "submission_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
