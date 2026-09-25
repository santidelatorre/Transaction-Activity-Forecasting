"""Materialize the locked, unmodified V3Model for the read-only product demo."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transaction_forecasting.product.evidence import select_cases
from transaction_forecasting.product.provenance import (
    DEFAULT_BUNDLE,
    ROOT,
    digest,
    runtime_versions,
    verify_source,
    write_json,
)
from transaction_forecasting.ubs.data import (
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    load_ubs_data,
    validate_submission,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v3.model import V3Model

DATA_FILES = (
    "train_transactions.jsonl",
    "train_labels.csv",
    "valid_transactions.jsonl",
    "valid_labels.csv",
    "test_transactions.jsonl",
    "sample_submission.csv",
)


def prepare(data_dir: Path, out: Path) -> dict:
    lock = verify_source()
    # A failed/interrupted run can never masquerade as a complete bundle.
    out.mkdir(parents=True, exist_ok=False)
    hashes = {name: digest(data_dir / name) for name in DATA_FILES}
    data = load_ubs_data(data_dir)
    if len(data.sample_submission) != 1000:
        raise ValueError("Expected exactly 1000 sample client IDs")
    print("Fitting frozen V3-A on TRAIN for measured VALID metrics", flush=True)
    validation_model = V3Model().fit(data.train_transactions, data.train_labels)
    valid_prediction = validation_model.predict(data.valid_transactions)
    metrics = evaluate_predictions(
        data.valid_labels.set_index("client_id")[TARGET_COLUMN], valid_prediction
    )
    write_json(
        out / "metrics.json",
        {
            "scope": "VALID",
            "fit_scope": "TRAIN",
            "validation_independent": False,
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "per_class": metrics["per_class"],
            "base_sha": lock["base_sha"],
            "note": (
                "VALID reused during team development; not an independent generalization estimate"
            ),
        },
    )
    print(f"VALID Macro-F1={metrics['macro_f1']:.12f}; fitting TRAIN+VALID", flush=True)
    model = V3Model().fit(
        pd.concat([data.train_transactions, data.valid_transactions], ignore_index=True),
        pd.concat([data.train_labels, data.valid_labels], ignore_index=True),
    )
    scores = model.predict_components(data.test_transactions)["A"]
    official = model.predict(data.test_transactions)
    pd.testing.assert_series_equal(official, scores.idxmax(axis=1))
    predictions = data.sample_submission[["client_id"]].copy()
    predictions[PREDICTION_COLUMN] = predictions.client_id.map(official)
    validate_submission(predictions, data.sample_submission, data.test_transactions)
    predictions.to_csv(out / "predictions.csv", index=False)
    scores.to_csv(out / "scores.csv", index_label="client_id")
    joblib.dump(model, out / "model.joblib", compress=3)
    # A round-trip check detects corrupt serialization and checks the actual predictor again.
    restored = joblib.load(out / "model.joblib")
    restored_scores = restored.predict_components(data.test_transactions)["A"]
    np.testing.assert_allclose(restored_scores, scores, rtol=0, atol=1e-12)
    cases = select_cases(scores, data.test_transactions, model.mapper_)
    write_json(out / "cases.json", cases)
    verify_source()
    if hashes != {name: digest(data_dir / name) for name in DATA_FILES}:
        raise RuntimeError("Data changed during preparation")
    manifest = {
        "model": lock,
        "fit_scope": "TRAIN+VALID",
        "prediction_scope": "TEST",
        "python": platform.python_version(),
        "versions": runtime_versions(),
        "product_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "data_hashes": hashes,
        "artifacts": {
            name: digest(out / name)
            for name in (
                "model.joblib",
                "predictions.csv",
                "scores.csv",
                "metrics.json",
                "cases.json",
            )
        },
        "runner_parity": "V3Model.predict == predict_components['A'].idxmax, all 1000 TEST clients",
    }
    write_json(out / "manifest.json", manifest)
    print(f"Ready: {out}; cases={cases}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/ubs_2026")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    prepare(args.data_dir, args.output_dir)
