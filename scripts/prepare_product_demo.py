"""Build Jaime's read-only demo from the final V4 TRAIN-only runner artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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
from transaction_forecasting.ubs.data import read_transactions, validate_submission
from transaction_forecasting.ubs.v4 import RECIPE, predict_probabilities


def prepare(data_dir: Path, out: Path, runner_dir: Path) -> dict:
    lock = verify_source()
    for phase, required in (("valid", "valid_results.json"), ("submission", "submission_v4.csv")):
        if not (runner_dir / required).exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_ubs_v4.py"),
                    "--phase",
                    phase,
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(runner_dir),
                ],
                cwd=ROOT,
                check=True,
            )
    provenance = json.loads((runner_dir / "frozen_recipe.json").read_text())
    metadata = json.loads((runner_dir / "model_metadata.json").read_text())
    if provenance["recipe"] != RECIPE or metadata["provenance"] != provenance:
        raise ValueError("Runner is not the final TRAIN-only recipe")
    for name, expected in provenance["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Runner source changed: {name}")
    for name, expected in provenance["train_sha256"].items():
        if digest(data_dir / name) != expected:
            raise ValueError(f"TRAIN data changed: {name}")
    if digest(runner_dir / "model.joblib") != metadata["model_sha256"]:
        raise ValueError("Runner model hash mismatch")
    out.mkdir(parents=True, exist_ok=False)
    model = joblib.load(runner_dir / "model.joblib")
    history = read_transactions(data_dir / "test_transactions.jsonl")
    scores = pd.read_csv(runner_dir / "test_probabilities.csv", index_col="client_id")
    np.testing.assert_allclose(
        predict_probabilities(model, history).reindex(scores.index), scores, atol=1e-12, rtol=0
    )
    predictions = pd.read_csv(runner_dir / "submission_v4.csv", dtype=str)
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype=str)
    if len(sample) != 1000:
        raise ValueError("Expected 1000 clients")
    validate_submission(predictions, sample, history)
    shutil.copyfile(runner_dir / "model.joblib", out / "model.joblib")
    shutil.copyfile(runner_dir / "submission_v4.csv", out / "predictions.csv")
    scores.to_csv(out / "scores.csv", index_label="client_id")
    metrics = json.loads((runner_dir / "valid_results.json").read_text())["metrics"]
    write_json(
        out / "metrics.json",
        {
            **metrics,
            "scope": "VALID",
            "fit_scope": "TRAIN",
            "validation_independent": False,
            "base_sha": lock["base_sha"],
            "note": "Reused VALID; secondary evidence only",
        },
    )
    write_json(out / "cases.json", select_cases(scores, history, model.mapper_))
    manifest = {
        "model": lock,
        "fit_scope": "TRAIN",
        "prediction_scope": "TEST",
        "versions": runtime_versions(),
        "runner_parity": "V4 runner == serialized V3-A scores; all 1000 TEST clients",
        "data_hashes": {
            name: digest(data_dir / name)
            for name in (
                "train_transactions.jsonl",
                "train_labels.csv",
                "test_transactions.jsonl",
                "sample_submission.csv",
                "valid_transactions.jsonl",
                "valid_labels.csv",
            )
        },
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
    }
    verify_source()
    write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/ubs_2026")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--runner-dir", type=Path, default=ROOT / "outputs/metrics/ubs_v4_final")
    args = parser.parse_args()
    print(prepare(args.data_dir, args.output_dir, args.runner_dir))
