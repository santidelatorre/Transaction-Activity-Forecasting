"""Frozen V4 synthesis: V3-A, TRAIN-only OOF / VALID / TEST submission.

No model/weight/threshold search and no TRAIN+VALID refit. Historical runners
remain unchanged. Use a fresh output directory whenever source or TRAIN changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import read_labels, read_transactions, validate_submission
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v4 import (
    RECIPE,
    assert_disjoint,
    fit_final_model,
    predict_probabilities,
    submission_frame,
    target_series,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/metrics/ubs_v4_final"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(phase, data_dir, out):
    started = perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), *sorted((ROOT / "src/transaction_forecasting/ubs").rglob("*.py"))]
    sources.append(ROOT / "src/transaction_forecasting/evaluation/official.py")
    stamps = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    # Submission never opens VALID labels, even for hashing.
    train_hashes = {
        name: digest(data_dir / name) for name in ("train_transactions.jsonl", "train_labels.csv")
    }
    provenance = {
        "recipe": RECIPE,
        "source_sha256": stamps,
        "train_sha256": train_hashes,
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "catboost", "joblib")
        },
    }
    freeze_path = out / "frozen_recipe.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != provenance:
        raise ValueError("Frozen source/TRAIN/environment changed; choose a fresh output directory")
    write_json(freeze_path, provenance)
    train = read_transactions(data_dir / "train_transactions.jsonl")
    labels = read_labels(data_dir / "train_labels.csv")
    target = target_series(labels)
    if phase in ("oof", "all"):
        if (out / "oof_results.json").exists():
            raise ValueError("OOF already recorded; choose a fresh output directory")
        blocks, folds = [], []
        for number, (fit, hold) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=42).split(target.index, target), 1
        ):
            fit_ids, hold_ids = target.index[fit], target.index[hold]
            model = fit_final_model(
                train.loc[train.client_id.isin(fit_ids)], labels.loc[labels.client_id.isin(fit_ids)]
            )
            p = predict_probabilities(model, train.loc[train.client_id.isin(hold_ids)])
            p.to_csv(out / f"fold_{number}_probabilities.csv", index_label="client_id")
            blocks.append(p)
            folds.append(evaluate_predictions(target.loc[hold_ids], p.idxmax(axis=1)))
            print("OOF fold", number, folds[-1]["macro_f1"], flush=True)
        probabilities = pd.concat(blocks).reindex(target.index)
        probabilities.to_csv(out / "oof_probabilities.csv", index_label="client_id")
        write_json(
            out / "oof_results.json",
            {"metrics": evaluate_predictions(target, probabilities.idxmax(axis=1)), "folds": folds},
        )
    if phase in ("valid", "submission", "all"):
        model_path, metadata_path = out / "model.joblib", out / "model_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata["provenance"] != provenance
                or digest(model_path) != metadata["model_sha256"]
            ):
                raise ValueError("Model provenance/hash mismatch")
            model = joblib.load(model_path)
        else:
            print("Fitting frozen V3-A on TRAIN only", flush=True)
            model = fit_final_model(train, labels)
            joblib.dump(model, model_path, compress=3)
            write_json(
                metadata_path, {"provenance": provenance, "model_sha256": digest(model_path)}
            )
        if phase in ("valid", "all"):
            if (out / "valid_results.json").exists():
                raise ValueError("VALID already recorded; no adaptive retries")
            valid = read_transactions(data_dir / "valid_transactions.jsonl")
            assert_disjoint(train, valid)
            p = predict_probabilities(model, valid)
            p.to_csv(out / "valid_probabilities.csv", index_label="client_id")
            # First VALID-label access, strictly after recipe freeze and prediction.
            truth = target_series(read_labels(data_dir / "valid_labels.csv"))
            result = evaluate_predictions(truth, p.idxmax(axis=1))
            write_json(
                out / "valid_results.json", {"metrics": result, "validation_independent": False}
            )
            print("VALID", result["macro_f1"], flush=True)
        if phase in ("submission", "all"):
            path = out / "submission_v4.csv"
            if path.exists():
                raise ValueError("Refusing to overwrite an existing submission")
            test = read_transactions(data_dir / "test_transactions.jsonl")
            assert_disjoint(train, test)
            sample = pd.read_csv(data_dir / "sample_submission.csv", dtype=str)
            p = predict_probabilities(model, test)
            p.to_csv(out / "test_probabilities.csv", index_label="client_id")
            submission = submission_frame(p, sample, test)
            submission.to_csv(path, index=False)
            validate_submission(pd.read_csv(path, dtype=str), sample, test)
            write_json(
                out / "submission_metadata.json",
                {
                    "recipe": RECIPE,
                    "submission_sha256": digest(path),
                    "rows": len(submission),
                    "model_sha256": digest(model_path),
                },
            )
    if stamps != {str(p.relative_to(ROOT)): digest(p) for p in sources}:
        raise RuntimeError("Source changed during run")
    if train_hashes != {name: digest(data_dir / name) for name in train_hashes}:
        raise RuntimeError("TRAIN changed during run")
    write_json(out / f"{phase}_runtime.json", {"seconds": perf_counter() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid", "submission", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/ubs_2026")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.phase, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
