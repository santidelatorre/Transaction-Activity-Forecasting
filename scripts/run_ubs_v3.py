"""Reproduce fixed V3 ablations with nested client isolation and untouched V2.

Run phases in order: oof, valid, submission (or all). Outputs remain local.
VALID cannot select the submission candidate; its name is frozen by TRAIN OOF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import (
    LABELS,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    read_labels,
    read_transactions,
    validate_submission,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v3.family_text import TextFamilyModel, focused_correction
from transaction_forecasting.ubs.v3.model import V3Model


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(data_dir):
    source = [Path(__file__), *Path("src/transaction_forecasting/ubs").rglob("*.py")]
    files = [*source, *sorted(data_dir.glob("*"))]
    return {
        str(path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if path.is_file()
    }


def bootstrap_delta(target, control, candidate, repeats=2000):
    positions = {label: n for n, label in enumerate(LABELS)}
    truth = np.array([positions[v] for v in target])
    a = np.array([positions[v] for v in control.reindex(target.index)])
    b = np.array([positions[v] for v in candidate.reindex(target.index)])
    rng = np.random.default_rng(42)

    def score(actual, prediction):
        cm = np.bincount(actual * 8 + prediction, minlength=64).reshape(8, 8)
        denominator = cm.sum(axis=0) + cm.sum(axis=1)
        return np.divide(
            2 * cm.diagonal(), denominator, out=np.zeros(8), where=denominator > 0
        ).mean()

    deltas = []
    for _ in range(repeats):
        take = rng.integers(0, len(target), len(target))
        deltas.append(score(truth[take], b[take]) - score(truth[take], a[take]))
    return {
        "delta": float(score(truth, b) - score(truth, a)),
        "percentile_95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "resamples": repeats,
        "seed": 42,
        "selection_corrected": False,
    }


def predictions(model, text_model, transactions, directory, prefix):
    components = model.predict_components(transactions)
    frame = pd.DataFrame({name: value.idxmax(axis=1) for name, value in components.items()})
    scores = text_model.predict_scores(transactions)
    frame["focused"] = focused_correction(frame.V2, scores)
    frame["text_positive_only"] = scores.idxmax(axis=1)
    for name, values in components.items():
        values.to_csv(directory / f"{prefix}_{name}_probabilities.csv", index_label="client_id")
    scores.to_csv(directory / f"{prefix}_text_scores.csv", index_label="client_id")
    return frame


def metrics(target, frame):
    return {name: evaluate_predictions(target, frame[name]) for name in frame}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid", "submission", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics/ubs_v3"))
    args = parser.parse_args()
    out = args.output_dir
    if out.resolve() in {
        Path("outputs/metrics/ubs_v1").resolve(),
        Path("outputs/metrics/ubs_v2").resolve(),
    }:
        raise ValueError("V3 cannot overwrite baseline artifacts")
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    stamps = fingerprint(args.data_dir)
    provenance = {
        "fingerprints": stamps,
        "python": platform.python_version(),
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost")
        },
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": 42,
        "validation_independent": False,
    }
    train = read_transactions(args.data_dir / "train_transactions.jsonl")
    labels = read_labels(args.data_dir / "train_labels.csv")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    if args.phase in ("oof", "all"):
        previous = out / "oof_provenance.json"
        if previous.exists() and json.loads(previous.read_text())["fingerprints"] != stamps:
            raise ValueError("Source/data changed: choose a fresh output directory")
        write_json(previous, provenance)
        frames, fold_metrics = [], []
        splitter = StratifiedKFold(5, shuffle=True, random_state=42)
        for fold, (fit_pos, hold_pos) in enumerate(splitter.split(target.index, target), 1):
            path = out / f"fold_{fold}_predictions.csv"
            fit_ids, hold_ids = target.index[fit_pos], target.index[hold_pos]
            if path.exists():
                frame = pd.read_csv(path, index_col="client_id")
            else:
                print(f"Fitting outer fold {fold}/5", flush=True)
                fit = train.loc[train.client_id.isin(fit_ids)]
                hold = train.loc[train.client_id.isin(hold_ids)]
                fit_labels = labels.loc[labels.client_id.isin(fit_ids)]
                model = V3Model().fit(fit, fit_labels)
                text_model = TextFamilyModel().fit(fit, fit_labels)
                frame = predictions(model, text_model, hold, out, f"fold_{fold}")
                frame.to_csv(path, index_label="client_id")
            report = metrics(target.loc[hold_ids], frame)
            fold_metrics.append(report)
            frames.append(frame)
            print(
                json.dumps({"fold": fold, **{k: v["macro_f1"] for k, v in report.items()}}),
                flush=True,
            )
        oof = pd.concat(frames).reindex(target.index)
        if not oof.index.is_unique or oof.isna().any().any():
            raise ValueError("OOF predictions are not complete/disjoint")
        report = metrics(target, oof)
        candidates = ("A", "B", "full", "ensemble", "focused")
        selected = max(candidates, key=lambda name: report[name]["macro_f1"])
        write_json(
            out / "oof_results.json",
            {
                "metrics": report,
                "folds": fold_metrics,
                "selected_candidate": selected,
                "bootstrap_vs_v2": {
                    name: bootstrap_delta(target, oof.V2, oof[name]) for name in candidates
                },
            },
        )
        oof.to_csv(out / "oof_predictions.csv", index_label="client_id")
        write_json(
            out / "frozen_selection.json",
            {
                "candidate": selected,
                "criterion": "maximum fixed-eight-class TRAIN OOF Macro-F1",
                "fingerprints": stamps,
                "valid_labels_used_for_this_selection": False,
            },
        )
    frozen = json.loads((out / "frozen_selection.json").read_text())
    if frozen["fingerprints"] != stamps:
        raise ValueError("Source or data differs from frozen OOF selection")
    if args.phase in ("valid", "all"):
        valid = read_transactions(args.data_dir / "valid_transactions.jsonl")
        if set(train.client_id).intersection(valid.client_id):
            raise ValueError("TRAIN/VALID client overlap")
        model = V3Model().fit(train, labels)
        text_model = TextFamilyModel().fit(train, labels)
        frame = predictions(model, text_model, valid, out, "valid")
        # Predictions are persisted before reading VALID labels.
        frame.to_csv(out / "validation_predictions.csv", index_label="client_id")
        valid_labels = read_labels(args.data_dir / "valid_labels.csv")
        truth = valid_labels.set_index("client_id")[TARGET_COLUMN]
        report = metrics(truth, frame)
        write_json(
            out / "valid_results.json",
            {
                "metrics": report,
                "oof_selected_candidate": frozen["candidate"],
                "bootstrap_vs_v2": {
                    name: bootstrap_delta(truth, frame.V2, frame[name])
                    for name in ("A", "B", "full", "ensemble", "focused")
                },
            },
        )
        model.mapper_.audit_.to_csv(out / "train_family_map.csv")
        for arm, fitted in model.models_.items():
            fitted.feature_importance().to_csv(out / f"importance_{arm}.csv", index=False)
        print(json.dumps({"VALID": {k: v["macro_f1"] for k, v in report.items()}}), flush=True)
    if args.phase in ("submission", "all"):
        valid = read_transactions(args.data_dir / "valid_transactions.jsonl")
        valid_labels = read_labels(args.data_dir / "valid_labels.csv")
        test = read_transactions(args.data_dir / "test_transactions.jsonl")
        if set(test.client_id).intersection(set(train.client_id) | set(valid.client_id)):
            raise ValueError("TEST overlaps training clients")
        merged = pd.concat([train, valid], ignore_index=True)
        merged_labels = pd.concat([labels, valid_labels], ignore_index=True)
        model = V3Model().fit(merged, merged_labels)
        text_model = TextFamilyModel().fit(merged, merged_labels)
        frame = predictions(model, text_model, test, out, "test")
        sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype=str)
        submission = sample[["client_id"]].copy()
        submission[PREDICTION_COLUMN] = submission.client_id.map(frame[frozen["candidate"]])
        validate_submission(submission, sample, test)
        submission.to_csv(out / "submission_v3.csv", index=False)
        validate_submission(pd.read_csv(out / "submission_v3.csv", dtype=str), sample, test)
    if stamps != fingerprint(args.data_dir):
        raise RuntimeError("Source/data changed while running")
    write_json(
        out / f"{args.phase}_provenance.json", {**provenance, "seconds": perf_counter() - started}
    )


if __name__ == "__main__":
    main()
