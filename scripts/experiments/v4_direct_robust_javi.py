"""TRAIN-only direct-description ablation; one frozen VALID evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.direct_robust import (
    DirectFeatures,
    LocalCorruptor,
    client_histories,
    normalized_view_weights,
    ordered_probabilities,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions

# Frozen before any VALID read. Fold instability is the mean SD across the three views.
ROBUST_WEIGHTS = {"clean": 0.25, "medium": 0.35, "severe": 0.40}
STABILITY_PENALTY = 0.10
VIEWS = tuple(ROBUST_WEIGHTS)
REPRESENTATIONS = ("R0", "R1", "R2", "R3", "R4", "R5")


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_for(name: str):
    if name == "ComplementNB":
        return ComplementNB(alpha=1.0)
    if name == "LogisticRegression":
        return LogisticRegression(C=1.0, max_iter=250, tol=0.01)
    raise ValueError(name)


def fit_model(name, matrices, target, client_ids, mode):
    if mode == "clean":
        matrix, labels, weights = matrices["clean"], target, None
    else:
        matrix = sparse.vstack([matrices[view] for view in VIEWS], format="csr")
        labels = np.tile(target, len(VIEWS))
        ids = list(client_ids) * len(VIEWS)
        weights = normalized_view_weights(ids) if mode == "normalized" else None
    return model_for(name).fit(matrix, labels, sample_weight=weights)


def make_folds(target: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(target.index, target))


def score(target: pd.Series, probs: np.ndarray) -> dict:
    guesses = pd.Series(np.asarray(LABELS)[np.argmax(probs, axis=1)], index=target.index)
    report = evaluate_predictions(target, guesses)
    clipped = np.clip(probs, 1e-12, 1)
    report["mean_entropy"] = float(np.mean(-np.sum(clipped * np.log(clipped), axis=1)))
    report["mean_confidence"] = float(np.mean(np.max(probs, axis=1)))
    return report


def diagnostics(target, arrays, folds):
    reports = {view: score(target, arrays[view]) for view in VIEWS}
    fold_scores = {
        view: [score(target.iloc[hold], arrays[view][hold])["macro_f1"] for _, hold in folds]
        for view in VIEWS
    }
    stability = float(np.mean([np.std(values) for values in fold_scores.values()]))
    robust = sum(ROBUST_WEIGHTS[view] * reports[view]["macro_f1"] for view in VIEWS)
    return {
        "views": reports,
        "fold_macro_f1": fold_scores,
        "fold_instability": stability,
        "robust_score": float(robust - STABILITY_PENALTY * stability),
    }


def candidate_key(model, representation, mode):
    return f"{model}/{representation}/{mode}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v4_direct_robust_javi")
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tracemalloc.start()
    source = [Path(__file__), Path("src/transaction_forecasting/ubs/direct_robust.py")]
    files = source + [
        args.data_dir / name for name in ("train_transactions.jsonl", "train_labels.csv")
    ]
    fingerprints = {str(path): sha256(path) for path in files}
    target_frame = read_labels(args.data_dir / "train_labels.csv")
    target = target_frame.set_index("client_id")[TARGET_COLUMN].sort_index()
    transactions = read_transactions(args.data_dir / "train_transactions.jsonl")
    if set(target.index) != set(transactions.client_id):
        raise ValueError("TRAIN client mismatch")
    ids = target.index
    histories = client_histories(transactions, ids)
    folds = make_folds(target)
    if args.phase in {"oof", "all"}:
        if (out / "valid_results.json").exists():
            raise ValueError("VALID already evaluated; use a fresh output directory")
        oof = {}
        fold_details = []
        for fold_number, (fit_pos, hold_pos) in enumerate(folds, 1):
            print(f"fold {fold_number}/5", flush=True)
            fit_histories = [histories[position] for position in fit_pos]
            hold_histories = [histories[position] for position in hold_pos]
            corruptor = LocalCorruptor().fit(fit_histories)
            features = DirectFeatures().fit(fit_histories)
            train_views = {view: corruptor.transform(fit_histories, view) for view in VIEWS}
            hold_views = {view: corruptor.transform(hold_histories, view) for view in VIEWS}
            fold_details.append(
                {
                    "fold": fold_number,
                    "fit_clients": len(fit_pos),
                    "hold_clients": len(hold_pos),
                    "vocabulary_coverage": features.coverage(hold_histories),
                    "dimensions": {rep: len(features.names(rep)) for rep in REPRESENTATIONS},
                }
            )
            y_fit = target.iloc[fit_pos].to_numpy()
            for representation in REPRESENTATIONS:
                fit_matrices = {
                    view: features.transform(train_views[view], representation) for view in VIEWS
                }
                hold_matrices = {
                    view: features.transform(hold_views[view], representation) for view in VIEWS
                }
                for model_name in ("ComplementNB", "LogisticRegression"):
                    # Robust modes on R4/R5 are predeclared, avoiding an adaptive model grid.
                    modes = (
                        ("clean", "augmented", "normalized")
                        if representation in {"R4", "R5"}
                        else ("clean",)
                    )
                    for mode in modes:
                        key = candidate_key(model_name, representation, mode)
                        fitted = fit_model(model_name, fit_matrices, y_fit, ids[fit_pos], mode)
                        slots = oof.setdefault(
                            key, {view: np.full((len(ids), len(LABELS)), np.nan) for view in VIEWS}
                        )
                        for view in VIEWS:
                            slots[view][hold_pos] = ordered_probabilities(
                                fitted, hold_matrices[view]
                            )
        if any(np.isnan(values).any() for slots in oof.values() for values in slots.values()):
            raise ValueError("Incomplete OOF probabilities")
        results = {key: diagnostics(target, arrays, folds) for key, arrays in oof.items()}
        selected = max(sorted(results), key=lambda key: results[key]["robust_score"])
        model_name, representation, mode = selected.split("/")
        frame = pd.DataFrame(oof[selected]["clean"], index=ids, columns=LABELS)
        frame.to_csv(out / "oof_probabilities.csv", index_label="client_id")
        for view in ("medium", "severe"):
            pd.DataFrame(oof[selected][view], index=ids, columns=LABELS).to_csv(
                out / f"oof_{view}_probabilities.csv", index_label="client_id"
            )
        save_json(
            out / "oof_results.json",
            {
                "candidates": results,
                "selected": selected,
                "folds": fold_details,
                "selection_note": "TRAIN OOF selection is optimistic; VALID evaluated once",
            },
        )
        save_json(
            out / "frozen_selection.json",
            {
                "selected": selected,
                "model": model_name,
                "representation": representation,
                "mode": mode,
                "robust_weights": ROBUST_WEIGHTS,
                "stability_penalty": STABILITY_PENALTY,
                "fingerprints": fingerprints,
                "fold_seed": 42,
                "valid_labels_used": False,
            },
        )
        print(f"frozen {selected}: {results[selected]['robust_score']:.5f}", flush=True)
    if args.phase in {"valid", "all"}:
        if (out / "valid_results.json").exists() or (out / "valid_probabilities.csv").exists():
            raise ValueError("VALID already predicted/evaluated; refusing another attempt")
        frozen = json.loads((out / "frozen_selection.json").read_text(encoding="utf-8"))
        if frozen["fingerprints"] != fingerprints:
            raise ValueError("Frozen source or TRAIN data changed")
        valid_tx = read_transactions(args.data_dir / "valid_transactions.jsonl")
        valid_ids = pd.Index(sorted(valid_tx.client_id.unique()), name="client_id")
        if set(ids).intersection(valid_ids):
            raise ValueError("TRAIN/VALID overlap")
        valid_histories = client_histories(valid_tx, valid_ids)
        corruptor = LocalCorruptor().fit(histories)
        features = DirectFeatures().fit(histories)
        representation = frozen["representation"]
        train_matrices = {
            view: features.transform(corruptor.transform(histories, view), representation)
            for view in VIEWS
        }
        fitted = fit_model(frozen["model"], train_matrices, target.to_numpy(), ids, frozen["mode"])
        valid_matrix = features.transform(valid_histories, representation)
        probabilities = ordered_probabilities(fitted, valid_matrix)
        prediction = pd.DataFrame(probabilities, index=valid_ids, columns=LABELS)
        prediction.to_csv(out / "valid_probabilities.csv", index_label="client_id")
        prediction.idxmax(axis=1).to_csv(out / "valid_predictions.csv", index_label="client_id")
        # The persistence boundary precedes the first VALID-label read.
        valid_target = read_labels(args.data_dir / "valid_labels.csv")
        truth = valid_target.set_index("client_id")[TARGET_COLUMN].reindex(valid_ids)
        if truth.isna().any() or len(truth) != len(valid_ids):
            raise ValueError("VALID labels do not match predictions")
        report = score(truth, probabilities)
        names = features.names(representation)
        if frozen["model"] == "LogisticRegression":
            importance = fitted.coef_
        else:
            importance = fitted.feature_log_prob_ - fitted.feature_log_prob_.mean(axis=0)
        top_features = {}
        for row, label in enumerate(fitted.classes_):
            indices = np.argsort(importance[row])[-12:][::-1]
            top_features[label] = [
                {
                    "feature": names[index],
                    "weight": float(importance[row, index]),
                    "kind": "generic"
                    if names[index].startswith(("generic_", "specific_"))
                    or names[index].removeprefix("desc:") in features.generic_
                    else "specific",
                }
                for index in indices
            ]
        save_json(
            out / "valid_results.json",
            {
                "selected": frozen["selected"],
                "metrics": report,
                "vocabulary_coverage": features.coverage(valid_histories),
                "feature_dimensionality": valid_matrix.shape[1],
                "top_train_fitted_features": top_features,
                "seconds": time.perf_counter() - started,
                "tracemalloc_peak_mib": tracemalloc.get_traced_memory()[1] / 2**20,
                "python": platform.python_version(),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            },
        )
        print(f"VALID Macro-F1 {report['macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
