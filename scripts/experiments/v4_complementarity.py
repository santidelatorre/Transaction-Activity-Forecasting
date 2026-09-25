"""TRAIN-only probability alignment and error diversity; no fitting or selection."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels
from transaction_forecasting.ubs.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/metrics/v4_synthesis"
RANKER = ROOT / "outputs/metrics/v4_soft_candidate_santiago"


def checked(frame, target):
    if not frame.index.is_unique or set(frame.index) != set(target.index):
        raise ValueError("Client coverage/uniqueness mismatch")
    if list(frame.columns) != list(LABELS):
        raise ValueError("Class order mismatch")
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
        raise ValueError("Invalid probabilities")
    return frame.reindex(target.index)


def main():
    target = (
        read_labels(ROOT / "data/raw/ubs_2026/train_labels.csv")
        .set_index("client_id")[TARGET_COLUMN]
        .sort_index()
    )
    paths = {
        "baseline": RANKER / "train_oof_baseline_probabilities.csv",
        "santiago": OUT / "santiago_oof_probabilities.csv",
        "santiago_linear": RANKER / "train_oof_linear_probabilities.csv",
        "javi": OUT / "javi_clean_oof_probabilities.csv",
        "esteban": OUT / "esteban_oof_probabilities.csv",
        "laura_hazard_support": OUT / "survival/hazard_support_oof_probabilities.csv",
        "laura_hybrid_median_gap": OUT / "survival/hybrid_median_gap_oof_probabilities.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    frames = {
        name: checked(pd.read_csv(path, index_col="client_id"), target)
        for name, path in paths.items()
        if path.exists()
    }
    baseline_reference = pd.concat(
        [
            pd.read_csv(
                ROOT / f"outputs/metrics/ubs_v3a_baseline/fold_{i}_A_probabilities.csv",
                index_col="client_id",
            )
            for i in range(1, 6)
        ]
    ).reindex(target.index)
    np.testing.assert_allclose(frames["baseline"], baseline_reference, atol=1e-15, rtol=0)
    common = target.rename("true_label").to_frame()
    fold_ids = pd.Series(index=target.index, dtype=int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(target.index, target))
    for fold, (_, hold) in enumerate(folds, 1):
        fold_ids.iloc[hold] = fold
    common["fold"] = fold_ids.astype(int)
    metrics, pairs = {}, {}
    for name, frame in frames.items():
        common = common.join(frame.add_prefix(name + "__"))
        prediction = frame.idxmax(axis=1)
        metrics[name] = evaluate_predictions(target, prediction)
        metrics[name]["fold_macro_f1"] = [
            evaluate_predictions(target.iloc[hold], prediction.iloc[hold])["macro_f1"]
            for _, hold in folds
        ]
    for left, right in combinations(frames, 2):
        a, b = frames[left].idxmax(axis=1), frames[right].idxmax(axis=1)
        ea, eb = a.ne(target), b.ne(target)
        oracle = a.where(~ea | eb, b)
        per_class = {}
        for label in LABELS:
            mask = target.eq(label)
            per_class[label] = {
                "right_corrects_left": int((mask & ea & ~eb).sum()),
                "right_regresses_left": int((mask & ~ea & eb).sum()),
                "both_wrong": int((mask & ea & eb).sum()),
            }
        pairs[f"{left}__{right}"] = {
            "agreement": float(a.eq(b).mean()),
            "disagreement": float(a.ne(b).mean()),
            "error_correlation": float(np.corrcoef(ea.astype(float), eb.astype(float))[0, 1]),
            "probability_correlation": float(
                np.corrcoef(frames[left].to_numpy().ravel(), frames[right].to_numpy().ravel())[0, 1]
            ),
            "mean_jensen_shannon_distance": float(
                np.mean(jensenshannon(frames[left].to_numpy(), frames[right].to_numpy(), axis=1))
            ),
            "both_correct": int((~ea & ~eb).sum()),
            "both_wrong": int((ea & eb).sum()),
            "right_corrects_left": int((ea & ~eb).sum()),
            "right_regresses_left": int((~ea & eb).sum()),
            "per_class": per_class,
            "DIAGNOSTIC_oracle_correct_if_either": evaluate_predictions(target, oracle),
        }
    common.to_csv(OUT / "aligned_train_oof.csv", index_label="client_id")
    result = {
        "level": "VERIFIED rescoring of audited/reproduced OOF",
        "clients": len(target),
        "class_order": list(LABELS),
        "metrics": metrics,
        "pairs": pairs,
        "missing_local_probabilities": [*missing, "christian"],
        "notes": [
            "Oracle uses labels: diagnostic only; not deployable or an optimal Macro-F1 bound.",
            "SVD encoder is transductive; supervised heads exclude held clients.",
            "Ranker runner hash mismatch; model hashes match, one evidence fold regenerated.",
            "Javi stress is hash-seed dependent. No weight/threshold selected here.",
        ],
    }
    (OUT / "complementarity.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    for name, pair in pairs.items():
        if name.startswith("baseline__"):
            print(
                name,
                {
                    k: pair[k]
                    for k in (
                        "disagreement",
                        "error_correlation",
                        "right_corrects_left",
                        "right_regresses_left",
                    )
                },
            )


if __name__ == "__main__":
    main()
