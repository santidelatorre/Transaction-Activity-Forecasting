"""Reproduce the small, fixed V2.1 ablation set without changing frozen V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.models import RecurrenceHeuristic
from transaction_forecasting.ubs.provenance import provenance
from transaction_forecasting.ubs.v2 import IntegratedV2Model
from transaction_forecasting.ubs.v21 import (
    CrossFittedFamilyV21Model,
    MerchantHistoryV21Model,
)

OUT = Path("outputs/metrics/v21_ablation")


def _concise(metrics):
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "per_class_f1": {label: metrics["per_class"][label]["f1-score"] for label in LABELS},
        "confusion_matrix": metrics["confusion_matrix"],
        "prediction_distribution": metrics["prediction_distribution"],
    }


def _v2_arbitration_probabilities(model, transactions):
    components = model.predict_components(transactions)
    history = components["history"]
    base = model.mapping_.transform(transactions)
    raw = pd.DataFrame(
        RecurrenceHeuristic(-1.0, 1.0).predict_proba(base),
        index=base.index,
        columns=LABELS,
    )
    ordered = np.sort(history.to_numpy(), axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    confidence_weight = np.clip(0.50 - margin, 0.20, 0.45)[:, None]
    return {
        "v2": components["blend"],
        "raw_heuristic_25": 0.75 * history + 0.25 * raw,
        "confidence_raw": (1.0 - confidence_weight) * history + confidence_weight * raw,
    }


def _family_probabilities(model, transactions):
    return model.predict_components(transactions)


def _merchant_probabilities(model, transactions):
    return {"merchant_history": model.predict_proba(transactions)}


def _evaluate_oof(data, model_factory, component_names, probability_getter):
    target = data.train_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    predictions = {name: pd.Series(index=target.index, dtype=object) for name in component_names}
    folds = []
    split = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (fit_positions, held_positions) in enumerate(split.split(target, target), 1):
        fit_ids = target.index[fit_positions]
        held_ids = target.index[held_positions]
        fit_transactions = data.train_transactions.loc[
            data.train_transactions["client_id"].isin(fit_ids)
        ]
        held_transactions = data.train_transactions.loc[
            data.train_transactions["client_id"].isin(held_ids)
        ]
        fit_labels = data.train_labels.loc[data.train_labels["client_id"].isin(fit_ids)]
        model = model_factory().fit(fit_transactions, fit_labels)
        probabilities = probability_getter(model, held_transactions)
        row = {"fold": fold}
        for name in component_names:
            prediction = probabilities[name].idxmax(axis=1)
            predictions[name].loc[held_ids] = prediction.reindex(held_ids)
            row[name] = _concise(evaluate_predictions(target.loc[held_ids], prediction))
        folds.append(row)
        print(
            json.dumps(
                {
                    "fold": fold,
                    **{name: row[name]["macro_f1"] for name in component_names},
                }
            ),
            flush=True,
        )
    if any(prediction.isna().any() for prediction in predictions.values()):
        raise RuntimeError("Incomplete OOF prediction coverage")
    pooled = {
        name: _concise(evaluate_predictions(target, prediction))
        for name, prediction in predictions.items()
    }
    return folds, pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=("arbitration", "crossfit_family", "merchant_history"),
    )
    args = parser.parse_args()
    started = perf_counter()
    before = provenance()
    data = load_ubs_data("data/raw/ubs_2026")
    if args.candidate == "arbitration":
        names = ("v2", "raw_heuristic_25", "confidence_raw")
        factory = IntegratedV2Model
        getter = _v2_arbitration_probabilities
    elif args.candidate == "crossfit_family":
        names = ("family_model", "blend")
        factory = CrossFittedFamilyV21Model
        getter = _family_probabilities
    else:
        names = ("merchant_history",)
        factory = MerchantHistoryV21Model
        getter = _merchant_probabilities
    folds, pooled = _evaluate_oof(data, factory, names, getter)
    model = factory().fit(data.train_transactions, data.train_labels)
    valid_probabilities = getter(model, data.valid_transactions)
    valid_target = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    official = {
        name: _concise(evaluate_predictions(valid_target, values.idxmax(axis=1)))
        for name, values in valid_probabilities.items()
    }
    after = provenance()
    if before != after:
        raise RuntimeError("Source or data changed during the ablation")
    result = {
        "candidate": args.candidate,
        "hypothesis": {
            "arbitration": "Recover V1-only recurrence wins without overriding confident V2 rows.",
            "crossfit_family": (
                "Expose exact V1 family evidence to CatBoost without own-label rows."
            ),
            "merchant_history": (
                "Normalize unstable descriptions and use leave-one-client-out evidence."
            ),
        }[args.candidate],
        "fold_protocol": "5 stratified client folds inside train; seed 42; all state rebuilt",
        "folds": folds,
        "pooled_oof": pooled,
        "official_validation": official,
        "seconds": perf_counter() - started,
        "provenance": before,
        "limitation": "Official validation was already reused by prior model selection.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.candidate}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "pooled_oof": pooled, "official": official}, indent=2))


if __name__ == "__main__":
    main()
