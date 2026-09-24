"""Audit frozen V2 with client folds and paired official-validation uncertainty.

Internal OOF scores are not subtracted from the official V1 score. Bootstrap
intervals are conditional on this reused validation split and do not remove
selection bias. The audit never tunes a parameter or reads hidden test labels.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, PREDICTION_COLUMN, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.models import RecurrenceHeuristic
from transaction_forecasting.ubs.provenance import ROOT, provenance
from transaction_forecasting.ubs.v2 import IntegratedV2Model

OUT = ROOT / "outputs/metrics/v2_integration"


def main():
    before = provenance()
    data = load_ubs_data("data/raw/ubs_2026")
    target = data.train_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    predictions = {
        name: pd.Series(index=target.index, dtype=object)
        for name in ("v1_fixed", "history", "blend")
    }
    folds = []
    for fold, (train, heldout) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(target, target)
    ):
        fit_ids, held_ids = target.index[train], target.index[heldout]
        train_tx = data.train_transactions.loc[data.train_transactions.client_id.isin(fit_ids)]
        held_tx = data.train_transactions.loc[data.train_transactions.client_id.isin(held_ids)]
        train_labels = data.train_labels.loc[data.train_labels.client_id.isin(fit_ids)]
        assert not set(fit_ids).intersection(held_ids)
        model = IntegratedV2Model().fit(train_tx, train_labels)
        components = model.predict_components(held_tx)
        base = model.mapping_.transform(held_tx)
        fold_predictions = {
            "v1_fixed": pd.Series(RecurrenceHeuristic(-1.0, 1.0).predict(base), index=base.index),
            **{name: values.idxmax(axis=1) for name, values in components.items()},
        }
        row = {"fold": fold + 1, "fit_clients": len(fit_ids), "heldout_clients": len(held_ids)}
        for name, values in fold_predictions.items():
            predictions[name].loc[held_ids] = values.reindex(held_ids)
            row[name] = evaluate_predictions(target.loc[held_ids], values)
        folds.append(row)
        print(
            {"fold": fold + 1, **{name: row[name]["macro_f1"] for name in predictions}}, flush=True
        )
    if any(values.isna().any() for values in predictions.values()):
        raise RuntimeError("Incomplete out-of-fold coverage")
    pooled = {name: evaluate_predictions(target, values) for name, values in predictions.items()}
    ledger = json.loads((OUT / "results.json").read_text())
    valid_target = data.valid_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    guesses = {}
    for name in (
        "baseline",
        "catboost_history_raw",
        "catboost_history_temporal_blend",
        "temporal_periodicity",
    ):
        row = next(item for item in ledger if item["change"] == name)
        path = OUT / f"{row['step']:02d}_{name}_valid.csv"
        guesses[name] = (
            pd.read_csv(path, dtype=str, keep_default_na=False)
            .set_index("client_id")[PREDICTION_COLUMN]
            .reindex(valid_target.index)
            .to_numpy()
        )
    rng = np.random.default_rng(42)
    strata = [np.flatnonzero(valid_target.to_numpy() == label) for label in LABELS]
    samples = {name: [] for name in guesses if name != "baseline"}
    incremental = []
    for _ in range(2000):
        indices = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        scores = {
            name: f1_score(
                valid_target.iloc[indices],
                values[indices],
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
            for name, values in guesses.items()
        }
        for name in samples:
            samples[name].append(scores[name] - scores["baseline"])
        incremental.append(
            scores["catboost_history_temporal_blend"] - scores["catboost_history_raw"]
        )
    if before != provenance():
        raise RuntimeError("Source or data changed during the audit")
    result = {
        "folds": folds,
        "pooled_oof": pooled,
        "provenance": before,
        "fold_protocol": (
            "5 stratified client folds inside train, seed 42; "
            "all fit state rebuilt per fold; V1 bias fixed -1"
        ),
        "bootstrap": {
            name: np.quantile(values, [0.025, 0.5, 0.975]).tolist()
            for name, values in samples.items()
        },
        "blend_minus_history_bootstrap": np.quantile(incremental, [0.025, 0.5, 0.975]).tolist(),
        "bootstrap_replicates": 2000,
        "limitation": (
            "Conditional on reused official validation; no unbiased generalization estimate."
        ),
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "pooled": {k: v["macro_f1"] for k, v in pooled.items()},
                "bootstrap": result["bootstrap"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
