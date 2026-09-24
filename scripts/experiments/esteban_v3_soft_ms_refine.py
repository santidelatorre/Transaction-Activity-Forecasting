"""Refine music/streaming soft-boost alpha on StreamV3 and freeze best recipe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_text import StreamTextFamilyModel
from transaction_forecasting.ubs.v3 import StreamV3Model

OUT = Path("outputs/metrics/v3_max")


def soft_ms_boost(base: pd.DataFrame, text_scores: pd.DataFrame, alpha: float) -> pd.Series:
    logits = np.log(np.clip(base.reindex(columns=LABELS).to_numpy(), 1e-12, 1.0))
    for family in ("music", "streaming"):
        logits[:, LABELS.index(family)] += (
            alpha * text_scores[family].reindex(base.index).fillna(0.0).to_numpy()
        )
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    proba = weights / weights.sum(axis=1, keepdims=True)
    return pd.Series(np.asarray(LABELS)[proba.argmax(1)], index=base.index), pd.DataFrame(
        proba, index=base.index, columns=LABELS
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_ubs_data("data/raw/ubs_2026")
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]

    v3 = StreamV3Model(v2_blend=0.65).fit(data.train_transactions, data.train_labels)
    v3_proba = v3.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    text = StreamTextFamilyModel().fit(data.train_transactions, data.train_labels)
    text_scores, text_details = text.predict_scores(data.valid_transactions)
    text_scores = text_scores.reindex(yv.index).fillna(0.0)
    text_details = text_details.reindex(yv.index)

    results = {}
    best = None
    for alpha in [round(x, 2) for x in np.linspace(0.05, 1.5, 30)]:
        pred, proba = soft_ms_boost(v3_proba, text_scores, alpha)
        metrics = evaluate_predictions(yv, pred)
        payload = {
            "alpha": alpha,
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "per_class": {k: v["f1-score"] for k, v in metrics["per_class"].items()},
            "pred_dist": pred.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
        }
        results[f"alpha_{alpha}"] = payload
        if best is None or payload["macro_f1"] > best["macro_f1"]:
            best = payload
            best_pred = pred
            best_proba = proba
        music_f1 = payload["per_class"]["music"]
        print(f"alpha={alpha:.2f}  mf1={payload['macro_f1']:.4f}  music={music_f1:.3f}", flush=True)

    # Also hard override grid combined with best soft
    for thr in (0.80, 0.85, 0.90, 0.95):
        pred = best_pred.copy()
        conf = text_details["confidence"].fillna(0.0)
        tpred = text_details["prediction"].fillna("none")
        eligible = conf.ge(thr) & tpred.isin(("music", "streaming")) & pred.ne("none")
        pred.loc[eligible] = tpred.loc[eligible]
        metrics = evaluate_predictions(yv, pred)
        key = f"best_soft_then_hard_{thr}"
        results[key] = {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "per_class": {k: v["f1-score"] for k, v in metrics["per_class"].items()},
            "changed": int((pred != best_pred).sum()),
        }
        print(key, results[key]["macro_f1"], "changed", results[key]["changed"], flush=True)

    summary = {
        "best_soft": best,
        "note": "VALID-reported alpha for try branch; not OOF-frozen.",
        "results": results,
    }
    (OUT / "soft_ms_refine.json").write_text(json.dumps(summary, indent=2, default=str))
    best_pred.rename("predicted_next_recurring_merchant").reset_index().to_csv(
        OUT / "best_validation_predictions.csv", index=False
    )
    best_proba.to_csv(OUT / "best_validation_probabilities.csv")
    print(json.dumps({"best_soft": best}, indent=2, default=str))


if __name__ == "__main__":
    main()
