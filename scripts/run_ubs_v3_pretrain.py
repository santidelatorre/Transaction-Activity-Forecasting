#!/usr/bin/env python
"""Evaluate StreamV3PushEmbed (Push × unlabeled pretrain embeds) on official VALID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v3_pretrain import StreamV3PushEmbedModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-weight", type=float, default=0.85)
    parser.add_argument("--out", type=Path, default=Path("outputs/metrics/v3_pretrain"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]

    print("fitting StreamV3PushEmbed...", flush=True)
    model = StreamV3PushEmbedModel(push_weight=args.push_weight).fit(
        data.train_transactions, data.train_labels
    )
    pred = model.predict(data.valid_transactions)
    metrics = evaluate_predictions(yv.reindex(pred.index), pred)
    # Baselines from prior freezes (avoid refitting Push/V2 twice).
    v2_mf1 = 0.3915494559105542
    push_mf1 = 0.44454709747059196

    summary = {
        "model": "StreamV3PushEmbedModel",
        "push_weight": args.push_weight,
        "v2_macro_f1": v2_mf1,
        "push_macro_f1": push_mf1,
        "push_embed_macro_f1": float(metrics["macro_f1"]),
        "push_embed_accuracy": float(metrics["accuracy"]),
        "delta_vs_v2": float(metrics["macro_f1"] - v2_mf1),
        "delta_vs_push": float(metrics["macro_f1"] - push_mf1),
        "per_class_f1": {k: float(v["f1-score"]) for k, v in metrics["per_class"].items()},
        "prediction_distribution": pred.value_counts()
        .reindex(LABELS, fill_value=0)
        .astype(int)
        .to_dict(),
        "seconds": perf_counter() - started,
        "note": (
            "Unlabeled pretrain char-TFIDF→SVD client embeds + LR, "
            "arithmetically mixed with StreamV3Push. Weight selected on VALID grid."
        ),
    }
    (args.out / "esteban_v3_pretrain_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
