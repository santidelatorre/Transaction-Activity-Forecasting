"""Evaluate StreamV3Plus and mapper/override grids on official VALID."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v3 import StreamV3Model
from transaction_forecasting.ubs.v3_plus import StreamV3PlusModel

OUT = Path("outputs/metrics/v3_plus")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}

    print("fitting StreamV3Plus...", flush=True)
    for thr in (None, 0.75, 0.85, 0.90):
        model = StreamV3PlusModel(v2_blend=0.65, text_override_threshold=thr).fit(
            data.train_transactions, data.train_labels
        )
        if thr is None:
            pred = model.predict_components(data.valid_transactions)["blend"].idxmax(axis=1)
            key = "v3plus_no_override"
        else:
            pred = model.predict(data.valid_transactions)
            key = f"v3plus_ms_{thr}"
        metrics = evaluate_predictions(yv.reindex(pred.index), pred)
        results[key] = {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "per_class": {k: v["f1-score"] for k, v in metrics["per_class"].items()},
            "pred_dist": pred.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
        }
        print(key, results[key]["macro_f1"], flush=True)

    # Blend grid on plus components without override
    model = StreamV3PlusModel(v2_blend=0.65, text_override_threshold=None).fit(
        data.train_transactions, data.train_labels
    )
    comps = model.predict_components(data.valid_transactions)
    for w in (0.45, 0.55, 0.65, 0.75, 0.85):
        blend = w * comps["v2"].to_numpy() + (1 - w) * comps["stream_model"].to_numpy()
        pred = pd.DataFrame(blend, index=comps["v2"].index, columns=LABELS).idxmax(axis=1)
        metrics = evaluate_predictions(yv.reindex(pred.index), pred)
        key = f"v3plus_blend_{w}"
        results[key] = {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "per_class": {k: v["f1-score"] for k, v in metrics["per_class"].items()},
        }
        print(key, results[key]["macro_f1"], flush=True)

    # Reproduce StreamV3 for control in same process
    v3 = StreamV3Model(v2_blend=0.65).fit(data.train_transactions, data.train_labels)
    pred = v3.predict(data.valid_transactions)
    metrics = evaluate_predictions(yv.reindex(pred.index), pred)
    results["stream_v3_065"] = {"macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]}
    print("stream_v3_065", metrics["macro_f1"], flush=True)

    ranking = sorted(
        ((k, v["macro_f1"]) for k, v in results.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    summary = {"seconds": perf_counter() - started, "top": ranking, "results": results}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"top": ranking}, indent=2))


if __name__ == "__main__":
    main()
