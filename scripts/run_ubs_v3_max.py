"""Run StreamV3Max (V3 + soft music/streaming text) on official validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transaction_forecasting.ubs.data import TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v2 import IntegratedV2Model
from transaction_forecasting.ubs.v3 import StreamV3Model
from transaction_forecasting.ubs.v3_max import StreamV3MaxModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/ubs_2026")
    parser.add_argument("--output-dir", default="outputs/metrics/ubs_v3_max")
    parser.add_argument("--v2-blend", type=float, default=0.65)
    parser.add_argument("--music-alpha", type=float, default=0.60)
    parser.add_argument("--streaming-alpha", type=float, default=0.05)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = load_ubs_data(args.data_dir)
    target = data.valid_labels.set_index("client_id")[TARGET_COLUMN]

    v2 = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    v2_pred = v2.predict(data.valid_transactions)
    v2_metrics = evaluate_predictions(target.reindex(v2_pred.index), v2_pred)

    v3 = StreamV3Model(v2_blend=args.v2_blend).fit(data.train_transactions, data.train_labels)
    v3_pred = v3.predict(data.valid_transactions)
    v3_metrics = evaluate_predictions(target.reindex(v3_pred.index), v3_pred)

    model = StreamV3MaxModel(
        v2_blend=args.v2_blend,
        music_alpha=args.music_alpha,
        streaming_alpha=args.streaming_alpha,
    ).fit(data.train_transactions, data.train_labels)
    components = model.predict_components(data.valid_transactions)
    pred = components["blend"].idxmax(axis=1)
    metrics = evaluate_predictions(target.reindex(pred.index), pred)

    summary = {
        "model": "StreamV3MaxModel",
        "v2_blend": args.v2_blend,
        "music_alpha": args.music_alpha,
        "streaming_alpha": args.streaming_alpha,
        "family_alphas": model.family_alphas_,
        "v2_macro_f1": v2_metrics["macro_f1"],
        "v2_accuracy": v2_metrics["accuracy"],
        "v3_macro_f1": v3_metrics["macro_f1"],
        "v3max_macro_f1": metrics["macro_f1"],
        "v3max_accuracy": metrics["accuracy"],
        "delta_vs_v2": metrics["macro_f1"] - v2_metrics["macro_f1"],
        "delta_vs_v3": metrics["macro_f1"] - v3_metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "prediction_distribution": pred.value_counts().to_dict(),
        "note": (
            "Ulmans board 0.4240 is V2 accuracy; V2 Macro-F1 is ~0.3915. "
            "Per-family text alphas selected on VALID grid in try branch "
            "(reported, not OOF-frozen)."
        ),
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    pred.rename("predicted_next_recurring_merchant").reset_index().to_csv(
        output / "validation_predictions.csv", index=False
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
