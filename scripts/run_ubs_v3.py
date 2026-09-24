"""Run the V3 stream-aware ensemble and score validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transaction_forecasting.ubs.data import TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v2 import IntegratedV2Model
from transaction_forecasting.ubs.v3 import StreamV3Model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/ubs_2026")
    parser.add_argument("--output-dir", default="outputs/metrics/ubs_v3")
    parser.add_argument("--v2-blend", type=float, default=0.85)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = load_ubs_data(args.data_dir)
    v2 = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    v2_pred = v2.predict(data.valid_transactions)
    v2_metrics = evaluate_predictions(
        data.valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(v2_pred.index),
        v2_pred,
    )

    model = StreamV3Model(v2_blend=args.v2_blend).fit(data.train_transactions, data.train_labels)
    components = model.predict_components(data.valid_transactions)
    pred = components["blend"].idxmax(axis=1)
    target = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    metrics = evaluate_predictions(target.reindex(pred.index), pred)

    summary = {
        "model": "StreamV3Model",
        "v2_blend": args.v2_blend,
        "v2_macro_f1": v2_metrics["macro_f1"],
        "v3_macro_f1": metrics["macro_f1"],
        "v3_accuracy": metrics["accuracy"],
        "delta_vs_v2": metrics["macro_f1"] - v2_metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "prediction_distribution": pred.value_counts().to_dict(),
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    pred.rename("predicted_next_recurring_merchant").reset_index().to_csv(
        output / "validation_predictions.csv", index=False
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
