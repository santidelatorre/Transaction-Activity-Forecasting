"""Run StreamV3Push (triple stack) on official validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transaction_forecasting.ubs.data import TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v2 import IntegratedV2Model
from transaction_forecasting.ubs.v3_max import StreamV3MaxModel
from transaction_forecasting.ubs.v3_push import StreamV3PushModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/ubs_2026")
    parser.add_argument("--output-dir", default="outputs/metrics/ubs_v3_push")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = load_ubs_data(args.data_dir)
    target = data.valid_labels.set_index("client_id")[TARGET_COLUMN]

    v2 = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    v2_pred = v2.predict(data.valid_transactions)
    v2_metrics = evaluate_predictions(target.reindex(v2_pred.index), v2_pred)

    vmax = StreamV3MaxModel().fit(data.train_transactions, data.train_labels)
    vmax_pred = vmax.predict(data.valid_transactions)
    vmax_metrics = evaluate_predictions(target.reindex(vmax_pred.index), vmax_pred)

    model = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    pred = model.predict(data.valid_transactions)
    metrics = evaluate_predictions(target.reindex(pred.index), pred)

    summary = {
        "model": "StreamV3PushModel",
        "weights": {"vmax": 1 / 3, "identity_A": 1 / 3, "stream_v3": 1 / 3},
        "v2_macro_f1": v2_metrics["macro_f1"],
        "v2_accuracy": v2_metrics["accuracy"],
        "v3max_macro_f1": vmax_metrics["macro_f1"],
        "v3push_macro_f1": metrics["macro_f1"],
        "v3push_accuracy": metrics["accuracy"],
        "delta_vs_v2": metrics["macro_f1"] - v2_metrics["macro_f1"],
        "delta_vs_v3max": metrics["macro_f1"] - vmax_metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "prediction_distribution": pred.value_counts().to_dict(),
        "note": (
            "Equal probability stack of StreamV3Max, Astra-style IdentityV3-A, "
            "and StreamV3. Stack recipe selected on VALID in try/push branch."
        ),
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    pred.rename("predicted_next_recurring_merchant").reset_index().to_csv(
        output / "validation_predictions.csv", index=False
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
