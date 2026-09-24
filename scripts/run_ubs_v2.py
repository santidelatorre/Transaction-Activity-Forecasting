"""Run the frozen V2 recipe independently of V1, then refit for a valid submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from transaction_forecasting.ubs.data import (
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    load_ubs_data,
    validate_submission,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.provenance import digest, provenance
from transaction_forecasting.ubs.v2 import IntegratedV2Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-directory", default="outputs/metrics/ubs_v2")
    parser.add_argument("--submission", default="outputs/predictions/submission_v2.csv")
    args = parser.parse_args()
    metrics_dir, submission_path = Path(args.metrics_directory), Path(args.submission)
    # V1 artifact preservation is enforced even when custom output paths are supplied.
    if metrics_dir.resolve() == Path("outputs/metrics/ubs_v1").resolve():
        raise ValueError("V2 cannot overwrite V1 metrics")
    if submission_path.resolve() == Path("outputs/predictions/submission_v1.csv").resolve():
        raise ValueError("V2 cannot overwrite V1 submission")
    started = perf_counter()
    before = provenance()
    data = load_ubs_data("data/raw/ubs_2026")
    model = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    components = model.predict_components(data.valid_transactions)
    target = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    prediction = components["blend"].idxmax(axis=1).rename(PREDICTION_COLUMN)
    metrics = evaluate_predictions(target, prediction)
    component_metrics = {
        name: evaluate_predictions(target, probabilities.idxmax(axis=1))
        for name, probabilities in components.items()
    }
    # Validation predictions above are frozen before any train+valid refit.
    transactions = pd.concat([data.train_transactions, data.valid_transactions], ignore_index=True)
    labels = pd.concat([data.train_labels, data.valid_labels], ignore_index=True)
    final_model = IntegratedV2Model().fit(transactions, labels)
    test_prediction = final_model.predict(data.test_transactions)
    submission = data.sample_submission[["client_id"]].copy()
    submission[PREDICTION_COLUMN] = submission.client_id.map(test_prediction)
    validate_submission(submission, data.sample_submission, data.test_transactions)
    if before != provenance():
        raise RuntimeError("Source or data changed during the run")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    validate_submission(
        pd.read_csv(submission_path, dtype=str, keep_default_na=False),
        data.sample_submission,
        data.test_transactions,
    )
    valid_path = metrics_dir / "validation_predictions.csv"
    prediction.rename_axis("client_id").reset_index().to_csv(valid_path, index=False)
    summary = {
        "recipe": "75% history CatBoost 300/depth4/.05/balanced/42 + 25% periodicity heuristic",
        "metrics": metrics,
        "component_metrics": component_metrics,
        "feature_count": len(model.feature_names_),
        "feature_names": model.feature_names_,
        "validation_clients": len(prediction),
        "submission_rows": len(submission),
        "final_refit_clients": len(labels),
        "final_refit_uses_train_and_valid": True,
        "submission_sha256": digest(submission_path),
        "validation_sha256": digest(valid_path),
        "provenance": before,
        "seconds": perf_counter() - started,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k
                not in (
                    "provenance",
                    "feature_names",
                    "component_metrics",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
