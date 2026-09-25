"""Generate a submission directly from the frozen model; no agent is imported."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from transaction_forecasting.product.provenance import (
    DEFAULT_BUNDLE,
    ROOT,
    digest,
    verify_bundle,
    write_json,
)
from transaction_forecasting.ubs.data import (
    PREDICTION_COLUMN,
    read_transactions,
    validate_submission,
)


def build_submission(bundle: Path, data_dir: Path, output: Path) -> dict:
    manifest = verify_bundle(bundle, data_dir)
    model = joblib.load(bundle / "model.joblib")
    history = read_transactions(data_dir / "test_transactions.jsonl")
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype=str)
    if len(sample) != 1000 or sample.client_id.nunique() != 1000:
        raise ValueError("Exactly 1000 unique sample IDs are required")
    predictions = model.predict(history)
    result = sample[["client_id"]].copy()
    result[PREDICTION_COLUMN] = result.client_id.map(predictions)
    validate_submission(result, sample, history)
    pd.testing.assert_frame_equal(result, pd.read_csv(bundle / "predictions.csv", dtype=str))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep old submissions intact; a different output requires a different name.
    with output.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    validate_submission(pd.read_csv(output, dtype=str), sample, history)
    metadata = {
        "model_version": manifest["model"]["model_version"],
        "base_sha": manifest["model"]["base_sha"],
        "model_artifact_sha256": manifest["artifacts"]["model.joblib"],
        "submission_sha256": digest(output),
        "rows": 1000,
        "columns": result.columns.tolist(),
        "agent_used": False,
        "fit_scope": manifest["fit_scope"],
        "runner": manifest["model"]["runner"],
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/ubs_2026")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/predictions/submission_v3a_product.csv"
    )
    args = parser.parse_args()
    print(build_submission(args.bundle, args.data_dir, args.output))
