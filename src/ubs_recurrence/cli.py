"""Small train/evaluate/submit interface; all final feature computation starts from raw data."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .augmentation import corrupt_transactions
from .data import (
    LABELS,
    PREDICTION,
    ROOT,
    aligned_target,
    transactions,
    validate_submission,
)
from .evaluation import metrics, record, source_fingerprint
from .model import FamilyForecaster, build_features
from .price_prior import learn_price_profiles


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    df = transactions("train", use_cache=False)
    profiles = learn_price_profiles(use_cache=False)
    views = []
    legacy_views = []
    for scenario in ["original", "valid_like", "test_like"]:
        d = df if scenario == "original" else corrupt_transactions(df, scenario, 2026)
        x, _, legacy = build_features(
            d, profiles, min_count=args.min_count, return_legacy=True
        )
        views.append(x)
        legacy_views.append(legacy)
        print("Built raw-data training view", scenario, x.shape, flush=True)
    ids = views[0].index.get_level_values(0).unique().to_numpy()
    y = aligned_target(ids)
    model = FamilyForecaster(
        tuple(args.seeds), args.min_count, args.legacy_weight, args.device
    ).fit(views, y, legacy_views)
    artifact = {
        "model": model,
        "profiles": profiles,
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "source_sha256": source_fingerprint(),
            "labels": LABELS,
            "training_split": "train only",
            "training_clients": len(ids),
            "seeds": args.seeds,
            "min_count": args.min_count,
            "legacy_weight": args.legacy_weight,
            "xgboost_device": args.device,
            "raw_manifest": json.loads(
                (ROOT / "reports/data_manifest.json").read_text()
            ),
            "seconds": time.perf_counter() - started,
        },
    }
    joblib.dump(artifact, out / "model.joblib")
    (out / "metadata.json").write_text(json.dumps(artifact["metadata"], indent=2))
    print("Model artifact", out / "model.joblib", flush=True)


def predict(args):
    started = time.perf_counter()
    artifact = joblib.load(args.model)
    model = artifact["model"]
    split = "test" if args.command == "submit" else "valid"
    df = transactions(split, use_cache=False)
    features, streams, legacy = build_features(
        df, artifact["profiles"], min_count=model.min_count, return_legacy=True
    )
    ids = features.index.get_level_values(0).unique().to_numpy()
    p = model.predict_proba(features, legacy)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    predictions = pd.DataFrame(p, columns=["p_" + l for l in LABELS])
    predictions.insert(0, "client_id", ids)
    predictions[PREDICTION] = np.array(LABELS)[p.argmax(axis=1)]
    predictions.to_csv(out / "probabilities.csv", index=False)
    streams.to_parquet(out / "evidence_streams.parquet", index=False)
    features.to_parquet(out / "evidence_features.parquet")
    if args.command == "submit":
        sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
        submission = (
            predictions.set_index("client_id")[[PREDICTION]]
            .reindex(sample.client_id)
            .reset_index()
        )
        validate_submission(submission, sample)
        submission.to_csv(out / "submission.csv", index=False)
        reread = pd.read_csv(out / "submission.csv")
        validate_submission(reread, sample)
        receipt = {
            "rows": len(reread),
            "exact_ids": True,
            "duplicates": int(reread.client_id.duplicated().sum()),
            "missing": int(reread.isna().sum().sum()),
            "labels": reread[PREDICTION].value_counts().to_dict(),
            "sha256": hashlib.sha256((out / "submission.csv").read_bytes()).hexdigest(),
        }
        (out / "submission_validation.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
    else:
        # Freeze predictions before accessing labels, even during reproduction.
        with (ROOT / "reports/holdout_access_log.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "batch": str(out),
                        "purpose": args.purpose,
                        "predictions_frozen_before_labels": True,
                        "prediction_sha256": hashlib.sha256(
                            (out / "probabilities.csv").read_bytes()
                        ).hexdigest(),
                    }
                )
                + "\n"
            )
        y = aligned_target(ids, "valid", allow_holdout=True)
        result = metrics(y, p)
        result.update({"model_metadata": artifact["metadata"], "purpose": args.purpose})
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {"macro_f1": result["macro_f1"], "accuracy": result["accuracy"]},
                indent=2,
            )
        )
        record(
            "final_" + out.name,
            ids,
            y,
            p,
            runtime=time.perf_counter() - started,
            split="official_validation",
            metadata={
                "hypothesis": "Frozen recurrence/refund ensemble transfers to the official shifted distribution",
                "features": "Fresh raw-data compact and complementary family features",
                "model": "Frozen final family forecaster",
                "parameters": artifact["metadata"],
                "seed": artifact["metadata"]["seeds"],
                "protocol": args.purpose
                + "; predictions written before label read; official clients excluded from every model fit",
                "status": "evaluated",
                "conclusion": "Frozen final evaluation or exact reproduction; all classes and clients retained",
            },
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--output", required=True)
    t.add_argument("--seeds", type=int, nargs="+", default=[42, 17, 2026])
    t.add_argument("--min-count", type=int, choices=[2, 3], default=3)
    t.add_argument("--legacy-weight", type=float, default=0.25)
    t.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    e = sub.add_parser("evaluate")
    e.add_argument("--model", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--purpose", default="Frozen final model evaluation")
    s = sub.add_parser("submit")
    s.add_argument("--model", required=True)
    s.add_argument("--output", required=True)
    args = ap.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
