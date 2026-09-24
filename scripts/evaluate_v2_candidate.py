"""Evaluate one frozen candidate on the official split, keeping an append-only ledger.

This is an integration experiment, not an independent holdout estimate. Validation
has already been used by the team. Never change V1 outputs or silently select a
new recipe here. Generated features, predictions and hashes stay under outputs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from time import perf_counter

import numpy as np
import pandas as pd

from transaction_forecasting.experiment_tracking import ExperimentLogger
from transaction_forecasting.ubs.data import LABELS, PREDICTION_COLUMN, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.features import ClientFeatureBuilder
from transaction_forecasting.ubs.models import RecurrenceHeuristic
from transaction_forecasting.ubs.provenance import ROOT, digest, provenance

OUT = ROOT / "outputs/metrics/v2_integration"
REFERENCE = 0.2710242658492452


def inputs():
    data = load_ubs_data(ROOT / "data/raw/ubs_2026")
    builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
    key = hashlib.sha256(
        (
            digest(ROOT / "src/transaction_forecasting/ubs/features.py")
            + json.dumps(provenance()["data"], sort_keys=True)
        ).encode()
    ).hexdigest()[:20]
    cache = OUT / "cache" / key
    cache.mkdir(parents=True, exist_ok=True)
    matrices = []
    for split in ("train", "valid"):
        path = cache / f"{split}.parquet"
        if path.exists():
            matrix = pd.read_parquet(path)
        else:
            matrix = builder.transform(getattr(data, f"{split}_transactions"))
            matrix.to_parquet(path)
        matrices.append(matrix)
    x_train, x_valid = matrices
    y_train = data.train_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_train.index)
    y_valid = data.valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_valid.index)
    return data, builder, x_train, x_valid, y_train, y_valid, cache


def temporal_matrix(base, transactions, builder, cache, split, blocks):
    from transaction_forecasting.ubs.temporal_features import (
        apply_temporal_blocks,
        temporal_streams,
    )

    source = digest(ROOT / "src/transaction_forecasting/ubs/temporal_features.py")[:16]
    path = cache / f"{split}_temporal_{source}.parquet"
    if path.exists():
        streams = pd.read_parquet(path)
    else:
        streams = temporal_streams(transactions)
        streams.to_parquet(path)
    return apply_temporal_blocks(base, streams, builder.description_lift_, blocks)


def record(name, source, commit, metrics, prediction, previous, decision, notes, elapsed, before):
    after = provenance()
    if before != after:
        raise RuntimeError("Inputs/code changed during evaluation; result not accepted")
    ledger_path = OUT / "results.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    step = len(ledger)
    prediction_path = OUT / f"{step:02d}_{name}_valid.csv"
    prediction.rename(PREDICTION_COLUMN).rename_axis("client_id").reset_index().to_csv(
        prediction_path, index=False
    )
    row = {
        "step": step,
        "source_branch": source,
        "source_commit": commit,
        "change": name,
        **metrics,
        "delta_previous": metrics["macro_f1"] - previous,
        "delta_v1": metrics["macro_f1"] - REFERENCE,
        "decision": decision,
        "notes": notes,
        "seconds": elapsed,
        "provenance": before,
        "prediction_sha256": digest(prediction_path),
    }
    logger = ExperimentLogger(db_path=OUT / "experiments.sqlite3", repo_root=ROOT)
    row["experiment_id"] = logger.log_experiment(
        description=name,
        model_name=name,
        model_version="integration-v2",
        features=[name],
        hyperparameters={"seed": 42, "provenance": before},
        metrics={
            **metrics,
            "f1_per_class": {k: v["f1-score"] for k, v in metrics["per_class"].items()},
        },
        result=decision,
        baseline_macro_f1=REFERENCE,
        compute_seconds=elapsed,
        notes=notes,
        risk_notes="Official validation reused for selection; not independent.",
    )
    ledger.append(row)
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    headers = [
        "step",
        "source_branch",
        "source_commit",
        "change",
        "macro_f1",
        "delta_previous",
        "delta_v1",
        "accuracy",
        "decision",
        "notes",
    ]
    lines = [
        "# V2 Integration Results",
        "",
        "Official eight-class validation; V1 reference "
        "0.2710242658492452. Rejected trials are never enabled by default. "
        "delta_previous compares with the named stable/control score passed for that trial.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for result in ledger:
        lines.append(
            "| "
            + " | ".join(
                f"{result[h]:.9f}" if isinstance(result[h], float) else str(result[h])
                for h in headers
            )
            + " |"
        )
    lines += [
        "",
        "Full per-class F1, confusion matrices, predictions and provenance: "
        "`outputs/metrics/v2_integration/results.json` (local, ignored).",
        "",
    ]
    (ROOT / "reports/v2_integration_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: row[k] for k in headers}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=(
            "baseline",
            "tracking",
            "quality_gate",
            "temporal_intervals",
            "temporal_periodicity",
            "temporal_activity",
            "temporal_horizon",
            "merchant_blend",
            "catboost_v1_raw",
            "catboost_v1_calibrated",
            "catboost_history_raw",
            "catboost_history_calibrated",
            "catboost_history_temporal_blend",
        ),
    )
    parser.add_argument("--previous", type=float, default=REFERENCE)
    parser.add_argument("--source", default="baseline")
    parser.add_argument("--commit", default="0199a8b")
    parser.add_argument("--notes", default="Frozen V1 none_bias=-1; temperature=1.")
    args = parser.parse_args()
    started = perf_counter()
    before = provenance()
    data, builder, x_train, x_valid, y_train, y_valid, cache = inputs()
    model = RecurrenceHeuristic(none_bias=-1.0, temperature=1.0)
    if args.candidate.startswith("temporal_"):
        blocks = tuple(args.candidate.removeprefix("temporal_").split("+"))
        x_valid = temporal_matrix(x_valid, data.valid_transactions, builder, cache, "valid", blocks)
    elif args.candidate == "merchant_blend":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        from transaction_forecasting.ubs.text_v2 import MerchantFeatureBuilder

        merchant = MerchantFeatureBuilder().fit(data.train_transactions, data.train_labels)
        train = x_train.join(merchant.transform(data.train_transactions, training=True))
        valid = x_valid.join(merchant.transform(data.valid_transactions))
        imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
        train_matrix = scaler.fit_transform(imputer.fit_transform(train))
        valid_matrix = scaler.transform(imputer.transform(valid))
        classifier = LogisticRegression(
            C=0.3,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            tol=1e-5,
        ).fit(train_matrix, y_train)
        raw = classifier.predict_proba(valid_matrix)
        probabilities = np.column_stack(
            [raw[:, list(classifier.classes_).index(label)] for label in LABELS]
        )
        blended = 0.75 * model.predict_proba(x_valid) + 0.25 * probabilities
        prediction = pd.Series(np.asarray(LABELS)[blended.argmax(axis=1)], index=x_valid.index)
    elif args.candidate.startswith("catboost_"):
        from transaction_forecasting.ubs.v2 import (
            HistoryFeatureBuilder,
            calibrate_probabilities,
            make_model,
        )

        safe = args.candidate.startswith("catboost_history_")
        heuristic_features = x_valid
        if safe:
            history = HistoryFeatureBuilder().fit(data.train_transactions)
            x_train = history.transform(data.train_transactions)
            x_valid = history.transform(data.valid_transactions)
        key = hashlib.sha256(
            json.dumps(
                {
                    "versions": before["versions"],
                    "v2": digest(ROOT / "src/transaction_forecasting/ubs/v2.py"),
                    "safe": safe,
                    "columns": x_train.columns.tolist(),
                    "recipe": "300/4/.05/balanced/42",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:20]
        path = cache / f"catboost_{key}.npy"
        if path.exists():
            probabilities = np.load(path, allow_pickle=False)
        else:
            probabilities = make_model().fit(x_train, y_train).predict_proba(x_valid)
            np.save(path, probabilities, allow_pickle=False)
        if args.candidate.endswith("_calibrated"):
            probabilities = calibrate_probabilities(probabilities)
        if args.candidate == "catboost_history_temporal_blend":
            temporal = temporal_matrix(
                heuristic_features,
                data.valid_transactions,
                builder,
                cache,
                "valid",
                ("periodicity",),
            )
            probabilities = 0.75 * probabilities + 0.25 * model.predict_proba(temporal)
        prediction = pd.Series(
            np.asarray(LABELS)[probabilities.argmax(axis=1)], index=x_valid.index
        )
    elif args.candidate not in ("baseline", "tracking", "quality_gate"):
        raise ValueError("Unknown candidate")
    if args.candidate != "merchant_blend" and not args.candidate.startswith("catboost_"):
        prediction = pd.Series(model.predict(x_valid), index=x_valid.index)
    metrics = evaluate_predictions(y_valid, prediction)
    if args.candidate in ("baseline", "tracking", "quality_gate"):
        if abs(metrics["macro_f1"] - REFERENCE) > 1e-12:
            raise RuntimeError("V1 reproduction failed")
        decision = "BASELINE" if args.candidate == "baseline" else "KEEP_TOOLING"
    else:
        decision = "KEEP_PROVISIONAL" if metrics["macro_f1"] > args.previous else "REJECT"
    if args.candidate.startswith("catboost_v1_"):
        decision = "REJECT_SELF_LABEL"
    record(
        args.candidate,
        args.source,
        args.commit,
        metrics,
        prediction,
        args.previous,
        decision,
        args.notes,
        perf_counter() - started,
        before,
    )


if __name__ == "__main__":
    main()
