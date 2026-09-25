"""Rebuild the frozen imported model and compare it on the historical UBS VALID.

No search, early stopping, threshold fitting, or model selection is performed.
Historical source is read from pinned Git objects, without checking out a branch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ubs_recurrence.augmentation import corrupt_transactions
from ubs_recurrence.data import (
    CUTOFF,
    LABELS,
    PREDICTION,
    ROOT,
    TARGET,
    aligned_target,
    labels,
    transactions,
)
from ubs_recurrence.evaluation import metrics
from ubs_recurrence.model import FamilyForecaster, build_features
from ubs_recurrence.official import score_predictions, validate_submission
from ubs_recurrence.price_prior import learn_price_profiles

REFERENCE_COMMIT = "96409b7940a991fbda5235b85ffeb40652a4087b"
IMPORTED_COMMIT = "e4aa4c58175242e198cefd32d6ac4558145523af"
V1_COMMIT = "0199a8b7c8b2c790a9d3447b156f2088708c2864"
RECIPE = {"seeds": [42, 17, 2026], "min_count": 3, "legacy_weight": 0.25}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def source_hashes():
    paths = [
        *sorted((ROOT / "src/ubs_recurrence").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/validate_submission.py",
        ROOT / "pyproject.toml",
        ROOT / "requirements-lock.txt",
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def verify_data(data_dir):
    manifest = json.loads((ROOT / "reports/data_manifest.json").read_text())
    actual = {name: digest(data_dir / name) for name in manifest}
    if actual != {name: item["sha256"] for name, item in manifest.items()}:
        raise ValueError("Raw files differ from the official pinned dataset")
    return actual


def audit_histories(data_dir, out):
    audit, client_sets, histories = {}, {}, {}
    for split in ["train", "valid", "test", "unlabeled_pretrain"]:
        frame = transactions(split, use_cache=False, data_dir=data_dir)
        if frame.isna().any().any() or frame.duplicated().any():
            raise ValueError(f"Missing values or duplicate transactions: {split}")
        if not frame.timestamp.lt(CUTOFF).all():
            raise ValueError("Temporal leakage")
        ids = sorted(frame.client_id.unique())
        client_sets[split] = set(ids)
        hashes = []
        for _, group in frame.groupby("client_id", sort=True):
            values = pd.util.hash_pandas_object(
                group.drop(columns="client_id").sort_index(axis=1), index=False
            ).to_numpy()
            hashes.append(hashlib.sha256(values.tobytes()).hexdigest())
        if len(set(hashes)) != len(ids):
            raise ValueError(f"Duplicate complete client histories: {split}")
        histories[split] = set(hashes)
        (out / f"{split}_ids.txt").write_text("\n".join(ids) + "\n")
        audit[split] = {
            "clients": len(ids),
            "transactions": len(frame),
            "first_id": ids[0],
            "last_id": ids[-1],
            "ids_sha256": digest(out / f"{split}_ids.txt"),
            "timestamp_min": str(frame.timestamp.min()),
            "timestamp_max": str(frame.timestamp.max()),
            "nulls": 0,
            "duplicate_transactions": 0,
            "duplicate_histories": 0,
        }
    for i, left in enumerate(client_sets):
        for right in list(client_sets)[i + 1 :]:
            if (
                client_sets[left] & client_sets[right]
                or histories[left] & histories[right]
            ):
                raise ValueError(
                    f"Cross-split duplicate clients/histories: {left}, {right}"
                )
    expected_counts = {
        "train": 2000,
        "valid": 1000,
        "test": 1000,
        "unlabeled_pretrain": 10000,
    }
    if {k: len(v) for k, v in client_sets.items()} != expected_counts:
        raise ValueError("Unexpected split sizes")
    return audit


def fit_model(frame, y, profiles, device):
    views, legacy_views = [], []
    for scenario in ["original", "valid_like", "test_like"]:
        altered = (
            frame
            if scenario == "original"
            else corrupt_transactions(frame, scenario, 2026)
        )
        features, _, legacy = build_features(altered, profiles, return_legacy=True)
        views.append(features)
        legacy_views.append(legacy)
        print("Built", scenario, features.shape, flush=True)
    ids = views[0].index.get_level_values(0).unique()
    if not ids.equals(y.index):
        raise ValueError("Training labels are not aligned by client")
    model = FamilyForecaster(device=device).fit(views, y.to_numpy(), legacy_views)
    if set(model.training_ids_) != set(y.index):
        raise ValueError("Unexpected supervised training clients")
    return model


def predict(model, profiles, frame):
    features, _, legacy = build_features(frame, profiles, return_legacy=True)
    ids = features.index.get_level_values(0).unique()
    if set(ids) & set(model.training_ids_):
        raise ValueError("Scoring on fitted clients is prohibited")
    probabilities = model.predict_proba(features, legacy)
    result = pd.DataFrame(probabilities, columns=["p_" + label for label in LABELS])
    result.insert(0, "client_id", ids)
    result[PREDICTION] = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    return result


def historical_source(out):
    """Materialize immutable source files; never change a Git worktree or ref."""
    source = out / "reference_source"
    names = git("ls-tree", "-r", "--name-only", REFERENCE_COMMIT).splitlines()
    hashes = {}
    tests = {
        "tests/test_ubs_v1.py",
        "tests/test_ubs_v2.py",
        "tests/test_official.py",
        "tests/test_submission_validator.py",
        "tests/test_evaluation_compatibility.py",
    }
    for name in names:
        if not (
            name.startswith("src/transaction_forecasting/")
            or name in tests
            or name == "scripts/validate_submission.py"
        ):
            continue
        content = subprocess.check_output(
            ["git", "show", f"{REFERENCE_COMMIT}:{name}"], cwd=ROOT
        )
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError("Historical source was modified")
        destination.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    sys.path.insert(0, str(source / "src"))
    return hashes


def run_baselines(data_dir, out):
    provenance = historical_source(out)
    from transaction_forecasting.ubs.data import load_ubs_data
    from transaction_forecasting.ubs.features import ClientFeatureBuilder
    from transaction_forecasting.ubs.models import RecurrenceHeuristic
    from transaction_forecasting.ubs.v2 import IntegratedV2Model

    data = load_ubs_data(data_dir)
    builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
    features = builder.transform(data.valid_transactions)
    # Historical selected recipe, frozen here; do not repeat VALID tuning.
    v1 = RecurrenceHeuristic(none_bias=-1.0, temperature=1.0).predict_proba(features)
    first = pd.DataFrame(
        {"client_id": features.index, PREDICTION: np.asarray(LABELS)[v1.argmax(axis=1)]}
    )
    model = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    second = (
        model.predict(data.valid_transactions)
        .rename(PREDICTION)
        .rename_axis("client_id")
        .reset_index()
    )
    for name, frame in [("V1", first), ("V2", second)]:
        frame.to_csv(out / f"{name}_validation.csv", index=False)
    return {"V1": first, "V2": second}, provenance


def evaluate(labels_frame, predictions):
    # Reuse the historical evaluator, including exact coverage and ID alignment.
    result = score_predictions(labels_frame, predictions[["client_id", PREDICTION]])
    aligned = predictions.set_index("client_id").loc[labels_frame.client_id]
    encoding = {label: i for i, label in enumerate(LABELS)}
    y = labels_frame[TARGET].map(encoding).to_numpy()
    pred = aligned[PREDICTION].map(encoding).to_numpy()
    independent = metrics(y, np.eye(8)[pred])
    if not np.isclose(result["macro_f1"], independent["macro_f1"], rtol=0, atol=1e-15):
        raise ValueError("Historical and imported metric implementations disagree")
    result["true_distribution"] = {
        k: v["support"] for k, v in result["per_class"].items()
    }
    result["prediction_distribution"] = {
        k: v["predicted_count"] for k, v in result["per_class"].items()
    }
    return result, pred


def bootstrap(y, predictions, *, repetitions=2000, seed=20260925):
    """Paired ordinary client bootstrap; fixed eight classes, no refitting."""
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in predictions}
    for _ in range(repetitions):
        ix = rng.integers(0, len(y), len(y))
        for name, pred in predictions.items():
            cm = np.bincount(y[ix] * 8 + pred[ix], minlength=64).reshape(8, 8)
            denominator = cm.sum(axis=0) + cm.sum(axis=1)
            f1 = np.divide(
                2 * cm.diagonal(), denominator, out=np.zeros(8), where=denominator != 0
            )
            draws[name].append(float(f1.mean()))
    result = {
        name: np.quantile(values, [0.025, 0.975]).tolist()
        for name, values in draws.items()
    }
    result["delta_stream_vs_V2"] = np.quantile(
        np.asarray(draws["Stream Identity"]) - np.asarray(draws["V2"]), [0.025, 0.975]
    ).tolist()
    return {
        "method": "paired client percentile bootstrap, conditional on fixed fitted models",
        "repetitions": repetitions,
        "seed": seed,
        "confidence": 0.95,
        "intervals": result,
    }


def publish_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite existing artifact: {path}")
    else:
        path.write_bytes(content)


def verify_run(out, data_dir):
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt["data_sha256"] != verify_data(data_dir):
        raise ValueError("Data changed")
    if receipt["source_sha256"] != source_hashes():
        raise ValueError("Source changed; start a fresh run")
    for name, expected in receipt["artifact_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Artifact changed: {name}")
    target = labels("valid", allow_holdout=True, data_dir=data_dir).reset_index()
    summary = json.loads((out / "metrics.json").read_text())
    for name, filename in [
        ("V1", "V1_validation.csv"),
        ("V2", "V2_validation.csv"),
        ("Stream Identity", "replica_1/validation.csv"),
    ]:
        actual, _ = evaluate(target, pd.read_csv(out / filename))
        if actual != summary["models"][name]:
            raise ValueError("Recalculated metrics changed")
    print(
        "Verified input/source/artifact hashes and recalculated all three metrics.",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="stream_identity_20260925")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/ubs_2026")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "outputs/predictions/submission_stream_identity.csv",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name) or args.replicas < 2:
        raise ValueError(
            "Safe run name and at least two raw-data replicas are required"
        )
    out = ROOT / "outputs/metrics/stream_identity" / args.run_name
    if args.verify:
        summary = verify_run(out, args.data_dir)
        print(
            json.dumps(
                {
                    k: {m: v[m] for m in ["macro_f1", "accuracy"]}
                    for k, v in summary["models"].items()
                },
                indent=2,
            )
        )
        return
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    frozen = {
        "started_at": datetime.now(UTC).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "imported_commit": IMPORTED_COMMIT,
        "recipe": RECIPE,
        "device": args.device,
        "replicas": args.replicas,
        "source_sha256": source_hashes(),
        "data_sha256": verify_data(args.data_dir),
        "selection": "No changes or selection using this benchmark's VALID labels",
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "catboost",
                "lightgbm",
                "xgboost",
            ]
        },
        "python": sys.version,
    }
    write_json(out / "freeze.json", frozen)
    audit = audit_histories(args.data_dir, out)
    print("Pinned data and all split/temporal/duplicate checks passed.", flush=True)
    replicas, test_replicas = [], []
    for i in range(1, args.replicas + 1):
        print("Raw-data replica", i, flush=True)
        folder = out / f"replica_{i}"
        folder.mkdir()
        train = transactions("train", use_cache=False, data_dir=args.data_dir)
        ids = pd.Index(sorted(train.client_id.unique()), name="client_id")
        y = pd.Series(aligned_target(ids, data_dir=args.data_dir), index=ids)
        profiles = learn_price_profiles(use_cache=False, data_dir=args.data_dir)
        model = fit_model(train, y, profiles, args.device)
        valid = transactions("valid", use_cache=False, data_dir=args.data_dir)
        prediction = predict(model, profiles, valid)
        prediction.to_csv(folder / "validation.csv", index=False)
        replicas.append(prediction)
        test_prediction = predict(
            model,
            profiles,
            transactions("test", use_cache=False, data_dir=args.data_dir),
        )
        test_prediction.to_csv(folder / "test.csv", index=False)
        test_replicas.append(test_prediction)
        write_json(
            folder / "metadata.json",
            {
                "training_clients": len(ids),
                "features": len(model.columns_),
                "none_features": len(model.none_columns_),
                "profiles": profiles,
                "prediction_sha256": digest(folder / "validation.csv"),
            },
        )
        del model
    reference = replicas[0]
    for replica in replicas[1:]:
        pd.testing.assert_frame_equal(reference, replica, check_exact=True)
    for replica in test_replicas[1:]:
        pd.testing.assert_frame_equal(test_replicas[0], replica, check_exact=True)
    # Only now open VALID labels; all predictions and the recipe are already persisted.
    write_json(
        out / "holdout_access.json",
        {
            "at": datetime.now(UTC).isoformat(),
            "purpose": "Fixed historical comparison, no model/threshold selection",
            "prediction_sha256": digest(out / "replica_1/validation.csv"),
            "all_replicas_identical": True,
        },
    )
    valid_labels = labels(
        "valid", allow_holdout=True, data_dir=args.data_dir
    ).reset_index()
    baseline_predictions, historical_hashes = run_baselines(args.data_dir, out)
    predictions = {**baseline_predictions, "Stream Identity": reference}
    results, encoded = {}, {}
    for name, prediction in predictions.items():
        results[name], encoded[name] = evaluate(valid_labels, prediction)
        print(name, results[name]["macro_f1"], results[name]["accuracy"], flush=True)
    fixture_path = ROOT / "reports/stream_identity_reference_predictions.csv"
    if fixture_path.exists():
        fixture = pd.read_csv(fixture_path).set_index("client_id")
        for name in ["V1", "V2"]:
            actual = predictions[name].set_index("client_id")[PREDICTION]
            pd.testing.assert_series_equal(
                actual.sort_index(), fixture[name].sort_index(), check_names=False
            )
    y_valid = (
        valid_labels[TARGET]
        .map({label: i for i, label in enumerate(LABELS)})
        .to_numpy()
    )
    uncertainty = bootstrap(y_valid, encoded)
    # Current branch's final_protocol.md and the later V4 delivery use TRAIN only.
    # Both independent full-TRAIN refits have already predicted TEST before labels.
    test = test_replicas[0]
    sample = pd.read_csv(
        args.data_dir / "sample_submission.csv", dtype=str, keep_default_na=False
    )
    submission = validate_submission(test[["client_id", PREDICTION]], sample)
    if not submission.client_id.equals(sample.client_id):
        raise ValueError("Submission order mismatch")
    submission.to_csv(out / "submission.csv", index=False)
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_submission.py",
            "--submission",
            str(out / "submission.csv"),
            "--sample",
            str(args.data_dir / "sample_submission.csv"),
        ],
        cwd=ROOT,
        check=True,
    )
    publish_bytes(args.submission, (out / "submission.csv").read_bytes())
    if frozen["source_sha256"] != source_hashes() or frozen[
        "data_sha256"
    ] != verify_data(args.data_dir):
        raise ValueError("Source/data changed during benchmark")
    summary = {
        "provenance": frozen,
        "reference_commit": REFERENCE_COMMIT,
        "V1_original_commit": V1_COMMIT,
        "audit": audit,
        "models": results,
        "bootstrap": uncertainty,
        "delta_vs_V1": results["Stream Identity"]["macro_f1"]
        - results["V1"]["macro_f1"],
        "delta_vs_V2": results["Stream Identity"]["macro_f1"]
        - results["V2"]["macro_f1"],
        "reproduction": {
            "raw_data_rebuilds": args.replicas,
            "exact_probabilities": True,
            "exact_predictions": True,
            "exact_test_probabilities": True,
            "exact_test_predictions": True,
        },
        "submission": {
            "valid": True,
            "rows": len(submission),
            "sample_order": True,
            "training_clients": len(ids),
            "sha256": digest(args.submission),
            "distribution": submission[PREDICTION].value_counts().to_dict(),
        },
        "status": "INVALID",
        "status_reason": "Historical VALID-informed rejection of decision biases; numeric reproduction does not undo selection exposure.",
        "seconds": perf_counter() - started,
    }
    write_json(out / "metrics.json", summary)
    write_json(out / "historical_source_sha256.json", historical_hashes)
    small = ROOT / "outputs/metrics/stream_identity/metrics.json"
    publish_bytes(small, (out / "metrics.json").read_bytes())
    files = [p for p in out.rglob("*") if p.is_file() and "__pycache__" not in str(p)]
    files += [args.submission.resolve(), small]
    receipt = {
        "source_sha256": source_hashes(),
        "data_sha256": verify_data(args.data_dir),
        "artifact_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
    }
    write_json(out / "receipt.json", receipt)
    print("Complete:", out, flush=True)


if __name__ == "__main__":
    main()
