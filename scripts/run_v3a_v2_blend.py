"""OOF-only scalar V2+A selection, one frozen VALID evaluation, and TEST refit.

Run from the repository root: --phase oof, then valid, then submission (or all).
Every artifact is create-only. Existing V1/V2/V3 outputs are read-only.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from run_ubs_v3 import bootstrap_delta
from sklearn.model_selection import StratifiedKFold
from validate_submission import inspect_submission

from transaction_forecasting.evaluation.official import validate_submission as validate_official
from transaction_forecasting.evaluation.probability_blend import (
    align_probabilities,
    blend_probabilities,
    select_alpha,
    validate_alpha,
)
from transaction_forecasting.ubs.data import (
    LABELS,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    read_labels,
    read_transactions,
    validate_submission,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.provenance import digest
from transaction_forecasting.ubs.v3.model import V3Model

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = tuple(n / 10 for n in range(11))
TIE_TOLERANCE = 0.0001
RESAMPLES = 2000
SEED = 42


def now():
    return datetime.now(UTC).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, frame, *, index=True):
    frame.to_csv(path, mode="x", index=index, index_label="client_id" if index else None)


def hashes(paths):
    return {str(Path(path).resolve()): digest(path) for path in paths}


def verify_hashes(stamps):
    for path, expected in stamps.items():
        if digest(path) != expected:
            raise ValueError(f"Fingerprint mismatch: {path}")


def check_output_directory(out, source):
    for protected in (source, *(ROOT / "outputs/metrics" / f"ubs_v{n}" for n in (1, 2, 3))):
        a, b = out.resolve(), protected.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("Cannot overwrite or nest inside existing baseline outputs")


def verify_source(source, data, phase):
    """Verify recorded source/data hashes, reading only TRAIN inputs in the OOF phase."""
    provenance_path = source / f"{phase}_provenance.json"
    provenance = read_json(provenance_path)
    if provenance["seed"] != SEED:
        raise ValueError("Source seed differs from 42")
    environment = {name: version(name) for name in provenance["versions"]}
    if environment != provenance["versions"] or platform.python_version() != provenance["python"]:
        raise ValueError("Use the original V3 Python and dependency versions")
    required_data = {"train_transactions.jsonl", "train_labels.csv"}
    if phase in ("valid", "submission"):
        required_data |= {"valid_transactions.jsonl", "valid_labels.csv"}
    if phase == "submission":
        required_data |= {"test_transactions.jsonl", "sample_submission.csv"}
    stamps, found_data, found_source = {}, set(), set()
    for path, expected in provenance["fingerprints"].items():
        if path.startswith(("src/", "scripts/")):
            resolved = ROOT / path
            found_source.add(path)
        elif Path(path).name in required_data:
            resolved = data / Path(path).name
            found_data.add(Path(path).name)
        else:
            continue
        stamps[str(resolved.resolve())] = expected
    required_source = {"scripts/run_ubs_v3.py"} | {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/transaction_forecasting/ubs").rglob("*.py")
    }
    if found_data != required_data or not required_source.issubset(found_source):
        raise ValueError("Incomplete source provenance")
    verify_hashes(stamps)
    stamps.update(hashes([provenance_path]))
    return stamps


def own_source_hashes():
    return hashes(
        [
            Path(__file__),
            ROOT / "src/transaction_forecasting/evaluation/probability_blend.py",
            ROOT / "src/transaction_forecasting/evaluation/official.py",
            ROOT / "scripts/validate_submission.py",
        ]
    )


def load_probabilities(path, clients):
    frame = pd.read_csv(
        path, index_col="client_id", dtype={"client_id": str}, float_precision="round_trip"
    )
    return align_probabilities(frame, clients)


def load_predictions(path):
    return pd.read_csv(path, index_col="client_id", dtype=str, keep_default_na=False)


def assert_predictions(actual, cached):
    if not cached.index.is_unique or set(actual.index) != set(cached.index):
        raise ValueError("Cached prediction client mismatch")
    if not actual.equals(cached.reindex(actual.index).rename(actual.name)):
        raise ValueError("Probabilities do not reproduce cached predictions")


def freeze_selection(out, payload):
    path = out / "frozen_selection.json"
    write_json(path, payload)
    with (out / "frozen_selection.sha256").open("x", encoding="utf-8") as stream:
        stream.write(digest(path) + "\n")


def load_frozen(out):
    path = out / "frozen_selection.json"
    expected = (out / "frozen_selection.sha256").read_text(encoding="utf-8").strip()
    if digest(path) != expected:
        raise ValueError("Frozen selection was modified")
    frozen = read_json(path)
    validate_alpha(frozen["alpha"])
    if frozen["valid_labels_used_for_selection"] or frozen["alpha"] not in ALPHAS:
        raise ValueError("Invalid OOF freeze")
    verify_hashes(frozen["fingerprints"])
    return frozen


def check_frozen_context(frozen, data, source):
    if (
        str(data.resolve()) != frozen["data_directory"]
        or str(source.resolve()) != frozen["source_directory"]
    ):
        raise ValueError("Use the data and source directories recorded in the freeze")


def run_oof(data, source, out):
    check_output_directory(out, source)
    # Reject even a partially populated experiment directory; never overwrite a freeze.
    out.mkdir(parents=True, exist_ok=False)
    stamps = {**verify_source(source, data, "oof"), **own_source_hashes()}
    labels = read_labels(data / "train_labels.csv")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    frames = {name: [] for name in ("V2", "A")}
    folds = pd.Series(index=target.index, dtype=int, name="fold")
    for fold, (_, hold) in enumerate(splitter.split(target.index, target), 1):
        clients = target.index[hold]
        pred_path = source / f"fold_{fold}_predictions.csv"
        cached = load_predictions(pred_path)
        stamps.update(hashes([pred_path]))
        for name in frames:
            path = source / f"fold_{fold}_{name}_probabilities.csv"
            values = load_probabilities(path, clients)
            assert_predictions(values.idxmax(axis=1), cached[name])
            frames[name].append(values)
            stamps.update(hashes([path]))
        folds.loc[clients] = fold
    probabilities = {
        name: align_probabilities(pd.concat(parts), target.index) for name, parts in frames.items()
    }
    cached_path, metrics_path = source / "oof_predictions.csv", source / "oof_results.json"
    cached, cached_metrics = load_predictions(cached_path), read_json(metrics_path)["metrics"]
    stamps.update(hashes([cached_path, metrics_path]))
    for name, values in probabilities.items():
        prediction = values.idxmax(axis=1)
        assert_predictions(prediction, cached[name])
        score = evaluate_predictions(target, prediction)["macro_f1"]
        if abs(score - cached_metrics[name]["macro_f1"]) > 1e-14:
            raise ValueError("Cached OOF endpoint metric mismatch")
    reports, rows, scores = {}, [], {}
    for alpha in ALPHAS:
        prediction = blend_probabilities(probabilities["V2"], probabilities["A"], alpha).idxmax(
            axis=1
        )
        report = evaluate_predictions(target, prediction)
        reports[f"{alpha:.2f}"] = report
        scores[alpha] = report["macro_f1"]
        rows.append(
            {
                "alpha": alpha,
                "macro_f1": report["macro_f1"],
                "accuracy": report["accuracy"],
                **{f"f1_{c}": report["per_class"][c]["f1-score"] for c in LABELS},
                **{f"count_{c}": report["prediction_distribution"][c] for c in LABELS},
            }
        )
    alpha = select_alpha(scores, TIE_TOLERANCE)
    selected = blend_probabilities(probabilities["V2"], probabilities["A"], alpha)
    predictions = pd.DataFrame({name: p.idxmax(axis=1) for name, p in probabilities.items()})
    predictions["blend"] = selected.idxmax(axis=1)
    predictions["target"], predictions["fold"] = target, folds.astype(int)
    for name, frame in {**probabilities, "blend": selected}.items():
        write_csv(out / f"oof_{name}_probabilities.csv", frame)
    write_csv(out / "oof_predictions.csv", predictions)
    write_csv(out / "oof_weight_search.csv", pd.DataFrame(rows), index=False)
    fold_reports = {
        str(fold): {
            name: evaluate_predictions(
                target.loc[folds == fold], predictions.loc[folds == fold, name]
            )
            for name in ("V2", "A", "blend")
        }
        for fold in range(1, 6)
    }
    write_json(
        out / "oof_results.json",
        {
            "metrics_by_alpha": reports,
            "selected_alpha": alpha,
            "folds": fold_reports,
            "class_order": list(LABELS),
            "clients": len(target),
        },
    )
    verify_hashes(stamps)
    stamps.update(hashes(out.iterdir()))
    freeze_selection(
        out,
        {
            "alpha": alpha,
            "oof_macro_f1": scores[alpha],
            "raw_maximum_macro_f1": max(scores.values()),
            "gap_to_raw_maximum": max(scores.values()) - scores[alpha],
            "grid": ALPHAS,
            "tie_tolerance": TIE_TOLERANCE,
            "tie_rule": "within tolerance: pure model first, then closest to 0.5, then lower alpha",
            "criterion": "maximum official eight-class TRAIN OOF Macro-F1; declared tie rule",
            "valid_labels_used_for_selection": False,
            "validation_independent": False,
            "validation_limitation": "The hypothesis was informed by historical VALID results",
            "seed": SEED,
            "cutoff": "2026-01-01",
            "created_utc": now(),
            "source_directory": str(source.resolve()),
            "data_directory": str(data.resolve()),
            "python": platform.python_version(),
            "versions": read_json(source / "oof_provenance.json")["versions"],
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "fingerprints": stamps,
            "cache_audit": "Original V3 recorded source/input hashes, not probability hashes; "
            "verified exact folds, IDs, argmax and endpoint metrics; cached bytes pinned here",
        },
    )
    print(f"OOF-selected alpha: {alpha:.2f}; OOF Macro-F1: {scores[alpha]:.12f}", flush=True)


def run_valid(data, source, out):
    frozen = load_frozen(out)  # Must complete before any VALID input, including hash reads.
    check_frozen_context(frozen, data, source)
    # This create-only marker prevents a second evaluation even after interruption.
    write_json(
        out / "valid_evaluation_started.json",
        {
            "frozen_sha256": digest(out / "frozen_selection.json"),
            "started_utc": now(),
            "alpha": frozen["alpha"],
        },
    )
    stamps = verify_source(source, data, "valid")
    transactions = read_transactions(data / "valid_transactions.jsonl")
    clients = pd.Index(sorted(transactions.client_id.unique()), name="client_id")
    train_ids = load_predictions(out / "oof_predictions.csv").index
    if len(train_ids.intersection(clients)):
        raise ValueError("TRAIN/VALID client overlap")
    probabilities = {}
    for name in ("V2", "A", "full", "ensemble"):
        path = source / f"valid_{name}_probabilities.csv"
        probabilities[name] = load_probabilities(path, clients)
        stamps.update(hashes([path]))
    expected = blend_probabilities(probabilities["V2"], probabilities["full"], 0.5)
    np.testing.assert_allclose(probabilities["ensemble"], expected, rtol=0, atol=1e-15)
    cached_path = source / "validation_predictions.csv"
    cached = load_predictions(cached_path)
    stamps.update(hashes([cached_path]))
    for name, values in probabilities.items():
        assert_predictions(values.idxmax(axis=1), cached[name])
    probabilities["blend"] = blend_probabilities(
        probabilities["V2"], probabilities["A"], frozen["alpha"]
    )
    frame = pd.DataFrame({name: values.idxmax(axis=1) for name, values in probabilities.items()})
    for name, values in probabilities.items():
        write_csv(out / f"valid_{name}_probabilities.csv", values)
    write_csv(out / "valid_predictions.csv", frame)
    write_csv(out / "valid_official.csv", frame["blend"].rename(PREDICTION_COLUMN).to_frame())
    # All predictions and alpha are on disk before labels are decoded for scoring.
    target = read_labels(data / "valid_labels.csv").set_index("client_id")[TARGET_COLUMN]
    metrics = {name: evaluate_predictions(target, frame[name]) for name in frame}
    result = {
        "alpha": frozen["alpha"],
        "frozen_sha256": digest(out / "frozen_selection.json"),
        "metrics": metrics,
        "class_order": list(LABELS),
        "clients": len(target),
        "delta_macro_f1": {
            name: metrics["blend"]["macro_f1"] - metrics[name]["macro_f1"]
            for name in ("V2", "A", "ensemble")
        },
        "relative_delta_vs_v2": (
            metrics["blend"]["macro_f1"] / metrics["V2"]["macro_f1"] - 1
            if metrics["V2"]["macro_f1"]
            else None
        ),
        "bootstrap": {
            name: bootstrap_delta(target, frame[name], frame["blend"], RESAMPLES)
            for name in ("V2", "A", "ensemble")
        },
        "completed_utc": now(),
        "validation_independent": False,
        "bootstrap_scope": "paired clients; fixed predictions; no multiple-selection correction",
    }
    verify_hashes(stamps)
    load_frozen(out)
    write_json(out / "valid_results.json", result)
    write_json(
        out / "valid_provenance.json",
        {"fingerprints": stamps, "artifacts": hashes(out.glob("valid*"))},
    )
    print(f"VALID Macro-F1: {metrics['blend']['macro_f1']:.12f}", flush=True)


def run_submission(data, source, out):
    frozen = load_frozen(out)
    check_frozen_context(frozen, data, source)
    result = read_json(out / "valid_results.json")
    if result["frozen_sha256"] != digest(out / "frozen_selection.json"):
        raise ValueError("VALID did not evaluate this frozen selection")
    valid_provenance = read_json(out / "valid_provenance.json")
    verify_hashes(valid_provenance["fingerprints"])
    verify_hashes(valid_provenance["artifacts"])
    write_json(out / "submission_started.json", {"started_utc": now(), "alpha": frozen["alpha"]})
    stamps = verify_source(source, data, "submission")
    train, valid, test = (
        read_transactions(data / f"{part}_transactions.jsonl")
        for part in ("train", "valid", "test")
    )
    for left, right in ((train, valid), (train, test), (valid, test)):
        if set(left.client_id).intersection(right.client_id):
            raise ValueError("Submission partitions overlap")
    labels = pd.concat(
        [read_labels(data / f"{part}_labels.csv") for part in ("train", "valid")], ignore_index=True
    )
    # Same unchanged API/recipe as run_ubs_v3 submission. VALID labels enter only this final refit.
    print("Refitting unchanged V3Model (includes V2 and A) on TRAIN+VALID, seed 42", flush=True)
    model = V3Model().fit(pd.concat([train, valid], ignore_index=True), labels)
    components = model.predict_components(test)
    differences = {}
    for name in ("V2", "A"):
        reference_path = source / f"test_{name}_probabilities.csv"
        reference = load_probabilities(reference_path, components[name].index)
        stamps.update(hashes([reference_path]))
        differences[name] = float(np.abs(components[name] - reference).to_numpy().max())
        np.testing.assert_allclose(components[name], reference, rtol=0, atol=1e-12)
        assert_predictions(components[name].idxmax(axis=1), reference.idxmax(axis=1))
        write_csv(out / f"test_{name}_probabilities.csv", components[name])
    blended = blend_probabilities(components["V2"], components["A"], frozen["alpha"])
    write_csv(out / "test_blend_probabilities.csv", blended)
    write_csv(out / "test_predictions.csv", blended.idxmax(axis=1).rename("blend").to_frame())
    sample = pd.read_csv(data / "sample_submission.csv", dtype=str, keep_default_na=False)
    submission = sample[["client_id"]].copy()
    submission[PREDICTION_COLUMN] = submission.client_id.map(blended.idxmax(axis=1))
    validate_submission(submission, sample, test)
    validate_official(submission, sample)
    path = out / "submission_v3a_v2_blend.csv"
    write_csv(path, submission, index=False)
    validate_submission(pd.read_csv(path, dtype=str, keep_default_na=False), sample, test)
    checks, passed = inspect_submission(path, data / "sample_submission.csv")
    if not passed:
        raise ValueError("Submission validator failed")
    verify_hashes(stamps)
    load_frozen(out)
    write_json(
        out / "submission_provenance.json",
        {
            "fingerprints": stamps,
            "alpha": frozen["alpha"],
            "frozen_sha256": digest(out / "frozen_selection.json"),
            "refit": "unchanged V3Model.fit(TRAIN+VALID); only V2 and A probabilities used",
            "fit_clients": len(labels),
            "seed": SEED,
            "completed_utc": now(),
            "maximum_absolute_difference_vs_original_test": differences,
            "validation": checks,
            "artifacts": hashes(out.glob("test*")) | hashes([path]),
            "submitted": False,
        },
    )
    print(f"submission path: {path}; validators: PASS ({len(submission)} rows)", flush=True)


def print_summary(out, tests="See verification log", recommendation="See reports/v3a_v2_blend.md"):
    frozen, valid = read_json(out / "frozen_selection.json"), read_json(out / "valid_results.json")
    metrics, bootstrap = valid["metrics"], valid["bootstrap"]
    summary = {
        "OOF-selected alpha": frozen["alpha"],
        "OOF Macro-F1": frozen["oof_macro_f1"],
        "VALID Macro-F1": metrics["blend"]["macro_f1"],
        "V2 VALID": metrics["V2"]["macro_f1"],
        "V3-A VALID": metrics["A"]["macro_f1"],
        "previous V2+full ensemble VALID": metrics["ensemble"]["macro_f1"],
        "new V2+A VALID": metrics["blend"]["macro_f1"],
        "delta vs V2": valid["delta_macro_f1"]["V2"],
        "delta vs V3-A": valid["delta_macro_f1"]["A"],
        "delta vs previous ensemble": valid["delta_macro_f1"]["ensemble"],
        "bootstrap interval vs V2": bootstrap["V2"]["percentile_95"],
        "bootstrap interval vs V3-A": bootstrap["A"]["percentile_95"],
        "bootstrap interval vs previous ensemble": bootstrap["ensemble"]["percentile_95"],
        "submission path": str(out / "submission_v3a_v2_blend.csv"),
        "tests": tests,
        "recommendation": recommendation,
    }
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid", "submission", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/metrics/ubs_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics/v3a_v2_blend"))
    args = parser.parse_args()
    check_output_directory(args.output_dir, args.source_dir)
    for phase, function in (("oof", run_oof), ("valid", run_valid), ("submission", run_submission)):
        if args.phase in (phase, "all"):
            function(args.data_dir, args.source_dir, args.output_dir)
    if args.phase in ("submission", "all"):
        print_summary(args.output_dir)


if __name__ == "__main__":
    main()
