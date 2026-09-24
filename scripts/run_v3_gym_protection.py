"""TRAIN-only threshold selection, then one frozen diagnostic VALID evaluation.

Existing V2/V3 OOF artifacts are reused; no TEST data is read. Outputs are new,
exclusive files. A failed gate CV is reported honestly, never auto-promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting import gym_protection as gate
from transaction_forecasting.ubs.data import TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v3.model import V3Model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def verify_hashes(stamps):
    for name, expected in stamps.items():
        if sha256(name) != expected:
            raise ValueError(f"Fingerprint mismatch: {name}")


def environment():
    return {
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost")
        },
    }


def read_probabilities(path):
    return pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")


def load_oof(data, source):
    """Verify saved source/TRAIN provenance, exact fold membership and argmax."""
    provenance = json.loads((source / "oof_provenance.json").read_text())
    stamps = {
        name: stamp
        for name, stamp in provenance["fingerprints"].items()
        if name.endswith(".py") or Path(name).name.startswith("train_")
    }
    for name in ("train_labels.csv", "train_transactions.jsonl"):
        if stamps.get(str(data / name)) != sha256(data / name):
            raise ValueError(f"TRAIN input differs from cached OOF: {name}")
    verify_hashes(stamps)
    env = environment()
    if env["python"] != provenance["python"] or env["packages"] != provenance["versions"]:
        raise ValueError("Environment differs from the OOF generation environment")
    target = read_labels(data / "train_labels.csv").set_index("client_id")[TARGET_COLUMN]
    target = target.sort_index()
    components = {name: [] for name in ("V2", "A")}
    folds = pd.Series(index=target.index, dtype=int)
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (_, hold) in enumerate(splitter.split(target.index, target), 1):
        expected_ids = target.index[hold]
        folds.loc[expected_ids] = fold
        predictions_path = source / f"fold_{fold}_predictions.csv"
        predictions = read_probabilities(predictions_path)
        stamps[str(predictions_path)] = sha256(predictions_path)
        for name in components:
            path = source / f"fold_{fold}_{name}_probabilities.csv"
            frame = read_probabilities(path)
            if not frame.index.is_unique or set(frame.index) != set(expected_ids):
                raise ValueError(f"Wrong OOF clients in {path}")
            frame, _ = gate.aligned_probabilities(frame, frame)
            if not frame.idxmax(axis=1).equals(predictions[name].reindex(frame.index)):
                raise ValueError(f"Cached predictions disagree with probabilities: {path}")
            stamps[str(path)] = sha256(path)
            components[name].append(frame)
    stamps[str(source / "oof_provenance.json")] = sha256(source / "oof_provenance.json")
    return (
        target,
        pd.concat(components["V2"]).reindex(target.index),
        pd.concat(components["A"]).reindex(target.index),
        folds.astype(int),
        stamps,
    )


def disagreement_analysis(target, v2, identity):
    p2, pa = v2.idxmax(axis=1), identity.idxmax(axis=1)
    groups = {
        "V2_only_gym": p2.eq("gym") & pa.ne("gym"),
        "A_only_gym": p2.ne("gym") & pa.eq("gym"),
        "both_gym": p2.eq("gym") & pa.eq("gym"),
        "true_gym_both_miss": target.eq("gym") & p2.ne("gym") & pa.ne("gym"),
    }
    result = {}
    for name, mask in groups.items():
        result[name] = {
            "clients": int(mask.sum()),
            "true_classes": target[mask].value_counts().to_dict(),
            "V2_gym_mean": float(v2.loc[mask, "gym"].mean()) if mask.any() else None,
            "A_gym_mean": float(identity.loc[mask, "gym"].mean()) if mask.any() else None,
            "V2_gym_higher": int((mask & v2.gym.gt(identity.gym)).sum()),
        }
    return result


def scores(target, v2, identity, candidate):
    return {
        name: evaluate_predictions(target, prediction)
        for name, prediction in {
            "V2": v2.idxmax(axis=1),
            "A": identity.idxmax(axis=1),
            "candidate": candidate,
        }.items()
    }


def train_phase(data, source, out):
    target, v2, identity, folds, stamps = load_oof(data, source)
    threshold, search = gate.select_gate(target, v2, identity)
    candidate = gate.route_gym(v2, identity, threshold)
    cv_prediction, cv_folds = gate.cross_validate_gate(target, v2, identity, folds)
    pooled = scores(target, v2, identity, candidate)
    cv = scores(target, v2, identity, cv_prediction)
    cv_delta = cv["candidate"]["macro_f1"] - cv["A"]["macro_f1"]
    for path in (
        Path(__file__),
        Path(gate.__file__),
        Path("src/transaction_forecasting/evaluation/official.py"),
    ):
        stamps[str(path)] = sha256(path)
    payload = {
        "metrics": pooled,
        "disagreements": disagreement_analysis(target, v2, identity),
        "selected_overrides": gate.override_analysis(target, identity.idxmax(axis=1), candidate),
        "gate_cv_metrics": cv,
        "gate_cv_folds": cv_folds,
        "gate_cv_overrides": gate.override_analysis(target, identity.idxmax(axis=1), cv_prediction),
        "gate_cv_macro_delta": cv_delta,
        "gate_cv_fully_nested_base_models": False,
    }
    write_json(out / "train_results.json", payload)
    pd.DataFrame(search).to_csv(out / "threshold_search.csv", index=False, mode="x")
    pd.DataFrame(
        {
            "fold": folds,
            "target": target,
            "V2": v2.idxmax(axis=1),
            "A": identity.idxmax(axis=1),
            "selected_gate": candidate,
            "gate_cv": cv_prediction,
        }
    ).to_csv(out / "oof_predictions.csv", index_label="client_id", mode="x")
    freeze = {
        "threshold": threshold,
        "dominance_margin": 0.0,
        "threshold_grid": [None, *gate.THRESHOLDS],
        "selection": "maximum pooled TRAIN OOF official Macro-F1; ties prefer fewer overrides",
        "rule": "V2 top1=gym, A top1!=gym, V2 gym>=threshold, V2 gym>=A top1 probability",
        "gate_cv_macro_delta": cv_delta,
        "train_cv_supported": cv_delta > 0,
        "purpose": "frozen diagnostic; negative CV prevents promotion regardless of VALID",
        "valid_labels_used_for_selection": False,
        "seed": 42,
        "created_utc": datetime.now(UTC).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "environment": environment(),
        "fingerprints": stamps,
    }
    verify_hashes(stamps)
    write_json(out / "frozen_policy.json", freeze)
    print(json.dumps({"threshold": threshold, "oof": pooled, "gate_cv_delta": cv_delta}))


def valid_phase(data, out):
    # Verify freeze/source/cache before even reading VALID histories; never tune here.
    freeze_path = out / "frozen_policy.json"
    frozen = json.loads(freeze_path.read_text())
    verify_hashes(frozen["fingerprints"])
    if environment() != frozen["environment"]:
        raise ValueError("Environment changed since freeze")
    # Exclusive marker also prevents rerunning an interrupted evaluation silently.
    write_json(
        out / "valid_started.json",
        {"freeze_sha256": sha256(freeze_path), "started_utc": datetime.now(UTC).isoformat()},
    )
    start = perf_counter()
    inputs = {
        str(data / name): sha256(data / name)
        for name in ("valid_transactions.jsonl", "valid_labels.csv")
    }
    train = read_transactions(data / "train_transactions.jsonl")
    labels = read_labels(data / "train_labels.csv")
    valid = read_transactions(data / "valid_transactions.jsonl")
    if set(train.client_id) & set(valid.client_id):
        raise ValueError("TRAIN/VALID overlap")
    # Reuse unchanged V3 API. It fits all original arms; only V2/A are used here.
    model = V3Model().fit(train, labels)
    components = model.predict_components(valid)
    v2, identity = gate.aligned_probabilities(components["V2"], components["A"])
    candidate = gate.route_gym(v2, identity, frozen["threshold"])
    frame = pd.DataFrame({"V2": v2.idxmax(axis=1), "A": identity.idxmax(axis=1)})
    frame["candidate"] = candidate
    for name, values in (("V2", v2), ("A", identity)):
        values.to_csv(out / f"valid_{name}_probabilities.csv", index_label="client_id", mode="x")
    frame.to_csv(out / "valid_predictions.csv", index_label="client_id", mode="x")
    # First semantic access to VALID labels is after all predictions are saved.
    target = read_labels(data / "valid_labels.csv").set_index("client_id")[TARGET_COLUMN]
    result = {
        "metrics": scores(target, v2, identity, candidate),
        "overrides": gate.override_analysis(target, frame.A, candidate),
        "threshold": frozen["threshold"],
        "freeze_sha256": sha256(freeze_path),
        "input_fingerprints": inputs,
        "probability_fingerprints": {
            name: sha256(out / f"valid_{name}_probabilities.csv") for name in ("V2", "A")
        },
        "seconds": perf_counter() - start,
        "environment": environment(),
        "validation_independent": False,
    }
    verify_hashes(frozen["fingerprints"])
    verify_hashes(inputs)
    write_json(out / "valid_results.json", result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("train", "valid"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument("--oof-dir", type=Path, default=Path("outputs/metrics/ubs_v3"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v3_gym_protection")
    )
    args = parser.parse_args()
    if args.phase == "train":
        # Fresh directory required; never overwrite previous runs or V1/V2/V3.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        train_phase(args.data_dir, args.oof_dir, args.output_dir)
    else:
        valid_phase(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
