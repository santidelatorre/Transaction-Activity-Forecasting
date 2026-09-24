"""Portable diagnostic replay of original router/gym functions on fresh base OOF.

The router report renderer needs Python 3.12; the gym CLI's literal path hash
lookup is platform-dependent. This adapter changes neither predictive function
nor evaluator. Run with the pinned snapshot's src first on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels
from transaction_forecasting.ubs.evaluation import evaluate_predictions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def verified_base(source, data):
    """Check content hashes, original fold membership and stored argmax/metrics."""
    truth = (
        read_labels(data / "train_labels.csv").set_index("client_id")[TARGET_COLUMN].sort_index()
    )
    parts = {name: [] for name in ("V2", "A")}
    folds = pd.Series(index=truth.index, dtype=int)
    evidence = {}
    for phase in ("oof", "valid"):
        path = source / f"{phase}_provenance.json"
        evidence[str(path)] = digest(path)
        provenance = json.loads(path.read_text())
        assert provenance["seed"] == 42
        for name, expected in provenance["fingerprints"].items():
            original = Path(name)
            if original.suffix == ".py":
                candidate = original
            elif original.name.startswith("train_"):
                candidate = data / original.name
            else:
                continue
            if digest(candidate) != expected:
                raise ValueError(f"Source/input mismatch: {candidate}")
            evidence[str(candidate)] = expected
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (_, hold) in enumerate(splitter.split(truth.index, truth), 1):
        ids = truth.index[hold]
        folds.loc[ids] = fold
        saved = pd.read_csv(source / f"fold_{fold}_predictions.csv", index_col="client_id")
        for name in parts:
            path = source / f"fold_{fold}_{name}_probabilities.csv"
            frame = pd.read_csv(path, index_col="client_id", float_precision="round_trip")
            assert frame.index.is_unique and set(frame.index) == set(ids)
            assert frame.columns.tolist() == list(LABELS)
            assert frame.idxmax(axis=1).equals(saved[name].rename(None))
            evidence[str(path)] = digest(path)
            parts[name].append(frame)
    parts = {name: pd.concat(values).reindex(truth.index) for name, values in parts.items()}
    expected = json.loads((source / "oof_results.json").read_text())["metrics"]
    for name, frame in parts.items():
        score = evaluate_predictions(truth, frame.idxmax(axis=1))["macro_f1"]
        assert abs(score - expected[name]["macro_f1"]) < 1e-14
    return truth, parts, folds.astype(int), evidence


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    truth, base, folds, evidence = verified_base(args.source_dir, args.data_dir)
    v2, identity = base["V2"], base["A"]
    predictions = pd.DataFrame({"V2": v2.idxmax(axis=1), "A": identity.idxmax(axis=1)})
    if args.name == "router":
        from transaction_forecasting.ubs.v3 import disagreement_gate as module

        candidates = {}
        for name, factory in module.gate_model_factories().items():
            prediction, gate_folds = module.cross_fitted_gate_predictions(
                truth, v2, identity, factory
            )
            candidates[name] = prediction
        selected = max(
            candidates, key=lambda name: evaluate_predictions(truth, candidates[name])["macro_f1"]
        )
        predictions["candidate"] = candidates[selected]
        predictions["fixed_confidence"] = module.fixed_confidence_route(v2, identity)
        fitted = module.fit_final_gate(truth, v2, identity, module.gate_model_factories()[selected])
        predictions["fold"] = gate_folds.astype(int)
        freeze = {"selected": selected, "threshold": 0.5}
    else:
        from transaction_forecasting import gym_protection as module

        selected, search = module.select_gate(truth, v2, identity)
        predictions["selected_gate"] = module.route_gym(v2, identity, selected)
        predictions["candidate"], details = module.cross_validate_gate(truth, v2, identity, folds)
        predictions["fold"] = folds
        freeze = {"threshold": selected, "search": search, "folds": details}
    evidence[str(Path(module.__file__))] = digest(Path(module.__file__))
    predictions.to_csv(out / "oof_predictions.csv", index_label="client_id")
    write_json(out / "frozen_replay.json", {**freeze, "fingerprints": evidence})
    write_json(
        out / "oof_results.json",
        {
            name: evaluate_predictions(truth, predictions[name])
            for name in predictions
            if name != "fold"
        },
    )
    valid = {
        name: pd.read_csv(
            args.source_dir / f"valid_{name}_probabilities.csv",
            index_col="client_id",
            float_precision="round_trip",
        )
        for name in ("V2", "A")
    }
    frame = pd.DataFrame({name: values.idxmax(axis=1) for name, values in valid.items()})
    if args.name == "router":
        features = module.gate_features(valid["V2"], valid["A"])
        frame["candidate"] = module.route_with_model(fitted, features, frame.V2, frame.A)
        frame["fixed_confidence"] = module.fixed_confidence_route(valid["V2"], valid["A"])
    else:
        frame["candidate"] = module.route_gym(valid["V2"], valid["A"], selected)
    frame.to_csv(out / "valid_predictions.csv", index_label="client_id")
    # Labels are decoded after the frozen replay predictions exist.
    valid_truth = read_labels(args.data_dir / "valid_labels.csv").set_index("client_id")[
        TARGET_COLUMN
    ]
    report = {name: evaluate_predictions(valid_truth, frame[name]) for name in frame}
    write_json(out / "valid_results.json", report)
    print({name: value["macro_f1"] for name, value in report.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("router", "gym"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
