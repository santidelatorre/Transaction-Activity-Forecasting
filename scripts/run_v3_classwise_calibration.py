"""Cross-fit compact V3-A decision offsets on verified local TRAIN OOF artifacts.

Run --phase train first, then --phase valid exactly once. VALID cannot select
parameters; the second phase consumes a fingerprinted, frozen TRAIN selection.
No TEST inputs are opened. The original V3 runner can generate the base artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v3.calibration import (
    BOUND,
    GRID,
    MIN_GAIN,
    PENALTY,
    apply_offsets,
    cross_fit_calibration,
    select_calibration,
    validate_probabilities,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_base(probability_dir, data_dir, phase):
    """Verify predictive source/input fingerprints without accessing TEST files."""
    provenance = probability_dir / f"{'oof' if phase == 'train' else 'valid'}_provenance.json"
    saved = json.loads(provenance.read_text(encoding="utf-8"))
    if saved["seed"] != 42:
        raise ValueError("Expected the original seed-42 client folds")
    checked = {str(provenance): digest(provenance)}
    for name, expected in saved["fingerprints"].items():
        original = Path(name)
        if original.suffix == ".py":
            path = original
        elif original.name in (
            "train_transactions.jsonl",
            "train_labels.csv",
            *(["valid_transactions.jsonl"] if phase == "valid" else []),
        ):
            path = data_dir / original.name
        else:
            continue
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Base provenance mismatch: {path}")
        checked[str(path)] = actual
    required = {
        str(Path("scripts/run_ubs_v3.py")),
        str(Path("src/transaction_forecasting/ubs/v3/model.py")),
        str(data_dir / "train_labels.csv"),
        str(data_dir / "train_transactions.jsonl"),
    }
    if not required.issubset(checked):
        raise ValueError("Incomplete base provenance")
    return checked


def source_fingerprints():
    return {
        str(path): digest(path)
        for path in (
            Path(__file__).resolve(),
            Path("src/transaction_forecasting/ubs/v3/calibration.py"),
            Path("src/transaction_forecasting/evaluation/official.py"),
            Path("reports/handoff/v3_classwise_calibration_protocol_esteban.md"),
        )
    }


def load_probabilities(path, expected_ids=None):
    frame = pd.read_csv(path, dtype={"client_id": str}).set_index("client_id")
    validate_probabilities(frame)
    if expected_ids is not None:
        if set(frame.index) != set(expected_ids):
            raise ValueError(f"Probability clients differ from expected fold: {path}")
        frame = frame.reindex(expected_ids)
    return frame


def comparison(target, control, raw, calibrated):
    prediction = pd.DataFrame({"V2": control, "raw": raw, "calibrated": calibrated})
    scores = {name: evaluate_predictions(target, prediction[name]) for name in prediction}
    changed = raw.ne(calibrated)
    before, after = raw.eq(target), calibrated.eq(target)
    transitions = pd.crosstab(raw, calibrated).reindex(index=LABELS, columns=LABELS, fill_value=0)
    by_class = {}
    for label in LABELS:
        take = target.eq(label)
        by_class[label] = {
            "changed": int((changed & take).sum()),
            "incorrect_to_correct": int((~before & after & take).sum()),
            "correct_to_incorrect": int((before & ~after & take).sum()),
            "incorrect_to_other_incorrect": int((changed & ~before & ~after & take).sum()),
        }
    return {
        "clients": len(target),
        "metrics": scores,
        "macro_f1_delta": scores["calibrated"]["macro_f1"] - scores["raw"]["macro_f1"],
        "per_class_delta": {
            label: scores["calibrated"]["per_class"][label]["f1-score"]
            - scores["raw"]["per_class"][label]["f1-score"]
            for label in LABELS
        },
        "true_distribution": target.value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "changes": {
            "total": int(changed.sum()),
            "incorrect_to_correct": int((~before & after).sum()),
            "correct_to_incorrect": int((before & ~after).sum()),
            "incorrect_to_other_incorrect": int((changed & ~before & ~after).sum()),
            "unchanged_correct": int((~changed & before).sum()),
            "unchanged_incorrect": int((~changed & ~before).sum()),
        },
        "changes_by_true_class": by_class,
        "transition_matrix_raw_to_calibrated": transitions.to_numpy().tolist(),
    }


def train_phase(args):
    out = args.output_dir
    if (out / "frozen_calibration.json").exists():
        raise ValueError("TRAIN is already frozen; use a fresh output directory")
    checks = verify_base(args.probability_dir, args.data_dir, "train")
    truth = read_labels(args.data_dir / "train_labels.csv").set_index("client_id")[TARGET_COLUMN]
    truth = truth.sort_index()
    folds = pd.Series(index=truth.index, dtype=int, name="fold")
    raw_parts, v2_parts = [], []
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (_, hold) in enumerate(splitter.split(truth.index, truth), 1):
        ids = truth.index[hold]
        folds.loc[ids] = fold
        for name, parts in (("A", raw_parts), ("V2", v2_parts)):
            path = args.probability_dir / f"fold_{fold}_{name}_probabilities.csv"
            parts.append(load_probabilities(path, ids))
            checks[str(path)] = digest(path)
    raw = pd.concat(raw_parts).reindex(truth.index)
    v2 = pd.concat(v2_parts).reindex(truth.index)
    validate_probabilities(raw)
    validate_probabilities(v2)
    raw.to_csv(out / "train_oof_A_probabilities.csv", index_label="client_id")
    v2.to_csv(out / "train_oof_V2_probabilities.csv", index_label="client_id")
    calibrated, fold_parameters = cross_fit_calibration(raw, truth, folds)
    frame = pd.DataFrame(
        {
            "fold": folds.astype(int),
            "target": truth,
            "V2": v2.idxmax(axis=1),
            "raw": raw.idxmax(axis=1),
            "calibrated": calibrated.idxmax(axis=1),
        }
    )
    calibrated.to_csv(out / "train_cross_fitted_probabilities.csv", index_label="client_id")
    frame.to_csv(out / "train_cross_fitted_predictions.csv", index_label="client_id")
    report = comparison(truth, frame.V2, frame.raw, frame.calibrated)
    report["folds"] = []
    for fitted in fold_parameters:
        take = folds.eq(fitted["fold"])
        report["folds"].append(
            {
                **fitted,
                "evaluation": comparison(
                    truth.loc[take],
                    frame.V2.loc[take],
                    frame.raw.loc[take],
                    frame.calibrated.loc[take],
                ),
            }
        )
    final = select_calibration(raw, truth, folds)
    write_json(out / "train_results.json", report)
    write_json(out / "final_selection_diagnostics.json", final)
    frozen = {
        "classes": list(LABELS),
        "parameters": final["parameters"],
        "offsets": final["offsets"],
        "weights": final["weights"],
        "direction": final["direction"],
        "identity_bound": BOUND,
        "penalty": PENALTY,
        "min_gain": MIN_GAIN,
        "grid_size": len(GRID),
        "valid_labels_used": False,
        "fully_nested_base_model_cv": False,
        "source_fingerprints": source_fingerprints(),
        "base_fingerprints": checks,
        "train_results_sha256": digest(out / "train_results.json"),
        "final_selection_sha256": digest(out / "final_selection_diagnostics.json"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
    }
    write_json(out / "frozen_calibration.json", frozen)
    print(
        json.dumps(
            {
                "TRAIN": {k: v["macro_f1"] for k, v in report["metrics"].items()},
                "parameters": final["parameters"],
                "offsets": final["offsets"],
            }
        ),
        flush=True,
    )


def valid_phase(args):
    out = args.output_dir
    # Exclusive marker prevents repeated scoring, including after a partial failure.
    if (out / "valid_evaluation_started.json").exists():
        raise ValueError("VALID evaluation already started; do not repeatedly score this run")
    frozen_path = out / "frozen_calibration.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["source_fingerprints"] != source_fingerprints():
        raise ValueError("Calibration source/protocol/evaluator changed after TRAIN freeze")
    for name, expected in frozen["base_fingerprints"].items():
        if digest(name) != expected:
            raise ValueError(f"TRAIN base input changed after freeze: {name}")
    for filename, key in (
        ("train_results.json", "train_results_sha256"),
        ("final_selection_diagnostics.json", "final_selection_sha256"),
    ):
        if digest(out / filename) != frozen[key]:
            raise ValueError("TRAIN selection artifacts changed after freeze")
    checks = verify_base(args.probability_dir, args.data_dir, "valid")
    raw_path = args.probability_dir / "valid_A_probabilities.csv"
    v2_path = args.probability_dir / "valid_V2_probabilities.csv"
    raw = load_probabilities(raw_path)
    v2 = load_probabilities(v2_path, raw.index)
    train_ids = read_labels(args.data_dir / "train_labels.csv").client_id
    if set(train_ids).intersection(raw.index):
        raise ValueError("TRAIN/VALID client overlap")
    calibrated = apply_offsets(raw, frozen["offsets"])
    predictions = pd.DataFrame(
        {
            "V2": v2.idxmax(axis=1),
            "raw": raw.idxmax(axis=1),
            "calibrated": calibrated.idxmax(axis=1),
        }
    )
    predictions.to_csv(out / "valid_predictions.csv", index_label="client_id")
    calibrated.to_csv(out / "valid_calibrated_probabilities.csv", index_label="client_id")
    checks.update({str(path): digest(path) for path in (raw_path, v2_path, frozen_path)})
    with (out / "valid_evaluation_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"frozen_sha256": digest(frozen_path), "fingerprints": checks}, handle, indent=2)
    # First VALID-label access: all choices and predictions have already been persisted.
    label_path = args.data_dir / "valid_labels.csv"
    truth = read_labels(label_path).set_index("client_id")[TARGET_COLUMN]
    report = comparison(truth, predictions.V2, predictions.raw, predictions.calibrated)
    report["valid_labels_sha256"] = digest(label_path)
    report["frozen_sha256"] = digest(frozen_path)
    write_json(out / "valid_results.json", report)
    print(
        json.dumps(
            {
                "VALID": {k: v["macro_f1"] for k, v in report["metrics"].items()},
                "delta": report["macro_f1_delta"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "valid"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument("--probability-dir", type=Path, default=Path("outputs/metrics/ubs_v3"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v3_classwise_calibration_esteban")
    )
    args = parser.parse_args()
    if args.output_dir.resolve() == args.probability_dir.resolve():
        raise ValueError("Calibration output must not overwrite base artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    (train_phase if args.phase == "train" else valid_phase)(args)
    write_json(args.output_dir / f"{args.phase}_timing.json", {"seconds": perf_counter() - start})


if __name__ == "__main__":
    main()
