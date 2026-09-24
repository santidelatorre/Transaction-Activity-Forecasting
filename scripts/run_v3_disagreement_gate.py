"""Reproduce the leakage-controlled V2 versus V3-A disagreement gate study.

Run ``oof`` before ``valid``.  The first phase generates base-model OOF
probabilities, cross-fits the gate, and freezes the learned architecture using
TRAIN only.  The second phase persists VALID predictions before opening labels.
Generated probabilities and metrics stay under ignored ``outputs/`` paths.
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

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v3.disagreement_gate import (
    V2V3AComponents,
    cross_fitted_gate_predictions,
    fit_final_gate,
    fixed_confidence_route,
    gate_features,
    gate_model_factories,
    route_with_model,
    routing_diagnostics,
)

METHOD_LABELS = {
    "V2": "V2",
    "V3-A": "V3-A",
    "fixed_confidence": "Fixed confidence routing",
    "learned_gate": "Learned disagreement gate",
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(data_dir: Path) -> dict[str, str]:
    paths = [
        data_dir / "train_transactions.jsonl",
        data_dir / "train_labels.csv",
        data_dir / "valid_transactions.jsonl",
        data_dir / "valid_labels.csv",
        Path(__file__),
        Path("src/transaction_forecasting/ubs/v2.py"),
        Path("src/transaction_forecasting/ubs/v3/features.py"),
        Path("src/transaction_forecasting/ubs/v3/model.py"),
        Path("src/transaction_forecasting/ubs/v3/disagreement_gate.py"),
    ]
    return {path.as_posix(): sha256(path) for path in paths}


def read_probability(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="client_id").reindex(columns=LABELS)


def score_methods(target: pd.Series, predictions: pd.DataFrame) -> dict[str, object]:
    return {name: evaluate_predictions(target, predictions[name]) for name in predictions}


def fold_scores(
    target: pd.Series, predictions: pd.DataFrame, fold_assignment: pd.Series
) -> list[dict[str, object]]:
    results = []
    for fold in sorted(fold_assignment.unique()):
        clients = fold_assignment.index[fold_assignment.eq(fold)]
        results.append(
            {
                "fold": int(fold),
                "clients": len(clients),
                "metrics": score_methods(target.loc[clients], predictions.loc[clients]),
            }
        )
    return results


def generate_base_oof(
    train: pd.DataFrame, labels: pd.DataFrame, out: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    v2_blocks, v3a_blocks, fold_blocks = [], [], []
    for fold, (fit_positions, hold_positions) in enumerate(
        splitter.split(target.index, target), start=1
    ):
        v2_path = out / f"base_fold_{fold}_V2_probabilities.csv"
        v3a_path = out / f"base_fold_{fold}_V3A_probabilities.csv"
        fit_ids = target.index[fit_positions]
        hold_ids = target.index[hold_positions]
        if v2_path.exists() and v3a_path.exists():
            v2, v3a = read_probability(v2_path), read_probability(v3a_path)
            if set(v2.index) != set(hold_ids) or set(v3a.index) != set(hold_ids):
                raise ValueError(f"Stale base fold {fold} client set")
        else:
            print(f"Fitting V2/V3-A outer fold {fold}/5", flush=True)
            fit_transactions = train.loc[train.client_id.isin(fit_ids)]
            hold_transactions = train.loc[train.client_id.isin(hold_ids)]
            fit_labels = labels.loc[labels.client_id.isin(fit_ids)]
            model = V2V3AComponents().fit(fit_transactions, fit_labels)
            v2, v3a = model.predict_probabilities(hold_transactions)
            v2.to_csv(v2_path, index_label="client_id")
            v3a.to_csv(v3a_path, index_label="client_id")
        v2_blocks.append(v2)
        v3a_blocks.append(v3a)
        fold_blocks.append(pd.Series(fold, index=hold_ids, name="base_fold"))
    v2 = pd.concat(v2_blocks).reindex(target.index)
    v3a = pd.concat(v3a_blocks).reindex(target.index)
    base_fold = pd.concat(fold_blocks).reindex(target.index).astype(int)
    if v2.isna().any().any() or v3a.isna().any().any() or not v2.index.is_unique:
        raise RuntimeError("Base OOF probabilities are incomplete")
    v2.to_csv(out / "oof_V2_probabilities.csv", index_label="client_id")
    v3a.to_csv(out / "oof_V3A_probabilities.csv", index_label="client_id")
    return v2, v3a, base_fold


def run_oof(data_dir: Path, out: Path, stamps: dict[str, str]) -> None:
    train = read_transactions(data_dir / "train_transactions.jsonl")
    labels = read_labels(data_dir / "train_labels.csv")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    v2, v3a, base_fold = generate_base_oof(train, labels, out)
    v2_prediction = v2.idxmax(axis=1)
    v3a_prediction = v3a.idxmax(axis=1)
    fixed = fixed_confidence_route(v2, v3a)

    factories = gate_model_factories()
    candidate_predictions: dict[str, pd.Series] = {}
    candidate_folds: dict[str, pd.Series] = {}
    candidate_metrics: dict[str, object] = {}
    for name, factory in factories.items():
        prediction, gate_fold = cross_fitted_gate_predictions(target, v2, v3a, factory)
        candidate_predictions[name] = prediction
        candidate_folds[name] = gate_fold
        candidate_metrics[name] = evaluate_predictions(target, prediction)
    selected = max(factories, key=lambda name: candidate_metrics[name]["macro_f1"])
    learned = candidate_predictions[selected]
    gate_fold = candidate_folds[selected]
    predictions = pd.DataFrame(
        {
            "V2": v2_prediction,
            "V3-A": v3a_prediction,
            "fixed_confidence": fixed,
            "learned_gate": learned,
        },
        index=target.index,
    )
    predictions.to_csv(out / "oof_predictions.csv", index_label="client_id")
    fold_table = pd.DataFrame({"base_fold": base_fold, "gate_fold": gate_fold})
    fold_table.to_csv(out / "oof_fold_assignment.csv", index_label="client_id")
    metrics = score_methods(target, predictions)
    results = {
        "metrics": metrics,
        "learned_candidate_metrics": candidate_metrics,
        "selected_learned_gate": selected,
        "folds": fold_scores(target, predictions, gate_fold),
        "routing": {
            "fixed_confidence": routing_diagnostics(target, v2_prediction, v3a_prediction, fixed),
            "learned_gate": routing_diagnostics(target, v2_prediction, v3a_prediction, learned),
        },
    }
    write_json(out / "oof_results.json", results)
    write_json(
        out / "frozen_selection.json",
        {
            "selected_learned_gate": selected,
            "selection_criterion": "maximum fixed-eight-class TRAIN cross-fitted Macro-F1",
            "candidate_hyperparameters": "fixed in source before VALID",
            "gate_threshold": 0.5,
            "gate_crossfit_folds": 5,
            "gate_crossfit_seed": 314159,
            "base_oof_folds": 5,
            "base_oof_seed": 42,
            "valid_labels_used_for_selection": False,
            "fingerprints": stamps,
        },
    )
    print(
        json.dumps(
            {
                "TRAIN_OOF": {name: row["macro_f1"] for name, row in metrics.items()},
                "selected_gate": selected,
            }
        ),
        flush=True,
    )


def run_valid(data_dir: Path, out: Path, stamps: dict[str, str]) -> None:
    frozen = json.loads((out / "frozen_selection.json").read_text(encoding="utf-8"))
    if frozen["fingerprints"] != stamps:
        raise ValueError("Source or data differs from the TRAIN-frozen gate selection")
    train = read_transactions(data_dir / "train_transactions.jsonl")
    labels = read_labels(data_dir / "train_labels.csv")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    oof_v2 = read_probability(out / "oof_V2_probabilities.csv").reindex(target.index)
    oof_v3a = read_probability(out / "oof_V3A_probabilities.csv").reindex(target.index)
    selected = frozen["selected_learned_gate"]
    factory = gate_model_factories()[selected]
    gate = fit_final_gate(target, oof_v2, oof_v3a, factory)

    valid_transactions = read_transactions(data_dir / "valid_transactions.jsonl")
    if set(train.client_id).intersection(valid_transactions.client_id):
        raise ValueError("TRAIN/VALID clients overlap")
    base = V2V3AComponents().fit(train, labels)
    valid_v2, valid_v3a = base.predict_probabilities(valid_transactions)
    valid_v2.to_csv(out / "valid_V2_probabilities.csv", index_label="client_id")
    valid_v3a.to_csv(out / "valid_V3A_probabilities.csv", index_label="client_id")
    v2_prediction = valid_v2.idxmax(axis=1)
    v3a_prediction = valid_v3a.idxmax(axis=1)
    fixed = fixed_confidence_route(valid_v2, valid_v3a)
    learned = route_with_model(
        gate, gate_features(valid_v2, valid_v3a), v2_prediction, v3a_prediction
    )
    predictions = pd.DataFrame(
        {
            "V2": v2_prediction,
            "V3-A": v3a_prediction,
            "fixed_confidence": fixed,
            "learned_gate": learned,
        },
        index=valid_v2.index,
    )
    # Persist every frozen prediction before opening VALID labels.
    predictions.to_csv(out / "validation_predictions.csv", index_label="client_id")

    valid_labels = read_labels(data_dir / "valid_labels.csv")
    valid_target = valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(predictions.index)
    metrics = score_methods(valid_target, predictions)
    results = {
        "metrics": metrics,
        "selected_learned_gate": selected,
        "routing": {
            "fixed_confidence": routing_diagnostics(
                valid_target, v2_prediction, v3a_prediction, fixed
            ),
            "learned_gate": routing_diagnostics(
                valid_target, v2_prediction, v3a_prediction, learned
            ),
        },
    }
    write_json(out / "valid_results.json", results)
    render_report(out)
    print(
        json.dumps({"VALID": {name: row["macro_f1"] for name, row in metrics.items()}}),
        flush=True,
    )


def metric_table(metrics: dict[str, object]) -> str:
    lines = ["| Method | Macro-F1 | Accuracy |", "|---|---:|---:|"]
    for key in METHOD_LABELS:
        row = metrics[key]
        lines.append(f"| {METHOD_LABELS[key]} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} |")
    return "\n".join(lines)


def per_class_table(metrics: dict[str, object], candidate: str = "learned_gate") -> str:
    lines = [
        "| Actual class | V2 F1 | V3-A F1 | Gate F1 | Gate - V3-A |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        v2 = metrics["V2"]["per_class"][label]["f1-score"]
        v3a = metrics["V3-A"]["per_class"][label]["f1-score"]
        gate = metrics[candidate]["per_class"][label]["f1-score"]
        lines.append(f"| {label} | {v2:.6f} | {v3a:.6f} | {gate:.6f} | {gate-v3a:+.6f} |")
    return "\n".join(lines)


def confusion_table(method: str, row: dict[str, object]) -> str:
    lines = [f"**{METHOD_LABELS[method]}**", "", "| true \\ pred | " + " | ".join(LABELS) + " |"]
    lines.append("|---|" + "---:|" * len(LABELS))
    for label, values in zip(LABELS, row["confusion_matrix"], strict=True):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def routing_table(routing: dict[str, object]) -> str:
    lines = [
        "| Actual class | Disagree | V2 only | V3-A only | Both wrong | Route acc. "
        "| To V2 | To V3-A | Changed | Correct | Incorrect |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = {"ALL": routing["overall"], **routing["by_actual_class"]}
    for label, row in rows.items():
        lines.append(
            f"| {label} | {row['disagreements']} | {row['v2_correct_only']} | "
            f"{row['v3a_correct_only']} | {row['both_wrong']} | "
            f"{row['routing_accuracy_on_disagreements']:.4f} | {row['routed_to_v2']} | "
            f"{row['routed_to_v3a']} | {row['changed_vs_v3a']} | "
            f"{row['correct_changes']} | {row['incorrect_changes']} |"
        )
    return "\n".join(lines)


def render_report(out: Path) -> None:
    oof = json.loads((out / "oof_results.json").read_text(encoding="utf-8"))
    valid = json.loads((out / "valid_results.json").read_text(encoding="utf-8"))
    selected = oof["selected_learned_gate"]
    oof_delta = oof["metrics"]["learned_gate"]["macro_f1"] - oof["metrics"]["V3-A"]["macro_f1"]
    valid_delta = (
        valid["metrics"]["learned_gate"]["macro_f1"] - valid["metrics"]["V3-A"]["macro_f1"]
    )
    fold_wins = sum(
        row["metrics"]["learned_gate"]["macro_f1"] > row["metrics"]["V3-A"]["macro_f1"]
        for row in oof["folds"]
    )
    if oof_delta > 0 and valid_delta > 0 and fold_wins >= 3:
        recommendation = "PROMOTE"
    elif oof_delta > 0 or valid_delta > 0:
        recommendation = "PROMISING"
    else:
        recommendation = "REJECT"
    fold_lines = [
        "| Gate fold | Clients | V2 | V3-A | Fixed confidence | Learned gate | Gate - V3-A |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in oof["folds"]:
        scores = {name: value["macro_f1"] for name, value in row["metrics"].items()}
        fold_lines.append(
            f"| {row['fold']} | {row['clients']} | {scores['V2']:.6f} | {scores['V3-A']:.6f} | "
            f"{scores['fixed_confidence']:.6f} | {scores['learned_gate']:.6f} | "
            f"{scores['learned_gate']-scores['V3-A']:+.6f} |"
        )
    candidate_lines = [
        "| Learned candidate | TRAIN cross-fitted Macro-F1 | Accuracy |",
        "|---|---:|---:|",
    ]
    for name, row in oof["learned_candidate_metrics"].items():
        suffix = " (selected)" if name == selected else ""
        candidate_lines.append(
            f"| {name}{suffix} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} |"
        )
    confusion_oof = "\n\n".join(
        confusion_table(name, oof["metrics"][name]) for name in METHOD_LABELS
    )
    confusion_valid = "\n\n".join(
        confusion_table(name, valid["metrics"][name]) for name in METHOD_LABELS
    )
    routing = valid["routing"]["learned_gate"]["overall"]
    report = f"""# V3 V2/V3-A disagreement gate — Ginestar

## 1. Executive summary

The frozen learned gate is **{selected}**. On TRAIN it changes V3-A Macro-F1 by
**{oof_delta:+.6f}** under a second five-fold client-level cross-fitting layer. On
previously unopened VALID it changes Macro-F1 by **{valid_delta:+.6f}**. It routes
{routing['disagreements']} disagreements with accuracy
**{routing['routing_accuracy_on_disagreements']:.4f}**, changing
{routing['changed_vs_v3a']} V3-A predictions ({routing['correct_changes']} correct,
{routing['incorrect_changes']} incorrect, {routing['both_wrong_changes']} between
two wrong classes). Recommendation: **{recommendation}**.

This is a hard per-client selector. It is not a global probability blend and it
does not interpolate V2 and V3-A scores.

## 2. Hypothesis

V2 and V3-A encode different evidence. V3-A adds cross-fitted merchant-family
identity features, while V2 retains the older history representation. If their
errors differ systematically, inference-time confidence geometry may identify
which prediction to trust for an individual client.

## 3. Anti-leakage design

- Five outer stratified TRAIN folds (seed 42) produce V2 and V3-A probabilities.
  Neither base model sees its held client's label.
- A second five-fold stratified split (seed 314159) evaluates every learned gate.
  Each routed TRAIN prediction comes from a gate fitted without that client.
- Only TRAIN OOF results select the gate family. All candidate hyperparameters and
  the 0.5 decision threshold are fixed in source.
- The final gate trains on decisive TRAIN OOF disagreements only. VALID predictions
  are written to disk before `valid_labels.csv` is read.
- Client ID is an index only. No IDs, VALID labels, TEST labels, or post-cutoff
  transactions enter a model feature.

The TRAIN figure after selecting among three cross-fitted gates remains a model
selection estimate, not a pristine external estimate. VALID is the untouched
confirmation set for this workstream.

## 4. Construction of OOF predictions

Each base fold fits the unchanged `IntegratedV2Model` and the frozen V3-A
identity-only arm. V3-A uses the same 75% CatBoost / 25% periodicity composition
as the discovery study. Its supervised family map is also internally cross-fitted
for the base-fold training matrix. OOF CSVs contain one probability vector per
TRAIN client and are ignored by Git.

## 5. Gate architecture

Agreements preserve the shared class. On a disagreement, the fixed rule selects
the higher maximum probability. Learned gates predict `choose V3-A` versus
`choose V2`. Their training target exists only when exactly one system is correct;
both-wrong cases are excluded from fitting and retained as routing failures during
evaluation. The frozen shortlist is logistic regression, a depth-3 tree, and a
small histogram gradient booster.

{"\n".join(candidate_lines)}

## 6. Feature set

The gate uses 27 inference-time numeric features: both eight-class probability
vectors, each model's maximum probability, top-1/top-2 margin, normalized entropy,
the V3-A-minus-V2 differences for those summaries, and both `none` probabilities.
No transaction feature was added because the V3-A probability vector already
summarizes family evidence and the initial test should keep gate capacity low.

## 7. TRAIN OOF results

{metric_table(oof['metrics'])}

### TRAIN per-class F1

{per_class_table(oof['metrics'])}

### TRAIN confusion matrices

{confusion_oof}

## 8. VALID results

{metric_table(valid['metrics'])}

### VALID per-class F1

{per_class_table(valid['metrics'])}

### VALID confusion matrices

{confusion_valid}

## 9. Disagreement decomposition

### TRAIN OOF learned gate

{routing_table(oof['routing']['learned_gate'])}

### VALID learned gate

{routing_table(valid['routing']['learned_gate'])}

## 10. Per-class changes

The per-class tables above compare F1, which includes both false positives from
other actual classes and false negatives inside each row. The routing tables give
the complementary client counts by actual class. A class should be treated as a
real gain only when its F1 improves without being driven by a tiny number of
changes.

## 11. Fold stability

{"\n".join(fold_lines)}

The learned gate beats V3-A in **{fold_wins}/5** gate folds. These are the
additional gate folds, not the outer folds that generated the base probabilities.

## 12. Comparison against V2 and V3-A

The fixed-confidence router is the no-training control. The learned router is
only useful if it improves on V3-A out of sample and the gain is not isolated to
one fold or one tiny class. VALID is reported once after freezing; its result did
not alter the selected gate, features, threshold, or targeted classes.

## 13. Recommendation

**{recommendation}.** The decision rule was fixed in code: promote only when the
gate improves both TRAIN cross-fitted and VALID Macro-F1 and wins at least three
of five gate folds; mark promising when exactly one aggregate split improves;
otherwise reject. A promotion still requires normal PR review and must not replace
the frozen V2/V3-A artifacts silently.

## 14. Reproduction commands

```powershell
python scripts/run_v3_disagreement_gate.py --phase oof
python scripts/run_v3_disagreement_gate.py --phase valid
python -m pytest -q
python -m ruff check .
```

Generated evidence is under `{out.as_posix()}` and intentionally ignored. The
tracked report contains no client IDs, probabilities, dataset rows, or predictions.
"""
    report_path = Path("reports/handoff/v3_disagreement_gate_ginestar.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics/v3_disagreement_gate_ginestar"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    stamps = fingerprints(args.data_dir)
    if args.phase in ("oof", "all"):
        old = args.output_dir / "frozen_selection.json"
        if old.exists():
            previous = json.loads(old.read_text(encoding="utf-8"))
            if previous["fingerprints"] != stamps:
                raise ValueError("Existing output was produced by different source/data")
        run_oof(args.data_dir, args.output_dir, stamps)
    if args.phase in ("valid", "all"):
        run_valid(args.data_dir, args.output_dir, stamps)
    write_json(
        args.output_dir / f"{args.phase}_provenance.json",
        {
            "commit_before_experiment": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "python": platform.python_version(),
            "versions": {
                package: version(package)
                for package in ("numpy", "pandas", "scikit-learn", "catboost")
            },
            "fingerprints": stamps,
            "seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
