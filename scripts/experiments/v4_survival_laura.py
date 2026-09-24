"""TRAIN-only selection then a single frozen VALID evaluation for Laura's V4."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import (
    LABELS,
    TARGET_COLUMN,
    read_labels,
    read_transactions,
)
from transaction_forecasting.ubs.survival import (
    FamilyEvidence,
    RecurrenceModel,
    aggregate,
    median_gap_probabilities,
    pseudo_cutoffs,
    stream_features,
)
from transaction_forecasting.ubs.v3.model import V3Model


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def metrics(truth, probabilities):
    probabilities = probabilities.reindex(truth.index)
    if probabilities.isna().any().any() or not probabilities.index.is_unique:
        raise ValueError("Incomplete probabilities")
    predicted = probabilities.idxmax(axis=1)
    confident = probabilities.max(axis=1).ge(0.8)
    report = classification_report(
        truth, predicted, labels=LABELS, output_dict=True, zero_division=0
    )
    return dict(
        macro_f1=f1_score(truth, predicted, labels=LABELS, average="macro"),
        per_family={c: report[c] for c in LABELS},
        event_accuracy=float((truth.ne("none") == predicted.ne("none")).mean()),
        coverage=float(predicted.ne("none").mean()),
        high_confidence_coverage=float(confident.mean()),
        precision_high_confidence=float((predicted[confident] == truth[confident]).mean())
        if confident.any()
        else None,
        brier=float(
            np.mean(
                np.sum(
                    (
                        probabilities.to_numpy()
                        - np.array([[int(t == c) for c in LABELS] for t in truth])
                    )
                    ** 2,
                    axis=1,
                )
            )
        ),
        logloss=float(
            log_loss(
                [LABELS.index(t) for t in truth], probabilities, labels=list(range(len(LABELS)))
            )
        ),
    )


def stream_metrics(frame, p):
    y, w = frame.event.to_numpy(), frame.weight.to_numpy()
    event = 1 - p[:, 3]
    selected = pd.DataFrame(
        {"client": frame.client_id, "cutoff": frame.cutoff, "score": event, "event": y}
    ).reset_index(drop=True)
    groups = selected.groupby(["client", "cutoff"])
    has_event = groups.event.max().eq(1)
    top = selected.loc[groups.score.idxmax()].set_index(["client", "cutoff"]).event
    observed = y.astype(bool)
    expected = (p[:, :3] @ np.array([15.0, 45.0, 75.0])) / np.maximum(event, 1e-12)
    return dict(
        auroc=float(roc_auc_score(y, event, sample_weight=w)) if len(set(y)) > 1 else None,
        pr_auc=float(average_precision_score(y, event, sample_weight=w)),
        brier=float(brier_score_loss(y, event, sample_weight=w)),
        logloss=float(
            log_loss(y, np.column_stack([1 - event, event]), labels=[0, 1], sample_weight=w)
        ),
        top_event_ranking_accuracy=float(top[has_event].mean()) if has_event.any() else None,
        temporal_mae_diagnostic=float(
            np.average(
                abs(expected[observed] - frame.time_to_next_event.to_numpy()[observed]),
                weights=w[observed],
            )
        )
        if observed.any()
        else None,
    )


def baseline(train, labels, hold, cache, prefix):
    path = cache / f"{prefix}_A_probabilities.csv"
    provenance = cache / "oof_provenance.json"
    if path.exists() and provenance.exists():
        stamps = json.loads(provenance.read_text())["fingerprints"]
        # Reuse only baseline outputs whose source and TRAIN data still match.
        for name, sha in stamps.items():
            if name.startswith("src/transaction_forecasting/ubs/") or "train_" in name:
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                    # The promotion changed only the selector and documentation.
                    if name != "src/transaction_forecasting/ubs/v3/model.py":
                        raise ValueError(f"Baseline fingerprint mismatch: {name}")
                    old = subprocess.check_output(["git", "show", f"df9fe41:{name}"])
                    if hashlib.sha256(old).hexdigest() != sha:
                        raise ValueError("Unknown baseline model provenance")

                    def component_ast(source):
                        tree = ast.parse(source)
                        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
                        return [
                            ast.dump(n)
                            for n in cls.body
                            if isinstance(n, ast.FunctionDef)
                            and n.name in ("fit", "predict_components")
                        ]

                    if component_ast(old) != component_ast(Path(name).read_bytes()):
                        raise ValueError("Baseline A implementation changed")
        p = pd.read_csv(path, index_col="client_id").reindex(columns=LABELS)
        if set(p.index) != set(hold.client_id):
            raise ValueError("Baseline fold clients mismatch")
        return p
    return V3Model().fit(train, labels).predict_components(hold)["A"]


def predict_variants(fit, target, snapshots, hold, clients):
    streams = stream_features(hold, seasonal=True)
    mapper = FamilyEvidence().fit(fit, target)
    evidence, confidence = mapper.transform(streams)
    output, models = {}, {}
    for kind, seasonal in (("recurrence", False), ("hazard", False), ("hazard", True)):
        name = kind + ("_seasonal" if seasonal else "")
        model = RecurrenceModel(kind, seasonal).fit(snapshots)
        models[name] = model
        p = model.predict(streams)
        for mode in ("competing", "support", "pooled"):
            output[f"{name}_{mode}"] = aggregate(streams, p, evidence, confidence, clients, mode)
    output["median_gap"] = aggregate(
        streams, median_gap_probabilities(streams), evidence, confidence, clients
    )
    # Export stream-level primary predictions, including the explicit active assumption.
    audit = streams.copy()
    audit["P_active"] = 1.0
    p = models["hazard_seasonal"].predict(streams)
    audit["P_event_within_90"] = 1 - p[:, 3]
    audit["identity_confidence"] = confidence
    for i, family in enumerate(evidence.T):
        audit[f"family_{[c for c in LABELS if c != 'none'][i]}"] = family
    return output, models, audit


def fingerprints(data):
    paths = [
        Path(__file__),
        *Path("src/transaction_forecasting/ubs").rglob("*.py"),
        data / "train_transactions.jsonl",
        data / "train_labels.csv",
    ]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["oof", "valid"], default="oof")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/metrics/ubs_v3"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v4_survival_laura")
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stamp = fingerprints(args.data_dir)
    train = read_transactions(args.data_dir / "train_transactions.jsonl")
    labels = read_labels(args.data_dir / "train_labels.csv")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    cutoffs = pd.date_range("2024-04-01", "2025-07-01", freq="QS", tz="UTC")
    snapshot_fingerprint = {
        k: v
        for k, v in stamp.items()
        if k.startswith("src/") or k.endswith("train_transactions.jsonl")
    }
    snapshot_path = out / "snapshots.parquet"
    snapshot_stamp = out / "snapshot_fingerprints.json"
    if (
        snapshot_path.exists()
        and snapshot_stamp.exists()
        and json.loads(snapshot_stamp.read_text()) == snapshot_fingerprint
    ):
        snapshots = pd.read_parquet(snapshot_path)
    else:
        snapshots = pseudo_cutoffs(train, cutoffs, seasonal=True)
        snapshots.to_parquet(snapshot_path, index=False)
        write(snapshot_stamp, snapshot_fingerprint)
    if args.phase == "oof":
        if (out / "frozen.json").exists():
            raise ValueError("Frozen experiment exists; use a new directory")
        all_predictions, diagnostics, fold_scores = {}, [], []
        for fold, (a, b) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=42).split(target.index, target), 1
        ):
            print(f"Fold {fold}/5", flush=True)
            fit_ids, hold_ids = target.index[a], target.index[b]
            fit, hold = (
                train.loc[train.client_id.isin(fit_ids)],
                train.loc[train.client_id.isin(hold_ids)],
            )
            fit_snap = snapshots.loc[snapshots.client_id.isin(fit_ids)]
            hold_snap = snapshots.loc[snapshots.client_id.isin(hold_ids)]
            variants, models, audit = predict_variants(
                fit, target.loc[fit_ids], fit_snap, hold, hold_ids
            )
            audit.to_parquet(out / f"fold_{fold}_streams.parquet", index=False)
            variants["baseline"] = baseline(
                fit,
                labels.loc[labels.client_id.isin(fit_ids)],
                hold,
                args.baseline_dir,
                f"fold_{fold}",
            )
            # Predeclared blends, no adaptive weights or NONE thresholds.
            for name, p in list(variants.items()):
                if name != "baseline":
                    variants[f"hybrid_{name}"] = 0.75 * variants["baseline"] + 0.25 * p
            fold_report = {}
            for name, p in variants.items():
                p.to_csv(out / f"fold_{fold}_{name}.csv")
                all_predictions.setdefault(name, []).append(p)
                fold_report[name] = metrics(target.loc[hold_ids], p)["macro_f1"]
            fold_scores.append(fold_report)
            for name, model in models.items():
                p = model.predict(hold_snap)
                diagnostics.append(dict(fold=fold, model=name, **stream_metrics(hold_snap, p)))
            print(json.dumps(fold_report), flush=True)
        scores = {}
        for name, blocks in all_predictions.items():
            p = pd.concat(blocks).reindex(target.index)
            p.to_csv(out / f"oof_{name}_probabilities.csv")
            scores[name] = metrics(target, p)
        selected = max(scores, key=lambda name: scores[name]["macro_f1"])
        write(
            out / "oof_metrics.json", dict(metrics=scores, folds=fold_scores, streams=diagnostics)
        )
        write(
            out / "frozen.json",
            dict(
                selected=selected,
                fingerprints=stamp,
                cutoffs=[str(c) for c in cutoffs],
                class_order=list(LABELS),
                criterion="TRAIN OOF macro-F1; fixed .75 baseline/.25 temporal hybrids",
                high_confidence_threshold=0.8,
                validation_used=False,
            ),
        )
        print(f"Frozen: {selected}", flush=True)
    else:
        frozen = json.loads((out / "frozen.json").read_text())
        if frozen["fingerprints"] != stamp:
            raise ValueError("Source/TRAIN changed after freeze")
        # Exclusive marker prevents repeated VALID access, including interrupted runs.
        with (out / "valid_started.json").open("x") as handle:
            json.dump({"frozen": frozen["selected"]}, handle)
        valid = read_transactions(args.data_dir / "valid_transactions.jsonl")
        if set(valid.client_id) & set(train.client_id):
            raise ValueError("TRAIN/VALID overlap")
        clients = sorted(valid.client_id.unique())
        variants, _, audit = predict_variants(train, target, snapshots, valid, clients)
        audit.to_parquet(out / "valid_streams.parquet", index=False)
        variants["baseline"] = baseline(train, labels, valid, args.baseline_dir, "valid")
        for name, p in list(variants.items()):
            if name != "baseline":
                variants[f"hybrid_{name}"] = 0.75 * variants["baseline"] + 0.25 * p
        for name, p in variants.items():
            p.to_csv(out / f"valid_{name}_probabilities.csv")
        truth = read_labels(args.data_dir / "valid_labels.csv").set_index("client_id")[
            TARGET_COLUMN
        ]
        write(out / "valid_metrics.json", {name: metrics(truth, p) for name, p in variants.items()})
        predicted = variants[frozen["selected"]].idxmax(axis=1).reindex(truth.index)
        errors = pd.DataFrame({"truth": truth, "predicted": predicted})
        errors["error"] = np.select(
            [
                ~errors.index.isin(audit.client_id),
                predicted.eq(truth),
                predicted.eq("none") | truth.eq("none"),
            ],
            ["no detected stream", "correct", "horizon or inactive (unidentifiable)"],
            default="wrong family",
        )
        errors.to_csv(out / "valid_errors.csv")
        print(
            json.dumps({k: metrics(truth, v)["macro_f1"] for k, v in variants.items()}), flush=True
        )


if __name__ == "__main__":
    main()
