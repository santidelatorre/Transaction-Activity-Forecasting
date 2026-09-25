"""Nested TRAIN evaluation, freeze, then one-shot VALID for the soft candidate ranker.

Run from repository root with the package installed (or PYTHONPATH=src).
No test partition or submission path is used. Outputs and caches stay local.
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

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.soft_candidate_ranker import (
    CandidateBatch,
    EvidenceProvider,
    SoftCandidateRanker,
    SourcePartition,
    client_folds,
    client_target,
    combine_batches,
    degrade_history,
    raw_family_probabilities,
)

BASE_BRANCH = "origin/baseline/v4-frozen"
BASE_SHA = "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"
BASE_MACRO_F1 = 0.424111097737
KINDS = ("linear", "catboost")
POLICY = (
    "CatBoost only if pooled OOF Macro-F1 >= linear + 0.005, wins >=4/5 folds, "
    "corruption Macro-F1 >= linear - 0.005 and none F1 >= linear - 0.01; otherwise linear. "
    "Never select using VALID, tune prevalence, or automatically replace the frozen baseline."
)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(data):
    # In particular, do not open/hash VALID labels before freeze.
    paths = [Path(__file__), *sorted(Path("src/transaction_forecasting/ubs").rglob("*.py"))]
    paths += [Path("src/transaction_forecasting/evaluation/official.py")]
    paths += [data / "train_transactions.jsonl", data / "train_labels.csv"]
    return {str(p).replace("\\", "/"): digest(p) for p in paths}


def save_batch(directory, name, batch):
    batch.validate().features.to_parquet(directory / f"{name}.parquet")
    write_json(
        directory / f"{name}_sources.json",
        [
            {"fit": sorted(source.fit_clients), "predict": sorted(source.prediction_clients)}
            for source in batch.sources
        ],
    )


def load_batch(directory, name):
    sources = json.loads((directory / f"{name}_sources.json").read_text())
    return CandidateBatch(
        pd.read_parquet(directory / f"{name}.parquet"),
        tuple(SourcePartition(frozenset(s["fit"]), frozenset(s["predict"])) for s in sources),
    ).validate()


def cached_inner_oof(out, name, transactions, labels):
    target = client_target(transactions, labels)
    batches = []
    for number, (fit_ids, hold_ids) in enumerate(client_folds(target, 3), 1):
        key = f"{name}_inner_{number}"
        if (out / f"{key}_sources.json").exists():
            batch = load_batch(out, key)
        else:
            print(f"{key}: fitting disjoint baseline and mapping", flush=True)
            provider = EvidenceProvider().fit(
                transactions.loc[transactions.client_id.isin(fit_ids)],
                labels.loc[labels.client_id.isin(fit_ids)],
            )
            batch = provider.transform(transactions.loc[transactions.client_id.isin(hold_ids)])
            save_batch(out, key, batch)
        if batch.sources != (SourcePartition(frozenset(fit_ids), frozenset(hold_ids)),):
            raise ValueError("Cached inner fold provenance differs from requested partition")
        batches.append(batch)
    return combine_batches(batches)


def metrics(target, probabilities):
    if not probabilities.index.is_unique or set(probabilities.index) != set(target.index):
        raise ValueError("Probability clients must exactly match target")
    probabilities = probabilities.reindex(target.index)
    if list(probabilities.columns) != list(LABELS):
        raise ValueError("Invalid class order")
    values = probabilities.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
        raise ValueError("Invalid normalized probabilities")
    prediction = probabilities.idxmax(axis=1)
    report = evaluate_predictions(target, prediction)
    top2 = np.asarray(LABELS)[np.argsort(-values, axis=1, kind="stable")[:, :2]]
    report["rank_accuracy"] = report["accuracy"]
    report["top_2_recall"] = float((top2 == target.to_numpy()[:, None]).any(axis=1).mean())
    report["none_false_positives"] = int((prediction.eq("none") & target.ne("none")).sum())
    report["none_false_negatives"] = int((prediction.ne("none") & target.eq("none")).sum())
    return report


def diagnostics(target, predictions, batch):
    context = batch.features.xs("none", level="candidate").reindex(target.index)
    masks = {
        "no_family_evidence": context.evidence_family_count.eq(0),
        "single_family_evidence": context.evidence_family_count.eq(1),
        "multiple_family_evidence": context.evidence_family_count.gt(1),
        "high_textual_specificity": context.textual_specificity.ge(np.log(3)),
        "low_textual_specificity": context.textual_specificity.lt(np.log(3)),
        "high_recurrence_confidence": context.strongest_recurrence.ge(1),
        "low_recurrence_confidence": context.strongest_recurrence.lt(1),
        "ambiguous_positive_clients": context.evidence_family_count.gt(1) & target.ne("none"),
    }
    coverage = batch.wide("mapped_event_count").reindex(target.index).gt(0)
    covered = np.array([coverage.at[client, label] for client, label in target.items()])
    positive = target.ne("none").to_numpy()
    segments = {}
    for name, mask in masks.items():
        ids = target.index[mask]
        segments[name] = {
            "clients": len(ids),
            "metrics": {
                name: metrics(target.loc[ids], p.loc[ids]) for name, p in predictions.items()
            }
            if len(ids)
            else {},
        }
    baseline = predictions["baseline"].reindex(target.index).idxmax(axis=1)
    complementary = {}
    for name, probabilities in predictions.items():
        guess = probabilities.reindex(target.index).idxmax(axis=1)
        complementary[name] = {
            "candidate_only_correct": int((guess.eq(target) & baseline.ne(target)).sum()),
            "baseline_only_correct": int((guess.ne(target) & baseline.eq(target)).sum()),
            "both_wrong": int((guess.ne(target) & baseline.ne(target)).sum()),
            "disagreement": int(guess.ne(baseline).sum()),
        }
    return {
        "candidate_availability": 1.0,
        "positive_target_evidence_coverage": float(covered[positive].mean())
        if positive.any()
        else None,
        "any_family_evidence_coverage": float(context.evidence_family_count.gt(0).mean()),
        "segments": segments,
        "complementary_errors": complementary,
    }


def probabilities_for(batch, models):
    return {
        "baseline": batch.wide("baseline_probability"),
        "raw_family": raw_family_probabilities(batch),
        **{name: model.predict_proba(batch) for name, model in models.items()},
    }


def save_predictions(out, prefix, predictions, models=None, batch=None):
    for name, probabilities in predictions.items():
        probabilities.to_csv(out / f"{prefix}_{name}_probabilities.csv", index_label="client_id")
    pd.DataFrame({name: p.idxmax(axis=1) for name, p in predictions.items()}).to_csv(
        out / f"{prefix}_predictions.csv", index_label="client_id"
    )
    if models is not None:
        for name, model in models.items():
            model.predict_scores(batch).to_csv(out / f"{prefix}_{name}_scores.csv")


def run_oof(train, labels, out, stamps):
    target = client_target(train, labels)
    clean, corrupt, batches, fold_reports = [], [], [], []
    for number, (fit_ids, hold_ids) in enumerate(client_folds(target), 1):
        name = f"fold_{number}"
        fit = train.loc[train.client_id.isin(fit_ids)]
        hold = train.loc[train.client_id.isin(hold_ids)]
        fit_labels = labels.loc[labels.client_id.isin(fit_ids)]
        inner = cached_inner_oof(out, name, fit, fit_labels)
        # No outer validation label may enter any inner feature generator.
        for source in inner.sources:
            if (source.fit_clients | source.prediction_clients) & set(hold_ids):
                raise ValueError("Outer fold leakage into inner baseline/mapping")
        keys = [f"{name}_hold", f"{name}_corrupt"]
        if all((out / f"{key}_sources.json").exists() for key in keys):
            batch, degraded = (load_batch(out, key) for key in keys)
        else:
            print(f"{name}: fitting outer baseline and mapping", flush=True)
            provider = EvidenceProvider().fit(fit, fit_labels)
            batch = provider.transform(hold)
            degraded = provider.transform(degrade_history(hold))
            for key, value in zip(keys, (batch, degraded), strict=True):
                save_batch(out, key, value)
        expected_source = (SourcePartition(frozenset(fit_ids), frozenset(hold_ids)),)
        if batch.sources != expected_source or degraded.sources != expected_source:
            raise ValueError("Outer baseline provenance mismatch")
        models = {kind: SoftCandidateRanker(kind).fit(inner, target.loc[fit_ids]) for kind in KINDS}
        predictions = probabilities_for(batch, models)
        damaged = probabilities_for(degraded, models)
        save_predictions(out, name, predictions, models, batch)
        save_predictions(out, f"{name}_corrupt", damaged)
        fold_reports.append(
            {key: metrics(target.loc[hold_ids], p) for key, p in predictions.items()}
        )
        print(
            json.dumps({"fold": number, **{k: v["macro_f1"] for k, v in fold_reports[-1].items()}}),
            flush=True,
        )
        clean.append(predictions)
        corrupt.append(damaged)
        batches.append(batch)
    merged = combine_batches(batches)
    save_batch(out, "train_oof_candidates", merged)
    oof = {name: pd.concat([p[name] for p in clean]).reindex(target.index) for name in clean[0]}
    damaged = {
        name: pd.concat([p[name] for p in corrupt]).reindex(target.index) for name in clean[0]
    }
    result = {name: metrics(target, p) for name, p in oof.items()}
    stress = {name: metrics(target, p) for name, p in damaged.items()}
    wins = sum(f["catboost"]["macro_f1"] > f["linear"]["macro_f1"] for f in fold_reports)
    use_catboost = (
        result["catboost"]["macro_f1"] >= result["linear"]["macro_f1"] + 0.005
        and wins >= 4
        and stress["catboost"]["macro_f1"] >= stress["linear"]["macro_f1"] - 0.005
        and result["catboost"]["per_class"]["none"]["f1-score"]
        >= result["linear"]["per_class"]["none"]["f1-score"] - 0.01
    )
    selected = "catboost" if use_catboost else "linear"
    save_predictions(out, "train_oof", oof)
    save_predictions(out, "train_oof_corrupt", damaged)
    for kind in KINDS:
        pd.concat(
            [
                pd.read_csv(out / f"fold_{i}_{kind}_scores.csv", index_col="client_id")
                for i in range(1, 6)
            ]
        ).reindex(target.index).to_csv(out / f"train_oof_{kind}_scores.csv")
    oof[selected].to_csv(out / "train_oof_probabilities.csv", index_label="client_id")
    oof[selected].idxmax(axis=1).rename("predicted_next_recurring_merchant").to_csv(
        out / "train_oof_selected_predictions.csv", index_label="client_id"
    )
    summary = {
        "base_branch": BASE_BRANCH,
        "base_sha": BASE_SHA,
        "base_valid_macro_f1": BASE_MACRO_F1,
        "selected": selected,
        "selection_policy": POLICY,
        "catboost_fold_wins": wins,
        "class_order": list(LABELS),
        "outer_folds": 5,
        "inner_folds": 3,
        "baseline_mapping_folds": 5,
        "metrics": result,
        "folds": fold_reports,
        "corruption": {
            "method": "25% event thinning; first event retained; seed 2026",
            "metrics": stress,
        },
        "diagnostics": diagnostics(target, oof, merged),
        "validation_independent": False,
        "valid_labels_used_for_selection": False,
        "normalization": "pointwise binary raw logits -> per-client softmax, temperature=1",
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "frozen_selection.json",
        {
            "selected": selected,
            "policy": POLICY,
            "fingerprints": stamps,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "valid_labels_used": False,
            "oof_summary_sha256": digest(out / "summary.json"),
        },
    )
    print(
        json.dumps(
            {
                "frozen": selected,
                "oof": {name: report["macro_f1"] for name, report in result.items()},
                "catboost_fold_wins": wins,
            }
        ),
        flush=True,
    )


def run_valid(train, labels, data, out, stamps):
    frozen = json.loads((out / "frozen_selection.json").read_text())
    if (
        frozen["fingerprints"] != stamps
        or frozen["policy"] != POLICY
        or frozen["valid_labels_used"]
    ):
        raise ValueError("VALID requires an unchanged TRAIN-only freeze")
    if frozen["oof_summary_sha256"] != digest(out / "summary.json"):
        raise ValueError("OOF summary changed after freeze")
    if (out / "valid_results.json").exists():
        raise ValueError("VALID already evaluated in this output directory")
    train_oof = load_batch(out, "train_oof_candidates")
    target = client_target(train, labels)
    selected = frozen["selected"]
    ranker = SoftCandidateRanker(selected).fit(train_oof, target)
    print(f"Frozen {selected}; fitting full TRAIN baseline/mapping for VALID", flush=True)
    provider = EvidenceProvider().fit(train, labels)
    valid = read_transactions(data / "valid_transactions.jsonl")
    batch = provider.transform(valid)
    save_batch(out, "valid_candidates", batch)
    predictions = probabilities_for(batch, {selected: ranker})
    save_predictions(out, "valid", predictions, {selected: ranker}, batch)
    predictions[selected].to_csv(out / "valid_probabilities.csv", index_label="client_id")
    predictions[selected].idxmax(axis=1).rename("predicted_next_recurring_merchant").to_csv(
        out / "valid_selected_predictions.csv", index_label="client_id"
    )
    # First and only read of VALID labels: all modeling and predictions are complete.
    valid_labels = read_labels(data / "valid_labels.csv")
    truth = client_target(valid, valid_labels)
    report = {
        "selected": selected,
        "frozen_at_utc": frozen["frozen_at_utc"],
        "metrics": {name: metrics(truth, p) for name, p in predictions.items()},
        "diagnostics": diagnostics(truth, predictions, batch),
        "data_fingerprints": {
            name: digest(data / name) for name in ("valid_transactions.jsonl", "valid_labels.csv")
        },
        "valid_labels_used_for_selection": False,
        "validation_independent": False,
    }
    write_json(out / "valid_results.json", report)
    summary = json.loads((out / "summary.json").read_text())
    write_json(out / "final_summary.json", {**summary, "valid": report})
    print(
        json.dumps({"valid": {name: m["macro_f1"] for name, m in report["metrics"].items()}}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v4_soft_candidate_santiago")
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stamps = fingerprint(args.data_dir)
    provenance = out / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text())["fingerprints"] != stamps:
        raise ValueError("Source/TRAIN data changed; choose a fresh output directory")
    write_json(
        provenance,
        {
            "fingerprints": stamps,
            "policy": POLICY,
            "python": platform.python_version(),
            "versions": {
                name: version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost")
            },
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], text=True
            ).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    )
    train = read_transactions(args.data_dir / "train_transactions.jsonl")
    labels = read_labels(args.data_dir / "train_labels.csv")
    if args.phase in ("oof", "all"):
        if (out / "frozen_selection.json").exists():
            print("Existing immutable TRAIN freeze; skipping selection", flush=True)
        else:
            run_oof(train, labels, out, stamps)
    if args.phase in ("valid", "all"):
        run_valid(train, labels, args.data_dir, out, stamps)


if __name__ == "__main__":
    main()
