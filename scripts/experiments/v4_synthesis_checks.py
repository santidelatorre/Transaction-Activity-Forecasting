"""Reproduce selected V4 controls from audited, pinned source snapshots.

Run audit_v4_branches.py --snapshots first. This script never reads VALID/TEST
labels. It does not change branch recipes or search for hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import transaction_forecasting.ubs as ubs
from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/metrics/v4_synthesis"
DATA = ROOT / "data/raw/ubs_2026"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def checked_probabilities(frame, target):
    if not frame.index.is_unique or set(frame.index) != set(target.index):
        raise ValueError("OOF client mismatch")
    if list(frame.columns) != list(LABELS):
        raise ValueError("Class order mismatch")
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1):
        raise ValueError("Invalid probabilities")
    return frame.reindex(target.index)


def snapshot(name, runner):
    source = OUT / "snapshots" / name
    ubs.__path__.append(str(source / "src/transaction_forecasting/ubs"))
    spec = importlib.util.spec_from_file_location(name + "_runner", source / runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint_matches(source, expected):
    raw = source.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {
        hashlib.sha256(v).hexdigest() for v in (raw, lf, lf.replace(b"\n", b"\r\n"))
    }


def check_ranker(labels):
    module = snapshot("santiago", "scripts/experiments/v4_soft_candidate_santiago.py")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    saved = ROOT / "outputs/metrics/v4_soft_candidate_santiago"
    provenance = json.loads((saved / "provenance.json").read_text())
    hash_checks = {}
    for name, expected in provenance["fingerprints"].items():
        relative = name.split("Transaction Activity Forecasting/")[-1]
        source = (
            ROOT / relative
            if relative.startswith("data/")
            else OUT / "snapshots/santiago" / relative
        )
        hash_checks[relative] = fingerprint_matches(source, expected)
    mismatches = [name for name, matches in hash_checks.items() if not matches]
    if mismatches not in ([], ["scripts/experiments/v4_soft_candidate_santiago.py"]):
        raise ValueError(f"Stale candidate evidence: {hash_checks}")
    regenerated_max_delta = None
    reports, reproduced = [], []
    for fold, (fit_ids, hold_ids) in enumerate(module.client_folds(target), 1):
        inner = module.combine_batches(
            [module.load_batch(saved, f"fold_{fold}_inner_{n}") for n in range(1, 4)]
        )
        hold = module.load_batch(saved, f"fold_{fold}_hold")
        for source in inner.sources:
            if (source.fit_clients | source.prediction_clients) & set(hold_ids):
                raise ValueError("Outer label contamination")
        if set(inner.features.index.get_level_values(0)) != set(fit_ids):
            raise ValueError("Inner clients differ from fold")
        if hold.sources != (module.SourcePartition(frozenset(fit_ids), frozenset(hold_ids)),):
            raise ValueError("Wrong outer source")
        if fold == 1:
            # Independently regenerate a complete 400-client evidence block. The
            # archived runner hash differs; never silently claim it matched.
            train = read_transactions(DATA / "train_transactions.jsonl")
            provider = module.EvidenceProvider().fit(
                train.loc[train.client_id.isin(fit_ids)],
                labels.loc[labels.client_id.isin(fit_ids)],
            )
            fresh = provider.transform(train.loc[train.client_id.isin(hold_ids)])
            pd.testing.assert_frame_equal(fresh.features, hold.features, atol=1e-12, rtol=0)
            regenerated_max_delta = float(np.abs(fresh.features - hold.features).to_numpy().max())
        model = module.SoftCandidateRanker("catboost").fit(inner, target.loc[fit_ids])
        p = checked_probabilities(model.predict_proba(hold), target.loc[hold_ids])
        expected = pd.read_csv(
            saved / f"fold_{fold}_catboost_probabilities.csv", index_col="client_id"
        )
        delta = float(np.abs(p - expected.reindex(p.index)).to_numpy().max())
        np.testing.assert_allclose(p, expected.reindex(p.index), atol=1e-12, rtol=0)
        reports.append(
            {
                "fold": fold,
                "max_probability_delta": delta,
                **evaluate_predictions(target.loc[hold_ids], p.idxmax(axis=1)),
            }
        )
        reproduced.append(p)
        print("ranker fold", fold, reports[-1]["macro_f1"], flush=True)
    p = checked_probabilities(pd.concat(reproduced), target)
    p.to_csv(OUT / "santiago_oof_probabilities.csv", index_label="client_id")
    baseline = checked_probabilities(
        pd.read_csv(saved / "train_oof_baseline_probabilities.csv", index_col="client_id"), target
    )
    # Fixed equal average, no weight search and no VALID access.
    averaged = (baseline + p) / 2
    averaged.to_csv(OUT / "equal_average_oof_probabilities.csv", index_label="client_id")
    result = {
        "fingerprints": hash_checks,
        "folds": reports,
        "runner_hash_mismatch": bool(mismatches),
        "regenerated_fold_1_features_max_delta": regenerated_max_delta,
        "ranker": evaluate_predictions(target, p.idxmax(axis=1)),
        "baseline": evaluate_predictions(target, baseline.idxmax(axis=1)),
        "equal_average": evaluate_predictions(target, averaged.idxmax(axis=1)),
        "refit_scope": "ranker heads; hashed nested evidence reused",
        "valid_labels_read": False,
    }
    write(OUT / "ranker_reproduction.json", result)
    return result


def check_direct(labels, train):
    module = snapshot("javi", "scripts/experiments/v4_direct_robust_javi.py")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    histories = module.client_histories(train, target.index)
    views = module.VIEWS
    slots = {v: np.full((len(target), 8), np.nan) for v in views}
    folds = module.make_folds(target)
    for fold, (fit, hold) in enumerate(folds, 1):
        fit_rows, hold_rows = [histories[i] for i in fit], [histories[i] for i in hold]
        corruptor = module.LocalCorruptor().fit(fit_rows)
        features = module.DirectFeatures().fit(fit_rows)
        matrices = {v: features.transform(corruptor.transform(fit_rows, v), "R4") for v in views}
        model = module.fit_model(
            "ComplementNB", matrices, target.iloc[fit].to_numpy(), target.index[fit], "normalized"
        )
        for view in views:
            matrix = features.transform(corruptor.transform(hold_rows, view), "R4")
            slots[view][hold] = module.ordered_probabilities(model, matrix)
        print("direct fold", fold, flush=True)
    for view in views:
        frame = checked_probabilities(
            pd.DataFrame(slots[view], index=target.index, columns=LABELS), target
        )
        frame.to_csv(OUT / f"javi_{view}_oof_probabilities.csv", index_label="client_id")
    result = module.diagnostics(target, slots, folds)
    result["recipe"] = "Original frozen ComplementNB/R4/normalized; no rerun of 20-arm selection"
    write(OUT / "direct_reproduction.json", result)
    return result


def check_svd(labels, train):
    snapshot("esteban", "scripts/experiments/v4_selfsupervised_esteban.py")
    from transaction_forecasting.ubs.representation import (
        DenoisingEncoder,
        client_grouped_oof,
        label_free_client_stats,
    )

    unlabeled = read_transactions(DATA / "unlabeled_pretrain_transactions.jsonl")
    if set(unlabeled.client_id) & set(train.client_id):
        raise ValueError("Unlabeled/TRAIN overlap")
    sample = unlabeled.sample(n=min(80000, len(unlabeled)), random_state=42)
    encoder = DenoisingEncoder(seed=42).fit(pd.concat([sample, train], ignore_index=True))
    features = encoder.client_features(train, mode="clean").join(label_free_client_stats(train))
    target = labels.set_index("client_id")[TARGET_COLUMN]
    p = checked_probabilities(client_grouped_oof(features, target, seed=42), target)
    p.to_csv(OUT / "esteban_oof_probabilities.csv", index_label="client_id")
    result = {
        "metrics": evaluate_predictions(target, p.idxmax(axis=1)),
        "fit_descriptions": encoder.fit_descriptions_,
        "components": encoder.n_components_,
        "transductive": True,
        "valid_labels_read": False,
        "recipe": "Original frozen clean_embed_plus_stats; no model reselection",
    }
    write(OUT / "svd_reproduction.json", result)
    return result


def check_survival(labels, train):
    module = snapshot("laura", "scripts/experiments/v4_survival_laura.py")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    snapshots = module.pseudo_cutoffs(
        train, pd.date_range("2024-04-01", "2025-07-01", freq="QS", tz="UTC"), seasonal=True
    )
    np.testing.assert_allclose(snapshots.groupby("client_id").weight.sum(), 1)
    blocks, folds, diagnostics = {}, [], []
    for fold, (a, b) in enumerate(
        module.StratifiedKFold(5, shuffle=True, random_state=42).split(target.index, target), 1
    ):
        fit_ids, hold_ids = target.index[a], target.index[b]
        fit_snap = snapshots.loc[snapshots.client_id.isin(fit_ids)]
        hold_snap = snapshots.loc[snapshots.client_id.isin(hold_ids)]
        variants, models, _ = module.predict_variants(
            train.loc[train.client_id.isin(fit_ids)],
            target.loc[fit_ids],
            fit_snap,
            train.loc[train.client_id.isin(hold_ids)],
            hold_ids,
        )
        baseline = checked_probabilities(
            pd.read_csv(
                ROOT / f"outputs/metrics/ubs_v3a_baseline/fold_{fold}_A_probabilities.csv",
                index_col="client_id",
            ),
            target.loc[hold_ids],
        )
        for name, p in list(variants.items()):
            variants[f"hybrid_{name}"] = 0.75 * baseline + 0.25 * p
        variants["baseline"] = baseline
        reports = {}
        for name, p in variants.items():
            p = checked_probabilities(p, target.loc[hold_ids])
            blocks.setdefault(name, []).append(p)
            reports[name] = evaluate_predictions(target.loc[hold_ids], p.idxmax(axis=1))
        folds.append(reports)
        for name, model in models.items():
            diagnostics.append(
                {
                    "fold": fold,
                    "model": name,
                    **module.stream_metrics(hold_snap, model.predict(hold_snap)),
                }
            )
        print("survival fold", fold, reports["hazard_support"]["macro_f1"], flush=True)
    results = {}
    directory = OUT / "survival"
    directory.mkdir(exist_ok=True)
    for name, pieces in blocks.items():
        p = checked_probabilities(pd.concat(pieces), target)
        p.to_csv(directory / f"{name}_oof_probabilities.csv", index_label="client_id")
        results[name] = evaluate_predictions(target, p.idxmax(axis=1))
    result = {
        "metrics": results,
        "folds": folds,
        "stream_diagnostics": diagnostics,
        "snapshots": len(snapshots),
        "clients": int(snapshots.client_id.nunique()),
        "cutoffs": {str(k): int(v) for k, v in snapshots.groupby("cutoff").size().items()},
        "valid_labels_read": False,
        "baseline": "Existing baseline OOF cache; exact fold IDs checked",
    }
    write(OUT / "survival_reproduction.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("ranker", "direct", "svd", "survival"))
    args = parser.parse_args()
    started = perf_counter()
    labels = read_labels(DATA / "train_labels.csv")
    if args.check == "ranker":
        result = check_ranker(labels)
    else:
        train = read_transactions(DATA / "train_transactions.jsonl")
        result = {"direct": check_direct, "svd": check_svd, "survival": check_survival}[args.check](
            labels, train
        )
    write(OUT / f"{args.check}_runtime.json", {"seconds": perf_counter() - started})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
