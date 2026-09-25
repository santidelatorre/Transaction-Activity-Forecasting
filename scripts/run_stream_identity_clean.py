"""Auditable stages: prepare/develop/stability/freeze/predict/evaluate/report.

Only evaluate can open VALID labels; a durable exclusive receipt prevents reuse.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from ubs_recurrence.clean_protocol import (
    RAW_COLUMNS,
    SCENARIOS,
    SEEDS,
    ComponentTrainer,
    Recipe,
    candidate_index,
    candidates,
    category_schema,
    client_folds,
    compact_matrix,
    corruption_view,
    digest,
    feature_banks,
    file_hash,
    legacy_parameters,
    project_schema,
    verify_frozen,
    write_json,
)
from ubs_recurrence.data import LABELS, PREDICTION, ROOT, aligned_target, transactions
from ubs_recurrence.evaluation import metrics
from ubs_recurrence.features import GENERIC, MCC, PATTERNS
from ubs_recurrence.model import none_features, parameters
from ubs_recurrence.official import score_predictions, validate_submission
from ubs_recurrence.price_prior import learn_price_profiles
from ubs_recurrence.templates import TEMPLATES

OUT = ROOT / "outputs/stream_identity_clean"
RAW = ROOT / "data/raw/ubs_2026"
CONFIG = ROOT / "configs/stream_identity_clean_frozen.json"
REPORT = ROOT / "reports/stream_identity_clean_protocol.md"
HISTORICAL = 0.6194934236214421
V2 = 0.391549456


def now():
    return datetime.now(UTC).isoformat()


def log(message):
    print(now(), message, flush=True)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_hashes():
    paths = sorted((ROOT / "src/ubs_recurrence").glob("*.py"))
    paths += [Path(__file__).resolve()]
    return {p.relative_to(ROOT).as_posix(): source_hash(p) for p in paths}


def source_hash(path):
    # Git checkout line endings must not invalidate unchanged frozen Python code.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "xgboost",
                "pyarrow",
                "pytest",
                "ruff",
                "joblib",
                "psutil",
            )
        },
        "device": "cpu",
        "threads_per_estimator": 4,
    }


def memory():
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1024**2


def save_predictions(path, ids, p, folds=None):
    if len(set(ids)) != len(ids) or p.shape != (len(ids), 8):
        raise ValueError("Probability alignment failure")
    frame = pd.DataFrame(p, columns=["p_" + label for label in LABELS])
    frame.insert(0, "client_id", ids)
    if folds is not None:
        frame["fold"] = folds
    frame[PREDICTION] = np.array(LABELS)[p.argmax(axis=1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def assessed(y, p, fold_ids=None):
    result = metrics(y, p)
    result["mean_entropy"] = float(
        -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1).mean()
    )
    result["mean_confidence"] = float(p.max(axis=1).mean())
    result["brier_multiclass"] = float(np.square(p - np.eye(8)[y]).sum(axis=1).mean())
    if fold_ids is not None:
        scores = [
            metrics(y[fold_ids == f], p[fold_ids == f])["macro_f1"] for f in range(5)
        ]
        result["fold_macro_f1"] = scores
        result["fold_mean"] = float(np.mean(scores))
        result["fold_std"] = float(np.std(scores))
    return result


def assert_development():
    if CONFIG.exists() or (OUT / "valid_access_receipt.json").exists():
        raise ValueError("Development is closed after freezing")


def prepare(*, frozen=False):
    if frozen:
        envelope = frozen_checked()
        if (OUT / "train_cache.joblib").exists() or (
            OUT / "valid_access_receipt.json"
        ).exists():
            raise ValueError("Frozen rebuild requires a fresh output directory")
    else:
        assert_development()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if not (OUT / "initial_state.json").exists():
        preserved = list((ROOT / "reports").glob("*stream_identity*"))
        preserved += [
            ROOT / "reports/final_metrics.json",
            ROOT / "reports/final_candidate_config.json",
            ROOT / "outputs/predictions/submission_stream_identity.csv",
        ]
        snapshot = {
            "created_at": now(),
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain=v1"),
            "historical_f1": HISTORICAL,
            "preserved_artifacts": {
                p.relative_to(ROOT).as_posix(): file_hash(p)
                for p in preserved
                if p.exists() and p != REPORT
            },
        }
        write_json(OUT / "initial_state.json", snapshot, exclusive=True)
    data = transactions("train", use_cache=False, data_dir=RAW)
    ids = np.sort(data.client_id.unique())
    y = aligned_target(ids, data_dir=RAW)
    profiles = (
        envelope["configuration"]["profiles"]
        if frozen
        else learn_price_profiles(use_cache=False, data_dir=RAW)
    )
    write_json(OUT / "profiles.json", profiles)
    log(f"TRAIN {len(ids)} clients; fitting no VALID covariates or labels")
    raw_views, banks = {}, {}
    for scenario in SCENARIOS:
        raw_views[scenario] = corruption_view(data, scenario)
        banks[scenario] = feature_banks(raw_views[scenario], profiles)
        log(
            f"Built {scenario} banks: {banks[scenario]['full']['compact'].shape}; peak {memory():.0f} MiB"
        )
    joblib.dump(
        {"ids": ids, "y": y, "raw": raw_views, "banks": banks, "profiles": profiles},
        OUT / "train_cache.joblib",
        compress=3,
    )
    receipt = {
        "created_at": now(),
        "environment": environment(),
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_mib": memory(),
        "train_clients": len(ids),
        "train_transactions": len(data),
        "data_hashes": {
            name: file_hash(RAW / name)
            for name in (
                "train_transactions.jsonl",
                "train_labels.csv",
                "unlabeled_pretrain_transactions.jsonl",
            )
        },
        "cache_sha256": file_hash(OUT / "train_cache.joblib"),
        "source_hashes": source_hashes(),
        "source_hash_kind": "utf8_universal_newlines",
        "candidate_catalog": [r.to_dict() for r in candidates()],
    }
    write_json(OUT / "prepared.json", receipt)


def rebuild():
    """Recreate only TRAIN features from an existing freeze, without selection."""
    prepare(frozen=True)


def load_cache():
    prepared = json.loads((OUT / "prepared.json").read_text())
    if file_hash(OUT / "train_cache.joblib") != prepared["cache_sha256"]:
        raise ValueError("Training cache was modified")
    # Feature source must be identical to preparation. Runner may gain tests/reporting.
    for path, sha in prepared["source_hashes"].items():
        hasher = (
            source_hash
            if prepared.get("source_hash_kind") == "utf8_universal_newlines"
            else file_hash
        )
        if path.startswith("src/") and hasher(ROOT / path) != sha:
            raise ValueError(f"Feature/model source changed since prepare: {path}")
    return joblib.load(OUT / "train_cache.joblib")


def run_cv(cache, recipes, split_seed, run_name, *, target=None):
    started = time.perf_counter()
    ids = cache["ids"]
    y = cache["y"] if target is None else target
    folder = OUT / run_name
    folder.mkdir(parents=True, exist_ok=True)
    # Every resume must have the identical data, folds, recipes and implementation.
    spec = {
        "seed": split_seed,
        "recipes": [r.to_dict() for r in recipes],
        "target_sha": digest(y.tolist()),
        "cache_sha": file_hash(OUT / "train_cache.joblib"),
        "source_hashes": source_hashes(),
    }
    signature = digest(spec)
    spec_path = folder / "specification.json"
    if spec_path.exists() and json.loads(spec_path.read_text())["sha256"] != signature:
        raise ValueError("Cannot resume a CV run with changed inputs/code")
    if not spec_path.exists():
        write_json(
            spec_path, {"sha256": signature, "specification": spec}, exclusive=True
        )
    folds = client_folds(ids, y, split_seed)
    predictions = {
        r.name: {s: np.zeros((len(ids), 8)) for s in SCENARIOS} for r in recipes
    }
    fold_ids = np.full(len(ids), -1)
    timings = {r.name: [] for r in recipes}
    for fold, (train, held) in enumerate(folds):
        fold_ids[held] = fold
        trainer = ComponentTrainer(cache["banks"], cache["raw"], ids[train], y[train])
        for recipe in recipes:
            path = folder / f"fold{fold}_{recipe.name}.joblib"
            if path.exists():
                artifact = joblib.load(path)
                if (
                    artifact["signature"] != signature
                    or artifact["held_ids"] != ids[held].tolist()
                ):
                    raise ValueError("Fold checkpoint mismatch")
            else:
                clock = time.perf_counter()
                model = trainer.fit(recipe)
                p = {
                    s: model.predict_proba(cache["banks"][s], ids[held])
                    for s in SCENARIOS
                }
                artifact = {
                    "signature": signature,
                    "held_ids": ids[held].tolist(),
                    "probabilities": p,
                    "seconds": time.perf_counter() - clock,
                    "features": list(model.models["rank"][0][1]),
                    "peak_rss_mib": memory(),
                }
                joblib.dump(artifact, path, compress=3)
                log(
                    f"{run_name} fold={fold + 1}/5 {recipe.name}: clean {assessed(y[held], p['original'])['macro_f1']:.5f}, {artifact['seconds']:.1f}s"
                )
            for scenario in SCENARIOS:
                predictions[recipe.name][scenario][held] = artifact["probabilities"][
                    scenario
                ]
            timings[recipe.name].append(artifact["seconds"])
    if (fold_ids < 0).any():
        raise ValueError("Incomplete OOF coverage")
    results = {}
    for recipe in recipes:
        results[recipe.name] = {
            "recipe": recipe.to_dict(),
            "runtime_seconds_incremental": sum(timings[recipe.name]),
            "scenarios": {},
        }
        for scenario, p in predictions[recipe.name].items():
            results[recipe.name]["scenarios"][scenario] = assessed(y, p, fold_ids)
            save_predictions(
                folder / f"{recipe.name}_{scenario}_oof.csv", ids, p, fold_ids
            )
    summary = {
        "split_seed": split_seed,
        "label_order": LABELS,
        "results": results,
        "runtime_seconds": sum(sum(values) for values in timings.values()),
        "current_invocation_seconds": time.perf_counter() - started,
        "peak_rss_mib": memory(),
    }
    write_json(folder / "summary.json", summary)
    return summary


def complexity(recipe):
    return len(recipe.seeds) * (1 + int(recipe.none_detector)) + 3 * int(
        recipe.legacy_weight > 0
    )


def develop():
    assert_development()
    catalog = candidates()
    summary = run_cv(load_cache(), catalog, 42, "development")
    winner = max(
        enumerate(catalog),
        key=lambda pair: (
            summary["results"][pair[1].name]["scenarios"]["original"]["macro_f1"],
            -complexity(pair[1]),
            -pair[0],
        ),
    )[1]
    write_json(
        OUT / "selected_train_only.json",
        {
            "selected_at": now(),
            "recipe": winner.to_dict(),
            "selection": "maximum original TRAIN OOF Macro-F1; exact tie: estimator count, catalog order",
            "oof_macro_f1": summary["results"][winner.name]["scenarios"]["original"][
                "macro_f1"
            ],
        },
    )
    log(f"TRAIN-only winner: {winner.name}")


def selected_recipe():
    fields = json.loads((OUT / "selected_train_only.json").read_text())["recipe"]
    fields["seeds"] = tuple(fields["seeds"])
    return Recipe(**fields)


def stability():
    assert_development()
    cache = load_cache()
    recipe = selected_recipe()
    base = json.loads((OUT / "development/summary.json").read_text())["results"][
        recipe.name
    ]
    scores = {"42": base["scenarios"]["original"]["macro_f1"]}
    # Same folds, change only model seed; every requested seed is retained/reported.
    singles = [
        replace(recipe, name=f"model_seed_{seed}", seeds=(seed,)) for seed in SEEDS
    ]
    permuted = np.random.default_rng(20260925).permutation(cache["y"])
    # Independent audits share immutable feature banks, with separate fold trainers
    # and output directories. This changes execution time, not the fixed recipe.
    with ThreadPoolExecutor(max_workers=2) as pool:
        partition_runs = {
            seed: pool.submit(run_cv, cache, [recipe], seed, f"stability_{seed}")
            for seed in (17, 2026)
        }
        single_run = pool.submit(run_cv, cache, singles, 42, "model_seed_stability")
        permutation_run = pool.submit(
            run_cv, cache, [recipe], 42, "target_permutation", target=permuted
        )
        invariant_run = pool.submit(invariance, cache)
        for seed, future in partition_runs.items():
            result = future.result()
            scores[str(seed)] = result["results"][recipe.name]["scenarios"]["original"][
                "macro_f1"
            ]
        model_results = single_run.result()
        permutation_result = permutation_run.result()
        invariant_run.result()
    model_scores = {
        str(seed): model_results["results"][f"model_seed_{seed}"]["scenarios"][
            "original"
        ]["macro_f1"]
        for seed in SEEDS
    }
    values = list(scores.values())
    result = {
        "partition_seed_macro_f1": scores,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": min(values),
        "max": max(values),
        "model_seed_macro_f1": model_scores,
        "model_seed_mean": float(np.mean(list(model_scores.values()))),
        "model_seed_std": float(np.std(list(model_scores.values()))),
    }
    write_json(OUT / "stability.json", result)
    sanity = {
        "permutation_seed": 20260925,
        "permuted_macro_f1": permutation_result["results"][recipe.name]["scenarios"][
            "original"
        ]["macro_f1"],
        "uniform_chance_accuracy": 0.125,
        "chance_scores": {},
    }
    for seed in SEEDS:
        random_p = np.eye(8)[
            np.random.default_rng(seed).integers(0, 8, len(cache["y"]))
        ]
        sanity["chance_scores"][str(seed)] = assessed(cache["y"], random_p)["macro_f1"]
    write_json(OUT / "permutation_sanity.json", sanity)


def freeze():
    assert_development()
    cache = load_cache()
    recipe = selected_recipe()
    for name in ("stability.json", "permutation_sanity.json", "invariance_sanity.json"):
        if not (OUT / name).exists():
            raise ValueError(f"Required sanity evidence absent: {name}")
    prepared = json.loads((OUT / "prepared.json").read_text())
    first = joblib.load(OUT / "development" / f"fold0_{recipe.name}.joblib")
    sanity = json.loads((OUT / "permutation_sanity.json").read_text())
    if sanity["permuted_macro_f1"] > 0.20:
        raise ValueError("Permutation score requires investigation before freezing")
    schema = category_schema(list(cache["raw"].values()))
    final_features = project_schema(
        compact_matrix(cache["banks"]["original"], recipe), schema
    )
    config = {
        "frozen_at": now(),
        "frozen_before_valid": True,
        "base_commit": git("rev-parse", "HEAD"),
        "recipe": recipe.to_dict(),
        "label_order": LABELS,
        "cutoff": "2026-01-01T00:00:00Z",
        "target": "target_next_recurring_merchant",
        "training": "official TRAIN only",
        "folds": {"n_splits": 5, "group": "client_id", "stratified": True, "seed": 42},
        "stability_seeds": list(SEEDS),
        "augmentation_seed": 2026,
        "augmentation_rates": {
            "medium": [0.30, 0.18, 0.06, 0.15],
            "severe": [0.50, 0.30, 0.10, 0.25],
        },
        "augmentation_rng": "canonical raw history SHA256 order, excluding client_id/target",
        "client_weighting": "equal number of views per client; equal per-view weights; all clients have equal total mass",
        "ranker_parameters": {
            str(seed): parameters(seed)
            | {"num_leaves": recipe.leaves, "objective": "lambdarank"}
            for seed in recipe.seeds
        },
        "legacy_parameters": {
            kind: legacy_parameters(kind) for kind in ("hard", "soft", "xgb")
        },
        "none_parameters": {
            str(seed): parameters(seed) | {"num_leaves": recipe.leaves}
            for seed in recipe.seeds
        },
        "profiles": cache["profiles"],
        "profile_source": "disjoint unlabeled_pretrain only",
        "semantic_patterns": PATTERNS,
        "mcc_rules": MCC,
        "templates": TEMPLATES,
        "generic": GENERIC,
        "feature_rules": {
            "amount_log_tolerance": 0.035,
            "min_events": 3,
            "broad_min_events": 2,
            "candidate_stream_top_k": 2,
            "stream_strength_decay_days": 60,
            "soft_semantic_weight": 3,
            "soft_mcc_weight": 1,
            "soft_smoothing": 0.05,
            "soft_relative_cut": 0.4,
            "soft_absolute_cut": 0.02,
            "refund_amount_log_tolerance": 0.04,
            "refund_after_days": 10,
            "activity_gap_range": [5, 120],
            "activity_multiple": 1.35,
            "rounded_periods": [14, 28, 30, 60, 90],
        },
        "example_fold_feature_columns": first["features"],
        "final_ranker_feature_columns": list(final_features.columns),
        "final_none_feature_columns": list(
            none_features(final_features, cache["ids"]).columns
        )
        if recipe.none_detector
        else [],
        "final_category_vocabulary": schema,
        "feature_schema_rule": "training-view category vocabulary only",
        "decision": "argmax, first class wins exact tie",
        "submission_minimum_valid_f1": 0.52,
        "selection": json.loads((OUT / "selected_train_only.json").read_text()),
        "source_sha256": source_hashes(),
        "source_hash_kind": "utf8_universal_newlines",
        "official_data_manifest": json.loads(
            (ROOT / "reports/data_manifest.json").read_text()
        ),
        "training_data_sha256": prepared["data_hashes"],
        "environment": environment(),
        "development_evidence": {
            name: file_hash(OUT / name)
            for name in (
                "development/summary.json",
                "stability.json",
                "permutation_sanity.json",
                "invariance_sanity.json",
            )
        },
    }
    write_json(
        CONFIG, {"sha256": digest(config), "configuration": config}, exclusive=True
    )
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Congelación antes de VALID\n\nFROZEN_BEFORE_VALID: YES\n\nConfig: `configs/stream_identity_clean_frozen.json`\n\nSHA256: `{digest(config)}`\n\nBase commit: `{config['base_commit']}`\n"
        )
    log(f"Frozen {recipe.name}: {digest(config)}")


def frozen_checked():
    envelope = verify_frozen(CONFIG)
    config = envelope["configuration"]
    for path, sha in config["source_sha256"].items():
        if source_hash(ROOT / path) != sha:
            raise ValueError(f"Frozen source changed: {path}")
    for name, sha in config["training_data_sha256"].items():
        if file_hash(RAW / name) != sha:
            raise ValueError(f"Frozen input changed: {name}")
    return envelope


def predict():
    envelope = frozen_checked()
    if (OUT / "valid_access_receipt.json").exists() or (
        OUT / "prediction_receipt.json"
    ).exists():
        raise ValueError("Predictions/evaluation already frozen")
    start = time.perf_counter()
    cache = load_cache()
    recipe = Recipe(**envelope["configuration"]["recipe"])
    model = ComponentTrainer(
        cache["banks"], cache["raw"], cache["ids"], cache["y"]
    ).fit(recipe)
    joblib.dump(model, OUT / "model.joblib", compress=3)
    id_sets = {"train": set(cache["ids"])}
    raw_hashes = {}
    for split in ("valid", "test"):
        data = transactions(split, use_cache=False, data_dir=RAW)
        ids = np.sort(data.client_id.unique())
        if any(set(ids) & previous for previous in id_sets.values()):
            raise ValueError("Overlapping split clients")
        id_sets[split] = set(ids)
        bank = feature_banks(data, cache["profiles"])
        p = model.predict_proba(bank, ids)
        save_predictions(OUT / f"{split}_probabilities.csv", ids, p)
        raw_hashes[f"{split}_transactions.jsonl"] = file_hash(
            RAW / f"{split}_transactions.jsonl"
        )
        if (
            raw_hashes[f"{split}_transactions.jsonl"]
            != envelope["configuration"]["official_data_manifest"][
                f"{split}_transactions.jsonl"
            ]["sha256"]
        ):
            raise ValueError(
                "Official transaction input differs from historical manifest"
            )
        log(
            f"Persisted {split} predictions ({len(ids)} clients); VALID labels unopened"
        )
    external = transactions("unlabeled_pretrain", use_cache=False, data_dir=RAW)
    if any(set(external.client_id) & previous for previous in id_sets.values()):
        raise ValueError("Pretrain overlaps a supervised split")
    predicted = pd.read_csv(OUT / "test_probabilities.csv")[["client_id", PREDICTION]]
    sample = pd.read_csv(RAW / "sample_submission.csv")
    submission = validate_submission(predicted, sample)
    submission.to_csv(OUT / "submission_candidate.csv", index=False)
    receipt = {
        "created_at": now(),
        "config_sha256": envelope["sha256"],
        "source_sha256": source_hashes(),
        "predictions_frozen_before_labels": True,
        "model_sha256": file_hash(OUT / "model.joblib"),
        "files": {
            f"{s}_probabilities.csv": file_hash(OUT / f"{s}_probabilities.csv")
            for s in ("valid", "test")
        },
        "data_sha256": raw_hashes,
        "split_overlap": False,
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_mib": memory(),
    }
    write_json(OUT / "prediction_receipt.json", receipt, exclusive=True)


def evaluate():
    envelope = frozen_checked()
    receipt = json.loads((OUT / "prediction_receipt.json").read_text())
    if receipt["config_sha256"] != envelope["sha256"]:
        raise ValueError("Prediction/config mismatch")
    for path, sha in receipt["files"].items():
        if file_hash(OUT / path) != sha:
            raise ValueError("Predictions changed after freeze")
    predictions = pd.read_csv(OUT / "valid_probabilities.csv")
    # This exclusive file is created BEFORE the sole VALID-label read. If anything
    # fails after it, automatic reruns are prohibited; no silent second attempt.
    write_json(
        OUT / "valid_access_receipt.json",
        {
            "opened_at": now(),
            "purpose": "single frozen evaluation",
            "config_sha256": envelope["sha256"],
            "prediction_receipt_sha256": file_hash(OUT / "prediction_receipt.json"),
        },
        exclusive=True,
    )
    labels_sha = file_hash(RAW / "valid_labels.csv")
    if (
        labels_sha
        != envelope["configuration"]["official_data_manifest"]["valid_labels.csv"][
            "sha256"
        ]
    ):
        raise ValueError("VALID labels differ from the historical manifest")
    labels = pd.read_csv(RAW / "valid_labels.csv")
    official = score_predictions(labels, predictions[["client_id", PREDICTION]])
    y = (
        labels.set_index("client_id")
        .loc[predictions.client_id, "target_next_recurring_merchant"]
        .map({v: k for k, v in enumerate(LABELS)})
        .to_numpy()
    )
    p = predictions[["p_" + label for label in LABELS]].to_numpy()
    result = assessed(y, p)
    if abs(official["macro_f1"] - result["macro_f1"]) > 1e-12:
        raise ValueError("Official metric mismatch")
    f1 = result["macro_f1"]
    result.update(
        {
            "official": official,
            "delta_vs_v2": f1 - V2,
            "delta_vs_historical": f1 - HISTORICAL,
            "robustness_status": "ROBUST_BREAKTHROUGH"
            if f1 >= 0.58
            else "STRONG"
            if f1 >= 0.52
            else "REAL_BUT_OVERFIT"
            if f1 >= 0.45
            else "MOSTLY_VALID_OVERFIT",
            "submission_valid": False,
            "submission_path": None,
            "evaluated_at": now(),
            "config_sha256": envelope["sha256"],
            "valid_labels_sha256": labels_sha,
        }
    )
    if f1 >= envelope["configuration"]["submission_minimum_valid_f1"]:
        submission = pd.read_csv(OUT / "submission_candidate.csv")
        sample = pd.read_csv(RAW / "sample_submission.csv")
        submission = validate_submission(submission, sample)
        path = ROOT / "outputs/predictions/submission_stream_identity_clean.csv"
        if path.exists():
            raise FileExistsError("Never overwrite a previous submission")
        submission.to_csv(path, index=False)
        validate_submission(pd.read_csv(path), sample)
        result.update(
            {
                "submission_valid": True,
                "submission_path": path.relative_to(ROOT).as_posix(),
                "submission_sha256": file_hash(path),
                "submission_rows": len(submission),
            }
        )
    write_json(OUT / "valid_metrics.json", result, exclusive=True)
    log(
        f"Single frozen VALID: F1={f1:.12f}; accuracy={result['accuracy']:.6f}; {result['robustness_status']}"
    )


def invariance(cache=None):
    assert_development()
    cache = load_cache() if cache is None else cache
    ids = cache["ids"]
    # Fixed TRAIN-only sample covers every label; no VALID descriptions are inspected.
    sample_ids = np.concatenate(
        [ids[np.flatnonzero(cache["y"] == k)[:3]] for k in range(8)]
    )
    raw = cache["raw"]["original"]
    sample = raw[raw.client_id.isin(sample_ids)]
    mapping = {
        cid: f"renamed_{n:04d}"
        for n, cid in enumerate(sorted(sample_ids, reverse=True))
    }
    altered = sample.assign(
        client_id=sample.client_id.map(mapping),
        accidental_target="poison",
        target_next_recurring_merchant="none",
    ).sample(frac=1, random_state=17)
    max_error = 0.0
    compared_banks = {}
    for scenario in SCENARIOS:
        original = corruption_view(sample, scenario)
        renamed = corruption_view(altered, scenario)
        reverse = {value: key for key, value in mapping.items()}
        restored = (
            renamed.assign(client_id=renamed.client_id.map(reverse))
            .sort_values(list(RAW_COLUMNS))
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(original, restored)
        original_bank = feature_banks(original, cache["profiles"])
        changed_bank = feature_banks(renamed, cache["profiles"])
        compared_banks[scenario] = (original_bank, changed_bank)
        for mode in original_bank:
            for kind, matrix in original_bank[mode].items():
                compare = changed_bank[mode][kind]
                index = candidate_index(
                    [mapping[c] for c in matrix.index.get_level_values(0).unique()]
                )
                delta = np.abs(
                    matrix.to_numpy()
                    - compare.reindex(index=index, columns=matrix.columns).to_numpy()
                ).max()
                max_error = max(max_error, float(delta))
                np.testing.assert_allclose(
                    matrix.to_numpy(),
                    compare.reindex(index=index, columns=matrix.columns).to_numpy(),
                    atol=1e-10,
                    rtol=1e-10,
                )
    # Hold sampled clients out and verify predictions, not only features.
    train_ids = np.array([cid for cid in ids if cid not in set(sample_ids)])
    y = pd.Series(cache["y"], index=ids).loc[train_ids].to_numpy()
    model = ComponentTrainer(cache["banks"], cache["raw"], train_ids, y).fit(
        selected_recipe()
    )
    prediction_errors = {}
    for scenario, (original_bank, changed_bank) in compared_banks.items():
        p = model.predict_proba(original_bank, sorted(sample_ids))
        q = model.predict_proba(
            changed_bank, [mapping[cid] for cid in sorted(sample_ids)]
        )
        np.testing.assert_allclose(p, q, atol=1e-12, rtol=1e-12)
        prediction_errors[scenario] = float(np.abs(p - q).max())
    # Pretrain and TRAIN only at this phase; VALID/TEST overlap checks are deferred.
    pre = transactions("unlabeled_pretrain", use_cache=False, data_dir=RAW)
    assert not set(pre.client_id) & set(ids)
    write_json(
        OUT / "invariance_sanity.json",
        {
            "clients": len(sample_ids),
            "scenarios": list(SCENARIOS),
            "max_feature_error": max_error,
            "max_probability_error": max(prediction_errors.values()),
            "probability_errors_by_scenario": prediction_errors,
            "rename_ids": True,
            "row_shuffle": True,
            "target_poisoning": True,
            "cutoff_checked": True,
            "train_pretrain_overlap": False,
        },
    )
    log("TRAIN-only rename/shuffle/poison and prediction invariance passed")


def verify():
    """Verify a frozen reproduction using prediction hashes, without labels."""
    frozen_checked()
    reference = json.loads(
        (ROOT / "reports/stream_identity_clean_summary.json").read_text()
    )
    for name, sha in reference["prediction_receipt"]["files"].items():
        if file_hash(OUT / name) != sha:
            raise ValueError(f"Frozen prediction mismatch: {name}")
    log("Frozen VALID/TEST predictions match recorded hashes; no labels read")


def report():
    # Reporting consumes persisted metrics; it never reopens labels or refits models.
    final = json.loads((OUT / "valid_metrics.json").read_text())
    development = json.loads((OUT / "development/summary.json").read_text())
    stability_data = json.loads((OUT / "stability.json").read_text())
    sanity = json.loads((OUT / "permutation_sanity.json").read_text())
    invariant = json.loads((OUT / "invariance_sanity.json").read_text())
    config = verify_frozen(CONFIG)
    name = config["configuration"]["recipe"]["name"]
    selected = development["results"][name]["scenarios"]
    committed_oof = {}
    for scenario in SCENARIOS:
        source = OUT / "development" / f"{name}_{scenario}_oof.csv"
        destination = ROOT / "reports" / f"stream_identity_clean_oof_{scenario}.csv"
        destination.write_bytes(source.read_bytes())
        committed_oof[destination.relative_to(ROOT).as_posix()] = file_hash(destination)
    full_score = development["results"]["full"]["scenarios"]["original"]["macro_f1"]
    lines = [
        "\n## Resultados ejecutados",
        "",
        "| Variant | OOF Macro-F1 | Accuracy | Δ vs full | Medium F1 | Severe F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, result in development["results"].items():
        s = result["scenarios"]
        m = s["original"]
        lines.append(
            f"| {variant} | {m['macro_f1']:.9f} | {m['accuracy']:.6f} | {m['macro_f1'] - full_score:+.6f} | {s['medium']['macro_f1']:.6f} | {s['severe']['macro_f1']:.6f} |"
        )
    lines += [
        "",
        f"Ganador TRAIN-only: **{name}**. Todas las métricas y OOF por cliente/clase se conservan en `outputs/stream_identity_clean/development/`.",
        "",
        "### Estabilidad TRAIN-only",
        "",
        f"Semillas de partición: `{stability_data['partition_seed_macro_f1']}`.",
        f"Mean={stability_data['mean']:.9f}; std={stability_data['std']:.9f}; min={stability_data['min']:.9f}; max={stability_data['max']:.9f}.",
        f"Semillas de modelo con folds fijos: `{stability_data['model_seed_macro_f1']}`; std={stability_data['model_seed_std']:.9f}.",
        "",
        "### Sanity checks",
        "",
        f"Permutación target: F1={sanity['permuted_macro_f1']:.9f}. Controles aleatorios: `{sanity['chance_scores']}`.",
        f"Rename/shuffle/target poisoning: `{invariant}`.",
        "Cutoff comprobado por lector de datos y transformador; splits TRAIN/VALID/TEST/pretrain disjuntos comprobados antes de la lectura final de labels.",
        "",
        "### Evaluación oficial única",
        "",
        "| Model | Macro-F1 | Accuracy |",
        "| --- | ---: | ---: |",
        "| V1 histórico | 0.271024266 | 0.266 |",
        "| V2 histórico | 0.391549456 | 0.424 |",
        "| Stream Identity historical | 0.619493424 | 0.647 |",
        f"| Stream Identity clean frozen | {final['macro_f1']:.12f} | {final['accuracy']:.6f} |",
        "",
        f"Delta vs V2: {final['delta_vs_v2']:+.12f}; delta vs histórico: {final['delta_vs_historical']:+.12f}.",
        "",
        "| Class | OOF F1 | VALID precision | VALID recall | VALID F1 | Support | Predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        row = final["classification_report"][label]
        lines.append(
            f"| {label} | {selected['original']['classification_report'][label]['f1-score']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1-score']:.6f} | {row['support']:.0f} | {final['prediction_frequency'][label]} |"
        )
    lines += [
        "",
        "Confusion matrix: filas reales, columnas predichas, orden `"
        + ", ".join(LABELS)
        + "`.",
        "",
        "```json",
        json.dumps(final["confusion_matrix"], indent=2),
        "```",
        "",
        "### Calibración y recursos",
        "",
        "| Split | Log loss | Brier | Entropy | Confidence |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for title, value in (("OOF original", selected["original"]), ("VALID", final)):
        lines.append(
            f"| {title} | {value['log_loss']:.6f} | {value['brier_multiclass']:.6f} | {value['mean_entropy']:.6f} | {value['mean_confidence']:.6f} |"
        )
    prepared = json.loads((OUT / "prepared.json").read_text())
    prediction = json.loads((OUT / "prediction_receipt.json").read_text())
    lines += [
        "",
        f"Preparación: {prepared['runtime_seconds']:.1f}s. Catálogo OOF: {development['runtime_seconds']:.1f}s; pico RSS {development['peak_rss_mib']:.1f} MiB. Ajuste final e inferencia: {prediction['runtime_seconds']:.1f}s; pico RSS {prediction['peak_rss_mib']:.1f} MiB.",
        f"Features del fold 0 elegido: {len(config['configuration']['example_fold_feature_columns'])}. CPU; entorno exacto en config. No se estiman comparaciones de tiempo con GPU histórica.",
        "",
        "### Conclusión metodológica",
        "",
        f"Clasificación orientativa: **{final['robustness_status']}**. La selección de esta ejecución usa sólo TRAIN y no realiza un segundo ajuste tras VALID. El score histórico sigue siendo reproducible; esta reconstrucción no convierte el VALID previamente observado en un holdout nuevo. La arquitectura y las hipótesis de shift heredadas siguen condicionando el experimento. No se encontró inclusión predictiva de client_id, target o transacciones futuras; sí existía contaminación histórica por selección sobre VALID.",
        "",
        f"Submission: `{final['submission_path']}`. Validación: {'YES' if final['submission_valid'] else 'NO'}; sin sobrescribir entregas anteriores.",
        "",
        "Probabilidades OOF original/medium/severe del ganador y métricas completas quedan disponibles para stacking en `reports/stream_identity_clean_oof_{original,medium,severe}.csv`. El JSON resumido versionado está en `reports/stream_identity_clean_summary.json`.",
        "",
        "### Resultado solicitado",
        "",
        f"STREAM_IDENTITY_HISTORICAL_VALID_F1: {HISTORICAL}",
        "",
        f"STREAM_IDENTITY_CLEAN_OOF_F1: {selected['original']['macro_f1']}",
        "",
        f"STREAM_IDENTITY_CLEAN_VALID_F1: {final['macro_f1']}",
        "",
        f"DELTA_CLEAN_VS_V2: {final['delta_vs_v2']}",
        "",
        f"DELTA_CLEAN_VS_HISTORICAL: {final['delta_vs_historical']}",
        "",
        "FROZEN_BEFORE_VALID: YES",
        "",
        f"ROBUSTNESS_STATUS: {final['robustness_status']}",
        "",
        f"SUBMISSION_VALID: {'YES' if final['submission_valid'] else 'NO'}",
        "",
    ]
    text = REPORT.read_text(encoding="utf-8")
    text = text.split("\n## Resultados ejecutados")[0]
    REPORT.write_text(text + "\n".join(lines), encoding="utf-8")
    write_json(
        ROOT / "reports/stream_identity_clean_summary.json",
        {
            "historical_macro_f1": HISTORICAL,
            "config_sha256": config["sha256"],
            "selected_name": name,
            "oof": selected,
            "ablations": development,
            "stability": stability_data,
            "sanity": sanity,
            "invariance": invariant,
            "valid": final,
            "oof_artifact_sha256": committed_oof,
            "initial_state": json.loads((OUT / "initial_state.json").read_text()),
            "prediction_receipt": prediction,
            "valid_access_receipt": json.loads(
                (OUT / "valid_access_receipt.json").read_text()
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "rebuild",
            "develop",
            "stability",
            "invariance",
            "freeze",
            "predict",
            "evaluate",
            "report",
            "verify",
        ),
    )
    args = parser.parse_args()
    if git("branch", "--show-current") != "research/import-v2-stream-identity":
        raise ValueError("Wrong branch for this experiment")
    globals()[args.stage]()


if __name__ == "__main__":
    main()
