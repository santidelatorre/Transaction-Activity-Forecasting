"""Paired TRAIN-only representation audit of the pinned stream-identity branch.

This runner never loads official VALID targets or changes the production model.
Its one-seed compact component is deliberately not the nine-estimator ensemble.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

from ubs_recurrence.augmentation import corrupt_transactions
from ubs_recurrence.data import LABELS, aligned_target, transactions
from ubs_recurrence.evaluation import metrics
from ubs_recurrence.model import FamilyForecaster, build_features, none_features
from ubs_recurrence.price_prior import learn_price_profiles
from ubs_recurrence.research_time import feature_groups, research_matrix

SOURCE_COMMIT = "e4aa4c58175242e198cefd32d6ac4558145523af"
VIEWS = ("original", "valid_like", "test_like")
VARIANTS = ("control", "no_temporal", "no_refund", "cycle_state")
REPORT = ROOT / "reports/stream_time_audit"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(paths):
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(paths)}


def initialize(out):
    manifest = json.loads((ROOT / "reports/data_manifest.json").read_text())
    inputs = {}
    for name in ("train_transactions.jsonl", "train_labels.csv", "unlabeled_pretrain_transactions.jsonl"):
        inputs[name] = sha(ROOT / "data/raw" / name)
        if inputs[name] != manifest[name]["sha256"]:
            raise ValueError(f"Pinned input mismatch: {name}")
    sources = [*sorted((ROOT / "src/ubs_recurrence").glob("*.py")), Path(__file__).resolve()]
    signature = {
        "source_branch": "research/import-v2-stream-identity", "source_commit": SOURCE_COMMIT,
        "input_sha256": inputs, "source_sha256": fingerprint(sources),
        "variants": VARIANTS, "views": VIEWS, "fold_seed": 42, "folds": 5,
        "model_seeds": [42], "legacy_weight": 0.0,
        "protocol_sha256": sha(REPORT / "protocol.md"),
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "pyarrow", "joblib")},
        "python": platform.python_version(),
    }
    signature = json.loads(json.dumps(signature))
    path = out / "provenance.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Research code/data/protocol changed; use a new --run-name")
    else:
        dump(path, {"signature": signature, "started_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    return signature


def prepare(out):
    cache = out / "features"
    cache.mkdir(exist_ok=True)
    metadata_path = cache / "manifest.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if all(sha(cache / name) == value for name, value in metadata.items()):
            return {v: pd.read_parquet(cache / f"{v}.parquet") for v in VIEWS}
        raise ValueError("Cached feature checksum mismatch")
    df = transactions("train", use_cache=False)
    unlabelled = transactions("unlabeled_pretrain", use_cache=False)
    if set(df.client_id).intersection(unlabelled.client_id):
        raise ValueError("Unlabeled price profiles overlap supervised clients")
    del unlabelled
    profiles = learn_price_profiles(use_cache=False)
    matrices = {}
    for view in VIEWS:
        started = time.perf_counter()
        current = df if view == "original" else corrupt_transactions(df, view, 2026)
        frame, streams = build_features(current, profiles, min_count=3)
        matrices[view] = frame
        frame.to_parquet(cache / f"{view}.parquet")
        if view == "original":
            streams.to_parquet(cache / "streams.parquet", index=False)
        print(f"Built {view}: {frame.shape}; {time.perf_counter()-started:.1f}s", flush=True)
    dump(metadata_path, {p.name: sha(p) for p in cache.glob("*.parquet")})
    return matrices


def f1_from_counts(y, pred, weights=None):
    cm = np.bincount(8 * y + pred, weights=weights, minlength=64).reshape(8, 8)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.divide(2 * cm.diagonal(), den, out=np.zeros(8), where=den > 0).mean())


def paired_interval(y, control, alternative, repeats=1000):
    rng = np.random.default_rng(20260925)
    differences = []
    a, b = control.argmax(axis=1), alternative.argmax(axis=1)
    for _ in range(repeats):
        sample = rng.integers(0, len(y), len(y))
        differences.append(f1_from_counts(y[sample], b[sample]) - f1_from_counts(y[sample], a[sample]))
    lo, hi = np.quantile(differences, [.025, .975])
    return [float(lo), float(hi)]


def attribution(model, frame, y, out, fold):
    """Native raw-score contributions on the first 32 held-out IDs, not targets."""
    ids = frame.index.get_level_values(0).unique().to_numpy()
    sample = frame.iloc[: min(len(ids), 32) * 8]
    ranker, none = model.models_[0]
    rc = np.asarray(ranker.booster_.predict(sample, pred_contrib=True))
    nx = none_features(sample, ids[:32]).reindex(columns=model.none_columns_).fillna(-999)
    nc = np.asarray(none.booster_.predict(nx, pred_contrib=True))
    raw_rank = np.asarray(ranker.predict(sample, raw_score=True))
    raw_none = np.asarray(none.predict(nx, raw_score=True))
    if not np.allclose(rc.sum(axis=1), raw_rank, atol=1e-7):
        raise ValueError("Ranker contributions fail raw-score reconstruction")
    if not np.allclose(nc.sum(axis=1), raw_none, atol=1e-7):
        raise ValueError("None contributions fail raw-logit reconstruction")
    probabilities = model.predict_proba(sample)
    rank_rows = []
    for j, cid in enumerate(ids[:32]):
        order = probabilities[j].argsort()[::-1]
        pos = j * 8 + int(order[0])
        ranked = np.argsort(np.abs(rc[pos, :-1]))[::-1][:8]
        none_top = np.argsort(np.abs(nc[j, :-1]))[::-1][:8]
        rank_rows.append({
            "client_id": cid, "fold": fold, "true_family": LABELS[int(y[j])],
            "prediction": LABELS[int(order[0])], "runner_up": LABELS[int(order[1])],
            "normalized_scores": dict(zip(LABELS, probabilities[j].tolist())),
            "scope": "one-seed compact component; raw contributions are not final probability contributions",
            "ranker_raw_score": float(raw_rank[pos]), "ranker_base_value": float(rc[pos, -1]),
            "ranker_sum_all_contributions": float(rc[pos].sum()),
            "ranker_top": [{"feature": sample.columns[k], "value": float(sample.iloc[pos, k]), "raw_contribution": float(rc[pos, k])} for k in ranked],
            "none_raw_logit": float(raw_none[j]), "none_base_value": float(nc[j, -1]),
            "none_sum_all_contributions": float(nc[j].sum()),
            "none_top": [{"feature": nx.columns[k], "value": float(nx.iloc[j, k]), "raw_contribution": float(nc[j, k])} for k in none_top],
        })
    dump(out / f"attribution_fold{fold}.json", {
        "sample_rule": "first 32 sorted held-out client IDs, chosen without labels",
        "ranker_global_sample_mean_abs": dict(zip(sample.columns, np.abs(rc[:, :-1]).mean(axis=0).tolist())),
        "none_global_sample_mean_abs": dict(zip(nx.columns, np.abs(nc[:, :-1]).mean(axis=0).tolist())),
        "examples": rank_rows,
    })


def sensitivity(model, frame, out, fold, view):
    ids = frame.index.get_level_values(0).unique().to_numpy()
    groups = feature_groups(frame.columns)
    p = model.predict_proba(frame)
    result = {"control": p}
    for name, cols in groups.items():
        if not cols:
            continue
        values = frame[cols].to_numpy().reshape(len(ids), 8, len(cols))
        for repeat in range(3):
            rng = np.random.default_rng(10000 + fold * 100 + repeat)
            changed = frame.copy()
            changed.loc[:, cols] = values[rng.permutation(len(ids))].reshape(-1, len(cols))
            result[f"permute_{name}_{repeat}"] = model.predict_proba(changed)
    cols = groups["temporal"]
    changed = frame.copy()
    values = frame[cols].to_numpy().reshape(len(ids), 8, len(cols)).copy()
    values[:, :7] = np.roll(values[:, :7], 1, axis=1)
    changed.loc[:, cols] = values.reshape(-1, len(cols))
    nx = none_features(frame, ids)
    changed_nx = none_features(changed, ids)
    if not np.allclose(nx.to_numpy(), changed_nx.to_numpy(), atol=1e-7):
        raise ValueError("Within-client rotation unexpectedly changed symmetric none evidence")
    rotated = model.predict_proba(changed)
    if not np.allclose(p[:, 7], rotated[:, 7], atol=1e-10):
        raise ValueError("Family rotation changed none score")
    result["rotate_family_time"] = rotated
    np.savez_compressed(out / f"sensitivity_fold{fold}_{view}.npz", **result)


def fit_all(out, matrices):
    ids = matrices["original"].index.get_level_values(0).unique().to_numpy()
    y = aligned_target(ids)
    columns = matrices["original"].columns
    matrices = {v: x.reindex(columns=columns).fillna(-999) for v, x in matrices.items()}
    dump(out / "feature_groups.json", feature_groups(columns))
    target_index = matrices["original"].index
    if any(not x.index.equals(target_index) for x in matrices.values()):
        raise ValueError("Client/candidate alignment differs across views")
    fold_ids = np.zeros(len(ids), dtype=int)
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (fit, held) in enumerate(splitter.split(ids, y)):
        fold_ids[held] = fold
        fit_rows = (fit[:, None] * 8 + np.arange(8)).ravel()
        held_rows = (held[:, None] * 8 + np.arange(8)).ravel()
        for variant in VARIANTS:
            completed = out / f"fold{fold}_{variant}.json"
            model_path = out / f"fold{fold}_{variant}.joblib"
            pred_path = out / f"fold{fold}_{variant}.npz"
            transformed = {v: research_matrix(x, variant) for v, x in matrices.items()}
            if completed.exists():
                checkpoint = json.loads(completed.read_text())
                if sha(model_path) != checkpoint["model_sha256"] or sha(pred_path) != checkpoint["prediction_sha256"]:
                    raise ValueError("Fold checkpoint checksum mismatch")
                model = joblib.load(model_path) if variant == "control" else None
                print(f"Resume verified fold {fold} {variant}", flush=True)
            else:
                started = time.perf_counter()
                model = FamilyForecaster(seeds=(42,), min_count=3, legacy_weight=0.0, device="cpu")
                model.fit([x.iloc[fit_rows] for x in transformed.values()], y[fit])
                predictions = {v: model.predict_proba(x.iloc[held_rows]) for v, x in transformed.items()}
                np.savez_compressed(pred_path, held=held, **predictions)
                joblib.dump(model, model_path)
                record = {
                    "fold": fold, "variant": variant, "train_clients": len(fit), "held_clients": len(held),
                    "feature_count": len(transformed["original"].columns), "seconds": time.perf_counter() - started,
                    "scores": {v: metrics(y[held], p) for v, p in predictions.items()},
                    "model_sha256": sha(model_path), "prediction_sha256": sha(pred_path),
                }
                dump(completed, record)
                print(f"Fold {fold} {variant}: " + str({v: round(s["macro_f1"], 5) for v, s in record["scores"].items()}) + f" ({record['seconds']:.1f}s)", flush=True)
            if variant == "control":
                if not (out / f"attribution_fold{fold}.json").exists():
                    attribution(model, transformed["original"].iloc[held_rows], y[held], out, fold)
                for view in VIEWS:
                    if not (out / f"sensitivity_fold{fold}_{view}.npz").exists():
                        sensitivity(model, transformed[view].iloc[held_rows], out, fold, view)
                print(f"Fold {fold} control attribution/sensitivity complete", flush=True)
    pd.DataFrame({"client_id": ids, "fold": fold_ids, "true_family": np.array(LABELS)[y]}).to_csv(out / "fold_assignments.csv", index=False)


def summarize(out):
    assignments = pd.read_csv(out / "fold_assignments.csv")
    ids = assignments.client_id.to_numpy()
    y = assignments.true_family.map(dict(zip(LABELS, range(8)))).to_numpy()
    predictions, rows, details = {}, [], {}
    for variant in VARIANTS:
        for view in VIEWS:
            p = np.zeros((len(ids), 8))
            folds = []
            for fold in range(5):
                with np.load(out / f"fold{fold}_{variant}.npz") as stored:
                    p[stored["held"]] = stored[view]
                folds.append(json.loads((out / f"fold{fold}_{variant}.json").read_text())["scores"][view]["macro_f1"])
            predictions[variant, view] = p
            result = metrics(y, p)
            delta = result["macro_f1"] - metrics(y, predictions["control", view])["macro_f1"]
            interval = [0.0, 0.0] if variant == "control" else paired_interval(y, predictions["control", view], p)
            details[f"{variant}/{view}"] = {**result, "fold_macro_f1": folds, "delta_vs_control": delta, "paired_95_interval": interval}
            rows.append({"variant": variant, "view": view, "macro_f1": result["macro_f1"], "accuracy": result["accuracy"], "delta_vs_control": delta, "paired_low": interval[0], "paired_high": interval[1], "none_predictions": result["prediction_frequency"]["none"], **{"f1_" + label: result["classification_report"][label]["f1-score"] for label in LABELS}})
            export = pd.DataFrame(p, columns=LABELS)
            export.insert(0, "client_id", ids)
            export.to_csv(out / f"oof_{variant}_{view}.csv", index=False)
    pd.DataFrame(rows).to_csv(REPORT / "results.csv", index=False)
    dump(REPORT / "results.json", details)
    sensitivity_rows = []
    for view in VIEWS:
        full = {}
        for fold in range(5):
            held = assignments.fold.to_numpy() == fold
            with np.load(out / f"sensitivity_fold{fold}_{view}.npz") as stored:
                for key in stored.files:
                    full.setdefault(key, np.zeros((len(ids), 8)))[held] = stored[key]
        reference = full["control"]
        for key, p in full.items():
            if key == "control":
                continue
            result = metrics(y, p)
            sensitivity_rows.append({"view": view, "intervention": key, "macro_f1": result["macro_f1"],
                                     "f1_loss": metrics(y, reference)["macro_f1"] - result["macro_f1"],
                                     "changed_predictions": int((p.argmax(1) != reference.argmax(1)).sum()),
                                     "max_none_score_change": float(np.max(np.abs(p[:, 7] - reference[:, 7])))})
    pd.DataFrame(sensitivity_rows).to_csv(REPORT / "sensitivity.csv", index=False)
    attr = [json.loads((out / f"attribution_fold{fold}.json").read_text()) for fold in range(5)]
    for key, filename in (("ranker_global_sample_mean_abs", "ranker_attribution.csv"), ("none_global_sample_mean_abs", "none_attribution.csv")):
        scores = pd.DataFrame([a[key] for a in attr]).mean().sort_values(ascending=False)
        scores.rename("mean_absolute_raw_score_contribution").rename_axis("feature").to_csv(REPORT / filename)
    dump(out / "local_explanations.json", [item for a in attr for item in a["examples"]])
    # Commit no client identifiers or row-level histories. Explain one correct and
    # one wrong case with anonymous references; detailed records stay ignored.
    examples = [item for a in attr for item in a["examples"]]
    selected = []
    for correct in (True, False):
        candidates = [e for e in examples if (e["prediction"] == e["true_family"]) == correct and e["prediction"] != "none"]
        if candidates:
            item = {k: v for k, v in candidates[0].items() if k != "client_id"}
            item["anonymous_case"] = "correct_example" if correct else "error_example"
            selected.append(item)
    dump(REPORT / "explanation_examples.json", selected)
    dump(REPORT / "provenance.json", json.loads((out / "provenance.json").read_text()))
    print(pd.DataFrame(rows)[["variant", "view", "macro_f1", "delta_vs_control"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="paired_01")
    parser.add_argument("--stage", choices=("prepare", "fit", "summarize", "all"), default="all")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("Run name must be a single directory component")
    out = ROOT / "outputs/stream_time_audit" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    initialize(out)
    if args.stage in ("prepare", "fit", "all"):
        matrices = prepare(out)
        if args.stage != "prepare":
            fit_all(out, matrices)
    if args.stage in ("summarize", "all"):
        summarize(out)


if __name__ == "__main__":
    main()
