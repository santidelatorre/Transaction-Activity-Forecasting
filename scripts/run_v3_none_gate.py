"""Select on TRAIN and evaluate one dedicated V3-A positive-vs-none gate.

Run ``--phase oof`` first.  It creates client-grouped base OOF predictions,
cross-fits every second-level candidate, and freezes the architecture and final
threshold.  Only ``--phase valid`` reads VALID labels, after persisting frozen
predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v3.features import FAMILIES
from transaction_forecasting.ubs.v3.none_gate import (
    IdentityV3Model,
    apply_none_gate,
    build_gate_features,
    gate_feature_names,
    make_gate_estimator,
)

SEED = 42
CANDIDATES = (
    {"name": "score_a", "kind": "score", "feature_set": "probability", "c": 1.0},
    {"name": "logit_prob_c01", "kind": "logistic", "feature_set": "probability", "c": 0.1},
    {"name": "logit_prob_c1", "kind": "logistic", "feature_set": "probability", "c": 1.0},
    {"name": "logit_compact_c01", "kind": "logistic", "feature_set": "compact", "c": 0.1},
    {"name": "logit_compact_c1", "kind": "logistic", "feature_set": "compact", "c": 1.0},
    {"name": "tree_compact", "kind": "tree", "feature_set": "compact", "c": 1.0},
)
THRESHOLDS = np.round(np.linspace(0.20, 0.80, 61), 2)


def _json_default(value):
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=_json_default), encoding="utf-8"
    )


def fingerprint(data_dir: Path) -> dict[str, str]:
    paths = [
        Path(__file__),
        Path("src/transaction_forecasting/evaluation/official.py"),
        *Path("src/transaction_forecasting/ubs").rglob("*.py"),
        data_dir / "train_transactions.jsonl",
        data_dir / "train_labels.csv",
    ]
    return {
        str(path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def feature_columns(features: pd.DataFrame, config: dict) -> list[str]:
    columns = gate_feature_names(config["feature_set"])
    return features.columns.tolist() if not columns else columns


def candidate_scores(config: dict, features: pd.DataFrame, estimator=None) -> pd.Series:
    if config["kind"] == "score":
        denominator = features["a_positive_probability"] + features["a_none_probability"]
        return features["a_positive_probability"].div(denominator.clip(lower=1e-12))
    columns = feature_columns(features, config)
    values = estimator.predict_proba(features[columns])[:, 1]
    return pd.Series(values, index=features.index)


def fit_estimator(config: dict, features: pd.DataFrame, positive: pd.Series):
    model = make_gate_estimator(config["kind"], config["c"])
    columns = feature_columns(features, config)
    return model.fit(features[columns], positive.astype(int))


def threshold_search(
    truth: pd.Series, positive_family: pd.Series, scores: pd.Series
) -> tuple[float, float]:
    results = []
    for threshold in THRESHOLDS:
        prediction = apply_none_gate(scores, positive_family, float(threshold))
        results.append((evaluate_predictions(truth, prediction)["macro_f1"], float(threshold)))
    best_score = max(score for score, _ in results)
    tied = [threshold for score, threshold in results if np.isclose(score, best_score)]
    return min(tied, key=lambda value: (abs(value - 0.5), value)), best_score


def inner_threshold(
    config: dict,
    features: pd.DataFrame,
    truth: pd.Series,
    positive_family: pd.Series,
) -> float:
    if config["kind"] == "score":
        scores = candidate_scores(config, features)
    else:
        scores = pd.Series(index=features.index, dtype=float)
        splitter = StratifiedKFold(4, shuffle=True, random_state=SEED + 100)
        for fit_pos, hold_pos in splitter.split(features.index, truth):
            fit_ids, hold_ids = features.index[fit_pos], features.index[hold_pos]
            model = fit_estimator(config, features.loc[fit_ids], truth.loc[fit_ids].ne("none"))
            scores.loc[hold_ids] = candidate_scores(config, features.loc[hold_ids], model)
        if scores.isna().any():
            raise RuntimeError("Incomplete inner gate predictions")
    return threshold_search(truth, positive_family, scores)[0]


def crossfit_candidate(
    config: dict,
    features: pd.DataFrame,
    truth: pd.Series,
    positive_family: pd.Series,
    split_indices: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.Series, pd.Series, list[dict[str, float]]]:
    prediction = pd.Series(index=truth.index, dtype=object)
    scores = pd.Series(index=truth.index, dtype=float)
    fold_details = []
    for fold, (fit_pos, hold_pos) in enumerate(split_indices, 1):
        fit_ids, hold_ids = truth.index[fit_pos], truth.index[hold_pos]
        threshold = inner_threshold(
            config, features.loc[fit_ids], truth.loc[fit_ids], positive_family.loc[fit_ids]
        )
        model = None
        if config["kind"] != "score":
            model = fit_estimator(config, features.loc[fit_ids], truth.loc[fit_ids].ne("none"))
        hold_scores = candidate_scores(config, features.loc[hold_ids], model)
        scores.loc[hold_ids] = hold_scores
        prediction.loc[hold_ids] = apply_none_gate(
            hold_scores, positive_family.loc[hold_ids], threshold
        )
        fold_details.append(
            {
                "fold": fold,
                "threshold": threshold,
                "macro_f1": evaluate_predictions(truth.loc[hold_ids], prediction.loc[hold_ids])[
                    "macro_f1"
                ],
            }
        )
    if prediction.isna().any() or scores.isna().any():
        raise RuntimeError("Incomplete second-level cross-fit")
    return prediction, scores, fold_details


def change_analysis(truth: pd.Series, baseline: pd.Series, candidate: pd.Series) -> dict:
    baseline, candidate = baseline.reindex(truth.index), candidate.reindex(truth.index)
    changed = baseline.ne(candidate)
    positive = truth.ne("none")
    candidate_metrics = evaluate_predictions(truth, candidate)
    return {
        "changed_vs_a": int(changed.sum()),
        "changes_into_none": int((changed & candidate.eq("none")).sum()),
        "changes_out_of_none": int((changed & baseline.eq("none")).sum()),
        "correct_changes": int((changed & candidate.eq(truth) & baseline.ne(truth)).sum()),
        "incorrect_changes": int((changed & baseline.eq(truth) & candidate.ne(truth)).sum()),
        "changed_wrong_to_wrong": int((changed & baseline.ne(truth) & candidate.ne(truth)).sum()),
        "positive_client_accuracy": float(candidate.loc[positive].eq(truth.loc[positive]).mean()),
        "none_precision": candidate_metrics["per_class"]["none"]["precision"],
        "none_recall": candidate_metrics["per_class"]["none"]["recall"],
        "none_f1": candidate_metrics["per_class"]["none"]["f1-score"],
    }


def bootstrap_delta(
    truth: pd.Series, baseline: pd.Series, candidate: pd.Series, repeats: int = 2000
) -> dict:
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(repeats):
        positions = rng.integers(0, len(truth), len(truth))
        sample_truth = truth.iloc[positions].reset_index(drop=True)
        sample_baseline = baseline.reindex(truth.index).iloc[positions].reset_index(drop=True)
        sample_candidate = candidate.reindex(truth.index).iloc[positions].reset_index(drop=True)
        deltas.append(
            evaluate_predictions(sample_truth, sample_candidate)["macro_f1"]
            - evaluate_predictions(sample_truth, sample_baseline)["macro_f1"]
        )
    return {
        "delta": evaluate_predictions(truth, candidate)["macro_f1"]
        - evaluate_predictions(truth, baseline)["macro_f1"],
        "percentile_95": np.quantile(deltas, [0.025, 0.975]),
        "resamples": repeats,
        "seed": SEED,
    }


def fold_stability(
    truth: pd.Series,
    v2: pd.Series,
    a: pd.Series,
    candidate: pd.Series,
    split_indices: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    rows = []
    for fold, (_, hold_pos) in enumerate(split_indices, 1):
        ids = truth.index[hold_pos]
        values = {
            name: evaluate_predictions(truth.loc[ids], prediction.loc[ids])["macro_f1"]
            for name, prediction in (("V2", v2), ("A", a), ("candidate", candidate))
        }
        rows.append(
            {"fold": fold, **values, "delta_candidate_vs_a": values["candidate"] - values["A"]}
        )
    delta = np.array([row["delta_candidate_vs_a"] for row in rows])
    return {
        "folds": rows,
        "candidate_vs_a_mean": float(delta.mean()),
        "candidate_vs_a_std": float(delta.std(ddof=1)),
        "folds_candidate_beats_a": int((delta > 0).sum()),
    }


def run_oof(data_dir: Path, output: Path, stamps: dict[str, str]) -> None:
    started = perf_counter()
    train = read_transactions(data_dir / "train_transactions.jsonl")
    labels = read_labels(data_dir / "train_labels.csv")
    truth = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base_splits = list(splitter.split(truth.index, truth))
    feature_blocks, a_blocks, v2_blocks = [], [], []
    fold_assignment = pd.Series(index=truth.index, dtype=int)
    for fold, (fit_pos, hold_pos) in enumerate(base_splits, 1):
        fit_ids, hold_ids = truth.index[fit_pos], truth.index[hold_pos]
        paths = {
            "features": output / f"base_fold_{fold}_gate_features.csv",
            "A": output / f"base_fold_{fold}_a_probabilities.csv",
            "V2": output / f"base_fold_{fold}_v2_probabilities.csv",
        }
        if all(path.exists() for path in paths.values()):
            features = pd.read_csv(paths["features"], index_col="client_id")
            a_prob = pd.read_csv(paths["A"], index_col="client_id")
            v2_prob = pd.read_csv(paths["V2"], index_col="client_id")
        else:
            print(f"Fitting identity base fold {fold}/5", flush=True)
            fit = train.loc[train.client_id.isin(fit_ids)]
            hold = train.loc[train.client_id.isin(hold_ids)]
            fit_labels = labels.loc[labels.client_id.isin(fit_ids)]
            model = IdentityV3Model().fit(fit, fit_labels)
            components, mapped = model.predict_components(hold)
            a_prob, v2_prob = components["A"], components["V2"]
            features = build_gate_features(a_prob, v2_prob, hold, mapped)
            features.to_csv(paths["features"], index_label="client_id")
            a_prob.to_csv(paths["A"], index_label="client_id")
            v2_prob.to_csv(paths["V2"], index_label="client_id")
        expected = set(hold_ids)
        if set(features.index) != expected or set(a_prob.index) != expected:
            raise ValueError(f"Stale or incomplete base fold {fold}")
        feature_blocks.append(features)
        a_blocks.append(a_prob)
        v2_blocks.append(v2_prob)
        fold_assignment.loc[hold_ids] = fold

    features = pd.concat(feature_blocks).reindex(truth.index)
    a_prob = pd.concat(a_blocks).reindex(truth.index)
    v2_prob = pd.concat(v2_blocks).reindex(truth.index)
    if features.isna().any().any() or a_prob.isna().any().any() or v2_prob.isna().any().any():
        raise RuntimeError("Incomplete base OOF data")
    a_prediction = a_prob.idxmax(axis=1)
    v2_prediction = v2_prob.idxmax(axis=1)
    positive_family = a_prob.loc[:, FAMILIES].idxmax(axis=1)

    gate_splitter = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
    gate_splits = list(gate_splitter.split(truth.index, truth))
    candidate_predictions = {}
    candidate_reports = {}
    candidate_score_frames = {}
    for config in CANDIDATES:
        print(f"Cross-fitting gate candidate {config['name']}", flush=True)
        prediction, scores, folds = crossfit_candidate(
            config, features, truth, positive_family, gate_splits
        )
        candidate_predictions[config["name"]] = prediction
        candidate_score_frames[config["name"]] = scores
        candidate_reports[config["name"]] = {
            "config": config,
            "metrics": evaluate_predictions(truth, prediction),
            "folds": folds,
            "change_analysis": change_analysis(truth, a_prediction, prediction),
        }
    selected_name = max(
        (config["name"] for config in CANDIDATES),
        key=lambda name: candidate_reports[name]["metrics"]["macro_f1"],
    )
    selected_config = next(config for config in CANDIDATES if config["name"] == selected_name)
    final_threshold, _ = threshold_search(
        truth, positive_family, candidate_score_frames[selected_name]
    )
    selected_prediction = candidate_predictions[selected_name]
    results = {
        "base_metrics": {
            "V2": evaluate_predictions(truth, v2_prediction),
            "A": evaluate_predictions(truth, a_prediction),
        },
        "candidate_reports": candidate_reports,
        "selected_candidate": selected_name,
        "selected_train_oof_metrics": evaluate_predictions(truth, selected_prediction),
        "selected_change_analysis": change_analysis(truth, a_prediction, selected_prediction),
        "stability": fold_stability(
            truth, v2_prediction, a_prediction, selected_prediction, gate_splits
        ),
        "bootstrap_candidate_vs_a": bootstrap_delta(truth, a_prediction, selected_prediction),
        "base_fold_assignment": fold_assignment.astype(int).value_counts().sort_index().to_dict(),
        "gate_crossfit_protocol": "5 outer gate folds; four inner folds select each fold threshold",
        "reported_oof_uses_final_threshold": False,
        "seconds": perf_counter() - started,
    }
    write_json(output / "train_oof_results.json", results)
    features.to_csv(output / "train_oof_gate_features.csv", index_label="client_id")
    a_prob.to_csv(output / "train_oof_a_probabilities.csv", index_label="client_id")
    v2_prob.to_csv(output / "train_oof_v2_probabilities.csv", index_label="client_id")
    pd.DataFrame({"V2": v2_prediction, "A": a_prediction, **candidate_predictions}).to_csv(
        output / "train_oof_predictions.csv", index_label="client_id"
    )
    write_json(
        output / "frozen_gate.json",
        {
            "selected_candidate": selected_name,
            "config": selected_config,
            "threshold": final_threshold,
            "feature_names": feature_columns(features, selected_config),
            "selection_criterion": "maximum second-level client-grouped TRAIN OOF Macro-F1",
            "threshold_criterion": "maximum final-class Macro-F1 on cross-fitted TRAIN gate scores",
            "valid_labels_used": False,
            "fingerprints": stamps,
        },
    )
    print(
        json.dumps(
            {
                "selected": selected_name,
                "threshold": final_threshold,
                "A": results["base_metrics"]["A"]["macro_f1"],
                "candidate": results["selected_train_oof_metrics"]["macro_f1"],
            }
        ),
        flush=True,
    )


def run_valid(data_dir: Path, output: Path, stamps: dict[str, str]) -> None:
    started = perf_counter()
    frozen = json.loads((output / "frozen_gate.json").read_text(encoding="utf-8"))
    if frozen["fingerprints"] != stamps or frozen["valid_labels_used"]:
        raise ValueError("Frozen TRAIN selection does not match current source/data")
    if (output / "valid_results.json").exists():
        raise ValueError("VALID was already evaluated; refusing an adaptive rerun")

    train = read_transactions(data_dir / "train_transactions.jsonl")
    labels = read_labels(data_dir / "train_labels.csv")
    valid = read_transactions(data_dir / "valid_transactions.jsonl")
    if set(train.client_id).intersection(valid.client_id):
        raise ValueError("TRAIN/VALID client overlap")
    model = IdentityV3Model().fit(train, labels)
    components, mapped = model.predict_components(valid)
    features = build_gate_features(components["A"], components["V2"], valid, mapped)

    train_features = pd.read_csv(output / "train_oof_gate_features.csv", index_col="client_id")
    train_truth = labels.set_index("client_id")[TARGET_COLUMN].reindex(train_features.index)
    config = frozen["config"]
    estimator = None
    if config["kind"] != "score":
        estimator = fit_estimator(config, train_features, train_truth.ne("none"))
    scores = candidate_scores(config, features, estimator)
    positive_family = components["A"].loc[:, FAMILIES].idxmax(axis=1)
    candidate = apply_none_gate(scores, positive_family, frozen["threshold"])
    predictions = pd.DataFrame(
        {
            "V2": components["V2"].idxmax(axis=1),
            "A": components["A"].idxmax(axis=1),
            "candidate": candidate,
        }
    )
    # Freeze predictions before VALID labels are read.
    features.to_csv(output / "valid_gate_features.csv", index_label="client_id")
    components["A"].to_csv(output / "valid_a_probabilities.csv", index_label="client_id")
    components["V2"].to_csv(output / "valid_v2_probabilities.csv", index_label="client_id")
    predictions.to_csv(output / "valid_predictions.csv", index_label="client_id")

    valid_labels = read_labels(data_dir / "valid_labels.csv")
    truth = valid_labels.set_index("client_id")[TARGET_COLUMN]
    report = {
        "metrics": {name: evaluate_predictions(truth, predictions[name]) for name in predictions},
        "change_analysis": change_analysis(truth, predictions["A"], predictions["candidate"]),
        "bootstrap_candidate_vs_a": bootstrap_delta(
            truth, predictions["A"], predictions["candidate"]
        ),
        "frozen_gate": frozen,
        "predictions_persisted_before_reading_valid_labels": True,
        "seconds": perf_counter() - started,
    }
    write_json(output / "valid_results.json", report)
    print(
        json.dumps({name: values["macro_f1"] for name, values in report["metrics"].items()}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/ubs_v3_none_gate_santiago")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamps = fingerprint(args.data_dir)
    write_json(
        args.output_dir / f"{args.phase}_provenance.json",
        {
            "phase": args.phase,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "seed": SEED,
            "fingerprints": stamps,
        },
    )
    if args.phase == "oof":
        if (args.output_dir / "valid_results.json").exists():
            raise ValueError("VALID exists; refusing to change the TRAIN selection")
        run_oof(args.data_dir, args.output_dir, stamps)
    else:
        run_valid(args.data_dir, args.output_dir, stamps)


if __name__ == "__main__":
    main()
