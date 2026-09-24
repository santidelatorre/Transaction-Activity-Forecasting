"""Prediction-level research diagnostics; never fits a deployable model or gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions

NAMES = ("V2", "A", "NoneGate", "Router", "Family", "Music", "Calibration", "Gym", "Jaime")


def aligned(truth, prediction):
    """Use the official scorer to enforce complete, unique client coverage."""
    evaluate_predictions(truth, prediction)
    return prediction.reindex(truth.index)


def changes(truth, control, candidate):
    control, candidate = aligned(truth, control), aligned(truth, candidate)
    changed = control.ne(candidate)
    before, after = control.eq(truth), candidate.eq(truth)

    def counts(mask):
        return {
            "changed": int((mask & changed).sum()),
            "corrections": int((mask & ~before & after).sum()),
            "regressions": int((mask & before & ~after).sum()),
            "wrong_to_wrong": int((mask & changed & ~before & ~after).sum()),
        }

    transitions = pd.DataFrame({"true": truth, "before": control, "after": candidate})
    return {
        **counts(pd.Series(True, index=truth.index)),
        "by_true_class": {label: counts(truth.eq(label)) for label in LABELS},
        "transitions": transitions.loc[changed]
        .value_counts()
        .rename("clients")
        .reset_index()
        .to_dict("records"),
    }


def pairwise(truth, left, right):
    left, right = aligned(truth, left), aligned(truth, right)
    a, b = left.eq(truth), right.eq(truth)
    return {
        "agreement": int(left.eq(right).sum()),
        "disagreement": int(left.ne(right).sum()),
        "both_correct": int((a & b).sum()),
        "left_only": int((a & ~b).sum()),
        "right_only": int((~a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),
        "error_jaccard": float((~a & ~b).sum() / (~a | ~b).sum()) if (~a | ~b).any() else 0.0,
    }


def oracle(truth, predictions):
    """DIAGNOSTIC ORACLE - NOT DEPLOYABLE.

    Recover truth whenever any member predicts it; otherwise retain the first
    member. Accuracy is the exact selection ceiling. Macro-F1 is this explicit
    fallback oracle's score, not a maximization over wrong-label assignments.
    Also return a loose Macro-F1 upper bound from unavoidable false positives.
    """
    frame = pd.DataFrame({name: aligned(truth, p) for name, p in predictions.items()})
    recoverable = frame.eq(truth, axis=0).any(axis=1)
    prediction = frame.iloc[:, 0].where(~recoverable, truth)
    scores = evaluate_predictions(truth, prediction)
    forced = ~recoverable & frame.nunique(axis=1).eq(1)
    upper = []
    for label in LABELS:
        tp = int((truth.eq(label) & recoverable).sum())
        support = int(truth.eq(label).sum())
        minimum_fp = int((forced & frame.iloc[:, 0].eq(label)).sum())
        upper.append(2 * tp / (support + tp + minimum_fp) if support + tp + minimum_fp else 0)
    return {
        "label": "DIAGNOSTIC ORACLE - NOT DEPLOYABLE",
        "members": list(predictions),
        "fallback": frame.columns[0],
        "accuracy_ceiling": scores["accuracy"],
        "macro_f1_fixed_fallback": scores["macro_f1"],
        "macro_f1_loose_upper_bound": float(np.mean(upper)),
        "recoverable": int(recoverable.sum()),
        "all_fail": int((~recoverable).sum()),
        "extra_over_first": int((recoverable & frame.iloc[:, 0].ne(truth)).sum()),
        "all_fail_by_true_class": truth.loc[~recoverable]
        .value_counts()
        .reindex(LABELS, fill_value=0)
        .to_dict(),
    }


def numeric_score(truth, prediction):
    cm = np.bincount(truth * 8 + prediction, minlength=64).reshape(8, 8)
    denominator = cm.sum(0) + cm.sum(1)
    return np.divide(2 * cm.diagonal(), denominator, out=np.zeros(8), where=denominator > 0).mean()


def bootstrap(truth, left, right, repeats=10000, seed=42):
    left, right = aligned(truth, left), aligned(truth, right)
    code = {label: number for number, label in enumerate(LABELS)}
    y, a, b = (s.map(code).to_numpy() for s in (truth, left, right))
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for repeat in range(repeats):
        take = rng.integers(0, len(y), len(y))
        values[repeat] = numeric_score(y[take], b[take]) - numeric_score(y[take], a[take])
    return {
        "delta": float(numeric_score(y, b) - numeric_score(y, a)),
        "interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
        "repeats": repeats,
        "seed": seed,
        "selection_corrected": False,
    }


def read_frame(path):
    return pd.read_csv(path, index_col="client_id", float_precision="round_trip")


def music_predictions(path, freeze):
    frame = read_frame(path)
    base = frame[[f"a_{label}" for label in LABELS]].copy()
    base.columns = LABELS
    original = base.idxmax(axis=1)
    if freeze["arm"] == "identity_only":
        return original
    arm = freeze["arm"]
    scores = frame[[f"{arm}_{label}" for label in ("music", "streaming", "other")]].copy()
    scores.columns = ["music", "streaming", "other"]
    confidence = scores[["music", "streaming"]].max(axis=1)
    eligible = (
        ~original.isin(["music", "streaming", "none"])
        & confidence.ge(freeze["threshold"])
        & (confidence - scores.other).ge(freeze["margin"])
    )
    return original.where(~eligible, scores[["music", "streaming"]].idxmax(axis=1))


def load_predictions(root):
    frames = {}
    music_policy = json.loads((root / "music/frozen_policy.json").read_text())["selected"]
    none_policy = json.loads((root / "none/frozen_gate.json").read_text())["selected_candidate"]
    for split in ("oof", "valid"):
        base_path = "oof_predictions.csv" if split == "oof" else "validation_predictions.csv"
        base = read_frame(root / "base" / base_path)
        predictions = {name: base[name] for name in ("V2", "A", "full", "ensemble")}
        none_path = "train_oof_predictions.csv" if split == "oof" else "valid_predictions.csv"
        predictions["NoneGate"] = read_frame(root / "none" / none_path)[
            none_policy if split == "oof" else "candidate"
        ]
        router = read_frame(root / "router_replay" / f"{split}_predictions.csv")
        predictions["Router"] = router.candidate
        predictions["FixedRouter"] = router.fixed_confidence
        predictions["Family"] = read_frame(root / "family" / f"{split}_predictions.csv").probability
        music_path = "oof_scores.csv" if split == "oof" else "valid_scores_frozen.csv"
        predictions["Music"] = music_predictions(root / "music" / music_path, music_policy)
        calibration_path = (
            "train_cross_fitted_predictions.csv" if split == "oof" else "valid_predictions.csv"
        )
        predictions["Calibration"] = read_frame(root / "calibration" / calibration_path).calibrated
        gym = read_frame(root / "gym_replay" / f"{split}_predictions.csv")
        predictions["Gym"] = gym.candidate
        if split == "oof":
            predictions["GymSelected"] = gym.selected_gate
        predictions["Jaime"] = read_frame(root / "blend" / f"{split}_predictions.csv").blend
        frames[split] = pd.DataFrame(predictions)
    return frames


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def history_transfer(data_dir):
    """Descriptive coverage diagnostics, not supervised parameter selection."""
    train = read_transactions(data_dir / "train_transactions.jsonl")
    valid = read_transactions(data_dir / "valid_transactions.jsonl")
    target = read_labels(data_dir / "train_labels.csv").set_index("client_id")[TARGET_COLUMN]
    target = target.sort_index()
    rows = []
    for split, frame in (("train", train), ("valid", valid)):
        payments = frame.loc[frame.direction.eq("out") & frame.type.eq("card_payment")]
        counts = frame.groupby("client_id").size()
        rows.append(
            {
                "split": split,
                "clients": int(frame.client_id.nunique()),
                "transactions": len(frame),
                "outgoing_payments": len(payments),
                "unique_descriptions": int(frame.description.nunique()),
                "events_per_client_quantiles": counts.quantile([0.05, 0.5, 0.95]).to_dict(),
                "unseen_payment_rate": float(
                    (~payments.description.isin(train.description)).mean()
                ),
            }
        )
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (fit, hold) in enumerate(splitter.split(target.index, target), 1):
        fit_rows = train.loc[train.client_id.isin(target.index[fit])]
        hold_rows = train.loc[train.client_id.isin(target.index[hold])]
        payments = hold_rows.loc[hold_rows.direction.eq("out") & hold_rows.type.eq("card_payment")]
        rows.append(
            {
                "split": f"oof_{fold}",
                "unseen_payment_rate": float(
                    (~payments.description.isin(fit_rows.description)).mean()
                ),
            }
        )
    return rows


def run(args):
    frames = load_predictions(args.runs)
    results = {}
    for split, frame in frames.items():
        label_split = "train" if split == "oof" else "valid"
        truth = (
            read_labels(args.data_dir / f"{label_split}_labels.csv")
            .set_index("client_id")[TARGET_COLUMN]
            .sort_index()
        )
        # Validate before reindexing so unexpected clients cannot disappear.
        for name in frame:
            aligned(truth, frame[name])
        frame = frame.reindex(truth.index)
        metrics = {name: evaluate_predictions(truth, frame[name]) for name in frame}
        folds = {}
        for name in frame:
            seed = 43 if name == "NoneGate" else 314159 if name == "Router" else 42
            values, deltas = [], []
            for _, hold in StratifiedKFold(5, shuffle=True, random_state=seed).split(
                truth.index, truth
            ):
                ids = truth.index[hold]
                score = evaluate_predictions(truth.loc[ids], frame.loc[ids, name])["macro_f1"]
                control = evaluate_predictions(truth.loc[ids], frame.loc[ids, "A"])["macro_f1"]
                values.append(score)
                deltas.append(score - control)
            folds[name] = {
                "seed": seed,
                "scores": values,
                "deltas": deltas,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas, ddof=1)),
                "wins": int((np.array(deltas) > 1e-14).sum()),
            }
        results[split] = {
            "metrics": metrics,
            "class_frequencies": {
                "truth": truth.value_counts().reindex(LABELS, fill_value=0).to_dict(),
                **{
                    name: frame[name].value_counts().reindex(LABELS, fill_value=0).to_dict()
                    for name in frame
                },
            },
            "changes": {name: changes(truth, frame.A, frame[name]) for name in frame},
            "pairs": {
                f"{a}|{b}": pairwise(truth, frame[a], frame[b]) for a, b in combinations(frame, 2)
            },
        }
        if split == "oof":
            results[split]["folds"] = folds
        else:
            base_misses = frame.A.ne(truth) & frame.V2.ne(truth)
            results[split]["novel_correct_vs_v2_a"] = {
                name: {
                    "total": int((base_misses & frame[name].eq(truth)).sum()),
                    "by_true_class": truth.loc[base_misses & frame[name].eq(truth)]
                    .value_counts()
                    .reindex(LABELS, fill_value=0)
                    .to_dict(),
                }
                for name in NAMES[2:]
            }
            results[split]["identity_errors"] = {
                "wrong_positive_family": int(
                    (truth.ne("none") & frame.A.ne("none") & frame.A.ne(truth)).sum()
                ),
                "positive_to_none": int((truth.ne("none") & frame.A.eq("none")).sum()),
                "none_to_positive": int((truth.eq("none") & frame.A.ne("none")).sum()),
            }
            sets = [
                ["A", "V2"],
                ["A", "Gym"],
                ["A", "FixedRouter"],
                ["A", "V2", "Gym"],
                ["A", "V2", "FixedRouter"],
                ["A", "Family"],
                ["A", "V2", "Family"],
                ["A", "V2", "Calibration"],
                ["Jaime", "NoneGate", "Router", "Family", "Music", "Calibration", "Gym"],
                ["A", "V2", *NAMES[2:]],
            ]
            results[split]["oracles"] = [
                oracle(truth, {name: frame[name] for name in names}) for names in sets
            ]
            comparisons = [
                ("A", "FixedRouter"),
                ("A", "Gym"),
                ("A", "Jaime"),
                ("Gym", "FixedRouter"),
                ("A", "ensemble"),
                ("V2", "A"),
                ("A", "Family"),
                ("A", "NoneGate"),
                ("A", "Calibration"),
            ]
            results[split]["bootstrap"] = {
                f"{a}|{b}": bootstrap(truth, frame[a], frame[b], args.repeats)
                for a, b in comparisons
            }
    music_transfer = {}
    for split, filename in (("oof", "oof_scores.csv"), ("valid", "valid_scores_frozen.csv")):
        scores = read_frame(args.runs / "music" / filename)
        confidence = scores[["text_recurrence_music", "text_recurrence_streaming"]].max(axis=1)
        music_transfer[split] = {
            "confidence_quantiles": confidence.quantile([0.5, 0.9, 0.95, 0.99]).to_dict(),
            "at_least_075": int(confidence.ge(0.75).sum()),
        }
    results["music_score_transfer"] = music_transfer
    results["history_transfer"] = history_transfer(args.data_dir)
    results["inputs"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in args.runs.rglob("*.csv")
        if "probabilities" in path.name or "predictions" in path.name or "scores" in path.name
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    oof, valid = results["oof"]["metrics"], results["valid"]["metrics"]
    rows = []
    for name in valid:
        score = valid[name]
        rows.append(
            [
                name,
                f"{oof[name]['macro_f1']:.12f}",
                f"{score['macro_f1']:.12f}",
                f"{oof[name]['macro_f1'] - oof['A']['macro_f1']:+.9f}",
                f"{score['macro_f1'] - valid['A']['macro_f1']:+.9f}",
                score["accuracy"],
                *[f"{score['per_class'][label]['f1-score']:.6f}" for label in LABELS],
                results["valid"]["changes"][name]["changed"],
            ]
        )
    text = table(
        [
            "Experiment",
            "OOF F1",
            "VALID F1",
            "OOF delta A",
            "VALID delta A",
            "Accuracy",
            *LABELS,
            "Changed A",
        ],
        rows,
    )
    (args.output_dir / "metrics.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs/metrics/v3_seven_way/runs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v3_seven_way/analysis")
    )
    parser.add_argument("--repeats", type=int, default=10000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
