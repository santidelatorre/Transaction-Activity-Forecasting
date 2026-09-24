"""Ablation: incremental value of unlabeled pretrain priors for UBS V2.

A) Frozen V2 baseline (fixed 75/25 history/periodicity blend)
B) Same V2 components; blend weight adapted with TRAIN-only description priors
C) Same V2 components; blend weight adapted with TRAIN + unlabeled priors

Priors never use labels. CatBoost and heuristic stay identical to V2; only the
per-client blend weight between history and periodicity changes. B -> C isolates
the 10k unlabeled clients.

Diagnosis: feeding prior features into CatBoost collapsed valid Macro-F1 via
train/valid description-mix shift on P(recurrent). Soft anti-none logit nudges
also selected beta=0 on train OOF. Adaptive blend weight is the remaining
unsupervised use that keeps the V2 model frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import (
    LABELS,
    TARGET_COLUMN,
    load_ubs_data,
    read_transactions,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.pretrain_priors import (
    DescriptionPriors,
    client_prior_features,
    fit_description_priors,
)
from transaction_forecasting.ubs.v2 import IntegratedV2Model

OUTPUT_DEFAULT = Path("outputs/metrics/v3_discovery/esteban_pretrain")
# gamma: how far to shift heuristic weight above the V2 default 0.25 when priors
# say the client's descriptions are recurrent. Selected on train OOF only.
GAMMA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
BASE_HEURISTIC_WEIGHT = 0.25


def _metrics_payload(metrics: dict[str, object], prediction: pd.Series) -> dict[str, object]:
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "per_class": metrics["per_class"],
        "prediction_distribution": prediction.value_counts()
        .reindex(LABELS, fill_value=0)
        .astype(int)
        .to_dict(),
    }


def _client_signals(transactions: pd.DataFrame, priors: DescriptionPriors) -> pd.DataFrame:
    features = client_prior_features(transactions, priors)
    global_rate = float(priors.global_defaults["p_recurrent"])
    signals = pd.DataFrame(index=features.index)
    lift = (features["pretrain_mean_p_recurrent"] - global_rate).clip(-1.0, 1.0)
    max_lift = (features["pretrain_max_p_recurrent"] - global_rate).clip(-1.0, 1.0)
    stability = (
        priors.global_defaults["gap_cv_prior"] - features["pretrain_mean_gap_cv_prior"]
    ).clip(-1.0, 1.0)
    residual = features.get(
        "pretrain_stream_mean_period_residual",
        pd.Series(0.0, index=features.index),
    )
    period_match = (1.0 - residual.clip(0.0, 2.0) / 2.0).fillna(0.0)
    short = (
        features.get(
            "pretrain_short_history_stream_count",
            pd.Series(0.0, index=features.index),
        ).clip(0.0, 5.0)
        / 5.0
    )
    # Positive => trust periodicity heuristic more (population says recurrent).
    signals["trust_periodicity"] = (
        0.5 * lift.clip(0.0, 1.0)
        + 0.2 * max_lift.clip(0.0, 1.0)
        + 0.15 * stability.clip(0.0, 1.0)
        + 0.15 * period_match * short
    ).clip(0.0, 1.0)
    signals["recurrence_lift"] = lift
    signals["mean_p_recurrent"] = features["pretrain_mean_p_recurrent"]
    signals["mean_typical_period_days"] = features["pretrain_mean_typical_period_days"]
    signals["mean_gap_cv_prior"] = features["pretrain_mean_gap_cv_prior"]
    signals["unseen_desc_share"] = features["pretrain_unseen_desc_share"]
    signals["short_history_stream_count"] = features.get(
        "pretrain_short_history_stream_count",
        pd.Series(0.0, index=features.index),
    )
    return signals


def adaptive_blend(
    history: pd.DataFrame,
    heuristic: pd.DataFrame,
    signals: pd.DataFrame,
    gamma: float,
) -> pd.DataFrame:
    """Per-client blend: heuristic_weight = 0.25 + gamma * trust_periodicity."""
    trust = signals.reindex(history.index)["trust_periodicity"].fillna(0.0).to_numpy()
    h_weight = np.clip(BASE_HEURISTIC_WEIGHT + gamma * trust, 0.05, 0.60)
    hist_weight = 1.0 - h_weight
    hist = history.reindex(columns=LABELS).to_numpy(dtype=float)
    heur = heuristic.reindex(history.index).reindex(columns=LABELS).to_numpy(dtype=float)
    blended = hist_weight[:, None] * hist + h_weight[:, None] * heur
    return pd.DataFrame(blended, index=history.index, columns=LABELS)


def _select_gamma_oof(
    train_transactions: pd.DataFrame,
    train_labels: pd.DataFrame,
    priors: DescriptionPriors,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[float, dict[str, object], pd.DataFrame, pd.DataFrame]:
    target = train_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    clients = np.asarray(target.index)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_history = pd.DataFrame(0.0, index=clients, columns=LABELS)
    oof_heuristic = pd.DataFrame(0.0, index=clients, columns=LABELS)
    for train_idx, valid_idx in splitter.split(clients, target.to_numpy()):
        fit_ids = set(clients[train_idx])
        hold_ids = clients[valid_idx]
        fit_tx = train_transactions[train_transactions["client_id"].isin(fit_ids)]
        hold_tx = train_transactions[train_transactions["client_id"].isin(hold_ids)]
        fit_labels = train_labels[train_labels["client_id"].isin(fit_ids)]
        components = IntegratedV2Model().fit(fit_tx, fit_labels).predict_components(hold_tx)
        # Recover heuristic from blend/history: blend = 0.75*h + 0.25*heur
        history = components["history"].reindex(hold_ids)
        blend = components["blend"].reindex(hold_ids)
        heuristic = (blend - 0.75 * history) / 0.25
        oof_history.loc[hold_ids] = history.to_numpy()
        oof_heuristic.loc[hold_ids] = heuristic.to_numpy()
    signals = _client_signals(train_transactions, priors).reindex(clients)
    best_gamma = 0.0
    best_score = -1.0
    grid: dict[str, float] = {}
    for gamma in GAMMA_GRID:
        pred = adaptive_blend(oof_history, oof_heuristic, signals, gamma).idxmax(axis=1)
        score = float(evaluate_predictions(target, pred)["macro_f1"])
        grid[str(gamma)] = score
        if score > best_score:
            best_score = score
            best_gamma = float(gamma)
    payload = {"selected_gamma": best_gamma, "oof_macro_f1": best_score, "grid": grid}
    return best_gamma, payload, oof_history, oof_heuristic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/ubs_2026")
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--alpha", type=float, default=5.0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    data = load_ubs_data(args.data_dir)
    unlabeled = read_transactions(Path(args.data_dir) / "unlabeled_pretrain_transactions.jsonl")
    valid_target = data.valid_labels.set_index("client_id")[TARGET_COLUMN].sort_index()

    v2 = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    components = v2.predict_components(data.valid_transactions)
    history = components["history"]
    blend_a = components["blend"]
    heuristic = (blend_a - 0.75 * history) / 0.25
    pred_a = blend_a.idxmax(axis=1)
    metrics_a = evaluate_predictions(valid_target.reindex(pred_a.index), pred_a)

    priors_b = fit_description_priors(
        data.train_transactions, alpha=args.alpha, includes_unlabeled=False
    )
    gamma, oof_b, _, _ = _select_gamma_oof(data.train_transactions, data.train_labels, priors_b)
    signals_b = _client_signals(data.valid_transactions, priors_b)
    proba_b = adaptive_blend(history, heuristic, signals_b, gamma)
    pred_b = proba_b.idxmax(axis=1)
    metrics_b = evaluate_predictions(valid_target.reindex(pred_b.index), pred_b)

    population = pd.concat([data.train_transactions, unlabeled], ignore_index=True)
    priors_c = fit_description_priors(population, alpha=args.alpha, includes_unlabeled=True)
    signals_c = _client_signals(data.valid_transactions, priors_c)
    # Same frozen gamma; only prior source changes.
    proba_c = adaptive_blend(history, heuristic, signals_c, gamma)
    pred_c = proba_c.idxmax(axis=1)
    metrics_c = evaluate_predictions(valid_target.reindex(pred_c.index), pred_c)

    # Secondary forced-gamma grid on VALID for B and C (reporting only; not used to select).
    forced_grid = {}
    for g in GAMMA_GRID:
        pb = adaptive_blend(history, heuristic, signals_b, g).idxmax(axis=1)
        pc = adaptive_blend(history, heuristic, signals_c, g).idxmax(axis=1)
        mb = evaluate_predictions(valid_target.reindex(pb.index), pb)
        mc = evaluate_predictions(valid_target.reindex(pc.index), pc)
        forced_grid[str(g)] = {
            "B_macro_f1": mb["macro_f1"],
            "C_macro_f1": mc["macro_f1"],
            "delta_C_minus_B": mc["macro_f1"] - mb["macro_f1"],
        }

    valid_desc = set(data.valid_transactions["description"].astype(str).unique())
    train_desc = set(priors_b.table.index.astype(str))
    pretrain_desc = set(priors_c.table.index.astype(str))
    unseen_vs_train = sorted(valid_desc.difference(train_desc))
    covered_by_pretrain = sorted(set(unseen_vs_train).intersection(pretrain_desc))

    per_class_delta = {
        label: float(
            metrics_c["per_class"][label]["f1-score"] - metrics_b["per_class"][label]["f1-score"]
        )
        for label in LABELS
    }
    benefited = sorted(
        [label for label, delta in per_class_delta.items() if delta > 1e-4],
        key=lambda label: per_class_delta[label],
        reverse=True,
    )
    hurt = sorted(
        [label for label, delta in per_class_delta.items() if delta < -1e-4],
        key=lambda label: per_class_delta[label],
    )
    delta_bc = float(metrics_c["macro_f1"] - metrics_b["macro_f1"])
    if delta_bc >= 0.01:
        conclusion = "PRETRAIN IS A MAJOR SOURCE OF SIGNAL"
    elif delta_bc >= 0.002:
        conclusion = "PRETRAIN PROVIDES A SMALL/MODERATE GAIN"
    else:
        conclusion = "PRETRAIN DOES NOT IMPROVE THE CURRENT SYSTEM"

    disagree = pred_b.reindex(pred_c.index) != pred_c
    pd.DataFrame(
        {
            "client_id": pred_c.index,
            "prediction_a_v2": pred_a.reindex(pred_c.index).to_numpy(),
            "prediction_b_train_priors": pred_b.reindex(pred_c.index).to_numpy(),
            "prediction_c_pretrain_priors": pred_c.to_numpy(),
            "actual": valid_target.reindex(pred_c.index).to_numpy(),
            "trust_b": signals_b.reindex(pred_c.index)["trust_periodicity"].to_numpy(),
            "trust_c": signals_c.reindex(pred_c.index)["trust_periodicity"].to_numpy(),
            "p_recurrent_b": signals_b.reindex(pred_c.index)["mean_p_recurrent"].to_numpy(),
            "p_recurrent_c": signals_c.reindex(pred_c.index)["mean_p_recurrent"].to_numpy(),
        }
    ).to_csv(output_dir / "validation_predictions_abc.csv", index=False)

    priors_b.table.to_csv(output_dir / "priors_train_only.csv")
    priors_c.table.to_csv(output_dir / "priors_train_plus_unlabeled.csv")
    signals_b.to_csv(output_dir / "valid_signals_B.csv")
    signals_c.to_csv(output_dir / "valid_signals_C.csv")
    priors_c.table.sort_values(["p_recurrent", "n_clients"], ascending=False).head(40)[
        [
            "n_clients",
            "n_transactions",
            "p_recurrent",
            "typical_period_days",
            "gap_cv_prior",
            "amount_cv_prior",
            "p_weekly",
            "p_biweekly",
            "p_monthly",
            "p_quarterly",
            "typical_mcc",
            "typical_type",
            "typical_currency",
        ]
    ].to_csv(output_dir / "top_recurrent_descriptions_C.csv")

    # Signal difference: how much unlabeled changes client trust scores.
    trust_delta = (signals_c["trust_periodicity"] - signals_b["trust_periodicity"]).abs()
    summary = {
        "experiment": "esteban_pretrain",
        "alpha": args.alpha,
        "seconds": perf_counter() - started,
        "unlabeled_clients": int(unlabeled["client_id"].nunique()),
        "unlabeled_transactions": int(len(unlabeled)),
        "prior_descriptions_train_only": int(len(priors_b.table)),
        "prior_descriptions_train_plus_unlabeled": int(len(priors_c.table)),
        "system": (
            "Identical V2 history/heuristic probabilities for A/B/C. B/C adapt the "
            "per-client heuristic blend weight with unsupervised description priors. "
            "CatBoost features unchanged."
        ),
        "diagnosis": (
            "CatBoost+prior-features collapsed valid MF1 (~0.15). Anti-none logit "
            "adjustment selected beta=0 on train OOF. Adaptive blend is the evaluated "
            "unsupervised prior use; gamma frozen on train OOF with train-only priors."
        ),
        "selected_gamma": gamma,
        "gamma_selection_oof_train_only": oof_b,
        "forced_gamma_valid_grid_reporting_only": forced_grid,
        "A_v2_baseline_valid": _metrics_payload(metrics_a, pred_a),
        "B_train_priors": {
            "valid": _metrics_payload(metrics_b, pred_b),
            "prior_source_clients": priors_b.source_clients,
            "includes_unlabeled": False,
            "mean_trust_periodicity": float(signals_b["trust_periodicity"].mean()),
        },
        "C_train_plus_unlabeled_priors": {
            "valid": _metrics_payload(metrics_c, pred_c),
            "prior_source_clients": priors_c.source_clients,
            "includes_unlabeled": True,
            "mean_trust_periodicity": float(signals_c["trust_periodicity"].mean()),
        },
        "delta_macro_f1_B_to_C": delta_bc,
        "delta_macro_f1_A_to_C": float(metrics_c["macro_f1"] - metrics_a["macro_f1"]),
        "delta_macro_f1_A_to_B": float(metrics_b["macro_f1"] - metrics_a["macro_f1"]),
        "per_class_f1_delta_B_to_C": per_class_delta,
        "classes_benefited_B_to_C": benefited,
        "classes_hurt_B_to_C": hurt,
        "prediction_flips_B_to_C": int(disagree.sum()),
        "trust_abs_delta_mean": float(trust_delta.mean()),
        "trust_abs_delta_p90": float(trust_delta.quantile(0.9)),
        "music_streaming": {
            "music_f1_A": metrics_a["per_class"]["music"]["f1-score"],
            "music_f1_B": metrics_b["per_class"]["music"]["f1-score"],
            "music_f1_C": metrics_c["per_class"]["music"]["f1-score"],
            "streaming_f1_A": metrics_a["per_class"]["streaming"]["f1-score"],
            "streaming_f1_B": metrics_b["per_class"]["streaming"]["f1-score"],
            "streaming_f1_C": metrics_c["per_class"]["streaming"]["f1-score"],
        },
        "rare_unseen": {
            "valid_unique_descriptions": len(valid_desc),
            "unseen_vs_train_count": len(unseen_vs_train),
            "unseen_vs_train_covered_by_pretrain": len(covered_by_pretrain),
            "coverage_rate_of_unseen": len(covered_by_pretrain) / max(len(unseen_vs_train), 1),
            "unseen_examples": unseen_vs_train[:20],
            "covered_examples": covered_by_pretrain[:20],
        },
        "conclusion": conclusion,
        "leakage_policy": (
            "Priors fit without labels. Gamma chosen on train OOF with train-only priors. "
            "Candidate C swaps prior source to train+unlabeled with the same gamma and the "
            "same frozen V2 component probabilities. Validation labels used only for scoring "
            "and for a reporting-only forced-gamma grid (not selection)."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
