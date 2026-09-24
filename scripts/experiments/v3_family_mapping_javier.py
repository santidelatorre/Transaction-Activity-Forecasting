"""Nested-client TRAIN ablations for V3-A family identity; VALID only after freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.v2 import IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3.features import cross_fitted_family_features
from transaction_forecasting.ubs.v3.identity import (
    IdentityMap,
    cross_fitted_identity_features,
    identity_features,
)

ARMS = {
    "baseline": None,
    "normalized": {"normalized": True},
    "probability": {"normalized": True, "probabilities": True},
    "char_probability": {"normalized": True, "probabilities": True, "char_alias": True},
}


def family_matrix(transactions, labels, history, arm):
    if arm == "baseline":
        family = cross_fitted_family_features(transactions, labels).filter(regex="^identity_")
    else:
        family = cross_fitted_identity_features(transactions, labels, **ARMS[arm])
    return pd.concat([history, family], axis=1)


def predict_arm(model, fit, labels, hold, arm, history_fit, history_hold, heuristic):
    if arm == "baseline":
        from transaction_forecasting.ubs.v3.features import FamilyMap, family_features

        family = family_features(FamilyMap().fit(fit, labels).transform(hold)).filter(
            regex="^identity_"
        )
    else:
        mapper = IdentityMap(**ARMS[arm]).fit(fit, labels)
        family = identity_features(
            mapper.transform(hold), probabilities=ARMS[arm].get("probabilities", False)
        )
    matrix = pd.concat([history_hold, family], axis=1)
    raw = pd.DataFrame(model.predict_proba(matrix), index=matrix.index, columns=LABELS)
    return (0.75 * raw + 0.25 * heuristic).idxmax(axis=1)


def run_fold(fit, labels, hold, arms):
    v2 = IntegratedV2Model().fit(fit, labels)
    history_fit = v2.history_.transform(fit)
    history_hold = v2.history_.transform(hold)
    v2_scores = v2.predict_components(hold)
    heuristic = (v2_scores["blend"] - 0.75 * v2_scores["history"]) / 0.25
    target = labels.set_index("client_id")[TARGET_COLUMN].reindex(history_fit.index)
    output = pd.DataFrame(index=history_hold.index)
    output["V2"] = v2_scores["blend"].idxmax(axis=1)
    counts = {"V2": history_fit.shape[1]}
    for arm in arms:
        matrix = family_matrix(fit, labels, history_fit, arm)
        counts[arm] = matrix.shape[1]
        model = make_model().fit(matrix, target)
        output[arm] = predict_arm(
            model, fit, labels, hold, arm, history_fit, history_hold, heuristic
        )
    return output, counts


def mapping_diagnostics(mapper, transactions):
    mapped = mapper.transform(transactions)
    outgoing = mapped.loc[mapped.direction.eq("out") & mapped.type.eq("card_payment")]
    known = outgoing.family.ne("unknown")
    evidence = outgoing.loc[known, "evidence"]
    result = {
        "map_descriptions": int(len(mapper.mapping_)),
        "mapped_descriptions": int(mapper.mapping_.ne("unknown").sum()),
        "outgoing_rows": int(len(outgoing)),
        "mapped_outgoing_rows": int(known.sum()),
        "outgoing_row_coverage": float(known.mean()),
        "outgoing_client_coverage": float(
            outgoing.loc[known, "client_id"].nunique() / outgoing.client_id.nunique()
        ),
        "evidence_quantiles": evidence.quantile([0, 0.25, 0.5, 0.75, 1]).to_dict(),
    }
    if mapper.probabilities:
        columns = [f"prob_{family}" for family in mapper.probability_.columns]
        confidence = outgoing[columns].max(axis=1)
        result["probability_max_quantiles"] = confidence.quantile([0, 0.25, 0.5, 0.75, 1]).to_dict()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v3_family_mapping_javier")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = read_transactions(args.data_dir / "train_transactions.jsonl")
    labels = read_labels(args.data_dir / "train_labels.csv")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    if args.phase == "oof":
        splitter = StratifiedKFold(5, shuffle=True, random_state=42)
        blocks = []
        feature_counts = None
        for number, (fit_pos, hold_pos) in enumerate(splitter.split(target.index, target), 1):
            fit_ids, hold_ids = target.index[fit_pos], target.index[hold_pos]
            fit = train.loc[train.client_id.isin(fit_ids)]
            hold = train.loc[train.client_id.isin(hold_ids)]
            fit_labels = labels.loc[labels.client_id.isin(fit_ids)]
            print(f"OOF fold {number}/5", flush=True)
            predictions, feature_counts = run_fold(fit, fit_labels, hold, ARMS)
            predictions.to_csv(args.output_dir / f"fold_{number}.csv", index_label="client_id")
            blocks.append(predictions)
        frame = pd.concat(blocks).reindex(target.index)
        frame.to_csv(args.output_dir / "oof_predictions.csv", index_label="client_id")
        metrics = {name: evaluate_predictions(target, frame[name]) for name in frame}
        selected = max(ARMS, key=lambda name: metrics[name]["macro_f1"])
        results = {"metrics": metrics, "feature_counts": feature_counts, "selected": selected}
        (args.output_dir / "oof_results.json").write_text(json.dumps(results, indent=2))
        (args.output_dir / "frozen_selection.json").write_text(
            json.dumps(
                {"selected": selected, "criterion": "TRAIN OOF Macro-F1", "seed": 42}, indent=2
            )
        )
        print({name: score["macro_f1"] for name, score in metrics.items()}, flush=True)
        print(f"Frozen: {selected}", flush=True)
    else:
        frozen = json.loads((args.output_dir / "frozen_selection.json").read_text())
        selected = frozen["selected"]
        valid = read_transactions(args.data_dir / "valid_transactions.jsonl")
        if set(train.client_id).intersection(valid.client_id):
            raise ValueError("TRAIN and VALID clients overlap")
        print(f"VALID baseline and frozen {selected}", flush=True)
        frame, counts = run_fold(train, labels, valid, ("baseline", selected))
        frame.to_csv(args.output_dir / "valid_predictions.csv", index_label="client_id")
        valid_labels = read_labels(args.data_dir / "valid_labels.csv")
        truth = valid_labels.set_index("client_id")[TARGET_COLUMN]
        scores = {name: evaluate_predictions(truth, frame[name]) for name in frame}
        mapper = IdentityMap(**(ARMS[selected] or {})).fit(train, labels)
        mapper.audit_.to_csv(args.output_dir / "train_map.csv")
        diagnostics = mapping_diagnostics(mapper, valid)
        (args.output_dir / "valid_results.json").write_text(
            json.dumps(
                {
                    "metrics": scores,
                    "feature_counts": counts,
                    "selected": selected,
                    "mapping_diagnostics": diagnostics,
                },
                indent=2,
            )
        )
        print({name: score["macro_f1"] for name, score in scores.items()}, flush=True)


if __name__ == "__main__":
    main()
