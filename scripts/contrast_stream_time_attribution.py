# ruff: noqa: E402
# Direct script execution requires adding src before importing project modules.
"""Explain positive-family score margins separately from the none detector.

The compact predictor overwrites the ranker's none score. A raw SHAP magnitude
for is_none can therefore be large without explaining its deployed none score.
These contrasts cancel shared positive-family offsets before ranking features.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from ubs_recurrence.data import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="paired_01")
    args = parser.parse_args()
    out = ROOT / "outputs/stream_time_audit" / args.run_name
    frame = pd.read_parquet(out / "features/original.parquet")
    folds = pd.read_csv(out / "fold_assignments.csv")
    contributions, examples, errors = [], [], []
    for fold in range(5):
        model = joblib.load(out / f"fold{fold}_control.joblib")
        selected = np.flatnonzero(folds.fold.to_numpy() == fold)[:32]
        positions = (selected[:, None] * 8 + np.arange(8)).ravel()
        x = frame.iloc[positions].reindex(columns=model.columns_)
        ranker = model.models_[0][0]
        raw = ranker.predict(x, raw_score=True).reshape(-1, 8)
        shap = np.asarray(ranker.booster_.predict(x, pred_contrib=True)).reshape(
            len(selected), 8, -1
        )
        p = model.predict_proba(x)
        for j, position in enumerate(selected):
            winner, runner = np.argsort(raw[j, :7])[::-1][:2]
            difference = shap[j, winner] - shap[j, runner]
            error = abs(
                float(difference.sum()) - float(raw[j, winner] - raw[j, runner])
            )
            if error > 1e-7:
                raise ValueError(
                    "Contributions do not reconstruct positive-family margin"
                )
            errors.append(error)
            contributions.append(np.abs(difference[:-1]))
            top = np.argsort(np.abs(difference[:-1]))[::-1][:10]
            prediction = LABELS[int(p[j].argmax())]
            examples.append(
                {
                    "fold": fold,
                    "client_id": folds.iloc[position].client_id,
                    "true_family": folds.iloc[position].true_family,
                    "prediction": prediction,
                    "positive_winner": LABELS[int(winner)],
                    "positive_runner_up": LABELS[int(runner)],
                    "positive_raw_margin": float(raw[j, winner] - raw[j, runner]),
                    "sum_all_margin_contributions": float(difference.sum()),
                    "none_score": float(p[j, 7]),
                    "positive_winner_score": float(p[j, winner]),
                    "scope": "conditional positive-family margin of one compact ranker; none explained separately",
                    "top_margin_contributions": [
                        {
                            "feature": x.columns[k],
                            "winner_value": float(x.iloc[j * 8 + winner, k]),
                            "runner_value": float(x.iloc[j * 8 + runner, k]),
                            "contribution_to_margin": float(difference[k]),
                        }
                        for k in top
                    ],
                }
            )
    report = ROOT / "reports/stream_time_audit"
    pd.Series(
        np.mean(contributions, axis=0),
        index=x.columns,
        name="mean_abs_positive_margin_contribution",
    ).sort_values(ascending=False).rename_axis("feature").to_csv(
        report / "ranker_margin_attribution.csv"
    )
    (out / "margin_explanations.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )
    chosen = []
    for correct in (True, False):
        subset = [
            e
            for e in examples
            if (e["true_family"] == e["prediction"]) == correct
            and e["prediction"] != "none"
        ]
        if subset:
            case = {k: v for k, v in subset[0].items() if k != "client_id"}
            case["anonymous_case"] = "correct_example" if correct else "error_example"
            chosen.append(case)
    result = {
        "sample": "first 32 held-out IDs in each of five folds; selection independent of labels",
        "n_clients": len(examples),
        "max_margin_reconstruction_error": max(errors),
        "warning": "Raw ranker is_none contribution is not the separately predicted none score. Shared positive-logit shifts cancel.",
        "examples": chosen,
    }
    (report / "margin_examples.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Verified {len(examples)} component margin explanations; max error {max(errors):.3g}"
    )


if __name__ == "__main__":
    main()
