# ruff: noqa: E402
# Direct script execution requires adding src before importing project modules.
"""One frozen control-refit reproducibility check after a historical-score discrepancy."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from ubs_recurrence.data import LABELS
from ubs_recurrence.model import FamilyForecaster


def main():
    out = ROOT / "outputs/stream_time_audit/paired_01"
    assignments = pd.read_csv(out / "fold_assignments.csv")
    target = assignments.true_family.map(dict(zip(LABELS, range(8)))).to_numpy()
    held = np.flatnonzero(assignments.fold.to_numpy() == 2)
    fit = np.flatnonzero(assignments.fold.to_numpy() != 2)
    tr = (fit[:, None] * 8 + np.arange(8)).ravel()
    va = (held[:, None] * 8 + np.arange(8)).ravel()
    matrices = {
        v: pd.read_parquet(out / f"features/{v}.parquet")
        for v in ("original", "valid_like", "test_like")
    }
    columns = matrices["original"].columns
    matrices = {v: m.reindex(columns=columns).fillna(-999) for v, m in matrices.items()}
    new = FamilyForecaster(seeds=(42,), min_count=3, legacy_weight=0.0, device="cpu")
    new.fit([m.iloc[tr] for m in matrices.values()], target[fit])
    previous = joblib.load(out / "fold2_control.joblib")
    results = {}
    for view, frame in matrices.items():
        before = previous.predict_proba(frame.iloc[va])
        after = new.predict_proba(frame.iloc[va])
        with np.load(out / "fold2_control.npz") as saved:
            cached = saved[view]
        results[view] = {
            "max_saved_vs_reloaded_probability_difference": float(
                np.abs(before - cached).max()
            ),
            "max_retrained_probability_difference": float(np.abs(before - after).max()),
            "retrained_identical_predictions": bool(
                np.array_equal(before.argmax(1), after.argmax(1))
            ),
            "retrained_bit_identical_probabilities": bool(
                np.array_equal(before, after)
            ),
        }
    report = {
        "scope": "one local control refit on fold 2; no official labels used",
        "results": results,
        "limitation": "This does not reproduce the historical nine-estimator official-validation experiment or identify the cause of small historical component score differences.",
    }
    (ROOT / "reports/stream_time_audit/local_reproducibility.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
