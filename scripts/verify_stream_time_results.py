# ruff: noqa: E402
# Direct script execution requires adding src before importing project modules.
"""Independently reconcile saved OOF evidence and client isolation."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from ubs_recurrence.data import LABELS


def main():
    out = ROOT / "outputs/stream_time_audit/paired_01"
    report = ROOT / "reports/stream_time_audit"
    target = (
        pd.read_csv(ROOT / "data/raw/train_labels.csv")
        .set_index("client_id")
        .target_next_recurring_merchant
    )
    assignments = pd.read_csv(out / "fold_assignments.csv").set_index("client_id")
    assert assignments.index.is_unique and set(assignments.index) == set(target.index)
    assert assignments.true_family.equals(
        target.reindex(assignments.index).rename("true_family")
    )
    results = json.loads((report / "results.json").read_text())
    max_error = 0.0
    for key, result in results.items():
        variant, view = key.split("/")
        frame = pd.read_csv(out / f"oof_{variant}_{view}.csv").set_index("client_id")
        assert frame.index.is_unique and set(frame.index) == set(target.index)
        values = frame[LABELS].to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all()
        np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-10)
        pred = np.array(LABELS)[values.argmax(axis=1)]
        truth = target.reindex(frame.index)
        actual = f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0)
        max_error = max(max_error, abs(actual - result["macro_f1"]))
        assert abs(actual - result["macro_f1"]) < 1e-12
        np.testing.assert_array_equal(
            confusion_matrix(truth, pred, labels=LABELS), result["confusion_matrix"]
        )
    verified_models = 0
    for fold in range(5):
        fit = set(assignments.index[assignments.fold.ne(fold)])
        held = set(assignments.index[assignments.fold.eq(fold)])
        assert len(fit) == 1600 and len(held) == 400 and not fit.intersection(held)
        for variant in ("control", "no_temporal", "no_refund", "cycle_state"):
            model = joblib.load(out / f"fold{fold}_{variant}.joblib")
            assert set(model.training_ids_) == fit
            verified_models += 1
    receipt = {
        "verified_aggregate_scores": len(results),
        "verified_client_isolated_models": verified_models,
        "unique_train_clients": len(target),
        "maximum_f1_reconciliation_error": max_error,
        "all_confusion_matrices_match": True,
        "all_probability_rows_normalized": True,
        "official_holdout_labels_loaded": False,
        "scope": "Independent reconstruction from saved CSVs and source TRAIN labels; no new predictions selected.",
    }
    (report / "verification.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
