"""Integrity and isolation of the TRAIN-only merchant-text stress API."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.corruption import (
    FAMILIES,
    CorruptionLevel,
    CorruptionSuite,
    shift_metrics,
)
from transaction_forecasting.ubs.corruption_evaluation import (
    StressDataset,
    client_folds,
    evaluate_under_corruption,
)
from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN


@pytest.fixture
def dataset():
    rows, labels = [], []
    for number in range(24):
        client = f"client-{number:02d}"
        labels.append(
            {"client_id": client, "cutoff_date": "2026-01-01", TARGET_COLUMN: LABELS[number % 8]}
        )
        for stream in range(4):
            for event in range(4):
                rows.append(
                    {
                        "client_id": client,
                        "description": f"merchant payment {stream}",
                        "timestamp": pd.Timestamp("2025-01-01", tz="UTC")
                        + pd.Timedelta(days=stream + 30 * event),
                        "amount": 10.0 + stream,
                        "fee": 0.0,
                        "mcc": str(5812 if stream < 3 else 7997),
                        "currency": "chf",
                        "type": "card_payment",
                        "direction": "out",
                    }
                )
        rows.append({**rows[-1], "description": "income", "direction": "in", "type": "topup"})
    return StressDataset(pd.DataFrame(rows), pd.DataFrame(labels))


def test_clean_exact_and_all_metadata_target_preserved(dataset):
    frame = dataset.history.merge(dataset.labels[["client_id", TARGET_COLUMN]], on="client_id")
    original = frame.copy(deep=True)
    suite = CorruptionSuite().fit(frame)
    pd.testing.assert_frame_equal(suite.transform(frame, "clean"), frame)
    for level in suite.levels:
        result = suite.transform(frame, level)
        pd.testing.assert_frame_equal(
            result.drop(columns="description"), frame.drop(columns="description")
        )
        assert result.timestamp.lt(CUTOFF).all()
        assert result.loc[result.direction.eq("in"), "description"].equals(
            frame.loc[frame.direction.eq("in"), "description"]
        )
    pd.testing.assert_frame_equal(frame, original)


def test_determinism_seed_order_and_labels_do_not_affect_text(dataset, monkeypatch):
    # Neither fit nor transform may perform file reads, including VALID labels.
    def forbidden(*args, **kwargs):
        raise AssertionError("Corruption accessed external files")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(pd, "read_json", forbidden)
    frame = dataset.history.assign(**{TARGET_COLUMN: "cloud"})
    suite = CorruptionSuite(seed=123).fit(frame)
    a = suite.transform(frame, "severe")
    pd.testing.assert_frame_equal(a, suite.transform(frame, "severe"))
    assert not a.description.equals(
        CorruptionSuite(seed=321).fit(frame).transform(frame, "severe").description
    )
    shuffled = frame.sample(frac=1, random_state=42)
    pd.testing.assert_frame_equal(a, suite.transform(shuffled, "severe").sort_index())
    mutated = frame.assign(**{TARGET_COLUMN: "none"})
    assert a.description.equals(
        CorruptionSuite(seed=123).fit(mutated).transform(mutated, "severe").description
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_family_contracts_and_no_silent_string_truncation(dataset, family):
    suite = CorruptionSuite().fit(dataset.history)
    level = dataclasses.replace(CorruptionLevel(), **{family: 1.0})
    result = suite.transform(dataset.history, level)
    outgoing = result.loc[result.direction.eq("out")]
    original = dataset.history.loc[dataset.history.direction.eq("out")]
    assert outgoing.description.ne(original.description).all()
    if family == "dropout":
        assert set(outgoing.description) == {"__description_missing__"}
    if family == "stream_mask":
        assert outgoing.groupby("client_id").description.nunique().eq(4).all()
        assert outgoing.groupby(["client_id", "description"]).size().eq(4).all()
    if family == "fragmentation":
        assert outgoing.groupby("client_id").description.nunique().eq(8).all()
    if family == "collision":
        assert outgoing.groupby("client_id").description.nunique().eq(2).all()
        assert outgoing.groupby("description").mcc.nunique().eq(1).all()
    combined = dataclasses.replace(level, dropout=1.0, decoration=1.0)
    assert set(
        suite.transform(dataset.history, combined).loc[result.direction.eq("out"), "description"]
    ) == {"__description_missing__"}


def test_rejects_future_null_and_invalid_probabilities(dataset):
    with pytest.raises(ValueError):
        CorruptionLevel(dropout=1.1)
    future = dataset.history.copy()
    future.loc[0, "timestamp"] = CUTOFF
    with pytest.raises(ValueError, match="strictly before"):
        CorruptionSuite().fit(future)
    fitted = CorruptionSuite().fit(dataset.history)
    with pytest.raises(ValueError, match="strictly before"):
        fitted.transform(future)
    future.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="Null"):
        fitted.transform(future)


def test_whole_stream_mask_includes_related_refunds_and_transfers(dataset):
    history = dataset.history.copy()
    extra = history.iloc[:2].copy()
    extra["type"] = ["refund", "transfer"]
    extra["direction"] = ["in", "out"]
    history = pd.concat([history, extra], ignore_index=True)
    suite = CorruptionSuite().fit(history)
    changed = suite.transform(history, CorruptionLevel(stream_mask=1.0))
    for _, group in history.groupby(["client_id", "description"]):
        if (group.type.eq("card_payment") & group.direction.eq("out")).any():
            assert changed.loc[group.index].description.nunique() == 1
            assert changed.loc[group.index].description.ne(group.description).all()
    # Opaque whole-stream renaming must preserve exact-description group sizes.
    before = history.groupby(["client_id", "description"]).size().sort_values().to_numpy()
    after = changed.groupby(["client_id", "description"]).size().sort_values().to_numpy()
    np.testing.assert_array_equal(before, after)


def test_opaque_stream_rename_preserves_baseline_numeric_history(dataset):
    from transaction_forecasting.ubs.v2 import HistoryFeatureBuilder

    frame = dataset.history
    suite = CorruptionSuite().fit(frame)
    changed = suite.transform(frame, CorruptionLevel(stream_mask=1.0))
    builder = HistoryFeatureBuilder().fit(frame)
    pd.testing.assert_frame_equal(builder.transform(frame), builder.transform(changed))


def test_shift_metrics_are_label_free_and_amounts_currency_safe(dataset):
    suite = CorruptionSuite().fit(dataset.history)
    level = CorruptionLevel(collision=1.0)
    changed = suite.transform(dataset.history, level)
    before = shift_metrics(dataset.history, dataset.history, suite, dataset.history)
    after = shift_metrics(changed, dataset.history, suite, dataset.history)
    assert before["unseen_description_rate"] == 0
    assert after["collided_result_stream_fraction"] > 0
    assert after["streams_per_client"] < before["streams_per_client"]
    assert before["category_distributions"] == after["category_distributions"]


def test_no_merchant_payments_is_identity_and_serializable(dataset):
    frame = dataset.history.loc[dataset.history.direction.eq("in")]
    suite = CorruptionSuite().fit(frame)
    changed = suite.transform(frame, "severe")
    pd.testing.assert_frame_equal(frame, changed)
    profile = shift_metrics(changed, frame, suite, frame)
    assert profile["payment_events"] == 0
    json.dumps(profile, allow_nan=False)


def test_folds_match_baseline_and_all_views_stay_held_out(dataset, tmp_path):
    assignments = client_folds(dataset, n_splits=3)
    target = dataset.labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    for fold, (_, held) in enumerate(
        StratifiedKFold(3, shuffle=True, random_state=42).split(target.index, target), 1
    ):
        assert set(assignments.index[assignments.eq(fold)]) == set(target.index[held])
    fits, held_calls = [], []

    class Spy:
        def fit(self, history, labels):
            self.ids = set(history.client_id)
            assert self.ids == set(labels.client_id)
            fits.append(self.ids)
            return self

        def predict_proba(self, history):
            ids = sorted(history.client_id.unique())
            assert not self.ids.intersection(ids)
            held_calls.append(set(ids))
            return pd.DataFrame(np.full((len(ids), 8), 1 / 8), index=ids, columns=LABELS)

    truth_before = dataset.labels.copy(deep=True)
    result = evaluate_under_corruption(
        Spy,
        dataset,
        CorruptionSuite(),
        n_splits=3,
        output_dir=tmp_path,
        progress=lambda *a, **kw: None,
    )
    assert len(fits) == 3
    assert len(held_calls) == 30
    for position in range(0, 30, 10):
        assert all(ids == held_calls[position] for ids in held_calls[position : position + 10])
    assert not held_calls[0] & held_calls[10]
    assert not held_calls[10] & held_calls[20]
    assert result["scorecards"]["model"]["severe"]["drop_absolute"] == 0
    pd.testing.assert_frame_equal(dataset.labels, truth_before)
    duplicate = pd.concat([assignments, assignments.iloc[:1]])
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_under_corruption(Spy, dataset, CorruptionSuite(), folds=duplicate, n_splits=3)
