"""Causality, grouping, weighting and probability-contract regression tests."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.model_selection import GroupKFold

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.survival import (
    FamilyEvidence,
    RecurrenceModel,
    aggregate,
    pseudo_cutoffs,
    stream_features,
)


def history():
    rows = []
    for client in ("a", "b", "c", "d"):
        for day in (
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
            "2024-06-30",
            "2024-07-01",
            "2024-08-01",
        ):
            rows.append(
                dict(
                    client_id=client,
                    timestamp=pd.Timestamp(day, tz="UTC"),
                    description="merchant",
                    currency="chf",
                    amount=-20.0,
                    direction="out",
                    type="card_payment",
                    mcc="1234",
                    fee=0.0,
                )
            )
    return pd.DataFrame(rows)


CUT = pd.Timestamp("2024-04-01", tz="UTC")


def test_strict_causal_cutoff_and_future_mutation():
    tx = history()
    with pytest.raises(ValueError, match="Future"):
        stream_features(tx, CUT)
    past = tx.loc[tx.timestamp.lt(CUT)]
    features = stream_features(past, CUT, True)
    changed = tx.copy()
    changed.loc[changed.timestamp.ge(CUT), "amount"] = 1e9
    assert_frame_equal(features, stream_features(changed.loc[changed.timestamp.lt(CUT)], CUT, True))
    assert features.recency.eq(31).all()
    assert features["count"].eq(3).all()


def test_exact_horizon_and_complete_followup():
    tx = history()
    tx = tx.loc[~tx.timestamp.eq(CUT)]
    snapshots = pseudo_cutoffs(tx, [CUT], seasonal=True)
    # June 30 is exactly 90 days after April 1 and is outside [cutoff, cutoff+90d).
    assert snapshots.time_to_next_event.eq(90).all()
    assert snapshots.event.eq(0).all()
    assert snapshots.bucket.eq(3).all()
    tx.loc[tx.timestamp.eq(pd.Timestamp("2024-06-30", tz="UTC")), "timestamp"] -= pd.Timedelta(
        seconds=1
    )
    assert pseudo_cutoffs(tx, [CUT]).event.eq(1).all()
    with pytest.raises(ValueError, match="fully observed"):
        pseudo_cutoffs(tx, [CUT], observation_end=CUT + pd.Timedelta(days=89))


def test_snapshot_weights_and_client_folds():
    tx = history()
    extra = tx.loc[tx.client_id.eq("a")].assign(description="second")
    snapshots = pseudo_cutoffs(pd.concat([tx, extra]), [CUT, CUT + pd.Timedelta(days=30)])
    np.testing.assert_allclose(snapshots.groupby("client_id").weight.sum(), 1)
    for fit, hold in GroupKFold(2).split(snapshots, groups=snapshots.client_id):
        assert not set(snapshots.iloc[fit].client_id) & set(snapshots.iloc[hold].client_id)


@pytest.mark.parametrize("kind", ["recurrence", "hazard"])
def test_class_order_determinism_and_isolation(kind):
    snapshots = pseudo_cutoffs(history(), [CUT, CUT + pd.Timedelta(days=30)], seasonal=True)
    fit = snapshots.loc[snapshots.client_id.isin(["a", "b"])]
    hold = snapshots.loc[snapshots.client_id.isin(["c", "d"])]
    a = RecurrenceModel(kind, True).fit(fit)
    b = RecurrenceModel(kind, True).fit(fit)
    np.testing.assert_allclose(a.predict(hold), b.predict(hold), atol=0, rtol=0)
    np.testing.assert_allclose(a.predict(hold).sum(axis=1), 1)
    with pytest.raises(ValueError, match="overlap"):
        a.predict(fit)
    # All these streams recur; absent no-event class must occupy the last column with zero.
    assert (a.predict(hold)[:, 3] == 0).all()


def test_none_survival_and_symmetric_competition():
    streams = pd.DataFrame({"client_id": ["a", "a"], "count": [5, 5]})
    evidence = np.eye(7)[:2]
    p = np.array([[0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5]])
    result = aggregate(streams, p, evidence, np.ones(2), ["a", "empty"])
    assert list(result) == list(LABELS)
    assert result.loc["a", "none"] == 0.25
    assert result.loc["empty", "none"] == 1
    np.testing.assert_allclose(result.sum(axis=1), 1)
    np.testing.assert_allclose(result.iloc[0, :2], [0.375, 0.375])
    assert_frame_equal(
        result, aggregate(streams, p[::-1], evidence[::-1], np.ones(2), ["a", "empty"])
    )


def test_family_mapping_excludes_holdout_clients():
    tx = history()
    fit = tx.loc[tx.client_id.isin(["a", "b"])]
    mapper = FamilyEvidence().fit(fit, pd.Series(["gym", "cloud"], index=["a", "b"]))
    with pytest.raises(ValueError, match="disjoint"):
        mapper.transform(fit)


def test_runner_logloss_uses_official_class_order():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/experiments/v4_survival_laura.py"
    spec = importlib.util.spec_from_file_location("survival_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    probabilities = pd.DataFrame(np.eye(len(LABELS)), columns=LABELS)
    assert runner.metrics(pd.Series(LABELS), probabilities)["logloss"] < 1e-12
