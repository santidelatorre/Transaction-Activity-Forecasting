import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN

from ubs_recurrence.augmentation import corrupt_transactions, mask_descriptions
from ubs_recurrence.data import CUTOFF, LABELS
from ubs_recurrence.ranking import ranking_features
from ubs_recurrence.streams import amount_components, extract_streams, family_features
from ubs_recurrence.templates import template_features


def history():
    rows = []
    for cid, base, desc, mcc in [
        ("A", 7, "cloud access", "5732"),
        ("B", 60, "phone contract", "4814"),
    ]:
        for n in range(5):
            rows.append(
                {
                    "client_id": cid,
                    "timestamp": CUTOFF - pd.Timedelta(days=10 + 30 * n),
                    "amount": base * (1 + n * 0.001),
                    "description": desc,
                    "mcc": mcc,
                    "currency": "chf",
                    "type": "card_payment",
                    "direction": "out",
                    "fee": 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["client_id", "timestamp"])


@pytest.mark.parametrize("min_samples", [2, 3])
def test_amount_components_equal_dbscan(min_samples):
    rng = np.random.default_rng(777)
    for _ in range(15):
        a = np.exp(rng.uniform(1, 5, 70))
        ours = {frozenset(c) for c in amount_components(a, min_samples=min_samples)}
        lab = DBSCAN(eps=0.035, min_samples=min_samples).fit_predict(
            np.log(a).reshape(-1, 1)
        )
        expected = {frozenset(np.flatnonzero(lab == k)) for k in set(lab) - {-1}}
        assert ours == expected


def test_masking_preserves_nontext_fields_and_is_order_invariant():
    d = history()
    a = mask_descriptions(d, 0.5, 42)
    b = mask_descriptions(d.sample(frac=1, random_state=12), 0.5, 42)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(
        a.drop(columns="description"),
        d.reset_index(drop=True).drop(columns="description"),
    )
    c = corrupt_transactions(d, "test_like", 42)
    for col in [
        "client_id",
        "timestamp",
        "amount",
        "direction",
        "type",
        "currency",
        "fee",
    ]:
        pd.testing.assert_series_equal(c[col], d.reset_index(drop=True)[col])


def test_features_have_no_identifier_dependence_and_include_every_candidate():
    d = history()
    ids = np.array(["A", "B"])
    T = template_features(d, ids)
    S = extract_streams(d, filter_background=True)
    F = family_features(S, ids)
    X = ranking_features(F, T, ids, clocks=False)
    assert X.shape[0] == 16
    assert X.index.tolist() == [(cid, f) for cid in ids for f in LABELS]
    assert not any("client_id" in c or "target" in c for c in X.columns)
    d2 = d.assign(client_id=d.client_id.map({"A": "Z", "B": "Y"}))
    T2 = template_features(d2, np.array(["Z", "Y"]))
    np.testing.assert_allclose(T.to_numpy(), T2.to_numpy())
    np.testing.assert_allclose(
        T.to_numpy(),
        template_features(d.sample(frac=1, random_state=7), ids).to_numpy(),
    )


def test_input_label_columns_do_not_affect_features():
    d = history()
    ids = np.array(["A", "B"])
    with_label = d.assign(target_next_recurring_merchant="fake")
    pd.testing.assert_frame_equal(
        template_features(d, ids), template_features(with_label, ids)
    )
