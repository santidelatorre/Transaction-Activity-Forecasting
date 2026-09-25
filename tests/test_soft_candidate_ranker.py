import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs import soft_candidate_ranker as module
from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN


def sample():
    rows, labels = [], []
    for label in LABELS:
        for number in range(10):
            client = f"{label}_{number}"
            labels.append({"client_id": client, TARGET_COLUMN: label})
            for month in (9, 10, 11, 12):
                rows.append(
                    {
                        "client_id": client,
                        "description": f"merchant {label}",
                        "timestamp": pd.Timestamp(f"2025-{month:02d}-01", tz="UTC"),
                        "amount": 10.0,
                        "currency": "chf",
                        "direction": "out",
                        "type": "card_payment",
                        "mcc": "5812",
                        "fee": 0.0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(labels)


class CheapBaseline:
    def fit(self, tx, labels):
        self.clients = set(labels.client_id)
        return self

    def predict_components(self, tx):
        assert not self.clients.intersection(tx.client_id)
        p = pd.DataFrame(1 / 8, index=sorted(tx.client_id.unique()), columns=LABELS)
        return {"baseline": p, "v2": p}


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(module, "FrozenV3A", CheapBaseline)
    tx, labels = sample()
    return module.EvidenceProvider().fit(tx, labels)


def test_no_self_label_mapping_or_in_sample_baseline(provider):
    tx, _ = sample()
    with pytest.raises(ValueError, match="OOF baseline"):
        provider.transform(tx)
    with pytest.raises(ValueError, match="Self-label"):
        provider.mapper_.evidence(tx)


def test_all_eight_even_without_evidence_and_no_client_feature(provider):
    tx, _ = sample()
    unseen = tx.assign(client_id="new_" + tx.client_id, description="never observed")
    batch = provider.transform(unseen)
    assert len(batch.features) == 8 * tx.client_id.nunique()
    assert batch.features.index.get_level_values("candidate")[:8].tolist() == list(LABELS)
    assert batch.features.xs("none", level="candidate").shape[0] == tx.client_id.nunique()
    assert "client_id" not in batch.features and TARGET_COLUMN not in batch.features
    assert np.isfinite(batch.features.to_numpy()).all()
    assert batch.features.mapped_event_count.eq(0).all()
    assert batch.features.no_family_evidence.eq(1).all()
    no_card = provider.transform(unseen.assign(type="transfer"))
    assert no_card.features.no_stream_recurs_proxy.eq(1).all()
    assert np.allclose(module.raw_family_probabilities(no_card).sum(axis=1), 1)


def test_batch_rejects_bad_provenance_order_and_nonfinite(provider):
    tx, _ = sample()
    batch = provider.transform(tx.assign(client_id="new_" + tx.client_id))
    clients = frozenset(batch.features.index.get_level_values("client_id"))
    with pytest.raises(ValueError, match="OOF baseline"):
        module.CandidateBatch(
            batch.features, (module.SourcePartition(clients, clients),)
        ).validate()
    with pytest.raises(ValueError, match="provenance"):
        module.CandidateBatch(batch.features, ()).validate()
    for bad in (batch.features.iloc[:-1], batch.features.iloc[::-1]):
        with pytest.raises(ValueError, match="eight candidates"):
            module.CandidateBatch(bad, batch.sources).validate()
    bad = batch.features.copy()
    bad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        module.CandidateBatch(bad, batch.sources).validate()
    bad = batch.features.assign(client_id=1)
    with pytest.raises(ValueError, match="Client ID"):
        module.CandidateBatch(bad, batch.sources).validate()


def test_crossfit_and_outer_isolation(monkeypatch):
    monkeypatch.setattr(module, "FrozenV3A", CheapBaseline)
    tx, labels = sample()
    target = module.client_target(tx, labels)
    fit_ids, hold_ids = next(module.client_folds(target, 2))
    # Excluded labels may change arbitrarily without entering any fit operation.
    fit = tx.loc[tx.client_id.isin(fit_ids)]
    fit_labels = labels.loc[labels.client_id.isin(fit_ids)]
    batch = module.cross_fitted_candidates(fit, fit_labels, folds=2)
    assert set(batch.features.index.get_level_values("client_id")) == set(fit_ids)
    for source in batch.sources:
        assert not source.fit_clients & source.prediction_clients
        assert not (source.fit_clients | source.prediction_clients) & set(hold_ids)


@pytest.mark.parametrize("kind", ["linear", "catboost"])
def test_official_relevance_and_finite_softmax(provider, kind, monkeypatch):
    monkeypatch.setattr(module, "FrozenV3A", CheapBaseline)
    tx, labels = sample()
    batch = module.cross_fitted_candidates(tx, labels, folds=2)
    target = labels.set_index("client_id")[TARGET_COLUMN]
    model = module.SoftCandidateRanker(kind).fit(batch, target)
    unseen = provider.transform(tx.assign(client_id="new_" + tx.client_id))
    p = model.predict_proba(unseen)
    assert list(p.columns) == list(LABELS)
    assert np.allclose(p.sum(axis=1), 1)
    assert np.isfinite(p.to_numpy()).all()
    assert p.idxmax(axis=1).equals(model.predict_scores(unseen).idxmax(axis=1))
    with pytest.raises(ValueError, match="excluded"):
        model.predict_proba(batch)


def test_history_contract_currency_and_corruption(provider):
    tx, _ = sample()
    unseen = tx.assign(client_id="new_" + tx.client_id)
    for timestamp in (CUTOFF, pd.NaT):
        with pytest.raises(ValueError):
            provider.transform(unseen.assign(timestamp=timestamp))
    streams = module.payment_streams(
        pd.concat([unseen, unseen.assign(currency="eur", amount=500)]),
        ["client_id", "description", "currency"],
    )
    assert len(streams) == 2 * unseen.client_id.nunique()
    assert streams.amount_cv.eq(0).all()
    damaged = module.degrade_history(unseen)
    pd.testing.assert_frame_equal(damaged, module.degrade_history(unseen))
    assert set(damaged.client_id) == set(unseen.client_id)
    assert len(damaged) < len(unseen)


def test_relevance_is_official_label_and_one_positive_per_client(provider, monkeypatch):
    tx, labels = sample()
    unseen = tx.assign(client_id="new_" + tx.client_id)
    batch = provider.transform(unseen)
    target = labels.assign(client_id="new_" + labels.client_id).set_index("client_id")[
        TARGET_COLUMN
    ]

    class Capture:
        def fit(self, features, relevance, **kwargs):
            actual = np.asarray(relevance).reshape(-1, 8)
            assert (actual.sum(axis=1) == 1).all()
            clients = features.index.get_level_values("client_id").unique()
            assert np.array_equal(np.asarray(LABELS)[actual.argmax(axis=1)], target.loc[clients])
            weights = kwargs["logisticregression__sample_weight"].reshape(-1, 8)
            assert np.allclose(weights, weights[:, :1])

    monkeypatch.setattr(module, "make_pipeline", lambda *args: Capture())
    module.SoftCandidateRanker().fit(batch, target.iloc[::-1])


def test_efficient_baseline_matches_reference_a(monkeypatch):
    from transaction_forecasting.ubs.v3 import model as reference

    class History:
        def transform(self, tx):
            return tx.groupby("client_id").size().to_frame("history_count")

    class V2:
        def fit(self, tx, labels):
            self.history_ = History()
            return self

        def predict_components(self, tx):
            p = pd.DataFrame(1 / 8, index=sorted(tx.client_id.unique()), columns=LABELS)
            return {"history": p, "blend": p}

    class Numeric:
        def fit(self, features, target):
            self.columns = features.columns
            return self

        def predict_proba(self, features):
            assert self.columns.equals(features.columns)
            values = np.ones((len(features), 8))
            values[:, 0] += features.sum(axis=1).to_numpy()
            return values / values.sum(axis=1, keepdims=True)

    for namespace in (module, reference):
        monkeypatch.setattr(namespace, "IntegratedV2Model", V2)
        monkeypatch.setattr(namespace, "make_model", Numeric)
    tx, labels = sample()
    unseen = tx.assign(client_id="new_" + tx.client_id)
    expected = reference.V3Model().fit(tx, labels).predict_components(unseen)["A"]
    actual = module.FrozenV3A().fit(tx, labels).predict_components(unseen)["baseline"]
    pd.testing.assert_frame_equal(actual, expected)


def test_runner_does_not_read_valid_before_freeze(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/experiments/v4_soft_candidate_santiago.py"
    spec = importlib.util.spec_from_file_location("v4_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for name in ("train_transactions.jsonl", "train_labels.csv"):
        (tmp_path / name).write_text("TRAIN only", encoding="utf-8")
    stamps = runner.fingerprint(tmp_path)
    assert not any(Path(p).name.startswith(("valid_", "test_")) for p in stamps)
    with pytest.raises(FileNotFoundError, match="frozen_selection"):
        runner.run_valid(None, None, tmp_path, tmp_path, stamps)
