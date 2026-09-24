import numpy as np
import pandas as pd
import pytest

from test_ubs_v2 import histories
from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.v21 import (
    CrossFittedFamilyV21Model,
    MerchantHistoryV21Model,
    cross_fitted_family_features,
)


def labels_for_histories(transactions):
    return pd.DataFrame(
        {"client_id": transactions.client_id.unique(), TARGET_COLUMN: np.repeat(LABELS, 2)}
    )


def test_cross_fitted_family_features_exclude_each_held_client(monkeypatch):
    from transaction_forecasting.ubs import v21

    original = v21.ClientFeatureBuilder
    partitions = []

    class AuditedBuilder(original):
        def fit(self, transactions, labels):
            self.audit_fit_ids = set(labels.client_id)
            return super().fit(transactions, labels)

        def transform(self, transactions):
            held_ids = set(transactions.client_id)
            partitions.append((self.audit_fit_ids, held_ids))
            assert self.audit_fit_ids.isdisjoint(held_ids)
            return super().transform(transactions)

    monkeypatch.setattr(v21, "ClientFeatureBuilder", AuditedBuilder)
    transactions = histories()
    features = cross_fitted_family_features(
        transactions, labels_for_histories(transactions), n_splits=2
    )
    assert len(partitions) == 2
    assert features.shape == (16, 72)
    assert np.isfinite(features.to_numpy()).all()


def test_v21_model_uses_family_block_and_refuses_fit_clients(monkeypatch):
    from transaction_forecasting.ubs import v21

    class InspectModel:
        def fit(self, features, target):
            assert features.index.equals(target.index)
            assert sum(column.startswith("family_") for column in features) == 72
            self.columns = features.columns.tolist()
            return self

        def predict_proba(self, features):
            assert features.columns.tolist() == self.columns
            return np.full((len(features), len(LABELS)), 1 / len(LABELS))

    monkeypatch.setattr(v21, "make_model", InspectModel)
    transactions = histories()
    model = CrossFittedFamilyV21Model(n_splits=2).fit(
        transactions, labels_for_histories(transactions)
    )
    with pytest.raises(ValueError, match="excluded"):
        model.predict(transactions)
    unseen = transactions.assign(client_id="unseen_" + transactions.client_id)
    components = model.predict_components(unseen)
    assert set(components) == {"family_model", "blend"}
    assert components["blend"].columns.tolist() == list(LABELS)
    assert np.allclose(components["blend"].sum(axis=1), 1)


def test_merchant_history_model_uses_leave_one_client_out_training(monkeypatch):
    from transaction_forecasting.ubs import v21

    calls = []
    original = v21.MerchantFeatureBuilder

    class AuditedMerchantBuilder(original):
        def transform(self, transactions, *, training=False):
            calls.append(training)
            return super().transform(transactions, training=training)

    class InspectModel:
        def fit(self, features, target):
            assert features.index.equals(target.index)
            assert "merchant_class_music_mean" in features
            self.columns = features.columns.tolist()
            return self

        def predict_proba(self, features):
            assert features.columns.tolist() == self.columns
            return np.full((len(features), len(LABELS)), 1 / len(LABELS))

    monkeypatch.setattr(v21, "MerchantFeatureBuilder", AuditedMerchantBuilder)
    monkeypatch.setattr(v21, "make_model", InspectModel)
    transactions = histories()
    model = MerchantHistoryV21Model().fit(transactions, labels_for_histories(transactions))
    assert calls == [True]
    unseen = transactions.assign(client_id="unseen_" + transactions.client_id)
    probabilities = model.predict_proba(unseen)
    assert calls == [True, False]
    assert np.allclose(probabilities.sum(axis=1), 1)
