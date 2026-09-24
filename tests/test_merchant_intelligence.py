import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.merchant_intelligence import (
    FamilyEvidence,
    MerchantIntelligence,
    build_fingerprints,
    cross_fitted_summaries,
    resolve_merchant,
)


def sample():
    rows, labels = [], []
    for j, label in enumerate(LABELS):
        for n in range(6):
            client = f"{label}_{n}"
            labels.append({"client_id": client, TARGET_COLUMN: label})
            for month in (9, 10, 11, 12):
                rows.append(
                    {
                        "client_id": client,
                        "description": f"merchant {label}",
                        "timestamp": pd.Timestamp(f"2025-{month:02d}-01", tz="UTC"),
                        "amount": 10.0 * (j + 1),
                        "direction": "out",
                        "type": "card_payment",
                        "currency": "chf",
                        "mcc": str(1000 + j),
                        "fee": 0.0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_deterministic_embedding_serialization_fit_transform(tmp_path):
    tx, _ = sample()
    a, b = MerchantIntelligence(), MerchantIntelligence()
    first = a.fit_transform(tx)
    second = b.fit(tx.sample(frac=1, random_state=3)).transform(tx)
    np.testing.assert_allclose(first.toarray(), second.toarray())
    np.testing.assert_allclose(first.toarray(), a.transform(tx).toarray())
    a.build_graph()
    a.save(tmp_path / "index.joblib")
    restored = MerchantIntelligence.load(tmp_path / "index.joblib")
    np.testing.assert_allclose(first.toarray(), restored.transform(tx).toarray())
    assert a.retrieve(tx) == restored.retrieve(tx)


def test_text_cannot_override_incompatible_behavior_and_graph_has_no_chains():
    tx, _ = sample()
    tx["description"] = "generic payment"
    index = MerchantIntelligence().fit(tx)
    index.build_graph()
    for stream, matches in zip(
        build_fingerprints(tx).itertuples(), index.retrieve(tx), strict=True
    ):
        assert all(index.nodes_.iloc[node].mcc_mode == stream.mcc_mode for node, *_ in matches)
    # Same text/MCC but amounts 10 -> 19 -> 35 form a possible chaining trap.
    source = tx.loc[tx.client_id.eq("cloud_0")].copy()
    chain = pd.concat(
        [source.assign(client_id=f"c{i}", amount=amount) for i, amount in enumerate((10, 19, 35))]
    )
    chain_index = MerchantIntelligence().fit(chain)
    chain_index.build_graph()
    for cluster in set(chain_index.clusters_):
        nodes = chain_index.nodes_.loc[chain_index.clusters_ == cluster]
        for _, row in nodes.iterrows():
            assert chain_index._compatible(row, nodes).all()


def test_alias_mask_and_unseen_currency_abstention():
    tx, _ = sample()
    index = MerchantIntelligence().fit(tx)
    one = tx.loc[tx.client_id.eq("music_0")]
    expected = index.retrieve(one)[0][0][0]
    for alias in ("", "** merchnt music #543", "totally new alias", "payment"):
        matches = index.retrieve(one.assign(description=alias))[0]
        assert expected in [match[0] for match in matches]
    assert index.retrieve(one.assign(currency="unseen_currency")) == [[]]
    result = resolve_merchant(one, resolver=index)[0]
    assert result["family_evidence"] is None
    assert result["merchant_embedding"]["size"] == index.transform(one).shape[1]


def test_label_leakage_self_client_rejection_class_order_and_presence():
    tx, labels = sample()
    index = MerchantIntelligence().fit(tx)
    mapper = FamilyEvidence(index).fit(labels.iloc[::-1])
    streams = build_fingerprints(tx)
    with pytest.raises(ValueError, match="excluded"):
        mapper.transform(streams)
    unseen = streams.assign(client_id="new_" + streams.client_id)
    posterior = mapper.transform(unseen)
    np.testing.assert_allclose(posterior.sum(axis=1), 1)
    np.testing.assert_array_equal(
        np.asarray(LABELS)[posterior.argmax(axis=1)],
        unseen.description.str.replace("merchant ", ""),
    )
    # Duplicating references cannot multiply client votes.
    duplicates = MerchantIntelligence().fit_fingerprints(pd.concat([streams, streams]))
    other = FamilyEvidence(duplicates).fit(labels)
    np.testing.assert_allclose(posterior, other.transform(unseen))
    # Label-like columns must not enter embeddings.
    poisoned = tx.assign(**{TARGET_COLUMN: "private-label", "family": "hidden"})
    np.testing.assert_allclose(index.transform(tx).toarray(), index.transform(poisoned).toarray())


def test_crossfit_own_label_cannot_affect_own_features(monkeypatch):
    tx, labels = sample()
    index = MerchantIntelligence().fit(tx)
    streams = build_fingerprints(tx)
    observed = []
    original = FamilyEvidence.transform

    def guarded(self, query, neighbors=None):
        assert not set(query.client_id) & self.fit_clients_
        observed.extend(query.client_id)
        return original(self, query, neighbors)

    monkeypatch.setattr(FamilyEvidence, "transform", guarded)
    a = cross_fitted_summaries(index, streams, labels)
    client = labels.client_id.iloc[0]
    changed = labels.copy()
    changed.loc[changed.client_id.eq(client), TARGET_COLUMN] = "music"
    b = cross_fitted_summaries(index, streams, changed)
    pd.testing.assert_series_equal(a.loc[client], b.loc[client])
    assert set(observed) == set(labels.client_id)
    assert not a.isna().any().any()


def test_future_history_and_duplicate_or_unknown_labels_rejected():
    tx, labels = sample()
    with pytest.raises(ValueError, match="cutoff"):
        MerchantIntelligence().fit(tx.assign(timestamp=CUTOFF))
    with pytest.raises(ValueError, match="disjoint"):
        MerchantIntelligence().fit(tx, tx)
    index = MerchantIntelligence().fit(tx)
    with pytest.raises(ValueError, match="Unique"):
        FamilyEvidence(index).fit(pd.concat([labels, labels.iloc[:1]]))
    with pytest.raises(ValueError, match="Unique"):
        FamilyEvidence(index).fit(labels.assign(**{TARGET_COLUMN: "invalid"}))


def test_control_arm_is_exactly_v3_a(monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/experiments/v4_merchant_intelligence_christian.py"
    )
    spec = importlib.util.spec_from_file_location("merchant_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from transaction_forecasting.ubs.v3.model import arm_matrix

    history = pd.DataFrame({"history": [1.0, 2.0]}, index=["a", "b"])
    family = pd.DataFrame({"identity_music_count": [1.0, 0.0]}, index=history.index)
    merchant = pd.DataFrame({"mi_music_mean": [0.4, 0.2]}, index=history.index)
    matrices = module.ControlledModel.matrices(history, family, merchant)
    pd.testing.assert_frame_equal(matrices["A_baseline"], arm_matrix(history, family, None, "A"))
    assert list(matrices["B_add_summary"]) == ["history", "identity_music_count", "mi_music_mean"]
    assert list(matrices["C_replace_identity"]) == ["history", "mi_music_mean"]


def experiment_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/experiments/v4_merchant_intelligence_christian.py"
    )
    spec = importlib.util.spec_from_file_location("merchant_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corruptions_rebuild_real_split_alias_timelines():
    tx, _ = sample()
    index = MerchantIntelligence().fit(tx)
    index.build_graph()
    diagnostics = experiment_module().corruption_diagnostics(
        index, build_fingerprints(tx), tx, sample_size=8
    )
    assert diagnostics["alias_fragmentation"]["query_fragments"] == 16
    for name in ("description_masking", "typo_decorations", "generic_replacement"):
        assert diagnostics[name]["query_fragments"] == 8
        assert diagnostics[name]["original_top1_recovered_at_k"] == 1
    assert diagnostics["alias_fragmentation"]["exact_description_recovery"] == 0


def test_runner_never_reads_valid_during_oof_and_allows_one_frozen_evaluation(
    tmp_path, monkeypatch
):
    module = experiment_module()
    data, output = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    tx, labels = sample()
    for split in ("train", "valid", "unlabeled_pretrain"):
        suffix = "" if split == "train" else "_" + split
        split_tx = tx.assign(client_id=tx.client_id + suffix)
        split_tx.to_json(
            data / f"{split}_transactions.jsonl", orient="records", lines=True, date_format="iso"
        )
        if split != "unlabeled_pretrain":
            labels.assign(client_id=labels.client_id + suffix, cutoff_date=CUTOFF).to_csv(
                data / f"{split}_labels.csv", index=False
            )

    reads = []
    original_read = module.read_labels

    def read(path):
        reads.append(Path(path).name)
        return original_read(path)

    class CheapModel:
        def fit(self, train, labels, index, streams, neighbors):
            self.family = FamilyEvidence(index).fit(labels)
            return self

        def predict(self, transactions, streams, neighbors):
            clients = pd.Index(sorted(transactions.client_id.unique()), name="client_id")
            posterior = self.family.transform(streams, neighbors)
            merchant = module.summary_features(streams, neighbors, posterior, clients)
            identity = pd.DataFrame(
                1.0, index=clients, columns=[f"identity_{label}_count" for label in LABELS]
            )
            probabilities = pd.DataFrame(1 / len(LABELS), index=clients, columns=LABELS)
            return (
                {name: probabilities.copy() for name in module.ARMS},
                identity,
                merchant,
                posterior,
            )

    monkeypatch.setattr(module, "read_labels", read)
    monkeypatch.setattr(module, "ControlledModel", CheapModel)
    argv = [module.__file__, "--data-dir", str(data), "--output-dir", str(output), "--phase"]
    monkeypatch.setattr(sys, "argv", [*argv, "oof"])
    module.main()
    assert reads == ["train_labels.csv"]
    frozen = json.loads((output / "frozen_selection.json").read_text())
    assert not frozen["valid_labels_used"]
    assert frozen["selected"] == "A_baseline"
    monkeypatch.setattr(sys, "argv", [*argv, "valid"])
    module.main()
    assert reads.count("valid_labels.csv") == 1
    with pytest.raises(ValueError, match="already started"):
        module.main()
    assert reads.count("valid_labels.csv") == 1
    assert json.loads((output / "frozen_selection.json").read_text()) == frozen
