import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import CUTOFF, LABELS, PREDICTION_COLUMN, TARGET_COLUMN
from transaction_forecasting.ubs.v3.family_text import focused_correction
from transaction_forecasting.ubs.v3.features import (
    FamilyMap,
    cross_fitted_family_features,
    family_features,
    payment_streams,
    recurrence_features,
)


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
                        "direction": "out",
                        "type": "card_payment",
                        "currency": "chf",
                        "mcc": "5812",
                        "fee": 0.0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_map_rejects_self_future_missing_and_duplicate_labels():
    tx, labels = sample()
    mapper = FamilyMap().fit(tx, labels)
    with pytest.raises(ValueError, match="excluded"):
        mapper.transform(tx)
    unseen = tx.assign(client_id="new_" + tx.client_id)
    for bad in (CUTOFF, pd.NaT):
        with pytest.raises(ValueError):
            mapper.transform(unseen.assign(timestamp=bad))
    with pytest.raises(ValueError, match="Unique labels"):
        FamilyMap().fit(tx, pd.concat([labels, labels.iloc[:1]]))
    mapped = mapper.transform(unseen)
    assert set(mapped.family) == {*LABELS} - {"none"} | {"unknown"}
    assert mapper.transform(unseen.assign(description="unseen")).family.eq("unknown").all()


def test_crossfit_never_transforms_clients_in_its_fit(monkeypatch):
    tx, labels = sample()
    seen = []
    original = FamilyMap.transform

    def checked(self, frame):
        assert not self.fit_clients_.intersection(frame.client_id)
        seen.extend(frame.client_id.unique())
        return original(self, frame)

    monkeypatch.setattr(FamilyMap, "transform", checked)
    result = cross_fitted_family_features(tx, labels)
    assert sorted(seen) == sorted(labels.client_id)
    assert not result.isna().any().any()
    assert "client_id" not in result and TARGET_COLUMN not in result
    assert all("none" not in column for column in result)


def test_recurring_payments_exclude_refunds_and_keep_currencies_separate():
    tx, _ = sample()
    one = tx.loc[tx.client_id.eq("music_0")]
    original = payment_streams(one, ["client_id", "description", "currency"])
    refund = one.assign(direction="in", type="refund", amount=1000)
    combined = payment_streams(pd.concat([one, refund]), ["client_id", "description", "currency"])
    pd.testing.assert_frame_equal(original, combined)
    two_currencies = pd.concat([one, one.assign(currency="eur", amount=500)])
    streams = payment_streams(two_currencies, ["client_id", "description", "currency"])
    assert len(streams) == 2 and streams.amount_cv.eq(0).all()


def test_alias_timeline_is_explicit_and_unseen_does_not_invent_evidence():
    tx, _ = sample()
    one = tx.loc[tx.client_id.eq("music_0")].copy()
    one["family"] = "music"
    one.loc[one.index[::2], "description"] = "second alias"
    features = family_features(one)
    assert features.iloc[0]["identity_music_aliases"] == 2
    assert features.iloc[0]["family_music_streams"] == 2
    assert features.iloc[0]["union_music_streams"] == 1
    unknown = family_features(one.assign(family="unknown"))
    assert not unknown.to_numpy().any()
    assert recurrence_features(one.assign(type="transfer")).eq(0).all().all()


def test_focused_correction_preserves_none_and_aligns_ids():
    prediction = pd.Series(["none", "gym", "cloud"], index=["a", "b", "c"])
    scores = pd.DataFrame(
        [[0.99, 0.01], [0.9, 0.1], [0.8, 0.2]],
        index=["a", "b", "c"],
        columns=["music", "streaming"],
    )
    result = focused_correction(prediction, scores.iloc[::-1])
    assert result.tolist() == ["none", "music", "cloud"]
    with pytest.raises(ValueError, match="exactly match"):
        focused_correction(prediction, scores.iloc[:2])
    assert np.isfinite(scores.to_numpy()).all()


def test_factorial_model_uses_crossfit_and_preserves_probability_contract(monkeypatch):
    from transaction_forecasting.ubs.v3 import model as module

    fitted_columns = []

    class History:
        def transform(self, tx):
            return tx.groupby("client_id").size().to_frame("history_count")

    class V2:
        def fit(self, tx, labels):
            self.clients = set(labels.client_id)
            self.history_ = History()
            return self

        def predict_components(self, tx):
            if self.clients.intersection(tx.client_id):
                raise ValueError("excluded")
            uniform = pd.DataFrame(1 / 8, index=sorted(tx.client_id.unique()), columns=LABELS)
            return {"history": uniform, "blend": uniform}

    class Classifier:
        def fit(self, features, target):
            assert features.index.equals(target.index)
            assert "client_id" not in features and TARGET_COLUMN not in features
            assert not features.isna().any().any()
            fitted_columns.append(features.columns.tolist())
            self.prediction = len(fitted_columns)
            return self

        def predict_proba(self, features):
            probabilities = np.full((len(features), 8), 0.025)
            probabilities[:, self.prediction] = 0.825
            return probabilities

    monkeypatch.setattr(module, "IntegratedV2Model", V2)
    monkeypatch.setattr(module, "make_model", Classifier)
    tx, labels = sample()
    model = module.V3Model().fit(tx, labels)
    assert [len(columns) for columns in fitted_columns] == [22, 14, 217]
    unseen = tx.assign(client_id="new_" + tx.client_id)
    components = model.predict_components(unseen)
    expected = components["A"].idxmax(axis=1)
    pd.testing.assert_series_equal(model.predict(unseen), expected)
    for name, frame in components.items():
        if name != "A":
            assert not frame.idxmax(axis=1).equals(expected)
    pd.testing.assert_frame_equal(components["AB"], components["full"])
    for frame in components.values():
        assert list(frame.columns) == list(LABELS)
        assert np.allclose(frame.sum(axis=1), 1)
        assert frame.index.is_unique
    with pytest.raises(ValueError, match="excluded"):
        model.predict(tx)


def test_reproduction_refuses_stale_input_copy(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/reproduce_v3_discovery.py"
    spec = importlib.util.spec_from_file_location("reproduce_v3", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, copy = tmp_path / "input.txt", tmp_path / "snapshot.txt"
    source.write_text("first", encoding="utf-8")
    first = module.copy_verified_input(source, copy)
    assert module.copy_verified_input(source, copy) == first
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from source"):
        module.copy_verified_input(source, copy)
    assert copy.read_text(encoding="utf-8") == "first"


def test_runner_freezes_a_despite_better_diagnostics_and_submits_only_a(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_ubs_v3.py"
    spec = importlib.util.spec_from_file_location("run_ubs_v3", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, out = tmp_path / "data", tmp_path / "outputs"
    data.mkdir()
    tx, labels = sample()
    for split in ("train", "valid", "test"):
        split_tx = tx.assign(client_id=tx.client_id + f"_{split}")
        split_tx.to_json(data / f"{split}_transactions.jsonl", orient="records", lines=True)
        split_labels = labels.assign(client_id=labels.client_id + f"_{split}", cutoff_date=CUTOFF)
        if split != "test":
            split_labels.to_csv(data / f"{split}_labels.csv", index=False)
        else:
            sample_submission = split_labels[["client_id"]].assign(**{PREDICTION_COLUMN: "none"})
            sample_submission.to_csv(data / "sample_submission.csv", index=False)

    class DiagnosticModel:
        def fit(self, transactions, labels):
            self.mapper_ = self
            self.audit_ = pd.DataFrame()
            self.models_ = {}
            return self

    def diagnostic_predictions(model, text_model, transactions, directory, prefix):
        clients = pd.Index(sorted(transactions.client_id.unique()), name="client_id")
        truth = clients.str.split("_").str[0]
        frame = pd.DataFrame(
            {arm: truth for arm in ("V2", "B", "AB", "full", "ensemble", "focused")},
            index=clients,
        )
        frame["A"] = "cloud"
        return frame

    monkeypatch.setattr(module, "V3Model", DiagnosticModel)
    monkeypatch.setattr(module, "TextFamilyModel", DiagnosticModel)
    monkeypatch.setattr(module, "predictions", diagnostic_predictions)
    monkeypatch.setattr(module, "bootstrap_delta", lambda *args: {})
    argv = [str(script), "--data-dir", str(data), "--output-dir", str(out), "--phase"]
    monkeypatch.setattr(sys, "argv", [*argv, "all"])
    module.main()
    frozen = json.loads((out / "frozen_selection.json").read_text())
    assert frozen["candidate"] == "A"
    assert frozen["valid_labels_used_for_this_selection"] is False
    assert "promoted V3-A baseline" in frozen["criterion"]
    oof = json.loads((out / "oof_results.json").read_text())
    assert oof["selected_candidate"] == "A"
    for arm in ("B", "AB", "full", "ensemble", "focused"):
        assert oof["metrics"][arm]["macro_f1"] > oof["metrics"]["A"]["macro_f1"]
    valid = json.loads((out / "valid_results.json").read_text())
    assert valid["oof_selected_candidate"] == "A"
    submission = pd.read_csv(out / "submission_v3.csv")
    assert submission[PREDICTION_COLUMN].eq("cloud").all()
    assert submission.client_id.tolist() == sample_submission.client_id.tolist()

    for phase in ("valid", "submission"):
        monkeypatch.setattr(sys, "argv", [*argv, phase])
        for arm in ("B", "AB", "full", "ensemble", "focused"):
            module.write_json(out / "frozen_selection.json", {**frozen, "candidate": arm})
            with pytest.raises(ValueError, match="requires frozen candidate A"):
                module.main()
