import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting import gym_protection as gate
from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN


def probabilities():
    v2 = pd.DataFrame(0.025, index=["a", "b", "c", "d"], columns=LABELS)
    identity = v2.copy()
    v2["gym"] = [0.7, 0.3, 0.6, 0.15]
    v2["none"] = 1 - v2.drop(columns="none").sum(axis=1)
    identity["none"] = [0.6, 0.6, 0.7, 0.7]
    identity["gym"] = 1 - identity.drop(columns="gym").sum(axis=1)
    return v2, identity


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_v3_gym_protection.py"
    spec = importlib.util.spec_from_file_location("gym_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_confident_dominant_v2_gym_is_routed_and_axes_align():
    v2, identity = probabilities()
    original = identity.copy()
    result = gate.route_gym(v2, identity.iloc[::-1, ::-1], 0.4)
    assert result.to_dict() == {"a": "gym", "b": "none", "c": "none", "d": "none"}
    pd.testing.assert_frame_equal(original, identity)
    pd.testing.assert_series_equal(gate.route_gym(v2, identity, None), identity.idxmax(axis=1))


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        gate.route_gym(*probabilities(), threshold)


@pytest.mark.parametrize("problem", ["ids", "duplicate", "columns", "nan", "sum", "negative"])
def test_probability_contract(problem):
    v2, identity = probabilities()
    if problem == "ids":
        identity = identity.iloc[:-1]
    elif problem == "duplicate":
        identity.index = ["a", "a", "c", "d"]
    elif problem == "columns":
        identity = identity.drop(columns="music")
    elif problem == "nan":
        identity.iloc[0, 0] = np.nan
    elif problem == "sum":
        identity.iloc[0, 0] = 0.5
    else:
        identity.iloc[0, 0] = -0.1
    with pytest.raises(ValueError):
        gate.route_gym(v2, identity, 0.4)


def test_selection_keeps_noop_when_overrides_harm():
    v2, identity = probabilities()
    target = identity.idxmax(axis=1)
    selected, _ = gate.select_gate(target, v2, identity)
    assert selected is None


def test_gate_cv_excludes_held_fold_labels_from_selection(monkeypatch):
    v2, identity = probabilities()
    target = pd.Series(["gym", "none", "gym", "none"], index=v2.index)
    folds = pd.Series([1, 1, 2, 2], index=v2.index)
    seen = []
    original = gate.select_gate

    def checked(labels, left, right):
        seen.append(set(labels.index))
        assert set(labels.index) == set(left.index) == set(right.index)
        return original(labels, left, right)

    monkeypatch.setattr(gate, "select_gate", checked)
    prediction, records = gate.cross_validate_gate(target, v2, identity, folds)
    assert seen == [{"c", "d"}, {"a", "b"}]
    assert prediction.index.equals(target.index) and not prediction.isna().any()
    assert len(records) == 2


def test_overrides_distinguish_harm_from_wrong_to_wrong():
    target = pd.Series(["gym", "none", "music"], index=["a", "b", "c"])
    original = pd.Series(["none", "none", "cloud"], index=target.index)
    candidate = pd.Series("gym", index=target.index)
    result = gate.override_analysis(target, original, candidate)
    assert result["correct_overrides"] == 1
    assert result["incorrect_overrides"] == 2
    assert result["wrong_to_wrong"] == 1
    assert result["classes_losing_correct_predictions"] == {"none": 1}


def test_artifact_tampering_and_existing_outputs_rejected(tmp_path):
    module = runner()
    path = tmp_path / "freeze.json"
    module.write_json(path, {"threshold": 0.4})
    stamp = {str(path): module.sha256(path)}
    module.verify_hashes(stamp)
    with pytest.raises(FileExistsError):
        module.write_json(path, {"threshold": 0.5})
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Fingerprint"):
        module.verify_hashes(stamp)


def test_valid_requires_freeze_and_persists_predictions_before_labels(tmp_path, monkeypatch):
    module = runner()
    with pytest.raises(FileNotFoundError):
        module.valid_phase(tmp_path, tmp_path)
    frozen = {"fingerprints": {}, "environment": module.environment(), "threshold": 0.4}
    (tmp_path / "frozen_policy.json").write_text(json.dumps(frozen), encoding="utf-8")
    for name in ("valid_transactions.jsonl", "valid_labels.csv"):
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    v2, identity = probabilities()
    train_labels = pd.DataFrame({"client_id": ["train"], TARGET_COLUMN: ["gym"]})

    def labels(path):
        if path.name == "valid_labels.csv":
            assert (tmp_path / "valid_predictions.csv").is_file()
            assert (tmp_path / "valid_started.json").is_file()
            return pd.DataFrame({"client_id": v2.index, TARGET_COLUMN: ["gym"] * len(v2)})
        return train_labels

    def transactions(path):
        ids = ["train"] if path.name.startswith("train") else v2.index
        return pd.DataFrame({"client_id": ids})

    class Model:
        def fit(self, tx, labels):
            assert set(labels.client_id) == {"train"}
            return self

        def predict_components(self, tx):
            return {"V2": v2, "A": identity}

    monkeypatch.setattr(module, "read_labels", labels)
    monkeypatch.setattr(module, "read_transactions", transactions)
    monkeypatch.setattr(module, "V3Model", Model)
    module.valid_phase(tmp_path, tmp_path)
    assert json.loads((tmp_path / "valid_results.json").read_text())["threshold"] == 0.4
    with pytest.raises(FileExistsError):
        module.valid_phase(tmp_path, tmp_path)
