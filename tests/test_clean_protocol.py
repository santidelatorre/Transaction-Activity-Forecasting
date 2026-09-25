"""Isolation, grouping, invariance and one-shot evaluation contracts."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from ubs_recurrence import clean_protocol as clean
from ubs_recurrence.data import LABELS, PREDICTION, TARGET
from ubs_recurrence.model import build_features


@pytest.fixture
def history():
    rows = []
    for client, shift in (("A", 0.0), ("B", 0.25)):
        for amount in (10.0, 20.0):
            for event in range(5):
                rows.append(
                    {
                        "client_id": client,
                        "timestamp": pd.Timestamp("2025-12-20", tz="UTC")
                        - pd.Timedelta(days=event * 30 + amount / 10),
                        "amount": amount + shift,
                        "currency": "chf",
                        "type": "card_payment",
                        "direction": "out",
                        "description": "cloud access",
                        "mcc": "5732",
                        "fee": 0.0,
                    }
                )
    return clean.canonical_transactions(pd.DataFrame(rows))


@pytest.fixture
def profiles():
    return {label: {"low": 1.0, "high": 1000.0} for label in LABELS[:-1]}


def test_fold_assignment_matches_baseline_and_groups_every_view():
    ids = np.array([f"C{i:04d}" for i in range(80)])
    y = np.tile(np.arange(8), 10)
    first = clean.client_folds(ids, y)
    second = clean.client_folds(ids, y)
    baseline = list(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y))
    for (train, held), (train2, held2), (_, expected) in zip(first, second, baseline):
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(held, held2)
        np.testing.assert_array_equal(held, expected)
        assert not set(np.tile(ids[train], 3)) & set(np.tile(ids[held], 3))
    np.testing.assert_array_equal(
        np.sort(np.concatenate([held for _, held in first])), np.arange(80)
    )
    with pytest.raises(ValueError):
        clean.client_folds(np.repeat("A", 80), y)


def test_category_vocabulary_is_fit_only(history):
    schema = clean.category_schema([history[history.client_id == "A"]])
    assert schema["mcc"] == ["5732"]
    validation = history[history.client_id == "B"].assign(mcc="UNSEEN")
    assert "UNSEEN" not in schema["mcc"]
    frame = pd.DataFrame(
        {"client_mcc_5732": [1], "client_mcc_UNSEEN": [1], "amount0_count": [2]}
    )
    projected = clean.project_schema(frame, schema)
    assert list(projected) == ["client_mcc_5732", "amount0_count"]
    assert validation.mcc.eq("UNSEEN").all()


def test_clean_path_is_identical_to_historical_features(history, profiles):
    historical, _, legacy = build_features(history, profiles, return_legacy=True)
    banks = clean.feature_banks(history, profiles)
    pd.testing.assert_frame_equal(banks["full"]["compact"], historical)
    pd.testing.assert_frame_equal(banks["full"]["legacy"], legacy["soft"])
    pd.testing.assert_frame_equal(banks["hard"]["legacy"], legacy["hard"])
    pd.testing.assert_frame_equal(clean.corruption_view(history, "original"), history)


@pytest.mark.parametrize("scenario", clean.SCENARIOS)
def test_corruption_ignores_ids_targets_and_input_order(history, scenario):
    first = clean.corruption_view(history, scenario)
    mapping = {"A": "Z", "B": "Y"}
    altered = history.assign(
        client_id=history.client_id.map(mapping), **{TARGET: "poison"}
    ).sample(frac=1, random_state=11)
    second = clean.corruption_view(altered, scenario)
    second["client_id"] = second.client_id.map({"Z": "A", "Y": "B"})
    pd.testing.assert_frame_equal(first, clean.canonical_transactions(second))
    assert TARGET not in second
    np.testing.assert_array_equal(first.amount, history.amount)


def test_features_are_local_to_clients_and_ignore_target(history, profiles):
    full = clean.feature_banks(history, profiles)
    train = clean.feature_banks(history[history.client_id == "A"], profiles)
    altered = history.assign(**{TARGET: "poison", "client_id_encoded": 123})
    poisoned = clean.feature_banks(altered, profiles)
    for mode in full:
        for kind, matrix in full[mode].items():
            pd.testing.assert_frame_equal(matrix, poisoned[mode][kind])
            pd.testing.assert_frame_equal(
                matrix.loc[train[mode][kind].index], train[mode][kind]
            )
            assert not any("client_id" in c or "target" in c for c in matrix)


@pytest.mark.parametrize("views", [1, 3, 4])
def test_client_weights_have_unit_total_for_any_view_count(views):
    weights = clean.view_weights(["A", "B", "C"], views)
    np.testing.assert_allclose(weights.reshape(views, 3).sum(axis=0), 1.0)


def test_output_class_order_and_none_probability_are_exact():
    scores = np.tile(np.arange(8), (3, 1))
    parts = {"rank": [scores], "none": [np.array([0.9, 0.1, 0.4])], "legacy": []}
    recipe = clean.Recipe(seeds=(42,), legacy_weight=0.0)
    p = clean.combine(parts, recipe, np.ones(8) / 8)
    assert p.shape == (3, 8)
    np.testing.assert_allclose(p[:, 7], [0.9, 0.1, 0.4])
    assert LABELS[7] == "none"
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    index = clean.candidate_index(["A"])
    matrix = pd.DataFrame({"family_index": np.arange(8)}, index=index)
    clean.check_features(matrix)
    with pytest.raises(ValueError, match="class order"):
        clean.check_features(matrix.iloc[::-1])
    with pytest.raises(ValueError, match="Forbidden"):
        clean.check_features(matrix.assign(client_id_encoded=1))


def test_future_and_missing_values_are_rejected(history):
    for altered in (
        history.assign(timestamp=pd.Timestamp("2026-01-01", tz="UTC")),
        history.assign(description=None),
    ):
        with pytest.raises(ValueError):
            clean.canonical_transactions(altered)


def test_freeze_hash_is_order_independent_and_detects_tampering(tmp_path):
    assert clean.digest({"a": 1, "b": 2}) == clean.digest({"b": 2, "a": 1})
    path = tmp_path / "frozen.json"
    config = {"recipe": "fixed", "seed": 42}
    clean.write_json(
        path, {"configuration": config, "sha256": clean.digest(config)}, exclusive=True
    )
    assert clean.verify_frozen(path)["configuration"] == config
    with pytest.raises(FileExistsError):
        clean.write_json(path, {}, exclusive=True)
    envelope = json.loads(path.read_text())
    envelope["configuration"]["seed"] = 17
    clean.write_json(path, envelope)
    with pytest.raises(ValueError, match="hash"):
        clean.verify_frozen(path)


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_stream_identity_clean.py"
    spec = importlib.util.spec_from_file_location("clean_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_valid_read_requires_persisted_predictions_and_blocks_retries(
    tmp_path, monkeypatch
):
    runner = load_runner()
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "RAW", tmp_path / "raw")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/valid_labels.csv").write_text(
        "synthetic fixture; parsing is interrupted"
    )
    monkeypatch.setattr(
        runner,
        "frozen_checked",
        lambda: {
            "sha256": "abc",
            "configuration": {
                "submission_minimum_valid_f1": 0.52,
                "official_data_manifest": {
                    "valid_labels.csv": {
                        "sha256": clean.file_hash(tmp_path / "raw/valid_labels.csv")
                    }
                },
            },
        },
    )
    frame = pd.DataFrame(np.eye(8), columns=["p_" + l for l in LABELS])
    frame.insert(0, "client_id", [f"C{i}" for i in range(8)])
    frame[PREDICTION] = LABELS
    frame.to_csv(tmp_path / "valid_probabilities.csv", index=False)
    clean.write_json(
        tmp_path / "prediction_receipt.json",
        {
            "config_sha256": "abc",
            "files": {
                "valid_probabilities.csv": clean.file_hash(
                    tmp_path / "valid_probabilities.csv"
                )
            },
        },
    )
    reads = []
    original = pd.read_csv

    def guard(path, *args, **kwargs):
        if Path(path).name == "valid_labels.csv":
            reads.append(path)
            assert (tmp_path / "valid_access_receipt.json").exists()
            assert (tmp_path / "valid_probabilities.csv").exists()
            raise RuntimeError("simulated interruption after first label access")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guard)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.evaluate()
    with pytest.raises(FileExistsError):
        runner.evaluate()
    assert len(reads) == 1


def test_development_cannot_resume_after_config_freeze(tmp_path, monkeypatch):
    runner = load_runner()
    path = tmp_path / "config.json"
    path.write_text("{}")
    monkeypatch.setattr(runner, "CONFIG", path)
    with pytest.raises(ValueError, match="closed"):
        runner.assert_development()


def test_evaluation_scores_persisted_predictions_once_without_retraining(
    tmp_path, monkeypatch
):
    runner = load_runner()
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "RAW", raw)
    monkeypatch.setattr(
        runner,
        "frozen_checked",
        lambda: {
            "sha256": "fixed",
            "configuration": {
                "submission_minimum_valid_f1": 1.1,
                "official_data_manifest": {
                    "valid_labels.csv": {
                        "sha256": clean.file_hash(raw / "valid_labels.csv")
                    }
                },
            },
        },
    )
    ids = [f"C{i}" for i in range(8)]
    prediction = pd.DataFrame(np.eye(8), columns=["p_" + label for label in LABELS])
    prediction.insert(0, "client_id", ids)
    prediction[PREDICTION] = LABELS
    prediction.to_csv(tmp_path / "valid_probabilities.csv", index=False)
    pd.DataFrame(
        {"client_id": ids, "cutoff_date": "2026-01-01", TARGET: LABELS}
    ).to_csv(raw / "valid_labels.csv", index=False)
    clean.write_json(
        tmp_path / "prediction_receipt.json",
        {
            "config_sha256": "fixed",
            "files": {
                "valid_probabilities.csv": clean.file_hash(
                    tmp_path / "valid_probabilities.csv"
                )
            },
        },
    )
    runner.evaluate()
    result = json.loads((tmp_path / "valid_metrics.json").read_text())
    assert result["macro_f1"] == 1.0
    assert result["official"]["label_order"] == LABELS
    with pytest.raises(FileExistsError):
        runner.evaluate()


def test_candidate_catalog_is_small_fixed_and_includes_required_ablations():
    catalog = clean.candidates()
    assert len(catalog) == 15
    names = {r.name for r in catalog}
    assert {
        "full",
        "no_unlabeled",
        "no_augmentations",
        "no_multiseed",
        "principal_lightgbm",
        "no_none_detector",
        "no_soft_assignment",
        "simple_postprocessing",
    } <= names
    no_prices = next(r for r in catalog if r.name == "no_unlabeled")
    assert clean.legacy_source(no_prices, "hard") == "hard"
    assert clean.legacy_source(no_prices, "soft") == "no_unlabeled"
    no_none = replace(clean.Recipe(), none_detector=False, legacy_weight=0.0)
    p = clean.combine({"rank": [np.zeros((2, 8))]}, no_none, np.ones(8) / 8)
    np.testing.assert_allclose(p, 1 / 8)


def test_all_estimators_fit_only_training_clients_and_project_unseen_categories(
    monkeypatch,
):
    ids = np.array([f"C{i:03d}" for i in range(48)])
    y = np.tile(np.arange(8), 6)
    index = clean.candidate_index(ids)
    rng = np.random.default_rng(7)
    matrix = pd.DataFrame(
        rng.normal(size=(len(index), 4)),
        index=index,
        columns=[
            "amount0_count",
            "amount0_last_age",
            "client_mcc_5732",
            "client_mcc_UNSEEN",
        ],
    )
    matrix["family_index"] = np.tile(np.arange(8), len(ids))
    matrix["is_none"] = matrix.family_index.eq(7).astype(int)
    raw = pd.DataFrame(
        {
            "client_id": ids,
            "type": "card_payment",
            "direction": "out",
            "currency": "chf",
            "mcc": ["5732"] * 40 + ["UNSEEN"] * 8,
        }
    )
    banks = {
        scenario: {
            mode: {"compact": matrix, "legacy": matrix}
            for mode in ("full", "hard", "no_unlabeled")
        }
        for scenario in clean.SCENARIOS
    }
    original_params, original_legacy = clean.parameters, clean.legacy_parameters
    monkeypatch.setattr(
        clean,
        "parameters",
        lambda seed: (
            original_params(seed) | {"n_estimators": 8, "min_child_samples": 2}
        ),
    )
    monkeypatch.setattr(
        clean,
        "legacy_parameters",
        lambda kind, seed=42: original_legacy(kind, seed) | {"n_estimators": 8},
    )
    trainer = clean.ComponentTrainer(
        banks, {s: raw for s in clean.SCENARIOS}, ids[:40], y[:40]
    )
    model = trainer.fit(clean.Recipe(seeds=(42,)))
    assert model.schema["mcc"] == ["5732"]
    assert "client_mcc_UNSEEN" not in model.models["rank"][0][1]
    p = model.predict_proba(banks["original"], ids[40:])
    assert p.shape == (8, 8)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="overlap"):
        model.predict_proba(banks["original"], ids[:8])


def test_source_hash_survives_git_checkout_line_endings(tmp_path):
    runner = load_runner()
    unix, windows = tmp_path / "unix.py", tmp_path / "windows.py"
    unix.write_bytes(b"x = 1\ny = 2\n")
    windows.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert runner.source_hash(unix) == runner.source_hash(windows)
    windows.write_bytes(b"x = 3\r\ny = 2\r\n")
    assert runner.source_hash(unix) != runner.source_hash(windows)


def test_stability_and_permutation_cannot_reselect_the_recipe(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "CONFIG", tmp_path / "not_frozen.json")
    monkeypatch.setattr(runner, "selected_recipe", lambda: clean.Recipe())
    cache = {"y": np.tile(np.arange(8), 10)}
    monkeypatch.setattr(runner, "load_cache", lambda: cache)
    monkeypatch.setattr(runner, "invariance", lambda data: None)
    selected = tmp_path / "selected_train_only.json"
    selected.write_text('{"selection": "must remain unchanged"}')
    clean.write_json(
        tmp_path / "development/summary.json",
        {"results": {"full": {"scenarios": {"original": {"macro_f1": 0.6}}}}},
    )
    calls = []

    def cv(data, recipes, seed, name, *, target=None):
        calls.append((name, seed, tuple(r.seeds for r in recipes), target is not None))
        score = 0.125 if target is not None else 0.6
        return {
            "results": {
                r.name: {"scenarios": {"original": {"macro_f1": score}}}
                for r in recipes
            }
        }

    monkeypatch.setattr(runner, "run_cv", cv)
    runner.stability()
    assert {c[0] for c in calls} == {
        "stability_17",
        "stability_2026",
        "model_seed_stability",
        "target_permutation",
    }
    assert next(c for c in calls if c[0] == "target_permutation")[3]
    assert (
        json.loads((tmp_path / "permutation_sanity.json").read_text())[
            "permuted_macro_f1"
        ]
        == 0.125
    )
    assert selected.read_text() == '{"selection": "must remain unchanged"}'
