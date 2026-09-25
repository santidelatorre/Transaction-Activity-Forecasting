"""Regression checks for historical scoring, strict delivery and split isolation."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ubs_recurrence.data import LABELS, PREDICTION, TARGET, transactions
from ubs_recurrence.model import build_features
from ubs_recurrence.official import score_predictions


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_scoring_aligns_ids_and_keeps_absent_classes():
    target = pd.DataFrame(
        {
            "client_id": ["A", "B"],
            "cutoff_date": ["2026-01-01"] * 2,
            TARGET: ["cloud", "none"],
        }
    )
    prediction = pd.DataFrame({"client_id": ["B", "A"], PREDICTION: ["none", "cloud"]})
    assert score_predictions(target, prediction)["macro_f1"] == 2 / 8
    for bad in [
        prediction.iloc[:1],
        pd.concat([prediction, prediction.iloc[:1]]),
        prediction.assign(client_id=["C", "A"]),
    ]:
        with pytest.raises(ValueError):
            score_predictions(target, bad)


def test_historical_validator_rejects_wrong_order_blanks_and_schema(tmp_path):
    validator = load_script("validate_submission")
    sample = pd.DataFrame({"client_id": ["A", "B"], PREDICTION: ["none", "cloud"]})
    sample_path, path = tmp_path / "sample.csv", tmp_path / "submission.csv"
    sample.to_csv(sample_path, index=False)
    sample.to_csv(path, index=False)
    assert validator.inspect_submission(path, sample_path)[1]
    for bad in [
        sample.iloc[::-1],
        sample.assign(**{PREDICTION: ["", "cloud"]}),
        sample.assign(**{PREDICTION: [None, "cloud"]}),
        sample.assign(client_id=["A", "A"]),
        sample.assign(extra="x"),
        sample.assign(**{PREDICTION: ["invalid", "cloud"]}),
    ]:
        bad.to_csv(path, index=False)
        assert not validator.inspect_submission(path, sample_path)[1]


def test_explicit_raw_input_bypasses_caches_and_rejects_cutoff(tmp_path, monkeypatch):
    from ubs_recurrence import data

    monkeypatch.setattr(data, "ROOT", tmp_path)
    cache = tmp_path / "data/cache"
    cache.mkdir(parents=True)
    pd.DataFrame(
        {"timestamp": pd.to_datetime(["2025-01-01"], utc=True), "client_id": ["cached"]}
    ).to_parquet(cache / "train.parquet")
    raw = tmp_path / "explicit"
    raw.mkdir()
    (raw / "train_transactions.jsonl").write_text(
        '{"timestamp":"2026-01-01T00:00:00Z","client_id":"raw","mcc":"5732"}\n'
    )
    with pytest.raises(ValueError, match="cutoff"):
        transactions("train", data_dir=raw)


def test_all_final_feature_blocks_ignore_labels_ids_and_row_order():
    rows = []
    for cid in ["A", "B"]:
        for amount in [10.0, 20.0]:
            for n in range(5):
                rows.append(
                    {
                        "client_id": cid,
                        "timestamp": pd.Timestamp("2025-12-20", tz="UTC")
                        - pd.Timedelta(days=30 * n + amount / 10),
                        "amount": amount,
                        "currency": "chf",
                        "type": "card_payment",
                        "direction": "out",
                        "description": "cloud access",
                        "mcc": "5732",
                        "fee": 0.0,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["client_id", "timestamp"])
    profiles = {label: {"low": 1, "high": 1000} for label in LABELS[:-1]}
    original, _, old = build_features(frame, profiles, return_legacy=True)
    altered = frame.assign(
        client_id=frame.client_id.map({"A": "Z", "B": "Y"}), **{TARGET: "poison"}
    ).sample(frac=1, random_state=7)
    features, _, legacy = build_features(altered, profiles, return_legacy=True)
    order = pd.MultiIndex.from_product(
        [["Z", "Y"], LABELS], names=["client_id", "family"]
    )
    np.testing.assert_allclose(original.to_numpy(), features.reindex(order).to_numpy())
    for name in old:
        np.testing.assert_allclose(
            old[name].to_numpy(), legacy[name].reindex(order).to_numpy()
        )
    assert not any("client_id" in c or TARGET in c for c in original.columns)


def test_bootstrap_is_paired_and_reproducible():
    runner = load_script("run_ubs_stream_identity")
    y = np.tile(np.arange(8), 5)
    predictions = {name: y for name in ["V1", "V2", "Stream Identity"]}
    result = runner.bootstrap(y, predictions, repetitions=30)
    assert result == runner.bootstrap(y, predictions, repetitions=30)
    assert result["intervals"]["delta_stream_vs_V2"] == [0, 0]
