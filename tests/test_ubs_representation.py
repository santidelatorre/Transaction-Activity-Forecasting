"""Checks for the label-free denoising encoder. No challenge labels, no real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.representation import (
    DenoisingEncoder,
    client_grouped_oof,
    gap_bucket,
    label_free_client_stats,
)


def _history() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, labels = [], []
    for label in LABELS:
        for number in range(6):
            client = f"{label}_{number}"
            labels.append({"client_id": client, TARGET_COLUMN: label})
            for month in (8, 9, 10, 11, 12):
                rows.append(
                    {
                        "client_id": client,
                        "description": f"merchant recurring {label} plan",
                        "timestamp": pd.Timestamp(f"2025-{month:02d}-15", tz="UTC"),
                        "amount": 20.0 + month,
                        "direction": "out",
                        "type": "card_payment",
                        "currency": "chf",
                        "mcc": "5812",
                        "fee": 0.0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(labels)


def _encoder() -> DenoisingEncoder:
    tx, _ = _history()
    return DenoisingEncoder(
        n_components=4,
        max_descriptions=100,
        max_features=300,
        min_df=1,
        seed=42,
    ).fit(tx)


def test_pretraining_refuses_challenge_labels():
    tx, labels = _history()
    tagged = tx.copy()
    tagged[TARGET_COLUMN] = "none"
    with pytest.raises(ValueError, match="Challenge labels"):
        DenoisingEncoder(n_components=4, min_df=1, max_features=200).fit(tagged)
    with pytest.raises(ValueError, match="Challenge labels"):
        DenoisingEncoder(n_components=4, min_df=1, max_features=200).fit(tx, labels)


def test_encoder_is_deterministic_and_reloadable(tmp_path: Path):
    first = _encoder()
    second = _encoder()
    tx, _ = _history()
    left = first.client_features(tx)
    right = second.client_features(tx)
    pd.testing.assert_frame_equal(left, right)
    path = tmp_path / "encoder.joblib"
    first.save(path)
    loaded = DenoisingEncoder.load(path)
    pd.testing.assert_frame_equal(left, loaded.client_features(tx))
    assert "client_id" not in left.columns
    assert left.shape[1] == (first.n_components_ + 4) * 2


def test_fixed_class_order_and_client_grouped_folds():
    tx, labels = _history()
    encoder = _encoder()
    features = encoder.client_features(tx)
    target = labels.set_index("client_id")[TARGET_COLUMN].reindex(features.index)
    oof = client_grouped_oof(features, target, n_splits=3, seed=42)
    assert list(oof.columns) == list(LABELS)
    assert oof.index.is_unique
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-6)
    assert gap_bucket(np.array([1, 10, 40, 120])).tolist() == [0, 1, 2, 3]


def test_no_post_cutoff_transactions():
    tx, _ = _history()
    leaked = tx.copy()
    leaked.loc[leaked.index[:1], "timestamp"] = CUTOFF
    with pytest.raises(ValueError, match="cutoff"):
        _encoder().client_features(leaked)


def test_stats_do_not_require_labels_and_drift_is_finite():
    tx, _ = _history()
    stats = label_free_client_stats(tx)
    assert TARGET_COLUMN not in stats.columns
    assert stats.index.is_unique
    encoder = _encoder()
    drift = encoder.drift(tx, n_rows=40)
    assert drift["n_rows"] == 40
    assert np.isfinite(drift["denoised_l2_mean"])
    assert np.isfinite(drift["raw_corrupt_l2_mean"])
    probe = encoder.gap_probe(tx, max_pairs=80)
    assert probe["n_pairs"] > 0
    same = encoder.contrastive(tx, n_clients=16)
    assert same["n_clients"] >= 8
    assert same["same_client_cosine"] >= same["cross_client_cosine"] - 0.05
