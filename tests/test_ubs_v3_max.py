"""Minimal tests for StreamV3Max soft music/streaming boost."""

from __future__ import annotations

import pandas as pd

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.v3_max import soft_family_boost


def test_soft_family_boost_increases_target_family():
    index = pd.Index(["c1", "c2"], name="client_id")
    base = pd.DataFrame(1.0 / len(LABELS), index=index, columns=LABELS)
    text = pd.DataFrame(0.0, index=index, columns=[c for c in LABELS if c != "none"])
    text.loc["c1", "music"] = 0.9
    out = soft_family_boost(base, text, alphas={"music": 2.0, "streaming": 0.1})
    assert out.loc["c1", "music"] > base.loc["c1", "music"]
    assert out.loc["c1"].idxmax() == "music"
    assert abs(out.sum(axis=1) - 1.0).max() < 1e-9


def test_soft_family_boost_respects_zero_alpha():
    index = pd.Index(["c1"], name="client_id")
    base = pd.DataFrame(1.0 / len(LABELS), index=index, columns=LABELS)
    text = pd.DataFrame(0.0, index=index, columns=[c for c in LABELS if c != "none"])
    text.loc["c1", "streaming"] = 0.99
    text.loc["c1", "music"] = 0.99
    out = soft_family_boost(base, text, alphas={"music": 2.0, "streaming": 0.0})
    assert out.loc["c1", "music"] > out.loc["c1", "streaming"]
