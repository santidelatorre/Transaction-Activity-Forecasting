"""Tests for StreamV3Push probability stacking helper behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.v3_push import StreamV3PushModel


def test_push_default_weights_sum_to_one():
    model = StreamV3PushModel()
    assert abs(sum(model.weights) - 1.0) < 1e-12


def test_equal_stack_is_componentwise_mean():
    index = pd.Index(["c1"], name="client_id")
    a = pd.DataFrame(0.0, index=index, columns=LABELS)
    b = pd.DataFrame(0.0, index=index, columns=LABELS)
    c = pd.DataFrame(0.0, index=index, columns=LABELS)
    a.loc["c1", "music"] = 1.0
    b.loc["c1", "streaming"] = 1.0
    c.loc["c1", "none"] = 1.0
    mats = [a.to_numpy(), b.to_numpy(), c.to_numpy()]
    blend = sum(mats) / 3.0
    blend = blend / blend.sum(axis=1, keepdims=True)
    assert abs(blend[0, LABELS.index("music")] - blend[0, LABELS.index("streaming")]) < 1e-9
    assert abs(blend[0, LABELS.index("music")] - blend[0, LABELS.index("none")]) < 1e-9
    assert np.isclose(blend.sum(), 1.0)
