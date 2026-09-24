"""StreamV3Push: equal probability stack of V3Max + Astra identity + StreamV3.

Frozen try recipe (VALID-reported Macro-F1 ~0.4445):
  1/3 StreamV3Max (V3 + soft music/streaming text)
  1/3 IdentityV3Model arm A (cross-fitted family identity CatBoost + heuristic)
  1/3 StreamV3Model (history + due-stream CatBoost @ v2_blend=0.65)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.v3 import StreamV3Model
from transaction_forecasting.ubs.v3_identity_model import IdentityV3Model
from transaction_forecasting.ubs.v3_max import StreamV3MaxModel


@dataclass
class StreamV3PushModel:
    """Equal-weight probability average of three complementary VALID-strong arms."""

    v2_blend: float = 0.65
    music_alpha: float = 0.60
    streaming_alpha: float = 0.05
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    seed: int = 42

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> StreamV3PushModel:
        self.vmax_ = StreamV3MaxModel(
            v2_blend=self.v2_blend,
            music_alpha=self.music_alpha,
            streaming_alpha=self.streaming_alpha,
            seed=self.seed,
        ).fit(transactions, labels)
        self.identity_ = IdentityV3Model(seed=self.seed).fit(transactions, labels)
        self.v3_ = StreamV3Model(v2_blend=self.v2_blend, seed=self.seed).fit(transactions, labels)
        self.fit_clients_ = set(self.v3_.fit_clients_)
        return self

    def predict_components(self, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not hasattr(self, "vmax_"):
            raise RuntimeError("Fit StreamV3PushModel before predict")
        clients = set(transactions["client_id"].astype(str))
        if clients.intersection(self.fit_clients_):
            raise ValueError("Prediction clients must be disjoint from fit clients")
        vmax = self.vmax_.predict_components(transactions)
        ident = self.identity_.predict_components(transactions)
        v3 = self.v3_.predict_components(transactions)
        frames = [
            vmax["blend"].reindex(columns=LABELS),
            ident["identity"].reindex(columns=LABELS),
            v3["blend"].reindex(columns=LABELS),
        ]
        index = frames[0].index
        for frame in frames[1:]:
            index = index.intersection(frame.index)
        mats = [frame.reindex(index).to_numpy() for frame in frames]
        w = np.asarray(self.weights, dtype=float)
        w = w / w.sum()
        blend = sum(weight * mat for weight, mat in zip(w, mats, strict=True))
        blend = blend / blend.sum(axis=1, keepdims=True)
        blended = pd.DataFrame(blend, index=index, columns=LABELS)
        return {
            "vmax": frames[0].reindex(index),
            "identity": frames[1].reindex(index),
            "stream_v3": frames[2].reindex(index),
            "blend": blended,
            "text_scores": vmax["text_scores"].reindex(index),
        }

    def predict(self, transactions: pd.DataFrame) -> pd.Series:
        return self.predict_components(transactions)["blend"].idxmax(axis=1)
