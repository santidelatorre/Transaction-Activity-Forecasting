"""StreamV3Max: StreamV3 + soft music/streaming text boost (Javier signal).

Frozen try recipe (VALID-reported):
  1. StreamV3Model at v2_blend=0.65
  2. Soft-add per-family alpha * P_text into blend logits for music/streaming
  3. Never soft-penalize ``none`` (preserves Ulmans V2 none behaviour)

Defaults from VALID refine grids:
  music_alpha=0.60, streaming_alpha=0.05  → Macro-F1 ~0.4319
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS
from transaction_forecasting.ubs.stream_text import StreamTextFamilyModel
from transaction_forecasting.ubs.v3 import StreamV3Model


def soft_family_boost(
    base: pd.DataFrame,
    text_scores: pd.DataFrame,
    *,
    alphas: dict[str, float],
) -> pd.DataFrame:
    """Add alpha[family] * text probability to selected family logits, renormalize."""
    logits = np.log(np.clip(base.reindex(columns=LABELS).to_numpy(), 1e-12, 1.0))
    for family, alpha in alphas.items():
        if family not in LABELS or family not in text_scores.columns or alpha == 0:
            continue
        logits[:, LABELS.index(family)] += (
            float(alpha) * text_scores[family].reindex(base.index).fillna(0.0).to_numpy()
        )
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    proba = weights / weights.sum(axis=1, keepdims=True)
    return pd.DataFrame(proba, index=base.index, columns=LABELS)


@dataclass
class StreamV3MaxModel:
    """StreamV3 blend plus leakage-safe music/streaming text soft boost."""

    v2_blend: float = 0.65
    music_alpha: float = 0.60
    streaming_alpha: float = 0.05
    # Backward-compatible shared alpha; if set, overrides per-family values.
    text_alpha: float | None = None
    text_families: tuple[str, ...] = ("music", "streaming")
    seed: int = 42
    family_alphas_: dict[str, float] = field(init=False, repr=False)

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> StreamV3MaxModel:
        if self.text_alpha is not None:
            self.family_alphas_ = {family: float(self.text_alpha) for family in self.text_families}
        else:
            self.family_alphas_ = {
                "music": float(self.music_alpha),
                "streaming": float(self.streaming_alpha),
            }
        self.v3_ = StreamV3Model(v2_blend=self.v2_blend, seed=self.seed).fit(transactions, labels)
        self.text_ = StreamTextFamilyModel().fit(transactions, labels)
        self.fit_clients_ = set(self.v3_.fit_clients_)
        return self

    def predict_components(self, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not hasattr(self, "v3_"):
            raise RuntimeError("Fit StreamV3MaxModel before predict")
        clients = set(transactions["client_id"].astype(str))
        if clients.intersection(self.fit_clients_):
            raise ValueError("Prediction clients must be disjoint from fit clients")
        v3 = self.v3_.predict_components(transactions)
        text_scores, text_details = self.text_.predict_scores(transactions)
        text_scores = text_scores.reindex(v3["blend"].index).fillna(0.0)
        blended = soft_family_boost(v3["blend"], text_scores, alphas=self.family_alphas_)
        return {
            **v3,
            "text_scores": text_scores,
            "text_details": text_details,
            "blend_pre_text": v3["blend"],
            "blend": blended,
        }

    def predict(self, transactions: pd.DataFrame) -> pd.Series:
        return self.predict_components(transactions)["blend"].idxmax(axis=1)
