"""StreamV3PushEmbed: Push stack blended with unlabeled-pretrain description embeds.

Frozen discovery recipe (VALID Macro-F1 ~0.4570):
  0.85 * StreamV3PushModel + 0.15 * multinomial LR on client bags of
  char-TFIDF→SVD embeddings fitted on unlabeled_pretrain descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.text_v2 import normalize_description
from transaction_forecasting.ubs.v3_push import StreamV3PushModel

DEFAULT_PRETRAIN = Path("data/raw/ubs_2026/unlabeled_pretrain_transactions.jsonl")

NOISE = {
    "salary",
    "atm withdrawal",
    "fresh foods",
    "grocery store",
    "neighborhood market",
    "coffee shop",
    "casual dining",
    "pharmacy",
    "hotel booking",
    "ride share",
    "merchant charge",
    "online marketplace",
    "electronics shop",
    "p2p send",
    "p2p receive",
}


def _load_pretrain_descriptions(path: Path, *, max_rows: int) -> list[str]:
    import json

    descs: list[str] = []
    with path.open() as handle:
        for i, line in enumerate(handle):
            if i >= max_rows:
                break
            payload = json.loads(line)
            descs.append(normalize_description(payload.get("description", "")))
    return sorted(set(descs))


@dataclass
class StreamV3PushEmbedModel:
    """Push probability blend with a pretrain-embedding client classifier."""

    push_weight: float = 0.85
    pretrain_path: Path = field(default_factory=lambda: DEFAULT_PRETRAIN)
    pretrain_max_rows: int = 200_000
    svd_components: int = 32
    seed: int = 42
    v2_blend: float = 0.65
    music_alpha: float = 0.60
    streaming_alpha: float = 0.05

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> StreamV3PushEmbedModel:
        self.push_ = StreamV3PushModel(
            v2_blend=self.v2_blend,
            music_alpha=self.music_alpha,
            streaming_alpha=self.streaming_alpha,
            seed=self.seed,
        ).fit(transactions, labels)
        self.fit_clients_ = set(self.push_.fit_clients_)

        if not self.pretrain_path.exists():
            raise FileNotFoundError(f"Missing pretrain file: {self.pretrain_path}")
        uniq = _load_pretrain_descriptions(self.pretrain_path, max_rows=self.pretrain_max_rows)
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30_000
        )
        matrix = self.vectorizer_.fit_transform(uniq)
        self.svd_ = TruncatedSVD(n_components=self.svd_components, random_state=self.seed)
        self.svd_.fit(matrix)

        y = labels.set_index("client_id")[TARGET_COLUMN]
        clients = y.index.astype(str)
        embeds = self._client_embeds(transactions, clients)
        self.embed_clf_ = LogisticRegression(
            max_iter=1500, class_weight="balanced", C=0.8, random_state=self.seed
        )
        self.embed_clf_.fit(embeds, y.to_numpy())
        self.embed_classes_ = list(self.embed_clf_.classes_)
        return self

    def _client_embeds(self, transactions: pd.DataFrame, clients: pd.Index) -> np.ndarray:
        frame = transactions.copy()
        frame["client_id"] = frame["client_id"].astype(str)
        frame["description"] = frame["description"].map(normalize_description)
        frame = frame[~frame["description"].isin(NOISE)]
        docs = frame.groupby("client_id")["description"].agg(
            lambda series: " ".join(series.astype(str).unique()[:40])
        )
        docs = docs.reindex(clients.astype(str), fill_value="")
        return normalize(self.svd_.transform(self.vectorizer_.transform(docs)))

    def predict_components(self, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not hasattr(self, "push_"):
            raise RuntimeError("Fit StreamV3PushEmbedModel before predict")
        clients = set(transactions["client_id"].astype(str))
        if clients.intersection(self.fit_clients_):
            raise ValueError("Prediction clients must be disjoint from fit clients")
        push_comp = self.push_.predict_components(transactions)
        push = push_comp["blend"].reindex(columns=LABELS)
        embeds = self._client_embeds(transactions, push.index)
        embed = pd.DataFrame(
            self.embed_clf_.predict_proba(embeds),
            index=push.index,
            columns=self.embed_classes_,
        ).reindex(columns=LABELS, fill_value=0.0)
        w = float(self.push_weight)
        blend = w * push.to_numpy() + (1.0 - w) * embed.to_numpy()
        blend = blend / np.clip(blend.sum(axis=1, keepdims=True), 1e-12, None)
        blended = pd.DataFrame(blend, index=push.index, columns=LABELS)
        return {
            "push": push,
            "embed": embed,
            "blend": blended,
            "text_scores": push_comp["text_scores"],
        }

    def predict(self, transactions: pd.DataFrame) -> pd.Series:
        return self.predict_components(transactions)["blend"].idxmax(axis=1)
