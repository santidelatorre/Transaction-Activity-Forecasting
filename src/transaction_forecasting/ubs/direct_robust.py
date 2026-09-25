"""Fold-local direct description features and reproducible textual shift."""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from transaction_forecasting.ubs.data import LABELS


def client_histories(transactions: pd.DataFrame, client_ids: pd.Index) -> list[pd.DataFrame]:
    """Return histories in caller order; IDs only locate rows and never become features."""
    if client_ids.has_duplicates:
        raise ValueError("Duplicate client IDs")
    grouped = dict(tuple(transactions.groupby("client_id", sort=False)))
    if set(grouped) != set(client_ids):
        raise ValueError("Client set mismatch")
    return [grouped[client] for client in client_ids]


def _unit(*parts: object) -> float:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


class LocalCorruptor:
    """Local TRAIN-derived shift simulator; replaceable by the official suite."""

    def fit(self, histories: list[pd.DataFrame]) -> LocalCorruptor:
        client_frequency: Counter[str] = Counter()
        for history in histories:
            client_frequency.update(set(history.description))
        self.generic_ = [name for name, _ in client_frequency.most_common(32)]
        return self

    def transform(self, histories: list[pd.DataFrame], level: str) -> list[pd.DataFrame]:
        if level == "clean":
            return [history.copy() for history in histories]
        if level not in {"medium", "severe"}:
            raise ValueError(level)
        intensity = {"medium": 0.35, "severe": 0.70}[level]
        result = []
        for history in histories:
            view = history.copy()
            client = history.client_id.iloc[0]
            stream_mask = {
                name: _unit(client, name, level, "stream") < intensity * 0.30
                for name in history.description.unique()
            }
            values = []
            for position, name in enumerate(history.description):
                roll = _unit(client, position, name, level, "row")
                if stream_mask[name] or roll < intensity * 0.18:
                    changed = "masked description"
                elif roll < intensity * 0.36 and self.generic_:
                    index = int(_unit(client, name, "generic") * len(self.generic_))
                    changed = self.generic_[index]
                elif roll < intensity * 0.68:
                    suffix = int(_unit(client, position, "alias") * 1000)
                    changed = f"{name} alias{suffix}"
                else:
                    changed = name
                values.append(changed)
            view["description"] = values
            result.append(view)
        return result


class DirectFeatures:
    """All learned vocabulary, document frequencies and IDF are fit on fold TRAIN."""

    def fit(self, histories: list[pd.DataFrame]) -> DirectFeatures:
        self._char_cache: dict[int, tuple[list[pd.DataFrame], sparse.csr_matrix]] = {}
        descriptions = [Counter(history.description) for history in histories]
        self.frequency_ = Counter()
        for counts in descriptions:
            self.frequency_.update(counts.keys())
        self.generic_ = {
            name for name, count in self.frequency_.items() if count >= max(3, len(histories) // 20)
        }
        self.full_ = DictVectorizer().fit(
            [{f"desc:{name}": 1 for name in counts} for counts in descriptions]
        )
        self.interaction_ = DictVectorizer().fit(self._interaction_dicts(histories))
        self.char_ = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            max_features=20000,
            sublinear_tf=True,
            dtype=np.float32,
        ).fit(self._documents(histories))
        return self

    @staticmethod
    def _documents(histories: list[pd.DataFrame]) -> list[str]:
        return [" \n ".join(history.description) for history in histories]

    @staticmethod
    def _interaction_dicts(histories: list[pd.DataFrame]) -> list[dict[str, float]]:
        return [
            {
                f"desc_mcc:{name}|{mcc}": min(count, 5)
                for (name, mcc), count in Counter(
                    zip(history.description, history.mcc, strict=True)
                ).items()
            }
            for history in histories
        ]

    def transform(self, histories: list[pd.DataFrame], representation: str) -> sparse.csr_matrix:
        if representation not in {"R0", "R1", "R2", "R3", "R4", "R5"}:
            raise ValueError(representation)
        counts = [Counter(history.description) for history in histories]
        binary = self.full_.transform([{f"desc:{name}": 1 for name in count} for count in counts])
        capped = self.full_.transform(
            [{f"desc:{name}": min(n, 5) for name, n in count.items()} for count in counts]
        )
        blocks: list[sparse.spmatrix] = [binary if representation == "R0" else capped]
        if representation in {"R2", "R3", "R4", "R5"}:
            key = id(histories)
            cached = self._char_cache.get(key)
            if cached is None or cached[0] is not histories:
                cached = (histories, self.char_.transform(self._documents(histories)))
                self._char_cache[key] = cached
            blocks.append(cached[1])
        if representation in {"R3", "R4"}:
            blocks.append(self.interaction_.transform(self._interaction_dicts(histories)))
        if representation in {"R4", "R5"}:
            numeric = []
            for count in counts:
                total = sum(count.values())
                shares = np.array(list(count.values()), dtype=float) / max(total, 1)
                numeric.append(
                    [
                        np.log1p(len(count)),
                        sum(n for name, n in count.items() if name in self.generic_)
                        / max(total, 1),
                        -float(np.sum(shares * np.log(shares + 1e-12))),
                        np.log1p(sum(n for n in count.values() if n >= 2)),
                        sum(n for name, n in count.items() if name not in self.generic_)
                        / max(total, 1),
                    ]
                )
            blocks.append(sparse.csr_matrix(np.asarray(numeric, dtype=np.float32)))
        # R5 combines full names, character signal and generic statistics,
        # dropping the high-dimensional MCC interactions for OOF comparison.
        return sparse.hstack(blocks, format="csr", dtype=np.float32)

    def names(self, representation: str) -> list[str]:
        names = list(self.full_.get_feature_names_out())
        if representation in {"R2", "R3", "R4", "R5"}:
            names += [f"char:{name}" for name in self.char_.get_feature_names_out()]
        if representation in {"R3", "R4"}:
            names += list(self.interaction_.get_feature_names_out())
        if representation in {"R4", "R5"}:
            names += [
                "unique_descriptions",
                "generic_share",
                "description_entropy",
                "recurrent_description_count",
                "specific_share",
            ]
        return names

    def coverage(self, histories: list[pd.DataFrame]) -> float:
        total = sum(len(history) for history in histories)
        known = sum(
            sum(name in self.frequency_ for name in history.description) for history in histories
        )
        return known / max(total, 1)


def normalized_view_weights(client_ids: list[str]) -> np.ndarray:
    """Every original client contributes total weight one, regardless of view count."""
    counts = Counter(client_ids)
    return np.array([1 / counts[client] for client in client_ids], dtype=float)


def ordered_probabilities(model: object, matrix: sparse.csr_matrix) -> np.ndarray:
    """Always emit the fixed official class order, including absent fit classes."""
    result = np.zeros((matrix.shape[0], len(LABELS)), dtype=float)
    raw = model.predict_proba(matrix)
    for column, label in enumerate(model.classes_):
        result[:, LABELS.index(label)] = raw[:, column]
    return result
