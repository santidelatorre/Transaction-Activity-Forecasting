"""Label-blind, deterministic merchant-text stress views and shift diagnostics.

Only streams anchored by outbound card payments change, including matching refund
and transfer events. Random keys use client IDs
for reproducibility, never as model features. Amounts, dates and all other columns
are copied untouched. Fit dictionaries on each outer training partition.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import CUTOFF

FAMILIES = ("dropout", "generic", "stream_mask", "fragmentation", "collision", "decoration")
REQUIRED = (
    "client_id",
    "description",
    "timestamp",
    "amount",
    "mcc",
    "type",
    "currency",
    "direction",
)


@dataclass(frozen=True)
class CorruptionLevel:
    dropout: float = 0.0
    generic: float = 0.0
    stream_mask: float = 0.0
    fragmentation: float = 0.0
    collision: float = 0.0
    decoration: float = 0.0
    aliases: int = 2

    def __post_init__(self):
        if any(not 0 <= getattr(self, name) <= 1 for name in FAMILIES) or self.aliases < 2:
            raise ValueError("Probabilities must be in [0,1] and aliases >=2")


LEVELS = {
    "clean": CorruptionLevel(),
    "mild": CorruptionLevel(0.02, 0.05, 0.05, 0.08, 0.04, 0.08, 2),
    "medium": CorruptionLevel(0.05, 0.12, 0.12, 0.18, 0.10, 0.18, 3),
    "severe": CorruptionLevel(0.10, 0.25, 0.25, 0.35, 0.20, 0.30, 4),
}


def validate_history(frame):
    if frame.empty or not set(REQUIRED).issubset(frame.columns):
        raise ValueError("Nonempty transaction history with required columns expected")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Null transaction values")
    timestamps = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    if timestamps.ge(CUTOFF).any():
        raise ValueError("History must be strictly before official cutoff")


def payments(frame):
    return frame.direction.eq("out") & frame.type.eq("card_payment")


def _tokens(value):
    return re.findall(r"[a-z0-9]+", str(value).lower())


def _digest(*parts):
    # Length-safe separation and a platform-independent seed, not Python hash().
    encoded = "".join(f"{len(str(part))}:{part}" for part in parts).encode()
    return hashlib.sha256(encoded).hexdigest()


class CorruptionSuite:
    """Fixed probabilities; TRAIN-derived generic tokens; no label argument.

    Each family selects using an independent deterministic hash. Selection keys
    always refer to the original stream, so probabilities are nested by severity.
    The composition order is generic, stream mask, collision, fragmentation,
    decoration, event dropout. Later families can overwrite earlier text.
    """

    def __init__(self, seed=20260925, levels=None):
        self.seed = seed
        self.levels = dict(LEVELS if levels is None else levels)
        if "clean" not in self.levels or self.levels["clean"] != CorruptionLevel():
            raise ValueError("Suite must include the exact clean identity level")

    def fit(self, history):
        validate_history(history)
        frame = history.loc[payments(history), ["description", "mcc"]].drop_duplicates()
        token_rows = [
            (token, str(row.mcc), row.description)
            for row in frame.itertuples()
            for token in set(_tokens(row.description))
        ]
        terms = pd.DataFrame(token_rows, columns=["token", "mcc", "description"])
        if terms.empty:
            self.generic_tokens_ = ()
        else:
            counts = terms.groupby("token").agg(
                contexts=("mcc", "nunique"), descriptions=("description", "nunique")
            )
            broad = counts.loc[counts.contexts.ge(3) & counts.descriptions.ge(10)]
            self.generic_tokens_ = tuple(
                broad.sort_values(
                    ["contexts", "descriptions", "token"], ascending=[False, False, True]
                )
                .head(12)
                .index
            )
        return self

    def config(self):
        return {
            "seed": self.seed,
            "levels": {k: asdict(v) for k, v in self.levels.items()},
            "scope": "whole client/description streams anchored by outgoing card payments",
            "fit_source": "outer TRAIN histories only",
            "generic_tokens": list(getattr(self, "generic_tokens_", ())),
            "generic_token_rule": "top12 tokens; >=3 MCCs and >=10 distinct descriptions",
            "order": [
                "generic",
                "stream_mask",
                "collision",
                "fragmentation",
                "decoration",
                "dropout",
            ],
        }

    def isolated(self, family, level="medium"):
        if family not in FAMILIES:
            raise ValueError("Unknown corruption family")
        source = self.levels[level]
        return replace(
            CorruptionLevel(), aliases=source.aliases, **{family: getattr(source, family)}
        )

    def transform(self, history, level="clean"):
        validate_history(history)
        config = self.levels[level] if isinstance(level, str) else level
        if not hasattr(self, "generic_tokens_"):
            raise RuntimeError("Fit suite on TRAIN histories before transforming")
        result = history.copy(deep=True)
        if config == CorruptionLevel():
            return result
        # Work in row positions: preserve even a duplicated caller index exactly.
        descriptions = history.description.to_numpy(copy=True)
        frame = history.reset_index(drop=True)
        keys = ["client_id", "description"]
        anchored = pd.MultiIndex.from_frame(frame.loc[payments(frame), keys])
        scoped = frame.loc[pd.MultiIndex.from_frame(frame[keys]).isin(anchored)]
        for (client, description), group in scoped.groupby(["client_id", "description"], sort=True):
            stream_key = _digest(self.seed, client, description)

            def draw(family, stream_key=stream_key):
                return int(_digest(stream_key, family)[:13], 16) / 16**13

            positions = group.index.to_numpy()
            values = np.full(len(group), str(description), dtype=object)
            if draw("generic") < config.generic:
                pool = self.generic_tokens_
                if pool:
                    start = int(stream_key[:8], 16) % len(pool)
                    generic = " ".join(
                        pool[(start + n) % len(pool)] for n in range(min(2, len(pool)))
                    )
                else:
                    generic = "__generic__"
                values[:] = generic
            if draw("stream_mask") < config.stream_mask:
                values[:] = f"__stream_{stream_key[:16]}__"
            if draw("collision") < config.collision:
                # Shared across descriptions only within the same MCC/type/currency/direction.
                values = np.array(
                    [
                        "__collision_"
                        + _digest(row.mcc, row.type, row.currency, row.direction)[:12]
                        + "__"
                        for row in group.itertuples()
                    ],
                    dtype=object,
                )
            if draw("fragmentation") < config.fragmentation and len(group) >= 2:
                # Contiguous temporal aliases, modelling an identity changing over time.
                order = np.argsort(group.timestamp.to_numpy(), kind="stable")
                buckets = np.empty(len(group), dtype=int)
                buckets[order] = (
                    np.arange(len(group)) * min(config.aliases, len(group)) // len(group)
                )
                values = np.array(
                    [
                        f"{value} #{bucket + 1}"
                        for value, bucket in zip(values, buckets, strict=True)
                    ],
                    dtype=object,
                )
            if draw("decoration") < config.decoration:
                # Purely syntactic: abbreviation plus punctuation, no external semantics.
                values = np.array(
                    [
                        "["
                        + " ".join(
                            token[:3] + "." if len(token) > 4 else token for token in _tokens(value)
                        )
                        + "]"
                        for value in values
                    ],
                    dtype=object,
                )
            for offset, row in enumerate(group.itertuples()):
                event_draw = int(_digest(stream_key, "dropout", row.timestamp)[:13], 16) / 16**13
                if event_draw < config.dropout:
                    values[offset] = "__description_missing__"
            descriptions[positions] = values
        result["description"] = descriptions
        return result


def shift_metrics(history, reference, suite, original=None):
    """Unlabeled outgoing-payment proxies; never infer real merchant truth."""
    validate_history(history)
    frame, ref = history.loc[payments(history)], reference.loc[payments(reference)]
    vocab, ref_vocab = set(frame.description), set(ref.description)
    freq = frame.description.value_counts(normalize=True)
    words = set(suite.generic_tokens_)
    generic = frame.description.map(
        lambda value: bool(_tokens(value)) and set(_tokens(value)) <= words
    )
    synthetic = frame.description.str.startswith(
        ("__generic", "__description_missing", "__collision")
    )
    counts = frame.groupby(["client_id", "description"]).size()
    monetary = frame.groupby(["client_id", "description", "currency"]).amount.agg(
        ["count", "mean", "std"]
    )
    cv = (monetary["std"] / monetary["mean"].abs().clip(lower=1e-9)).loc[monetary["count"].ge(2)]
    profile = {
        "clients": int(history.client_id.nunique()),
        "payment_events": len(frame),
        "generic_description_fraction": float((generic | synthetic).mean()) if len(frame) else None,
        "generic_proxy_definition": "TRAIN broad-token-only strings OR synthetic generic masks",
        "vocabulary_size": len(vocab),
        "vocabulary_overlap_jaccard": len(vocab & ref_vocab) / max(len(vocab | ref_vocab), 1),
        "unseen_description_rate": float((~frame.description.isin(ref_vocab)).mean())
        if len(frame)
        else None,
        "description_entropy_nats": float(-(freq * np.log(freq)).sum()),
        "streams_per_client": len(counts) / max(history.client_id.nunique(), 1),
        "singleton_stream_fraction": float(counts.eq(1).mean()) if len(counts) else None,
        "events_per_stream": float(counts.mean()) if len(counts) else None,
        "amount_cv_median_same_currency_repeated_streams": float(cv.median()) if len(cv) else None,
        "amount_cv_p90_same_currency_repeated_streams": float(cv.quantile(0.9))
        if len(cv)
        else None,
        "category_distributions": {
            name: {str(k): float(v) for k, v in frame[name].value_counts(normalize=True).items()}
            for name in ("mcc", "type", "currency")
        },
    }
    if original is not None:
        # Aligned untouched event positions provide exact synthetic fragmentation/collision audit.
        clean = original.loc[payments(original)]
        paired = pd.DataFrame(
            {
                "client": clean.client_id.to_numpy(),
                "before": clean.description.to_numpy(),
                "after": frame.description.to_numpy(),
            }
        )
        aliases = paired.groupby(["client", "before"]).after.nunique()
        merged = paired.groupby(["client", "after"]).before.nunique()
        profile.update(
            changed_event_fraction=float(history.description.ne(original.description).mean()),
            changed_payment_fraction=float(paired.before.ne(paired.after).mean())
            if len(paired)
            else None,
            mean_aliases_per_original_stream=float(aliases.mean()) if len(aliases) else None,
            fragmented_original_stream_fraction=float(aliases.gt(1).mean())
            if len(aliases)
            else None,
            collided_result_stream_fraction=float(merged.gt(1).mean()) if len(merged) else None,
        )
    return profile
