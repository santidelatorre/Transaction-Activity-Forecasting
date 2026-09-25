"""Reproducible compact family ranker with an independent none detector."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.special import softmax
from xgboost import XGBRanker

from .compact import compact_features, global_features
from .data import LABELS
from .payment_context import payment_context
from .ranking import ranking_features
from .streams import extract_streams, family_features
from .templates import template_features


def build_features(df, profiles, *, min_count=3, return_legacy=False):
    ids = np.sort(df.client_id.unique())
    templates = template_features(df, ids)
    streams = extract_streams(df, 0.035, True, min_count=min_count)
    families = family_features(streams, ids, price_profiles=profiles)
    ranking = ranking_features(families, templates, ids, clocks=False)
    compact = compact_features(ranking, global_features(df, ids))
    context = payment_context(df, ranking)
    features = pd.concat([compact, context], axis=1).fillna(-999)
    expected = pd.MultiIndex.from_product([ids, LABELS], names=["client_id", "family"])
    if not features.index.equals(expected):
        raise ValueError("Family/client alignment failure")
    if return_legacy:
        hard = ranking_features(
            family_features(streams, ids), templates, ids, clocks=False
        )
        return features, streams, {"hard": hard, "soft": ranking}
    return features, streams


def none_features(features, ids):
    subset = features.drop(
        columns=["family_index", "is_none"], errors="ignore"
    ).replace(-999, np.nan)
    agg = (
        subset.groupby(level=0, sort=False)
        .agg(["min", "max", "mean"])
        .reindex(ids)
        .fillna(-999)
    )
    agg.columns = ["__".join(c) for c in agg.columns]
    return agg


def parameters(seed):
    return {
        "n_estimators": 500,
        "num_leaves": 15,
        "learning_rate": 0.035,
        "min_child_samples": 120,
        "reg_lambda": 10,
        "colsample_bytree": 0.95,
        "n_jobs": 4,
        "verbosity": -1,
        "random_state": int(seed),
    }


@dataclass
class FamilyForecaster:
    seeds: tuple = (42, 17, 2026)
    min_count: int = 3
    legacy_weight: float = 0.25
    device: str = "cuda"

    def fit(self, views, y, legacy_views=None):
        self.columns_ = list(views[0].columns)
        reference = views[0].index
        for view in views:
            if not view.index.equals(reference):
                raise ValueError("Augmented clients misaligned")
        self.training_ids_ = reference.get_level_values(0).unique().to_numpy()
        if len(y) != len(self.training_ids_):
            raise ValueError("Target alignment failure")
        allx = pd.concat(
            [v.reindex(columns=self.columns_).fillna(-999) for v in views],
            ignore_index=True,
        )
        target = (np.repeat(y, 8) == np.tile(np.arange(8), len(y))).astype(int)
        target = np.tile(target, len(views))
        nx = pd.concat(
            [none_features(v, self.training_ids_) for v in views], ignore_index=True
        )
        self.none_columns_ = list(nx.columns)
        self.models_ = []
        for seed in self.seeds:
            ranker = LGBMRanker(**parameters(seed), objective="lambdarank")
            ranker.fit(allx, target, group=np.full(len(target) // 8, 8))
            none = LGBMClassifier(**parameters(seed))
            none.fit(nx, np.tile(np.asarray(y) == 7, len(views)))
            self.models_.append((ranker, none))
            print("Fitted model seed", seed, flush=True)
        self.legacy_models_ = []
        if self.legacy_weight:
            if legacy_views is None:
                raise ValueError("Legacy feature views are required")
            for kind in ["hard", "soft", "xgb"]:
                source = "soft" if kind == "xgb" else kind
                columns = list(legacy_views[0][source].columns)
                data = pd.concat(
                    [
                        v[source].reindex(columns=columns).fillna(-999)
                        for v in legacy_views
                    ],
                    ignore_index=True,
                )
                if kind == "xgb":
                    estimator = XGBRanker(
                        n_estimators=500,
                        max_depth=4,
                        learning_rate=0.04,
                        min_child_weight=10,
                        subsample=0.85,
                        colsample_bytree=0.9,
                        reg_lambda=10,
                        n_jobs=4,
                        random_state=42,
                        objective="rank:pairwise",
                        tree_method="hist",
                        device=self.device,
                        lambdarank_pair_method="mean",
                        lambdarank_num_pair_per_sample=4,
                    )
                    estimator.fit(
                        data.astype(np.float32),
                        target,
                        group=np.full(len(target) // 8, 8),
                    )
                else:
                    estimator = LGBMRanker(
                        n_estimators=450,
                        num_leaves=15,
                        learning_rate=0.035,
                        min_child_samples=105,
                        colsample_bytree=0.9,
                        reg_lambda=5,
                        n_jobs=4,
                        verbosity=-1,
                        random_state=42,
                        objective="lambdarank",
                    )
                    estimator.fit(data, target, group=np.full(len(target) // 8, 8))
                self.legacy_models_.append((estimator, columns, source))
                print("Fitted complementary expert", kind, flush=True)
        return self

    def predict_proba(self, features, legacy_views=None):
        ids = features.index.get_level_values(0).unique().to_numpy()
        expected = pd.MultiIndex.from_product(
            [ids, LABELS], names=["client_id", "family"]
        )
        if not features.index.equals(expected):
            raise ValueError("Prediction candidate order mismatch")
        x = features.reindex(columns=self.columns_).fillna(-999)
        nx = none_features(x, ids).reindex(columns=self.none_columns_).fillna(-999)
        predictions = []
        for ranker, none in self.models_:
            q = softmax(ranker.predict(x).reshape(-1, 8).astype(float), axis=1)
            pn = none.predict_proba(nx)[:, 1]
            q[:, :7] = (
                q[:, :7]
                / np.maximum(q[:, :7].sum(axis=1, keepdims=True), 1e-12)
                * (1 - pn[:, None])
            )
            q[:, 7] = pn
            predictions.append(q)
        probability = np.mean(predictions, axis=0)
        if self.legacy_weight:
            if legacy_views is None:
                raise ValueError("Legacy feature views are required")
            old = []
            for estimator, columns, source in self.legacy_models_:
                frame = (
                    legacy_views[source]
                    .reindex(index=features.index, columns=columns)
                    .fillna(-999)
                )
                old.append(
                    softmax(
                        estimator.predict(frame).reshape(-1, 8).astype(float), axis=1
                    )
                )
            probability = (
                1 - self.legacy_weight
            ) * probability + self.legacy_weight * np.mean(old, axis=0)
        return probability
