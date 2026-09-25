"""Isolated TRAIN-only reconstruction; the historical predictor is unchanged."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker

from .augmentation import corrupt_transactions
from .compact import compact_features, global_features
from .data import CUTOFF, LABELS, TARGET
from .model import none_features, parameters
from .payment_context import payment_context
from .ranking import ranking_features
from .streams import extract_streams, family_features
from .templates import template_features

SCENARIOS = ("original", "medium", "severe")
SEEDS = (42, 17, 2026)
RAW_COLUMNS = (
    "client_id",
    "timestamp",
    "amount",
    "currency",
    "type",
    "direction",
    "description",
    "mcc",
    "fee",
)


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def canonical_transactions(frame):
    """Whitelist prevents accidental target/ID-derived extra columns becoming features."""
    data = frame.loc[:, RAW_COLUMNS].copy()
    if data.isna().any().any() or (data.timestamp >= CUTOFF).any():
        raise ValueError("Missing raw values or prediction cutoff violation")
    if (data.amount <= 0).any():
        raise ValueError("Expected positive transaction amounts")
    return data.sort_values(list(RAW_COLUMNS), kind="stable").reset_index(drop=True)


def corruption_view(frame, scenario, seed=2026):
    """Same label-free rates as history, but RNG ordering is independent of IDs."""
    data = canonical_transactions(frame)
    if scenario == "original":
        return data
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    mapping = {}
    for client, rows in data.groupby("client_id", sort=False):
        content = rows.drop(columns="client_id").to_json(
            orient="records", date_unit="ns"
        )
        mapping[client] = hashlib.sha256(content.encode()).hexdigest()
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Duplicate histories require an explicit corruption policy")
    inverse = {value: key for key, value in mapping.items()}
    data["client_id"] = data.client_id.map(mapping)
    historical = {"medium": "valid_like", "severe": "test_like"}[scenario]
    data = corrupt_transactions(data, historical, seed)
    data["client_id"] = data.client_id.map(inverse)
    return canonical_transactions(data)


def client_folds(ids, y, seed=42):
    ids, y = np.asarray(ids), np.asarray(y)
    if len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ValueError("Folds require one labelled row per unique client")
    if not np.array_equal(ids, np.sort(ids)):
        raise ValueError("Canonical client order is required")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y))
    for train, held in folds:
        if set(ids[train]) & set(ids[held]):
            raise ValueError("Client overlap")
    return folds


def candidate_index(ids):
    return pd.MultiIndex.from_product([ids, LABELS], names=["client_id", "family"])


def check_features(frame):
    ids = frame.index.get_level_values(0).unique().to_numpy()
    if not frame.index.equals(candidate_index(ids)):
        raise ValueError("Output class order or client grouping mismatch")
    if not frame.columns.is_unique or any(
        "client_id" in c or "target" in c or c == TARGET for c in frame.columns
    ):
        raise ValueError("Forbidden predictive column")
    if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
        raise ValueError("Features must be numeric")
    return ids


def category_schema(frames):
    """Fit exclusively on the raw training histories, including training views."""
    return {
        col: sorted({str(value) for frame in frames for value in frame[col].unique()})
        for col in ("type", "direction", "currency", "mcc")
    }


def project_schema(frame, schema):
    allowed = {
        f"client_{col}_{value}" for col, values in schema.items() for value in values
    }
    prefixes = tuple(f"client_{col}_" for col in schema)
    return frame.loc[
        :, [c for c in frame if not c.startswith(prefixes) or c in allowed]
    ]


def feature_banks(frame, profiles):
    """Only fixed external profiles and within-client transforms; no fitted schema."""
    data = canonical_transactions(frame)
    ids = np.sort(data.client_id.unique())
    templates = template_features(data, ids)
    streams = extract_streams(data, 0.035, True, min_count=3)
    global_x = global_features(data, ids)
    neutral = {f: {"low": 0.01, "high": 1e12} for f in LABELS[:-1]}
    banks = {}
    for mode, price in (("full", profiles), ("no_unlabeled", neutral), ("hard", None)):
        ranking = ranking_features(
            family_features(streams, ids, price), templates, ids, clocks=False
        )
        if mode != "full":
            ranking = ranking.drop(
                columns=[c for c in ranking if "price" in c or c.startswith("prior_")]
            )
        compact = compact_features(ranking, global_x)
        context = payment_context(data, ranking)
        banks[mode] = {
            "compact": pd.concat([compact, context], axis=1).fillna(-999),
            "legacy": ranking,
        }
        for matrix in banks[mode].values():
            check_features(matrix)
    return banks


@dataclass(frozen=True)
class Recipe:
    name: str = "full"
    feature_mode: str = "full"
    augment: bool = True
    seeds: tuple = SEEDS
    none_detector: bool = True
    legacy_weight: float = 0.25
    leaves: int = 15
    payments: bool = True
    cadence: bool = True
    postprocess: str = "hierarchical"
    temperature: float = 1.0
    prior_bias: float = 0.0

    def to_dict(self):
        return asdict(self)


def candidates():
    base = Recipe()
    variants = [
        ("no_unlabeled", {"feature_mode": "no_unlabeled"}),
        ("no_augmentations", {"augment": False}),
        ("no_multiseed", {"seeds": (42,)}),
        (
            "principal_lightgbm",
            {"seeds": (42,), "legacy_weight": 0.0, "none_detector": False},
        ),
        ("no_none_detector", {"none_detector": False}),
        ("no_soft_assignment", {"feature_mode": "hard"}),
        (
            "simple_postprocessing",
            {"none_detector": False, "postprocess": "raw_logits"},
        ),
        ("no_payment_context", {"payments": False}),
        ("no_cadence_thresholds", {"cadence": False}),
        ("seven_leaves", {"leaves": 7}),
        ("no_legacy", {"legacy_weight": 0.0}),
        ("legacy_half", {"legacy_weight": 0.5}),
        ("temperature_1_5", {"temperature": 1.5}),
        ("train_prior_bias", {"prior_bias": 0.25}),
    ]
    return [base] + [replace(base, name=name, **fields) for name, fields in variants]


def compact_matrix(bank, recipe):
    matrix = bank[recipe.feature_mode]["compact"]
    remove = []
    if not recipe.payments:
        remove += [
            c
            for c in matrix
            if any(t in c for t in ("refund", "fee", "hour", "weekend"))
            and not c.startswith("client_")
        ]
    if not recipe.cadence:
        remove += [c for c in matrix if "_active" in c or "period_rounded" in c]
    return matrix.drop(columns=sorted(set(remove)))


def legacy_source(recipe, kind):
    if kind == "hard" and recipe.feature_mode in ("full", "no_unlabeled"):
        return "hard"
    return recipe.feature_mode


def legacy_parameters(kind, seed=42):
    if kind == "xgb":
        return {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.04,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_lambda": 10,
            "n_jobs": 4,
            "random_state": seed,
            "objective": "rank:pairwise",
            "tree_method": "hist",
            "device": "cpu",
            "lambdarank_pair_method": "mean",
            "lambdarank_num_pair_per_sample": 4,
        }
    return {
        "n_estimators": 450,
        "num_leaves": 15,
        "learning_rate": 0.035,
        "min_child_samples": 105,
        "colsample_bytree": 0.9,
        "reg_lambda": 5,
        "n_jobs": 4,
        "verbosity": -1,
        "random_state": seed,
        "objective": "lambdarank",
    }


def view_weights(ids, n_views, n_candidates=1):
    """Equal total mass per client; preserve historical per-client mass of one view."""
    if not len(ids) or n_views < 1:
        raise ValueError("Empty training views")
    return np.full(len(ids) * n_views * n_candidates, 1.0 / n_views)


def rank_target(y, n_views):
    return np.tile(
        (np.repeat(y, 8) == np.tile(np.arange(8), len(y))).astype(int), n_views
    )


def combine(parts, recipe, prior):
    if recipe.postprocess == "raw_logits":
        main = np.mean(parts["rank"], axis=0)
        if recipe.legacy_weight:
            main = (1 - recipe.legacy_weight) * main + recipe.legacy_weight * np.mean(
                parts["legacy"], axis=0
            )
        result = softmax(main / recipe.temperature, axis=1)
    else:
        probabilities = []
        for i, scores in enumerate(parts["rank"]):
            p = softmax(scores / recipe.temperature, axis=1)
            if recipe.none_detector:
                pn = parts["none"][i]
                p[:, :7] *= (1 - pn[:, None]) / np.maximum(
                    p[:, :7].sum(axis=1, keepdims=True), 1e-12
                )
                p[:, 7] = pn
            probabilities.append(p)
        result = np.mean(probabilities, axis=0)
        if recipe.legacy_weight:
            old = np.mean(
                [softmax(p / recipe.temperature, axis=1) for p in parts["legacy"]],
                axis=0,
            )
            result = (1 - recipe.legacy_weight) * result + recipe.legacy_weight * old
    if recipe.prior_bias:
        result = softmax(
            np.log(np.maximum(result, 1e-12)) - recipe.prior_bias * np.log(prior),
            axis=1,
        )
    if (
        result.shape[1] != 8
        or not np.isfinite(result).all()
        or not np.allclose(result.sum(axis=1), 1)
    ):
        raise ValueError("Invalid probabilities")
    return result


class ComponentTrainer:
    """Fold-scoped component reuse; never reuse a fitted model across folds."""

    def __init__(self, banks, raw_views, train_ids, y):
        self.banks = banks
        self.ids = np.asarray(train_ids)
        self.y = np.asarray(y)
        if len(self.y) != len(self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError("Training target alignment")
        self.schema = category_schema(
            [d[d.client_id.isin(self.ids)] for d in raw_views.values()]
        )
        self.cache = {}
        self.prior = (np.bincount(y, minlength=8) + 1) / (len(y) + 8)

    def fit_component(self, recipe, kind, seed):
        if kind in ("rank", "none"):
            key = (
                kind,
                recipe.feature_mode,
                recipe.augment,
                recipe.leaves,
                recipe.payments,
                recipe.cadence,
                seed,
            )
        else:
            key = (kind, legacy_source(recipe, kind), recipe.augment, seed)
        if key in self.cache:
            return self.cache[key]
        scenarios = SCENARIOS if recipe.augment else ("original",)
        frames = []
        for scenario in scenarios:
            bank = self.banks[scenario]
            matrix = (
                compact_matrix(bank, recipe)
                if kind in ("rank", "none")
                else bank[legacy_source(recipe, kind)]["legacy"]
            )
            matrix = project_schema(matrix, self.schema).reindex(
                candidate_index(self.ids)
            )
            check_features(matrix)
            if kind == "none":
                matrix = none_features(matrix, self.ids)
            frames.append(matrix)
        columns = list(frames[0].columns)
        x = pd.concat(
            [f.reindex(columns=columns).fillna(-999) for f in frames], ignore_index=True
        )
        # Identical clients have equal mass. Fixed global rescaling preserves the
        # historical regularization scale (all clients have all three views).
        if kind == "none":
            params = parameters(seed) | {"num_leaves": recipe.leaves}
            model = LGBMClassifier(**params)
            weights = view_weights(self.ids, len(frames)) * len(frames)
            model.fit(x, np.tile(self.y == 7, len(frames)), sample_weight=weights)
        else:
            params = (
                parameters(seed)
                | {"num_leaves": recipe.leaves, "objective": "lambdarank"}
                if kind == "rank"
                else legacy_parameters(kind, seed)
            )
            model = XGBRanker(**params) if kind == "xgb" else LGBMRanker(**params)
            kwargs = {"group": np.full(len(self.ids) * len(frames), 8)}
            model.fit(
                x.astype(np.float32) if kind == "xgb" else x,
                rank_target(self.y, len(frames)),
                **kwargs,
            )
        fitted = (model, columns)
        self.cache[key] = fitted
        return fitted

    def fit(self, recipe):
        models = {"rank": [], "none": [], "legacy": []}
        for seed in recipe.seeds:
            models["rank"].append(self.fit_component(recipe, "rank", seed))
            if recipe.none_detector:
                models["none"].append(self.fit_component(recipe, "none", seed))
        if recipe.legacy_weight:
            for kind in ("hard", "soft", "xgb"):
                models["legacy"].append((kind, self.fit_component(recipe, kind, 42)))
        return CleanForecaster(recipe, models, self.schema, self.prior, self.ids)


@dataclass
class CleanForecaster:
    recipe: Recipe
    models: dict
    schema: dict
    prior: np.ndarray
    training_ids: np.ndarray

    def predict_parts(self, bank, ids):
        if set(ids) & set(self.training_ids):
            raise ValueError("Prediction clients overlap model training")
        matrix = compact_matrix(bank, self.recipe).reindex(candidate_index(ids))
        check_features(matrix)
        parts = {"rank": [], "none": [], "legacy": []}
        for model, columns in self.models["rank"]:
            parts["rank"].append(
                model.predict(matrix.reindex(columns=columns).fillna(-999))
                .reshape(-1, 8)
                .astype(float)
            )
        for model, columns in self.models["none"]:
            x = none_features(matrix, ids).reindex(columns=columns).fillna(-999)
            parts["none"].append(model.predict_proba(x)[:, 1])
        for kind, (model, columns) in self.models["legacy"]:
            x = (
                bank[legacy_source(self.recipe, kind)]["legacy"]
                .reindex(index=candidate_index(ids), columns=columns)
                .fillna(-999)
            )
            parts["legacy"].append(model.predict(x).reshape(-1, 8).astype(float))
        return parts

    def predict_proba(self, bank, ids):
        return combine(self.predict_parts(bank, ids), self.recipe, self.prior)


def verify_frozen(config_path):
    envelope = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if digest(envelope["configuration"]) != envelope["sha256"]:
        raise ValueError("Frozen configuration hash mismatch")
    return envelope
