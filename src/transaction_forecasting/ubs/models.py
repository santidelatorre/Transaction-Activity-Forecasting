"""Interpretable heuristic and compact ML models for UBS V1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from transaction_forecasting.ubs.data import LABELS


def _reorder_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    positions = {str(label): position for position, label in enumerate(classes)}
    return np.column_stack([probabilities[:, positions[label]] for label in LABELS])


@dataclass
class RecurrenceHeuristic:
    """Rank train-learned family streams by cadence, recency, and amount stability."""

    none_bias: float = 0.0
    temperature: float = 1.0

    def _raw_scores(self, features: pd.DataFrame) -> np.ndarray:
        scores: list[np.ndarray] = []
        for label in LABELS:
            prefix = f"family_{label}"
            score = (
                np.log1p(features[f"{prefix}_recurrence_score"].to_numpy())
                + 0.15 * np.log1p(features[f"{prefix}_occurrences"].to_numpy())
                + 0.20 * features[f"{prefix}_description_lift"].to_numpy()
                + 0.10 * features[f"{prefix}_regularity"].to_numpy()
            )
            scores.append(score)
        matrix = np.column_stack(scores)
        matrix[:, LABELS.index("none")] += self.none_bias
        return matrix

    def tune(self, features: pd.DataFrame, target: pd.Series) -> RecurrenceHeuristic:
        """Select only calibration parameters on validation macro-F1."""
        best: tuple[float, float, float] | None = None
        for temperature in (0.5, 0.75, 1.0, 1.5, 2.0):
            for none_bias in np.arange(-2.0, 2.51, 0.25):
                self.temperature = float(temperature)
                self.none_bias = float(none_bias)
                predicted = self.predict(features)
                score = f1_score(target, predicted, labels=LABELS, average="macro")
                candidate = (
                    float(score),
                    -abs(float(none_bias)),
                    -abs(float(temperature) - 1.0),
                )
                if best is None or candidate > best:
                    best = candidate
                    best_parameters = (float(none_bias), float(temperature))
        self.none_bias, self.temperature = best_parameters
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = self._raw_scores(features) / self.temperature
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(LABELS)[self.predict_proba(features).argmax(axis=1)]


@dataclass
class ClientTextLogistic:
    """Regularized linear model over numeric aggregates plus train-only TF-IDF."""

    c: float = 1.0
    class_weight: str | None = None
    seed: int = 42
    max_text_features: int = 2500
    imputer: SimpleImputer = field(init=False)
    scaler: StandardScaler = field(init=False)
    vectorizer: TfidfVectorizer = field(init=False)
    model: LogisticRegression = field(init=False)
    numeric_columns_: list[str] = field(default_factory=list, init=False)

    def fit(
        self, features: pd.DataFrame, documents: pd.Series, target: pd.Series
    ) -> ClientTextLogistic:
        self.numeric_columns_ = features.columns.tolist()
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        numeric = self.imputer.fit_transform(features)
        numeric = self.scaler.fit_transform(numeric)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=self.max_text_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        text = self.vectorizer.fit_transform(documents)
        design = sparse.hstack([sparse.csr_matrix(numeric), text], format="csr")
        self.model = LogisticRegression(
            C=self.c,
            class_weight=self.class_weight,
            max_iter=2000,
            random_state=self.seed,
            solver="lbfgs",
            tol=1e-5,
        )
        self.model.fit(design, target)
        return self

    def _transform(self, features: pd.DataFrame, documents: pd.Series) -> sparse.csr_matrix:
        numeric = self.imputer.transform(features[self.numeric_columns_])
        numeric = self.scaler.transform(numeric)
        text = self.vectorizer.transform(documents)
        return sparse.hstack([sparse.csr_matrix(numeric), text], format="csr")

    def predict_proba(self, features: pd.DataFrame, documents: pd.Series) -> np.ndarray:
        probabilities = self.model.predict_proba(self._transform(features, documents))
        return _reorder_probabilities(probabilities, self.model.classes_)

    def predict(self, features: pd.DataFrame, documents: pd.Series) -> np.ndarray:
        return np.asarray(LABELS)[self.predict_proba(features, documents).argmax(axis=1)]

    def feature_importance(self) -> pd.DataFrame:
        text_names = self.vectorizer.get_feature_names_out().tolist()
        names = self.numeric_columns_ + [f"tfidf:{name}" for name in text_names]
        importance = np.abs(self.model.coef_).mean(axis=0)
        return pd.DataFrame({"feature": names, "importance": importance}).sort_values(
            "importance", ascending=False
        )


@dataclass
class CatBoostClientModel:
    """Gradient boosting on compact numeric client aggregates."""

    balanced: bool = False
    seed: int = 42
    iterations: int = 350
    depth: int = 6
    learning_rate: float = 0.05
    model: CatBoostClassifier = field(init=False)
    columns_: list[str] = field(default_factory=list, init=False)
    medians_: pd.Series = field(default_factory=pd.Series, init=False)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> CatBoostClientModel:
        self.columns_ = features.columns.tolist()
        self.medians_ = features.median().fillna(0.0)
        train = features.fillna(self.medians_)
        class_weights = None
        if self.balanced:
            weights = compute_class_weight(
                class_weight="balanced", classes=np.asarray(LABELS), y=target.to_numpy()
            )
            class_weights = {
                label: float(weight) for label, weight in zip(LABELS, weights, strict=True)
            }
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="MultiClass",
            random_seed=self.seed,
            class_weights=class_weights,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        self.model.fit(train, target)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = self.model.predict_proba(features[self.columns_].fillna(self.medians_))
        return _reorder_probabilities(probabilities, np.asarray(self.model.classes_))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(LABELS)[self.predict_proba(features).argmax(axis=1)]

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.columns_, "importance": self.model.get_feature_importance()}
        ).sort_values("importance", ascending=False)


def _balanced_sample_weights(target: pd.Series) -> np.ndarray:
    weights = compute_class_weight(
        class_weight="balanced", classes=np.asarray(LABELS), y=target.to_numpy()
    )
    mapping = {label: float(weight) for label, weight in zip(LABELS, weights, strict=True)}
    return target.map(mapping).to_numpy(dtype=float)


def _prepare_numeric_matrix(
    features: pd.DataFrame, columns: list[str], medians: pd.Series
) -> pd.DataFrame:
    return features[columns].fillna(medians)


@dataclass
class LightGBMClientModel:
    """LightGBM multiclass model on numeric client aggregates."""

    balanced: bool = False
    seed: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    model: Any = field(init=False)
    columns_: list[str] = field(default_factory=list, init=False)
    medians_: pd.Series = field(default_factory=pd.Series, init=False)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> LightGBMClientModel:
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as exc:  # pragma: no cover - environment specific
            raise RuntimeError(
                "LightGBM is unavailable in this environment (often missing libomp on macOS)"
            ) from exc

        self.columns_ = features.columns.tolist()
        self.medians_ = features.median().fillna(0.0)
        train = _prepare_numeric_matrix(features, self.columns_, self.medians_)
        sample_weight = _balanced_sample_weights(target) if self.balanced else None
        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            objective="multiclass",
            class_weight=None,
            random_state=self.seed,
            n_jobs=4,
            verbosity=-1,
        )
        self.model.fit(train, target, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        matrix = _prepare_numeric_matrix(features, self.columns_, self.medians_)
        probabilities = self.model.predict_proba(matrix)
        return _reorder_probabilities(probabilities, np.asarray(self.model.classes_))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(LABELS)[self.predict_proba(features).argmax(axis=1)]

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.columns_, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)


@dataclass
class XGBoostClientModel:
    """XGBoost multiclass model on numeric client aggregates."""

    balanced: bool = False
    seed: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1.0
    model: Any = field(init=False)
    columns_: list[str] = field(default_factory=list, init=False)
    medians_: pd.Series = field(default_factory=pd.Series, init=False)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> XGBoostClientModel:
        from xgboost import XGBClassifier

        self.columns_ = features.columns.tolist()
        self.medians_ = features.median().fillna(0.0)
        train = _prepare_numeric_matrix(features, self.columns_, self.medians_)
        sample_weight = _balanced_sample_weights(target) if self.balanced else None
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            objective="multi:softprob",
            num_class=len(LABELS),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
            verbosity=0,
        )
        encoded = pd.Categorical(target, categories=LABELS).codes
        self.model.fit(train, encoded, sample_weight=sample_weight)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        matrix = _prepare_numeric_matrix(features, self.columns_, self.medians_)
        probabilities = np.asarray(self.model.predict_proba(matrix), dtype=float)
        # Training used Categorical(LABELS).codes, so columns already follow LABELS.
        if probabilities.shape[1] != len(LABELS):
            raise ValueError("XGBoost probability width does not match LABELS")
        return probabilities

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(LABELS)[self.predict_proba(features).argmax(axis=1)]

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.columns_, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)


@dataclass
class TemperatureCalibrator:
    """Scale class logits on validation only; does not retrain the base model."""

    temperature: float = 1.0
    none_bias: float = 0.0

    def tune(
        self,
        probabilities: np.ndarray,
        target: pd.Series,
        *,
        temperatures: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
        none_biases: tuple[float, ...] | None = None,
    ) -> TemperatureCalibrator:
        if none_biases is None:
            none_biases = tuple(float(value) for value in np.arange(-1.5, 1.51, 0.25))
        best: tuple[float, float, float] | None = None
        best_parameters = (1.0, 0.0)
        for temperature in temperatures:
            for none_bias in none_biases:
                calibrated = self.apply(probabilities, temperature=temperature, none_bias=none_bias)
                predicted = np.asarray(LABELS)[calibrated.argmax(axis=1)]
                score = float(f1_score(target, predicted, labels=LABELS, average="macro"))
                candidate = (score, -abs(none_bias), -abs(temperature - 1.0))
                if best is None or candidate > best:
                    best = candidate
                    best_parameters = (float(temperature), float(none_bias))
        self.temperature, self.none_bias = best_parameters
        return self

    def apply(
        self,
        probabilities: np.ndarray,
        *,
        temperature: float | None = None,
        none_bias: float | None = None,
    ) -> np.ndarray:
        temperature = self.temperature if temperature is None else temperature
        none_bias = self.none_bias if none_bias is None else none_bias
        clipped = np.clip(probabilities, 1e-12, 1.0)
        logits = np.log(clipped)
        logits = logits / max(float(temperature), 1e-6)
        logits[:, LABELS.index("none")] += float(none_bias)
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def blend_probabilities(
    components: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Weighted soft-vote over aligned class probability matrices."""
    if len(components) != len(weights):
        raise ValueError("Probability components and weights must have the same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Blend weights must sum to a positive value")
    blended = np.zeros_like(components[0], dtype=float)
    for matrix, weight in zip(components, weights, strict=True):
        blended += (float(weight) / total) * matrix
    return blended


def predict_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS)[probabilities.argmax(axis=1)]
