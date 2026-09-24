"""Small, split-controlled comparison of UBS client-level classifiers.

This module does not construct features or choose a validation split. The caller
supplies client-indexed features, targets, partition IDs, and the official labels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelComparison:
    """Validation-only results; arrays follow ``validation_ids`` order."""

    summary: pd.DataFrame
    predictions: dict[str, pd.Series]
    probabilities: dict[str, pd.DataFrame]


def default_candidates(seed: int = 42) -> dict[str, Callable[[], object]]:
    """One fixed, modest setting per model; no validation tuning."""
    candidates: dict[str, Callable[[], object]] = {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
    }
    if find_spec("catboost") is not None:
        from catboost import CatBoostClassifier

        candidates["catboost"] = lambda: CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
    if find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        )
    if find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        candidates["xgboost"] = lambda: XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=4,
            tree_method="hist",
        )
    return candidates


def compare_classifiers(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    labels: Sequence[str],
    candidates: Mapping[str, Callable[[], object]] | None = None,
    include_ensemble: bool = True,
    seed: int = 42,
) -> ModelComparison:
    """Fit each candidate on identical train clients and score on identical valid clients.

    ``features`` must be numeric, one row per client, and built without validation
    labels or future transactions. Median imputation is fitted on train only.
    ``labels`` must be the fixed official vocabulary in its official order.
    """
    train_index = pd.Index(train_ids)
    valid_index = pd.Index(validation_ids)
    class_names = tuple(labels)
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("labels must be a nonempty unique class vocabulary")
    if train_index.empty or valid_index.empty:
        raise ValueError("Both partitions need at least one client")
    if not train_index.is_unique or not valid_index.is_unique:
        raise ValueError("Partition client IDs must be unique")
    if len(train_index.intersection(valid_index)):
        raise ValueError("Train and validation clients must be disjoint")
    if not features.index.is_unique or not target.index.is_unique:
        raise ValueError("Feature and target tables require unique client IDs")
    if features.empty or features.shape[1] == 0:
        raise ValueError("A nonempty numeric feature matrix is required")
    all_ids = train_index.append(valid_index)
    if not all_ids.isin(features.index).all() or not all_ids.isin(target.index).all():
        raise ValueError("Every partition client needs features and a target")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All comparison features must be numeric")
    values = features.loc[all_ids].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Features cannot contain infinite values")

    train_y = target.loc[train_index]
    valid_y = target.loc[valid_index]
    if train_y.isna().any() or valid_y.isna().any():
        raise ValueError("Targets cannot be missing")
    if not set(pd.concat([train_y, valid_y])).issubset(class_names):
        raise ValueError("Targets contain labels outside the supplied vocabulary")
    if set(train_y) != set(class_names):
        raise ValueError("Train must contain every supplied class for a fair multiclass comparison")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(features.loc[train_index])
    x_valid = imputer.transform(features.loc[valid_index])
    label_to_code = {label: code for code, label in enumerate(class_names)}
    y_train = train_y.map(label_to_code).to_numpy(dtype=int)
    labels_array = np.asarray(class_names)
    specs = default_candidates(seed) if candidates is None else dict(candidates)
    if not specs:
        raise ValueError("At least one candidate model is required")

    rows: list[dict[str, object]] = []
    predictions: dict[str, pd.Series] = {}
    probabilities: dict[str, pd.DataFrame] = {}
    for name, factory in specs.items():
        if name == "equal_probability_ensemble":
            raise ValueError("Candidate name is reserved for the ensemble")
        model = factory()
        model.fit(x_train, y_train)
        raw = np.asarray(model.predict_proba(x_valid), dtype=float)
        model_classes = np.asarray(model.classes_, dtype=int)
        if set(model_classes) != set(range(len(class_names))) or raw.shape != (
            len(valid_index),
            len(class_names),
        ):
            raise ValueError(f"{name} returned an incomplete class probability matrix")
        ordered = raw[:, np.argsort(model_classes)]
        if not np.isfinite(ordered).all():
            raise ValueError(f"{name} returned nonfinite probabilities")
        probabilities[name] = pd.DataFrame(ordered, index=valid_index, columns=class_names)
        predictions[name] = pd.Series(labels_array[ordered.argmax(axis=1)], index=valid_index)
        rows.append(
            {
                "model": name,
                "macro_f1": float(
                    f1_score(
                        valid_y,
                        predictions[name],
                        labels=class_names,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    if include_ensemble and len(probabilities) >= 2:
        name = "equal_probability_ensemble"
        average = np.mean([frame.to_numpy() for frame in probabilities.values()], axis=0)
        probabilities[name] = pd.DataFrame(average, index=valid_index, columns=class_names)
        predictions[name] = pd.Series(labels_array[average.argmax(axis=1)], index=valid_index)
        rows.append(
            {
                "model": name,
                "macro_f1": float(
                    f1_score(
                        valid_y,
                        predictions[name],
                        labels=class_names,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    return ModelComparison(
        summary=pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True),
        predictions=predictions,
        probabilities=probabilities,
    )
