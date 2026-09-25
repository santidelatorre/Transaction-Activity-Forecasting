"""Client-isolated stress evaluation for arbitrary history-to-client predictors.

``evaluate_under_corruption(model_factory, StressDataset(history, labels), suite)``
fits one fresh model per outer fold on CLEAN history. Inference views share the
held clients. It accepts indexed probability frames, dictionaries of components,
or indexed label predictions; no V3 dependency and no file-based label loading.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.evaluation.official import classification_metrics
from transaction_forecasting.ubs.corruption import FAMILIES, shift_metrics, validate_history
from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN


@dataclass(frozen=True)
class StressDataset:
    history: pd.DataFrame
    labels: pd.DataFrame


def client_folds(dataset, n_splits=5, seed=42):
    validate_history(dataset.history)
    labels = dataset.labels
    if labels.client_id.duplicated().any():
        raise ValueError("One label per client required")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    if set(target.index) != set(dataset.history.client_id) or not target.isin(LABELS).all():
        raise ValueError("History/label client mismatch or invalid labels")
    if not pd.to_datetime(labels.cutoff_date, utc=True).eq(CUTOFF).all():
        raise ValueError("Invalid label cutoff")
    if target.value_counts().min() < n_splits:
        raise ValueError("Not enough clients per class for requested folds")
    result = pd.Series(index=target.index, dtype=int, name="fold")
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (_, held) in enumerate(splitter.split(target.index, target), 1):
        result.iloc[held] = fold
    return result.astype(int)


def _predictions(fitted, history, ids):
    raw = (
        fitted.predict_proba(history)
        if hasattr(fitted, "predict_proba")
        else fitted.predict(history)
    )
    components = raw if isinstance(raw, dict) else {"model": raw}
    output = {}
    for name, value in components.items():
        if not isinstance(value, pd.Series | pd.DataFrame):
            raise ValueError("Predictors must return indexed client Series/DataFrames")
        if not value.index.is_unique or set(value.index) != set(ids):
            raise ValueError("Prediction client IDs must match held-out clients exactly")
        value = value.reindex(ids)
        if isinstance(value, pd.DataFrame):
            if list(value.columns) != list(LABELS):
                raise ValueError("Probabilities must use official label order")
            array = value.to_numpy(dtype=float)
            if (
                not np.isfinite(array).all()
                or (array < -1e-12).any()
                or (array > 1 + 1e-12).any()
                or not np.allclose(array.sum(axis=1), 1)
            ):
                raise ValueError("Invalid probabilities")
            prediction = value.idxmax(axis=1)
            confidence = {
                "max_probability_mean": float(array.max(axis=1).mean()),
                "entropy_nats_mean": float(
                    -(array * np.log(np.clip(array, 1e-12, 1))).sum(axis=1).mean()
                ),
                "none_probability_mean": float(array[:, LABELS.index("none")].mean()),
            }
        else:
            prediction, confidence = value, None
        if not prediction.isin(LABELS).all():
            raise ValueError("Invalid predicted labels")
        output[name] = (prediction, confidence, value)
    return output


def evaluate_under_corruption(
    model,
    dataset,
    suite,
    *,
    folds=None,
    n_splits=5,
    fold_seed=42,
    isolated=True,
    output_dir=None,
    progress=print,
):
    """Return pooled official scorecards, fold stability, shifts, and ablations.

    ``model`` is a zero-argument factory returning fit(history, labels)/predict[_proba].
    Split before creating views, fit suite dictionaries only on outer fit histories,
    then predict the same held-out clients for every view. No model augmentation or
    hyperparameter selection happens here. Supplied folds are checked for coverage.
    """
    started = perf_counter()
    canonical = client_folds(dataset, n_splits, fold_seed)
    folds = canonical if folds is None else folds
    if (
        not folds.index.is_unique
        or set(folds.index) != set(canonical.index)
        or folds.isna().any()
        or len(folds.unique()) < 2
    ):
        raise ValueError("Folds must assign every client exactly once")
    target = dataset.labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    labels_before = dataset.labels.copy(deep=True)
    out = Path(output_dir) if output_dir is not None else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        folds.to_csv(out / "fold_assignments.csv", index_label="client_id")
    views = dict(suite.levels)
    if isolated:
        views.update({f"only_{family}": suite.isolated(family) for family in FAMILIES})
    records, predictions, profiles, fit_times = [], {}, [], []
    for fold in sorted(folds.unique()):
        fit_ids, held_ids = folds.index[folds.ne(fold)], folds.index[folds.eq(fold)]
        fit = dataset.history.loc[dataset.history.client_id.isin(fit_ids)].copy()
        hold = dataset.history.loc[dataset.history.client_id.isin(held_ids)].copy()
        fit_labels = dataset.labels.loc[dataset.labels.client_id.isin(fit_ids)].copy()
        if set(fit.client_id) & set(hold.client_id):
            raise RuntimeError("Client crossed folds")
        fitted_suite = copy.deepcopy(suite).fit(fit)
        before_fit = perf_counter()
        progress(
            f"Fitting fold {fold}/{len(folds.unique())} on {len(fit_ids)} clean clients", flush=True
        )
        fitted = model().fit(fit, fit_labels)
        fit_times.append(
            {
                "fold": int(fold),
                "seconds": perf_counter() - before_fit,
                "fit_clients": len(fit_ids),
                "held_clients": len(held_ids),
                "suite": fitted_suite.config(),
            }
        )
        for view, config in views.items():
            start_view = perf_counter()
            changed = fitted_suite.transform(hold, config)
            pd.testing.assert_frame_equal(
                changed.drop(columns="description"), hold.drop(columns="description")
            )
            outputs = _predictions(fitted, changed, held_ids)
            profile = shift_metrics(changed, fit, fitted_suite, original=hold)
            profiles.append({"fold": int(fold), "view": view, **profile})
            for component, (prediction, confidence, raw) in outputs.items():
                score = classification_metrics(target.loc[held_ids], prediction)
                records.append(
                    {
                        "fold": int(fold),
                        "view": view,
                        "model": component,
                        "metrics": score,
                        "confidence": confidence,
                        "view_seconds_all_components": perf_counter() - start_view,
                    }
                )
                predictions.setdefault((component, view), []).append(prediction)
                if out:
                    raw.to_csv(out / f"fold_{fold}_{view}_{component}.csv", index_label="client_id")
            progress(
                f"fold {fold} / {view}: "
                + ", ".join(
                    f"{name}="
                    f"{classification_metrics(target.loc[held_ids], pred[0])['macro_f1']:.6f}"
                    for name, pred in outputs.items()
                ),
                flush=True,
            )
        pd.testing.assert_frame_equal(dataset.labels, labels_before)
    scorecards = {}
    for (component, view), pieces in predictions.items():
        pooled = pd.concat(pieces)
        if not pooled.index.is_unique or set(pooled.index) != set(target.index):
            raise RuntimeError("OOF coverage is incomplete or duplicated")
        pooled = pooled.reindex(target.index)
        scorecards.setdefault(component, {})[view] = classification_metrics(target, pooled)
        if out:
            pooled.rename("prediction").to_csv(
                out / f"oof_{view}_{component}.csv", index_label="client_id"
            )
    for component, cards in scorecards.items():
        clean = cards["clean"]
        for view, score in cards.items():
            scores = [
                r["metrics"]["macro_f1"]
                for r in records
                if r["model"] == component and r["view"] == view
            ]
            clean_scores = [
                r["metrics"]["macro_f1"]
                for r in records
                if r["model"] == component and r["view"] == "clean"
            ]
            score.update(
                drop_absolute=clean["macro_f1"] - score["macro_f1"],
                drop_relative=(clean["macro_f1"] - score["macro_f1"]) / clean["macro_f1"]
                if clean["macro_f1"]
                else None,
                accuracy_drop_absolute=clean["accuracy"] - score["accuracy"],
                fold_macro_f1_mean=float(np.mean(scores)),
                fold_macro_f1_std=float(np.std(scores, ddof=1)),
                fold_drop_absolute=(np.array(clean_scores) - scores).tolist(),
                per_class_f1_drop={
                    label: clean["per_class"][label]["f1"] - score["per_class"][label]["f1"]
                    for label in LABELS
                },
            )
    return {
        "schema_version": 1,
        "protocol": "clean fit; paired corrupt outer holdout",
        "suite": suite.config(),
        "fold_seed": fold_seed,
        "folds": len(folds.unique()),
        "scorecards": scorecards,
        "fold_results": records,
        "shift_by_fold": profiles,
        "fit_times": fit_times,
        "runtime_seconds": perf_counter() - started,
    }
