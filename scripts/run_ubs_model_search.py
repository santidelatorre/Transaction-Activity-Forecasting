"""Controlled UBS model search on the official V1 feature matrix.

Trains only on train labels, selects on validation macro-F1, then optionally
refits the winning recipe on train+valid for the test submission. Does not
learn vocabularies or hyperparameters from test.
"""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from transaction_forecasting.evaluation.experiments import save_experiment
from transaction_forecasting.ubs.data import (
    LABELS,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    load_ubs_data,
    validate_submission,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.features import ClientFeatureBuilder, build_client_documents
from transaction_forecasting.ubs.models import (
    CatBoostClientModel,
    ClientTextLogistic,
    LightGBMClientModel,
    RecurrenceHeuristic,
    TemperatureCalibrator,
    XGBoostClientModel,
    blend_probabilities,
    predict_from_probabilities,
)


def load_settings(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as config_file:
        return tomllib.load(config_file)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def indexed_prediction(values: np.ndarray, client_ids: pd.Index) -> pd.Series:
    return pd.Series(values, index=client_ids, name=PREDICTION_COLUMN)


def score(y_valid: pd.Series, prediction: np.ndarray, client_ids: pd.Index) -> dict[str, object]:
    return evaluate_predictions(y_valid, indexed_prediction(prediction, client_ids))


def record_row(
    *,
    name: str,
    family: str,
    metrics: dict[str, object],
    notes: str,
    train_seconds: float,
    inference_seconds: float,
    feature_count: int,
    params: dict[str, object],
) -> dict[str, object]:
    return {
        "model": name,
        "family": family,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "train_seconds": round(train_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "feature_count": feature_count,
        "params": params,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ubs_model_search.toml")
    parser.add_argument(
        "--skip-submission",
        action="store_true",
        help="Only run validation search; do not refit or write a test CSV.",
    )
    arguments = parser.parse_args()
    settings = load_settings(arguments.config)
    seed = int(settings["project"]["random_seed"])
    baseline_macro_f1 = float(settings["project"]["baseline_macro_f1"])
    random.seed(seed)
    np.random.seed(seed)

    data = load_ubs_data(settings["data"]["directory"])
    builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
    x_train = builder.transform(data.train_transactions)
    x_valid = builder.transform(data.valid_transactions)
    y_train = data.train_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_train.index)
    y_valid = data.valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_valid.index)
    if y_train.isna().any() or y_valid.isna().any():
        raise ValueError("Feature/label alignment produced missing targets")
    train_documents = build_client_documents(data.train_transactions, x_train.index)
    valid_documents = build_client_documents(data.valid_transactions, x_valid.index)

    search = settings["search"]
    outputs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, object]] = []
    metrics_dir = Path(settings["outputs"]["metrics_directory"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(settings["outputs"]["experiments_jsonl"])
    if jsonl_path.exists():
        jsonl_path.unlink()

    # --- Heuristic reference (same protocol as V1: calibrate on validation) ---
    started = perf_counter()
    heuristic = RecurrenceHeuristic().tune(x_valid, y_valid)
    train_seconds = perf_counter() - started
    started = perf_counter()
    heuristic_probabilities = heuristic.predict_proba(x_valid)
    heuristic_prediction = predict_from_probabilities(heuristic_probabilities)
    inference_seconds = perf_counter() - started
    heuristic_metrics = score(y_valid, heuristic_prediction, x_valid.index)
    outputs["recurrence_heuristic"] = {
        "kind": "heuristic",
        "probabilities": heuristic_probabilities,
        "prediction": heuristic_prediction,
        "model": heuristic,
        "feature_count": len(LABELS) * 4,
    }
    row = record_row(
        name="recurrence_heuristic",
        family="heuristic",
        metrics=heuristic_metrics,
        notes=f"none_bias={heuristic.none_bias:.2f}; temperature={heuristic.temperature:.2f}",
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        feature_count=len(LABELS) * 4,
        params={"none_bias": heuristic.none_bias, "temperature": heuristic.temperature},
    )
    rows.append(row)
    save_experiment(jsonl_path, {"split": "valid", **row})

    # --- Logistic ---
    if search.get("include_logistic", True):
        logistic_settings = settings["logistic"]
        for class_weight_setting in logistic_settings["class_weights"]:
            class_weight = None if class_weight_setting == "none" else str(class_weight_setting)
            for c_value in logistic_settings["c_values"]:
                name = f"logistic_c{float(c_value):g}_{class_weight_setting}"
                started = perf_counter()
                model = ClientTextLogistic(
                    c=float(c_value),
                    class_weight=class_weight,
                    seed=seed,
                    max_text_features=int(settings["features"]["tfidf_max_features"]),
                ).fit(x_train, train_documents, y_train)
                train_seconds = perf_counter() - started
                started = perf_counter()
                probabilities = model.predict_proba(x_valid, valid_documents)
                prediction = predict_from_probabilities(probabilities)
                inference_seconds = perf_counter() - started
                metrics = score(y_valid, prediction, x_valid.index)
                feature_count = len(model.numeric_columns_) + len(
                    model.vectorizer.get_feature_names_out()
                )
                outputs[name] = {
                    "kind": "logistic",
                    "probabilities": probabilities,
                    "prediction": prediction,
                    "model": model,
                    "feature_count": feature_count,
                }
                row = record_row(
                    name=name,
                    family="logistic",
                    metrics=metrics,
                    notes=f"C={float(c_value):g}; class_weight={class_weight_setting}",
                    train_seconds=train_seconds,
                    inference_seconds=inference_seconds,
                    feature_count=feature_count,
                    params={"c": float(c_value), "class_weight": class_weight_setting},
                )
                rows.append(row)
                save_experiment(jsonl_path, {"split": "valid", **row})

    # --- CatBoost grid ---
    if search.get("include_catboost", True):
        catboost_settings = settings["catboost"]
        for balanced in catboost_settings["balanced_options"]:
            for iterations in catboost_settings["iterations"]:
                for depth in catboost_settings["depths"]:
                    for learning_rate in catboost_settings["learning_rates"]:
                        name = (
                            f"catboost_i{iterations}_d{depth}_lr{float(learning_rate):g}"
                            f"_{'bal' if balanced else 'raw'}"
                        )
                        started = perf_counter()
                        model = CatBoostClientModel(
                            balanced=bool(balanced),
                            seed=seed,
                            iterations=int(iterations),
                            depth=int(depth),
                            learning_rate=float(learning_rate),
                        ).fit(x_train, y_train)
                        train_seconds = perf_counter() - started
                        started = perf_counter()
                        probabilities = model.predict_proba(x_valid)
                        prediction = predict_from_probabilities(probabilities)
                        inference_seconds = perf_counter() - started
                        metrics = score(y_valid, prediction, x_valid.index)
                        outputs[name] = {
                            "kind": "catboost",
                            "probabilities": probabilities,
                            "prediction": prediction,
                            "model": model,
                            "feature_count": len(model.columns_),
                        }
                        row = record_row(
                            name=name,
                            family="catboost",
                            metrics=metrics,
                            notes="numeric V1 features",
                            train_seconds=train_seconds,
                            inference_seconds=inference_seconds,
                            feature_count=len(model.columns_),
                            params={
                                "iterations": int(iterations),
                                "depth": int(depth),
                                "learning_rate": float(learning_rate),
                                "balanced": bool(balanced),
                            },
                        )
                        rows.append(row)
                        save_experiment(jsonl_path, {"split": "valid", **row})

    # --- LightGBM grid ---
    if search.get("include_lightgbm", True):
        lightgbm_settings = settings["lightgbm"]
        try:
            for balanced in lightgbm_settings["balanced_options"]:
                for n_estimators in lightgbm_settings["n_estimators"]:
                    for learning_rate in lightgbm_settings["learning_rates"]:
                        for num_leaves in lightgbm_settings["num_leaves"]:
                            name = (
                                f"lightgbm_n{n_estimators}_l{num_leaves}_lr{float(learning_rate):g}"
                                f"_{'bal' if balanced else 'raw'}"
                            )
                            started = perf_counter()
                            model = LightGBMClientModel(
                                balanced=bool(balanced),
                                seed=seed,
                                n_estimators=int(n_estimators),
                                learning_rate=float(learning_rate),
                                num_leaves=int(num_leaves),
                            ).fit(x_train, y_train)
                            train_seconds = perf_counter() - started
                            started = perf_counter()
                            probabilities = model.predict_proba(x_valid)
                            prediction = predict_from_probabilities(probabilities)
                            inference_seconds = perf_counter() - started
                            metrics = score(y_valid, prediction, x_valid.index)
                            outputs[name] = {
                                "kind": "lightgbm",
                                "probabilities": probabilities,
                                "prediction": prediction,
                                "model": model,
                                "feature_count": len(model.columns_),
                            }
                            row = record_row(
                                name=name,
                                family="lightgbm",
                                metrics=metrics,
                                notes="numeric V1 features",
                                train_seconds=train_seconds,
                                inference_seconds=inference_seconds,
                                feature_count=len(model.columns_),
                                params={
                                    "n_estimators": int(n_estimators),
                                    "num_leaves": int(num_leaves),
                                    "learning_rate": float(learning_rate),
                                    "balanced": bool(balanced),
                                },
                            )
                            rows.append(row)
                            save_experiment(jsonl_path, {"split": "valid", **row})
        except RuntimeError as exc:
            skipped = {
                "model": "lightgbm_skipped",
                "family": "lightgbm",
                "macro_f1": None,
                "accuracy": None,
                "notes": str(exc),
            }
            save_experiment(jsonl_path, {"split": "valid", **skipped})
            print(f"Skipping LightGBM grid: {exc}")

    # --- XGBoost grid ---
    if search.get("include_xgboost", True):
        xgboost_settings = settings["xgboost"]
        for balanced in xgboost_settings["balanced_options"]:
            for n_estimators in xgboost_settings["n_estimators"]:
                for learning_rate in xgboost_settings["learning_rates"]:
                    for max_depth in xgboost_settings["max_depths"]:
                        name = (
                            f"xgboost_n{n_estimators}_d{max_depth}_lr{float(learning_rate):g}"
                            f"_{'bal' if balanced else 'raw'}"
                        )
                        started = perf_counter()
                        model = XGBoostClientModel(
                            balanced=bool(balanced),
                            seed=seed,
                            n_estimators=int(n_estimators),
                            learning_rate=float(learning_rate),
                            max_depth=int(max_depth),
                        ).fit(x_train, y_train)
                        train_seconds = perf_counter() - started
                        started = perf_counter()
                        probabilities = model.predict_proba(x_valid)
                        prediction = predict_from_probabilities(probabilities)
                        inference_seconds = perf_counter() - started
                        metrics = score(y_valid, prediction, x_valid.index)
                        outputs[name] = {
                            "kind": "xgboost",
                            "probabilities": probabilities,
                            "prediction": prediction,
                            "model": model,
                            "feature_count": len(model.columns_),
                        }
                        row = record_row(
                            name=name,
                            family="xgboost",
                            metrics=metrics,
                            notes="numeric V1 features",
                            train_seconds=train_seconds,
                            inference_seconds=inference_seconds,
                            feature_count=len(model.columns_),
                            params={
                                "n_estimators": int(n_estimators),
                                "max_depth": int(max_depth),
                                "learning_rate": float(learning_rate),
                                "balanced": bool(balanced),
                            },
                        )
                        rows.append(row)
                        save_experiment(jsonl_path, {"split": "valid", **row})

    ml_names = [
        name
        for name, payload in outputs.items()
        if payload["kind"] in {"logistic", "catboost", "lightgbm", "xgboost"}
    ]
    if not ml_names:
        raise RuntimeError("Model search produced no ML candidates")

    # --- Temperature / none-bias calibration of each ML model on validation ---
    if search.get("include_calibration", True):
        calibration = settings["calibration"]
        temperatures = tuple(float(value) for value in calibration["temperatures"])
        none_biases = tuple(float(value) for value in calibration["none_biases"])
        for name in list(ml_names):
            calibrator = TemperatureCalibrator().tune(
                outputs[name]["probabilities"],
                y_valid,
                temperatures=temperatures,
                none_biases=none_biases,
            )
            started = perf_counter()
            probabilities = calibrator.apply(outputs[name]["probabilities"])
            prediction = predict_from_probabilities(probabilities)
            inference_seconds = perf_counter() - started
            metrics = score(y_valid, prediction, x_valid.index)
            calibrated_name = f"calibrated_{name}"
            outputs[calibrated_name] = {
                "kind": "calibrated",
                "base_name": name,
                "probabilities": probabilities,
                "prediction": prediction,
                "calibrator": calibrator,
                "feature_count": outputs[name]["feature_count"],
            }
            row = record_row(
                name=calibrated_name,
                family="calibration",
                metrics=metrics,
                notes=(
                    f"base={name}; temperature={calibrator.temperature:.2f}; "
                    f"none_bias={calibrator.none_bias:.2f}"
                ),
                train_seconds=0.0,
                inference_seconds=inference_seconds,
                feature_count=int(outputs[name]["feature_count"]),
                params={
                    "base": name,
                    "temperature": calibrator.temperature,
                    "none_bias": calibrator.none_bias,
                },
            )
            rows.append(row)
            save_experiment(jsonl_path, {"split": "valid", **row})

    # --- Soft ensembles: top ML + heuristic ---
    if search.get("include_ensembles", True):
        metrics_by_name = {row["model"]: row for row in rows}
        ranked_ml = sorted(
            ml_names,
            key=lambda name: (
                float(metrics_by_name[name]["macro_f1"]),
                float(metrics_by_name[name]["accuracy"]),
            ),
            reverse=True,
        )
        top_k = int(settings["ensemble"]["max_ml_components"])
        top_models = ranked_ml[:top_k]
        for heuristic_weight in settings["ensemble"]["heuristic_weights"]:
            remaining = 1.0 - float(heuristic_weight)
            if len(top_models) == 1:
                weight_map = {top_models[0]: remaining}
            else:
                weight_map = {
                    top_models[0]: remaining * 0.65,
                    top_models[1]: remaining * 0.35,
                }
            components = [heuristic_probabilities]
            weights = [float(heuristic_weight)]
            for model_name, weight in weight_map.items():
                components.append(outputs[model_name]["probabilities"])
                weights.append(float(weight))
            started = perf_counter()
            probabilities = blend_probabilities(components, weights)
            prediction = predict_from_probabilities(probabilities)
            inference_seconds = perf_counter() - started
            metrics = score(y_valid, prediction, x_valid.index)
            name = "ensemble_h" + f"{float(heuristic_weight):g}_" + "_".join(top_models)
            outputs[name] = {
                "kind": "ensemble",
                "probabilities": probabilities,
                "prediction": prediction,
                "components": ["recurrence_heuristic", *top_models],
                "weights": {
                    "recurrence_heuristic": float(heuristic_weight),
                    **{model_name: float(weight) for model_name, weight in weight_map.items()},
                },
                "feature_count": max(
                    outputs[model_name]["feature_count"] for model_name in top_models
                ),
            }
            row = record_row(
                name=name,
                family="ensemble",
                metrics=metrics,
                notes=json.dumps(outputs[name]["weights"], sort_keys=True),
                train_seconds=0.0,
                inference_seconds=inference_seconds,
                feature_count=int(outputs[name]["feature_count"]),
                params=outputs[name]["weights"],
            )
            rows.append(row)
            save_experiment(jsonl_path, {"split": "valid", **row})

    experiments = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    experiments.to_csv(metrics_dir / "experiments.csv", index=False)

    best_name = str(experiments.iloc[0]["model"])
    best_output = outputs[best_name]
    best_metrics = score(y_valid, best_output["prediction"], x_valid.index)
    delta = float(best_metrics["macro_f1"]) - baseline_macro_f1

    summary: dict[str, object] = {
        "baseline_macro_f1": baseline_macro_f1,
        "best_model": best_name,
        "best_macro_f1": best_metrics["macro_f1"],
        "best_accuracy": best_metrics["accuracy"],
        "delta_vs_baseline": delta,
        "improved_over_baseline": delta > 0,
        "per_class": best_metrics["per_class"],
        "n_experiments": len(experiments),
        "top_5": experiments.head(5)[["model", "family", "macro_f1", "accuracy", "notes"]].to_dict(
            orient="records"
        ),
        "leakage_policy": (
            "Features and supervised model parameters fit on train only. "
            "Heuristic bias/temperature and probability calibration/ensemble "
            "weights are selected on validation macro-F1. Test is prediction-only."
        ),
    }

    submission_path = Path(settings["outputs"]["submission"])
    if not arguments.skip_submission:
        all_transactions = pd.concat(
            [data.train_transactions, data.valid_transactions], ignore_index=True
        )
        all_labels = pd.concat([data.train_labels, data.valid_labels], ignore_index=True)
        final_builder = ClientFeatureBuilder().fit(all_transactions, all_labels)
        x_all = final_builder.transform(all_transactions)
        x_test = final_builder.transform(data.test_transactions)
        y_all = all_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_all.index)
        all_documents = build_client_documents(all_transactions, x_all.index)
        test_documents = build_client_documents(data.test_transactions, x_test.index)

        test_probabilities = _refit_and_predict(
            best_name=best_name,
            best_output=best_output,
            outputs=outputs,
            x_all=x_all,
            y_all=y_all,
            all_documents=all_documents,
            x_test=x_test,
            test_documents=test_documents,
            seed=seed,
            tfidf_max_features=int(settings["features"]["tfidf_max_features"]),
            heuristic=heuristic,
        )
        test_prediction = predict_from_probabilities(test_probabilities)
        prediction_by_client = pd.Series(test_prediction, index=x_test.index)
        submission = data.sample_submission[["client_id"]].copy()
        submission[PREDICTION_COLUMN] = submission["client_id"].map(prediction_by_client)
        validate_submission(submission, data.sample_submission, data.test_transactions)
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)
        written = pd.read_csv(submission_path, dtype={"client_id": str})
        validate_submission(written, data.sample_submission, data.test_transactions)
        summary["submission"] = str(submission_path.resolve())
        summary["submission_rows"] = len(submission)
        summary["final_refit_uses_train_and_valid"] = True

    save_json(metrics_dir / "summary.json", summary)
    top_table = experiments.head(12)[
        ["model", "family", "macro_f1", "accuracy", "notes"]
    ].to_string(index=False)
    report = "\n".join(
        [
            "# UBS model search results",
            "",
            f"- Baseline V1 macro-F1: `{baseline_macro_f1:.6f}`",
            f"- Best search macro-F1: `{float(best_metrics['macro_f1']):.6f}`",
            f"- Delta vs baseline: `{delta:+.6f}`",
            f"- Best model: `{best_name}`",
            f"- Experiments: `{len(experiments)}`",
            "",
            "## Top results",
            "",
            "```",
            top_table,
            "```",
            "",
            "## Leakage policy",
            "",
            str(summary["leakage_policy"]),
            "",
        ]
    )
    (metrics_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _refit_and_predict(
    *,
    best_name: str,
    best_output: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    x_all: pd.DataFrame,
    y_all: pd.Series,
    all_documents: pd.Series,
    x_test: pd.DataFrame,
    test_documents: pd.Series,
    seed: int,
    tfidf_max_features: int,
    heuristic: RecurrenceHeuristic,
) -> np.ndarray:
    kind = best_output["kind"]
    if kind == "heuristic":
        return RecurrenceHeuristic(
            none_bias=heuristic.none_bias,
            temperature=heuristic.temperature,
        ).predict_proba(x_test)

    if kind == "logistic":
        selected = best_output["model"]
        return (
            ClientTextLogistic(
                c=selected.c,
                class_weight=selected.class_weight,
                seed=seed,
                max_text_features=tfidf_max_features,
            )
            .fit(x_all, all_documents, y_all)
            .predict_proba(x_test, test_documents)
        )

    if kind == "catboost":
        selected = best_output["model"]
        return (
            CatBoostClientModel(
                balanced=selected.balanced,
                seed=seed,
                iterations=selected.iterations,
                depth=selected.depth,
                learning_rate=selected.learning_rate,
            )
            .fit(x_all, y_all)
            .predict_proba(x_test)
        )

    if kind == "lightgbm":
        selected = best_output["model"]
        return (
            LightGBMClientModel(
                balanced=selected.balanced,
                seed=seed,
                n_estimators=selected.n_estimators,
                learning_rate=selected.learning_rate,
                num_leaves=selected.num_leaves,
                min_child_samples=selected.min_child_samples,
            )
            .fit(x_all, y_all)
            .predict_proba(x_test)
        )

    if kind == "xgboost":
        selected = best_output["model"]
        return (
            XGBoostClientModel(
                balanced=selected.balanced,
                seed=seed,
                n_estimators=selected.n_estimators,
                learning_rate=selected.learning_rate,
                max_depth=selected.max_depth,
                min_child_weight=selected.min_child_weight,
            )
            .fit(x_all, y_all)
            .predict_proba(x_test)
        )

    if kind == "calibrated":
        base = outputs[best_output["base_name"]]
        base_probabilities = _refit_and_predict(
            best_name=best_output["base_name"],
            best_output=base,
            outputs=outputs,
            x_all=x_all,
            y_all=y_all,
            all_documents=all_documents,
            x_test=x_test,
            test_documents=test_documents,
            seed=seed,
            tfidf_max_features=tfidf_max_features,
            heuristic=heuristic,
        )
        calibrator = best_output["calibrator"]
        return calibrator.apply(base_probabilities)

    if kind == "ensemble":
        components = []
        weights = []
        weight_map = best_output["weights"]
        for component_name, weight in weight_map.items():
            component_output = outputs[component_name]
            component_probabilities = _refit_and_predict(
                best_name=component_name,
                best_output=component_output,
                outputs=outputs,
                x_all=x_all,
                y_all=y_all,
                all_documents=all_documents,
                x_test=x_test,
                test_documents=test_documents,
                seed=seed,
                tfidf_max_features=tfidf_max_features,
                heuristic=heuristic,
            )
            components.append(component_probabilities)
            weights.append(float(weight))
        return blend_probabilities(components, weights)

    raise ValueError(f"Unsupported model kind for refit: {kind}")


if __name__ == "__main__":
    main()
