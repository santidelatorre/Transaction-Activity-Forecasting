"""Train, compare, analyze, and export the UBS client-level V1 baseline."""

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

from transaction_forecasting.tracking_integration import record_validation_run
from transaction_forecasting.ubs.data import (
    LABELS,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    load_ubs_data,
    validate_submission,
)
from transaction_forecasting.ubs.evaluation import build_error_table, evaluate_predictions
from transaction_forecasting.ubs.features import ClientFeatureBuilder, build_client_documents
from transaction_forecasting.ubs.models import (
    CatBoostClientModel,
    ClientTextLogistic,
    RecurrenceHeuristic,
)


def load_settings(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as config_file:
        return tomllib.load(config_file)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def experiment_row(
    name: str,
    feature_set: str,
    class_weight: str,
    metrics: dict[str, object],
    notes: str,
    *,
    train_seconds: float,
    inference_seconds: float,
    feature_count: int,
    validation_clients: int,
) -> dict[str, object]:
    return {
        "model": name,
        "feature_set": feature_set,
        "class_weight": class_weight,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "feature_count": feature_count,
        "validation_clients": validation_clients,
        "notes": notes,
    }


def indexed_prediction(values: np.ndarray, client_ids: pd.Index) -> pd.Series:
    """Attach client IDs before scoring so row order cannot change the metric."""
    return pd.Series(values, index=client_ids, name=PREDICTION_COLUMN)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["float"]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.{digits}f}")
    headers = [str(column) for column in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def confusion_summary(metrics: dict[str, object]) -> list[dict[str, object]]:
    matrix = np.asarray(metrics["confusion_matrix"])
    pairs: list[dict[str, object]] = []
    for row, actual in enumerate(LABELS):
        for column, predicted in enumerate(LABELS):
            if row != column and matrix[row, column] > 0:
                pairs.append(
                    {
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(matrix[row, column]),
                    }
                )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)


def per_class_errors(metrics: dict[str, object]) -> dict[str, dict[str, int]]:
    matrix = np.asarray(metrics["confusion_matrix"])
    return {
        label: {
            "false_negatives": int(matrix[row, :].sum() - matrix[row, row]),
            "false_positives": int(matrix[:, row].sum() - matrix[row, row]),
        }
        for row, label in enumerate(LABELS)
    }


def render_report(
    experiments: pd.DataFrame,
    best_name: str,
    best_metrics: dict[str, object],
    feature_importance: pd.DataFrame,
    errors: pd.DataFrame,
    comparison_errors: pd.DataFrame,
    win_patterns: dict[str, object],
    ensemble_improvement: float,
    heuristic_metrics: dict[str, object],
    submission_path: Path,
) -> str:
    experiment_table = markdown_table(experiments)
    per_class = pd.DataFrame(best_metrics["per_class"]).T.reset_index(names="class")
    per_class_table = markdown_table(per_class)
    importance_table = markdown_table(feature_importance.head(15))
    difficult = per_class.sort_values("f1-score").head(3)["class"].tolist()
    confusions = confusion_summary(best_metrics)[:6]
    heuristic_only = int(comparison_errors["heuristic_only_win"].sum())
    model_only = int(comparison_errors["model_only_win"].sum())
    return f"""# UBS baseline V1 results

## Experiments

{experiment_table}

## Best validation result

- Model: `{best_name}`
- Macro-F1: {best_metrics["macro_f1"]:.6f}
- Accuracy: {best_metrics["accuracy"]:.6f}
- Heuristic macro-F1: {heuristic_metrics["macro_f1"]:.6f}
- Ensemble improvement over its ML component: {ensemble_improvement:.6f}
- Final submission refit: selected approach retrained on train + validation labels.

### Per-class metrics

{per_class_table}

### Most important features

{importance_table}

### Error analysis

- Hardest classes by F1: {", ".join(difficult)}
- Largest confusion pairs: {json.dumps(confusions)}
- False positives/negatives by class: {json.dumps(per_class_errors(best_metrics))}
- Cases only the heuristic gets right: {heuristic_only}
- Cases only the best ML model gets right: {model_only}
- Win-pattern feature means: {json.dumps(win_patterns)}

## V2 candidates

1. Cross-validation across client folds inside train for stabler feature and
   hyperparameter selection.
2. Candidate-level ranking of recurring streams before the final family classifier.
3. Character TF-IDF and calibrated class-specific thresholds for rare-family recall.
4. Unsupervised merchant normalization using the unlabeled pretrain file, without labels.

Submission: `{submission_path.resolve()}`
"""


def main() -> None:
    run_started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ubs_v1.toml")
    arguments = parser.parse_args()
    settings = load_settings(arguments.config)
    seed = int(settings["project"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)

    data = load_ubs_data(settings["data"]["directory"])
    feature_started = perf_counter()
    builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
    x_train = builder.transform(data.train_transactions)
    x_valid = builder.transform(data.valid_transactions)
    if not (x_train.index.is_unique and x_valid.index.is_unique):
        raise ValueError("Feature tables must contain exactly one unique row per client")
    y_train = data.train_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_train.index)
    y_valid = data.valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_valid.index)
    if y_train.isna().any() or y_valid.isna().any():
        raise ValueError("Feature/label alignment produced missing targets")
    train_documents = build_client_documents(data.train_transactions, x_train.index)
    valid_documents = build_client_documents(data.valid_transactions, x_valid.index)
    validation_feature_seconds = perf_counter() - feature_started

    outputs: dict[str, dict[str, Any]] = {}
    details: dict[str, dict[str, object]] = {}
    experiments: list[dict[str, object]] = []

    train_started = perf_counter()
    majority = str(y_train.mode().iloc[0])
    dummy_train_seconds = perf_counter() - train_started
    inference_started = perf_counter()
    dummy_prediction = np.repeat(majority, len(x_valid))
    dummy_probabilities = np.zeros((len(x_valid), len(LABELS)))
    dummy_probabilities[:, LABELS.index(majority)] = 1.0
    dummy_inference_seconds = perf_counter() - inference_started
    dummy_metrics = evaluate_predictions(
        y_valid, indexed_prediction(dummy_prediction, x_valid.index)
    )
    details["dummy_majority"] = dummy_metrics
    outputs["dummy_majority"] = {
        "prediction": dummy_prediction,
        "probabilities": dummy_probabilities,
        "kind": "dummy",
        "feature_count": 0,
    }
    experiments.append(
        experiment_row(
            "dummy_majority",
            "none",
            "none",
            dummy_metrics,
            f"Always predicts train majority: {majority}",
            train_seconds=dummy_train_seconds,
            inference_seconds=dummy_inference_seconds,
            feature_count=0,
            validation_clients=len(x_valid),
        )
    )

    train_started = perf_counter()
    heuristic = RecurrenceHeuristic().tune(x_valid, y_valid)
    heuristic_train_seconds = perf_counter() - train_started
    inference_started = perf_counter()
    heuristic_probabilities = heuristic.predict_proba(x_valid)
    heuristic_prediction = np.asarray(LABELS)[heuristic_probabilities.argmax(axis=1)]
    heuristic_inference_seconds = perf_counter() - inference_started
    heuristic_metrics = evaluate_predictions(
        y_valid, indexed_prediction(heuristic_prediction, x_valid.index)
    )
    details["recurrence_heuristic"] = heuristic_metrics
    outputs["recurrence_heuristic"] = {
        "prediction": heuristic_prediction,
        "probabilities": heuristic_probabilities,
        "kind": "heuristic",
        "model": heuristic,
        "feature_count": len(LABELS) * 4,
    }
    experiments.append(
        experiment_row(
            "recurrence_heuristic",
            "train-learned description lift + recurrence streams",
            "validation-calibrated none bias",
            heuristic_metrics,
            f"none_bias={heuristic.none_bias:.2f}; temperature={heuristic.temperature:.2f}",
            train_seconds=heuristic_train_seconds,
            inference_seconds=heuristic_inference_seconds,
            feature_count=len(LABELS) * 4,
            validation_clients=len(x_valid),
        )
    )

    logistic_settings = settings["logistic"]
    for class_weight_setting in logistic_settings["class_weights"]:
        class_weight = None if class_weight_setting == "none" else str(class_weight_setting)
        for c_value in logistic_settings["c_values"]:
            name = f"logistic_c{float(c_value):g}_{class_weight_setting}"
            train_started = perf_counter()
            model = ClientTextLogistic(
                c=float(c_value),
                class_weight=class_weight,
                seed=seed,
                max_text_features=int(settings["features"]["tfidf_max_features"]),
            ).fit(x_train, train_documents, y_train)
            train_seconds = perf_counter() - train_started
            inference_started = perf_counter()
            probabilities = model.predict_proba(x_valid, valid_documents)
            prediction = np.asarray(LABELS)[probabilities.argmax(axis=1)]
            inference_seconds = perf_counter() - inference_started
            metrics = evaluate_predictions(y_valid, indexed_prediction(prediction, x_valid.index))
            feature_count = len(model.numeric_columns_) + len(
                model.vectorizer.get_feature_names_out()
            )
            details[name] = metrics
            outputs[name] = {
                "prediction": prediction,
                "probabilities": probabilities,
                "kind": "logistic",
                "model": model,
                "feature_count": feature_count,
            }
            experiments.append(
                experiment_row(
                    name,
                    "client aggregates + train-only word/bigram TF-IDF",
                    str(class_weight_setting),
                    metrics,
                    f"C={float(c_value):g}",
                    train_seconds=train_seconds,
                    inference_seconds=inference_seconds,
                    feature_count=feature_count,
                    validation_clients=len(x_valid),
                )
            )

    catboost_settings = settings["catboost"]
    for balanced in (False, True):
        name = f"catboost_{'balanced' if balanced else 'normal'}"
        train_started = perf_counter()
        model = CatBoostClientModel(
            balanced=balanced,
            seed=seed,
            iterations=int(catboost_settings["iterations"]),
            depth=int(catboost_settings["depth"]),
            learning_rate=float(catboost_settings["learning_rate"]),
        ).fit(x_train, y_train)
        train_seconds = perf_counter() - train_started
        inference_started = perf_counter()
        probabilities = model.predict_proba(x_valid)
        prediction = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        inference_seconds = perf_counter() - inference_started
        metrics = evaluate_predictions(y_valid, indexed_prediction(prediction, x_valid.index))
        feature_count = len(model.columns_)
        details[name] = metrics
        outputs[name] = {
            "prediction": prediction,
            "probabilities": probabilities,
            "kind": "catboost",
            "model": model,
            "feature_count": feature_count,
        }
        experiments.append(
            experiment_row(
                name,
                "numeric client and recurrence aggregates",
                "balanced" if balanced else "none",
                metrics,
                (
                    f"iterations={catboost_settings['iterations']}; "
                    f"depth={catboost_settings['depth']}"
                ),
                train_seconds=train_seconds,
                inference_seconds=inference_seconds,
                feature_count=feature_count,
                validation_clients=len(x_valid),
            )
        )

    ml_names = [
        name for name, output in outputs.items() if output["kind"] in {"logistic", "catboost"}
    ]
    best_ml_name = max(
        ml_names, key=lambda name: (details[name]["macro_f1"], details[name]["accuracy"])
    )
    best_ml = outputs[best_ml_name]
    best_ensemble: tuple[float, float, np.ndarray, np.ndarray, dict[str, object]] | None = None
    train_started = perf_counter()
    for alpha in np.arange(0.05, 0.51, 0.05):
        probabilities = (1.0 - alpha) * best_ml["probabilities"] + alpha * heuristic_probabilities
        prediction = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        metrics = evaluate_predictions(y_valid, indexed_prediction(prediction, x_valid.index))
        candidate = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
        if best_ensemble is None or candidate > best_ensemble[:2]:
            best_ensemble = (
                candidate[0],
                candidate[1],
                probabilities,
                prediction,
                {**metrics, "alpha": float(alpha)},
            )
    assert best_ensemble is not None
    ensemble_train_seconds = perf_counter() - train_started
    alpha = float(best_ensemble[4]["alpha"])
    inference_started = perf_counter()
    ensemble_probabilities = (1.0 - alpha) * best_ml[
        "probabilities"
    ] + alpha * heuristic_probabilities
    ensemble_prediction = np.asarray(LABELS)[ensemble_probabilities.argmax(axis=1)]
    ensemble_inference_seconds = perf_counter() - inference_started
    ensemble_metrics = {
        **evaluate_predictions(y_valid, indexed_prediction(ensemble_prediction, x_valid.index)),
        "alpha": alpha,
    }
    ensemble_name = f"ensemble_{best_ml_name}_heuristic"
    details[ensemble_name] = ensemble_metrics
    outputs[ensemble_name] = {
        "prediction": ensemble_prediction,
        "probabilities": ensemble_probabilities,
        "kind": "ensemble",
        "base_name": best_ml_name,
        "alpha": ensemble_metrics["alpha"],
        "feature_count": best_ml["feature_count"],
    }
    ensemble_improvement = float(ensemble_metrics["macro_f1"] - details[best_ml_name]["macro_f1"])
    experiments.append(
        experiment_row(
            ensemble_name,
            "best ML probabilities + heuristic probabilities",
            "inherited",
            ensemble_metrics,
            f"alpha={ensemble_metrics['alpha']:.2f}; delta_macro_f1={ensemble_improvement:+.6f}",
            train_seconds=ensemble_train_seconds,
            inference_seconds=ensemble_inference_seconds,
            feature_count=int(best_ml["feature_count"]),
            validation_clients=len(x_valid),
        )
    )

    selectable_names = [name for name in outputs if name != ensemble_name]
    if ensemble_improvement > 0:
        selectable_names.append(ensemble_name)
    best_name = max(
        selectable_names,
        key=lambda name: (details[name]["macro_f1"], details[name]["accuracy"]),
    )
    best_output = outputs[best_name]
    best_metrics = details[best_name]

    record_validation_run(
        repo_root=Path(__file__).resolve().parents[1],
        experiments=experiments,
        details=details,
        settings=settings,
        selected_model=best_name,
    )

    # Model selection is complete. Refit the selected approach on every labelled
    # client before touching test predictions; test labels never exist or enter here.
    final_feature_started = perf_counter()
    all_transactions = pd.concat(
        [data.train_transactions, data.valid_transactions], ignore_index=True
    )
    all_labels = pd.concat([data.train_labels, data.valid_labels], ignore_index=True)
    final_builder = ClientFeatureBuilder().fit(all_transactions, all_labels)
    x_all = final_builder.transform(all_transactions)
    x_test = final_builder.transform(data.test_transactions)
    if not (x_all.index.is_unique and x_test.index.is_unique):
        raise ValueError("Final feature tables must contain one unique row per client")
    y_all = all_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_all.index)
    if y_all.isna().any():
        raise ValueError("Final feature/label alignment produced missing targets")
    all_documents = build_client_documents(all_transactions, x_all.index)
    test_documents = build_client_documents(data.test_transactions, x_test.index)
    final_feature_seconds = perf_counter() - final_feature_started

    final_train_started = perf_counter()
    if best_output["kind"] == "logistic":
        selected_model = best_output["model"]
        final_model = ClientTextLogistic(
            c=selected_model.c,
            class_weight=selected_model.class_weight,
            seed=selected_model.seed,
            max_text_features=selected_model.max_text_features,
        ).fit(x_all, all_documents, y_all)
        final_train_seconds = perf_counter() - final_train_started
        final_inference_started = perf_counter()
        test_probabilities = final_model.predict_proba(x_test, test_documents)
        final_inference_seconds = perf_counter() - final_inference_started
        importance = final_model.feature_importance()
    elif best_output["kind"] == "catboost":
        selected_model = best_output["model"]
        final_model = CatBoostClientModel(
            balanced=selected_model.balanced,
            seed=selected_model.seed,
            iterations=selected_model.iterations,
            depth=selected_model.depth,
            learning_rate=selected_model.learning_rate,
        ).fit(x_all, y_all)
        final_train_seconds = perf_counter() - final_train_started
        final_inference_started = perf_counter()
        test_probabilities = final_model.predict_proba(x_test)
        final_inference_seconds = perf_counter() - final_inference_started
        importance = final_model.feature_importance()
    elif best_output["kind"] == "heuristic":
        final_heuristic = RecurrenceHeuristic(
            none_bias=heuristic.none_bias,
            temperature=heuristic.temperature,
        )
        final_train_seconds = perf_counter() - final_train_started
        final_inference_started = perf_counter()
        test_probabilities = final_heuristic.predict_proba(x_test)
        final_inference_seconds = perf_counter() - final_inference_started
        importance = pd.DataFrame(
            {
                "feature": [
                    "family recurrence score",
                    "description lift",
                    "family occurrences",
                    "regularity",
                ],
                "importance": [1.0, 0.20, 0.15, 0.10],
            }
        )
    elif best_output["kind"] == "ensemble":
        base_output = outputs[best_output["base_name"]]
        if base_output["kind"] == "logistic":
            selected_model = base_output["model"]
            final_base = ClientTextLogistic(
                c=selected_model.c,
                class_weight=selected_model.class_weight,
                seed=selected_model.seed,
                max_text_features=selected_model.max_text_features,
            ).fit(x_all, all_documents, y_all)
        else:
            selected_model = base_output["model"]
            final_base = CatBoostClientModel(
                balanced=selected_model.balanced,
                seed=selected_model.seed,
                iterations=selected_model.iterations,
                depth=selected_model.depth,
                learning_rate=selected_model.learning_rate,
            ).fit(x_all, y_all)
        final_heuristic = RecurrenceHeuristic(
            none_bias=heuristic.none_bias,
            temperature=heuristic.temperature,
        )
        final_train_seconds = perf_counter() - final_train_started
        final_inference_started = perf_counter()
        if base_output["kind"] == "logistic":
            base_test = final_base.predict_proba(x_test, test_documents)
        else:
            base_test = final_base.predict_proba(x_test)
        alpha = float(best_output["alpha"])
        test_probabilities = (1.0 - alpha) * base_test + alpha * final_heuristic.predict_proba(
            x_test
        )
        final_inference_seconds = perf_counter() - final_inference_started
        importance = final_base.feature_importance()
    else:
        final_majority = str(y_all.mode().iloc[0])
        final_train_seconds = perf_counter() - final_train_started
        final_inference_started = perf_counter()
        test_probabilities = np.zeros((len(x_test), len(LABELS)))
        test_probabilities[:, LABELS.index(final_majority)] = 1.0
        final_inference_seconds = perf_counter() - final_inference_started
        importance = pd.DataFrame({"feature": ["train majority"], "importance": [1.0]})
    test_prediction = np.asarray(LABELS)[test_probabilities.argmax(axis=1)]

    prediction_by_client = pd.Series(test_prediction, index=x_test.index)
    submission = data.sample_submission[["client_id"]].copy()
    submission[PREDICTION_COLUMN] = submission["client_id"].map(prediction_by_client)
    validate_submission(submission, data.sample_submission, data.test_transactions)

    metrics_dir = Path(settings["outputs"]["metrics_directory"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_dir / "run_config.json", settings)
    submission_path = Path(settings["outputs"]["submission"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    written_submission = pd.read_csv(submission_path, dtype={"client_id": str})
    validate_submission(written_submission, data.sample_submission, data.test_transactions)
    experiments_frame = pd.DataFrame(experiments).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    )
    experiments_frame.to_csv(metrics_dir / "experiments.csv", index=False)
    save_json(metrics_dir / "validation_metrics.json", details)
    pd.DataFrame(best_metrics["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(
        metrics_dir / "best_confusion_matrix.csv"
    )
    importance.head(100).to_csv(metrics_dir / "feature_importance.csv", index=False)
    errors = build_error_table(
        x_valid.index,
        y_valid,
        best_output["prediction"],
        heuristic_prediction,
        best_output["probabilities"],
    )
    diagnostic_columns = [
        "n_transactions",
        "repeated_description_count",
        "best_recurrence_score",
        "regular_stream_count",
        "stable_amount_stream_count",
    ]
    errors = errors.join(x_valid[diagnostic_columns], on="client_id")
    errors.to_csv(metrics_dir / "validation_error_analysis.csv", index=False)
    comparison_errors = build_error_table(
        x_valid.index,
        y_valid,
        best_ml["prediction"],
        heuristic_prediction,
        best_ml["probabilities"],
    ).join(x_valid[diagnostic_columns], on="client_id")
    comparison_errors.to_csv(metrics_dir / "heuristic_vs_best_ml.csv", index=False)
    win_patterns = {
        "heuristic_only": comparison_errors.loc[
            comparison_errors["heuristic_only_win"], diagnostic_columns
        ]
        .mean()
        .round(4)
        .to_dict(),
        "ml_only": comparison_errors.loc[comparison_errors["model_only_win"], diagnostic_columns]
        .mean()
        .round(4)
        .to_dict(),
    }
    report = render_report(
        experiments_frame,
        best_name,
        best_metrics,
        importance,
        errors,
        comparison_errors,
        win_patterns,
        ensemble_improvement,
        heuristic_metrics,
        submission_path,
    )
    (metrics_dir / "report.md").write_text(report, encoding="utf-8")
    run_seconds = perf_counter() - run_started
    summary = {
        "best_model": best_name,
        "best_macro_f1": best_metrics["macro_f1"],
        "best_accuracy": best_metrics["accuracy"],
        "per_class": best_metrics["per_class"],
        "hardest_classes": sorted(
            LABELS, key=lambda label: best_metrics["per_class"][label]["f1-score"]
        )[:3],
        "largest_confusions": confusion_summary(best_metrics)[:6],
        "per_class_errors": per_class_errors(best_metrics),
        "heuristic_macro_f1": heuristic_metrics["macro_f1"],
        "heuristic_only_wins": int(comparison_errors["heuristic_only_win"].sum()),
        "model_only_wins": int(comparison_errors["model_only_win"].sum()),
        "best_ml_model": best_ml_name,
        "heuristic_vs_ml_heuristic_only_wins": int(comparison_errors["heuristic_only_win"].sum()),
        "heuristic_vs_ml_ml_only_wins": int(comparison_errors["model_only_win"].sum()),
        "win_patterns": win_patterns,
        "ensemble_improvement": ensemble_improvement,
        "ensemble_selected": best_name == ensemble_name,
        "top_features": importance.head(15).to_dict(orient="records"),
        "submission": str(submission_path.resolve()),
        "submission_rows": len(submission),
        "submission_prediction_distribution": submission[PREDICTION_COLUMN]
        .value_counts()
        .reindex(LABELS, fill_value=0)
        .astype(int)
        .to_dict(),
        "train_clients": len(x_train),
        "valid_clients": len(x_valid),
        "test_clients": len(x_test),
        "feature_count": x_train.shape[1],
        "final_refit_clients": len(x_all),
        "final_refit_uses_train_and_valid": True,
        "validation_feature_seconds": validation_feature_seconds,
        "final_feature_seconds": final_feature_seconds,
        "final_model_train_seconds": final_train_seconds,
        "final_inference_seconds": final_inference_seconds,
        "pipeline_seconds": run_seconds,
    }
    save_json(metrics_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
