"""Reproduce V1 validation predictions and diagnose error structure for handoff.

Writes tables and figures under Reports/. Does not change the competitive
submission or retrain alternate models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.features import ClientFeatureBuilder, build_client_documents
from transaction_forecasting.ubs.models import RecurrenceHeuristic

EXPECTED_MACRO_F1 = 0.2710242658492452
EXPECTED_ACCURACY = 0.2660
DIAGNOSTIC_FEATURES = [
    "n_transactions",
    "history_days",
    "days_since_last_transaction",
    "transactions_per_30d",
    "out_share",
    "amount_mean",
    "amount_std",
    "amount_median",
    "n_unique_description",
    "repeated_description_count",
    "best_recurrence_score",
    "regular_stream_count",
    "stable_amount_stream_count",
    "stream_days_since_last_min",
    "stream_regularity_mean",
    "periodicity_monthly_count",
    "weekend_share",
]


def indexed_prediction(values: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index, name="predicted_next_recurring_merchant")


def confusion_pairs(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"real_class": y_true.to_numpy(), "predicted_class": y_pred.to_numpy()})
    errors = frame.loc[frame["real_class"].ne(frame["predicted_class"])]
    counts = (
        errors.groupby(["real_class", "predicted_class"], sort=False)
        .size()
        .rename("number_errors")
        .reset_index()
    )
    support = y_true.value_counts()
    real_errors = errors.groupby("real_class").size()
    counts["percentage_of_real_class_errors"] = (
        counts["number_errors"] / counts["real_class"].map(real_errors)
    ).astype(float)
    counts["support_real_class"] = counts["real_class"].map(support).astype(int)
    return counts.sort_values("number_errors", ascending=False).reset_index(drop=True)


def effect_size(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if a.empty or b.empty:
        return float("nan")
    pooled = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2.0)
    if not pooled or np.isnan(pooled):
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def compare_groups(features: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in DIAGNOSTIC_FEATURES:
        if column not in features.columns:
            continue
        a = features.loc[mask_a, column]
        b = features.loc[mask_b, column]
        rows.append(
            {
                "feature": column,
                "median_a": float(a.median()) if len(a) else float("nan"),
                "median_b": float(b.median()) if len(b) else float("nan"),
                "mean_a": float(a.mean()) if len(a) else float("nan"),
                "mean_b": float(b.mean()) if len(b) else float("nan"),
                "effect_size_a_minus_b": effect_size(a, b),
                "n_a": int(mask_a.sum()),
                "n_b": int(mask_b.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "effect_size_a_minus_b", key=lambda s: s.abs(), ascending=False
    )


def save_confusion_figure(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("V1 recurrence heuristic — confusion matrix")
    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_f1_figure(per_class: dict[str, dict[str, float]], path: Path) -> None:
    f1 = [per_class[label]["f1-score"] for label in LABELS]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(LABELS, f1, color="#2c6eac")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("V1 per-class F1")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_pair_bars(pairs: pd.DataFrame, path: Path) -> None:
    top = pairs.head(8).copy()
    labels = top["real_class"] + "→" + top["predicted_class"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels[::-1], top["number_errors"][::-1], color="#c44e52")
    ax.set_xlabel("Validation errors")
    ax.set_title("Largest V1 confusion pairs")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_correct_incorrect_box(
    features: pd.DataFrame, correct: pd.Series, column: str, path: Path
) -> None:
    if column not in features.columns:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [
        features.loc[correct, column].dropna().to_numpy(),
        features.loc[~correct, column].dropna().to_numpy(),
    ]
    ax.boxplot(data, tick_labels=["correct", "incorrect"], showfliers=False)
    ax.set_title(f"{column}: correct vs incorrect")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/ubs_2026")
    parser.add_argument("--output-dir", default="Reports")
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    figures_dir = output_dir / "figures" / "error_analysis"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_ubs_data(arguments.data_dir)
    builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
    x_train = builder.transform(data.train_transactions)
    x_valid = builder.transform(data.valid_transactions)
    y_valid = data.valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(x_valid.index)
    if y_valid.isna().any():
        raise ValueError("Validation labels missing after feature alignment")

    # Reproduce the V1 selected model path: recurrence heuristic calibrated on valid.
    heuristic = RecurrenceHeuristic().tune(x_valid, y_valid)
    probabilities = heuristic.predict_proba(x_valid)
    prediction = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    metrics = evaluate_predictions(y_valid, indexed_prediction(prediction, x_valid.index))

    macro_f1 = float(metrics["macro_f1"])
    accuracy = float(metrics["accuracy"])
    if abs(macro_f1 - EXPECTED_MACRO_F1) > 1e-6 or abs(accuracy - EXPECTED_ACCURACY) > 1e-4:
        raise SystemExit(
            "Baseline mismatch: "
            f"got macro_f1={macro_f1}, accuracy={accuracy}; "
            f"expected macro_f1={EXPECTED_MACRO_F1}, accuracy={EXPECTED_ACCURACY}"
        )

    pred_series = pd.Series(prediction, index=x_valid.index, name="prediction")
    correct = y_valid.eq(pred_series)
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    pairs = confusion_pairs(y_valid, pred_series)
    pairs.to_csv(tables_dir / "confusion_pairs.csv", index=False)

    per_class_rows = []
    support = y_valid.value_counts()
    pred_counts = pred_series.value_counts()
    for label in LABELS:
        stats = metrics["per_class"][label]
        per_class_rows.append(
            {
                "class": label,
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1-score"],
                "support": int(support.get(label, 0)),
                "predicted_count": int(pred_counts.get(label, 0)),
                "overpredicted_ratio": float(
                    pred_counts.get(label, 0) / max(support.get(label, 1), 1)
                ),
            }
        )
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(tables_dir / "per_class_metrics.csv", index=False)

    # Top confusion pair diagnostics using feature medians.
    top_pairs = pairs.head(5)
    pair_reports: list[dict[str, object]] = []
    for row in top_pairs.itertuples(index=False):
        real = row.real_class
        predicted = row.predicted_class
        mask_real_pred = y_valid.eq(real) & pred_series.eq(predicted)
        mask_real_correct = y_valid.eq(real) & correct
        comparison = compare_groups(x_valid, mask_real_pred, mask_real_correct)
        comparison.to_csv(
            tables_dir / f"pair_{real}_to_{predicted}_vs_correct_{real}.csv", index=False
        )
        top_features = comparison.head(6).to_dict(orient="records")
        pair_reports.append(
            {
                "real_class": real,
                "predicted_class": predicted,
                "errors": int(row.number_errors),
                "percentage_of_real_class_errors": float(row.percentage_of_real_class_errors),
                "top_feature_deltas": top_features,
            }
        )

    # Correct vs incorrect overall and for weakest classes.
    overall_cmp = compare_groups(x_valid, correct, ~correct)
    overall_cmp.to_csv(tables_dir / "correct_vs_incorrect_overall.csv", index=False)
    weak = per_class.sort_values("f1").head(3)["class"].tolist()
    class_cmp: dict[str, list[dict[str, object]]] = {}
    for label in weak:
        mask_label = y_valid.eq(label)
        cmp_frame = compare_groups(x_valid, mask_label & correct, mask_label & ~correct)
        cmp_frame.to_csv(tables_dir / f"correct_vs_incorrect_{label}.csv", index=False)
        class_cmp[label] = cmp_frame.head(8).to_dict(orient="records")

    # Family score evidence: which family_*_recurrence_score wins among none→X errors.
    family_score_cols = [f"family_{label}_recurrence_score" for label in LABELS]
    available_family = [col for col in family_score_cols if col in x_valid.columns]
    family_win_notes: list[dict[str, object]] = []
    for row in top_pairs.itertuples(index=False):
        mask = y_valid.eq(row.real_class) & pred_series.eq(row.predicted_class)
        if mask.sum() == 0 or not available_family:
            continue
        subset = x_valid.loc[mask, available_family]
        argmax = subset.to_numpy().argmax(axis=1)
        winner = pd.Series([LABELS[i] for i in argmax]).value_counts(normalize=True)
        family_win_notes.append(
            {
                "pair": f"{row.real_class}->{row.predicted_class}",
                "n": int(mask.sum()),
                "top_family_score_share": winner.head(3).round(3).to_dict(),
                "mean_best_recurrence_score": float(
                    x_valid.loc[mask, "best_recurrence_score"].mean()
                )
                if "best_recurrence_score" in x_valid.columns
                else None,
            }
        )

    save_confusion_figure(matrix, figures_dir / "confusion_matrix.png")
    save_f1_figure(metrics["per_class"], figures_dir / "f1_by_class.png")
    save_pair_bars(pairs, figures_dir / "top_confusion_pairs.png")
    for column in (
        "best_recurrence_score",
        "repeated_description_count",
        "n_transactions",
        "days_since_last_transaction",
    ):
        save_correct_incorrect_box(
            x_valid,
            correct,
            column,
            figures_dir / f"correct_vs_incorrect_{column}.png",
        )

    # Feature inventory from the live builder output.
    feature_names = x_valid.columns.tolist()
    families = {
        "volume": [name for name in feature_names if name.startswith(("n_", "transactions_"))],
        "amounts": [name for name in feature_names if name.startswith("amount_") or "fee_" in name],
        "temporal": [
            name
            for name in feature_names
            if any(
                token in name
                for token in (
                    "days_",
                    "history_",
                    "dow_",
                    "weekend",
                    "monthly_",
                    "active_months",
                    "event_gap",
                    "frequency_last",
                )
            )
        ],
        "recurrence": [
            name
            for name in feature_names
            if any(
                token in name
                for token in (
                    "recurrence",
                    "stream_",
                    "periodicity_",
                    "regular_",
                    "stable_amount",
                    "repeated_",
                )
            )
        ],
        "family_scores": [name for name in feature_names if name.startswith("family_")],
        "categorical_shares": [
            name
            for name in feature_names
            if name.startswith(("mcc_", "type_", "currency_", "direction_"))
        ],
    }
    # Text is used by logistic, not by the winning heuristic.
    feature_inventory = {
        "winning_model": "recurrence_heuristic",
        "numeric_feature_count": len(feature_names),
        "families": {
            key: {"count": len(values), "examples": values[:8]} for key, values in families.items()
        },
        "text_in_winning_model": False,
        "text_available_in_v1_pipeline": "ClientTextLogistic TF-IDF over client documents",
        "documents_built_but_unused_by_winner": True,
    }

    overpredicted = per_class.sort_values("overpredicted_ratio", ascending=False).head(3)
    underpredicted = per_class.sort_values("recall").head(3)

    payload = {
        "baseline_verified": True,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "n_clients": int(len(y_valid)),
        "none_bias": heuristic.none_bias,
        "temperature": heuristic.temperature,
        "per_class": per_class.to_dict(orient="records"),
        "weakest_classes": weak,
        "strongest_classes": per_class.sort_values("f1", ascending=False).head(3)["class"].tolist(),
        "overpredicted_classes": overpredicted["class"].tolist(),
        "underpredicted_classes": underpredicted["class"].tolist(),
        "top_confusion_pairs": pair_reports,
        "family_score_on_errors": family_win_notes,
        "correct_vs_incorrect_overall_top": overall_cmp.head(10).to_dict(orient="records"),
        "correct_vs_incorrect_by_weak_class": class_cmp,
        "feature_inventory": feature_inventory,
        "train_clients": int(len(x_train)),
        "documents_note": (
            "build_client_documents is available; winning heuristic does not use raw text."
        ),
    }
    # Touch documents to prove availability without feeding the heuristic.
    _ = build_client_documents(data.valid_transactions, x_valid.index)

    (output_dir / "v1_error_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {"macro_f1": macro_f1, "accuracy": accuracy, "n_clients": len(y_valid)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
