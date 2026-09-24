"""Diagnose UBS V1 validation errors without changing or retraining the baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from transaction_forecasting.ubs.data import (
    LABELS,
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    load_ubs_data,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.features import (
    ClientFeatureBuilder,
    build_client_documents,
    build_recurrence_streams,
)

RANDOM_SEED = 42
EXPECTED_MACRO_F1 = 0.2710242658492452
EXPECTED_ACCURACY = 0.266
SELECTED_HEURISTIC_SUFFIXES = (
    "_recurrence_score",
    "_occurrences",
    "_description_lift",
    "_regularity",
)


def markdown_table(frame: pd.DataFrame, digits: int = 3, max_rows: int | None = None) -> str:
    """Render a DataFrame without requiring the optional tabulate dependency."""
    shown = frame.head(max_rows).copy() if max_rows else frame.copy()
    shown = shown.reset_index(drop=True)
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    headers = [str(column) for column in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def robust_effect_size(left: pd.Series, right: pd.Series) -> float:
    """Return a winsorized standardized mean difference: left minus right."""
    left_values = pd.to_numeric(left, errors="coerce").dropna().astype(float)
    right_values = pd.to_numeric(right, errors="coerce").dropna().astype(float)
    if len(left_values) < 2 or len(right_values) < 2:
        return 0.0
    combined = pd.concat([left_values, right_values], ignore_index=True)
    lower, upper = combined.quantile([0.01, 0.99])
    left_values = left_values.clip(lower, upper)
    right_values = right_values.clip(lower, upper)
    pooled_variance = (
        (len(left_values) - 1) * left_values.var(ddof=1)
        + (len(right_values) - 1) * right_values.var(ddof=1)
    ) / (len(left_values) + len(right_values) - 2)
    if not np.isfinite(pooled_variance) or pooled_variance <= 1e-12:
        return 0.0
    return float((left_values.mean() - right_values.mean()) / np.sqrt(pooled_variance))


def entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(probabilities * np.log2(probabilities)).sum())


def add_diagnostic_features(
    features: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, set[str]]:
    """Add analysis-only features that expose signals not represented clearly in V1."""
    diagnostics = pd.DataFrame(index=features.index)
    grouped = transactions.groupby("client_id", sort=False)
    description_counts = transactions.groupby(["client_id", "description"]).size()
    diagnostics["diag_description_entropy"] = grouped["description"].apply(entropy)
    diagnostics["diag_description_diversity_ratio"] = (
        grouped["description"].nunique() / grouped.size()
    )
    diagnostics["diag_top_description_share"] = (
        description_counts.groupby("client_id").max() / grouped.size()
    )
    diagnostics["diag_singleton_description_share"] = (
        description_counts.eq(1).groupby("client_id").mean()
    )

    ordered = transactions.copy()
    ordered["age_days"] = (
        (pd.Timestamp("2026-01-01", tz="UTC") - ordered["timestamp"]).dt.total_seconds().div(86400)
    )
    recent_30 = ordered[ordered["age_days"].le(30)]
    older_31_180 = ordered[ordered["age_days"].gt(30) & ordered["age_days"].le(180)]
    recent_count = recent_30.groupby("client_id").size().reindex(features.index, fill_value=0)
    older_count = older_31_180.groupby("client_id").size().reindex(features.index, fill_value=0)
    diagnostics["diag_recent_30_share"] = recent_count / features["n_transactions"].clip(lower=1)
    diagnostics["diag_activity_acceleration_30_vs_prior150"] = (recent_count / 30.0) / (
        older_count / 150.0
    ).replace(0, np.nan)
    diagnostics["diag_recent_description_diversity"] = (
        recent_30.groupby("client_id")["description"]
        .nunique()
        .reindex(features.index, fill_value=0)
    )
    recent_amount = recent_30.groupby("client_id")["amount"].median()
    older_amount = older_31_180.groupby("client_id")["amount"].median()
    diagnostics["diag_recent_amount_log_ratio"] = np.log1p(recent_amount).reindex(
        features.index
    ) - np.log1p(older_amount).reindex(features.index)

    streams = build_recurrence_streams(transactions)
    if not streams.empty:
        stream_grouped = streams.groupby("client_id", sort=False)
        phase_error = (
            streams["days_since_last"] - streams["median_interval_days"]
        ).abs() / streams["median_interval_days"].clip(lower=7.0)
        diagnostics["diag_best_stream_phase_error"] = phase_error.groupby(
            streams["client_id"]
        ).min()
        diagnostics["diag_due_stream_count"] = (
            phase_error.le(0.25).groupby(streams["client_id"]).sum()
        )
        diagnostics["diag_recent_recurring_streams_45d"] = stream_grouped["days_since_last"].apply(
            lambda values: int(values.le(45).sum())
        )
        diagnostics["diag_outgoing_recurring_streams"] = stream_grouped[
            "direction_out_share"
        ].apply(lambda values: int(values.ge(0.8).sum()))

    diagnostics = diagnostics.reindex(features.index).replace([np.inf, -np.inf], np.nan)
    diagnostic_names = set(diagnostics.columns)
    return pd.concat([features, diagnostics], axis=1), diagnostic_names


def feature_family(name: str) -> str:
    """Assign a human-readable family to a V1 or diagnostic feature."""
    if name.startswith("diag_"):
        if any(token in name for token in ("description", "entropy")):
            return "text / merchant-description"
        if "amount" in name:
            return "amount"
        return "temporal / recurrence"
    if name.startswith("amount_") or name.startswith("fee_") or "amount_" in name:
        return "amount"
    if any(token in name for token in ("description_lift", "_descriptions")):
        return "text"
    if "description" in name or "merchant" in name:
        return "merchant / description"
    if any(
        token in name
        for token in (
            "recurrence",
            "stream_",
            "periodicity_",
            "regularity",
            "_frequency",
            "_occurrences",
        )
    ):
        return "recurrence"
    if any(
        token in name
        for token in (
            "history",
            "days_since",
            "last_",
            "dow_",
            "weekend",
            "month",
            "event_gap",
            "active_months",
        )
    ):
        return "temporal"
    if any(token in name for token in ("n_transaction", "n_out", "n_in", "_count", "_share")):
        return "volume / composition"
    return "other"


def selected_by_v1(name: str) -> bool:
    return name.startswith("family_") and name.endswith(SELECTED_HEURISTIC_SUFFIXES)


def build_confusion_tables(
    actual: pd.Series, predicted: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = pd.crosstab(actual, predicted).reindex(index=LABELS, columns=LABELS, fill_value=0)
    errors: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for real_class in LABELS:
        class_errors = int(matrix.loc[real_class].sum() - matrix.loc[real_class, real_class])
        for predicted_class in LABELS:
            if real_class == predicted_class:
                continue
            count = int(matrix.loc[real_class, predicted_class])
            if count:
                errors.append(
                    {
                        "real_class": real_class,
                        "predicted_class": predicted_class,
                        "number_errors": count,
                        "pct_of_class_errors": 100.0 * count / max(class_errors, 1),
                    }
                )
    for left_index, left in enumerate(LABELS):
        for right in LABELS[left_index + 1 :]:
            left_to_right = int(matrix.loc[left, right])
            right_to_left = int(matrix.loc[right, left])
            pairs.append(
                {
                    "class_a": left,
                    "class_b": right,
                    "a_to_b": left_to_right,
                    "b_to_a": right_to_left,
                    "total_confusions": left_to_right + right_to_left,
                }
            )
    error_table = pd.DataFrame(errors).sort_values(
        ["number_errors", "pct_of_class_errors"], ascending=False
    )
    pair_table = pd.DataFrame(pairs).sort_values("total_confusions", ascending=False)
    return matrix, error_table, pair_table


def correct_vs_incorrect(analysis: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in LABELS:
        class_rows = analysis[analysis["actual"].eq(label)]
        correct = class_rows[class_rows["correct"]]
        incorrect = class_rows[~class_rows["correct"]]
        for feature in feature_columns:
            effect = robust_effect_size(correct[feature], incorrect[feature])
            rows.append(
                {
                    "class": label,
                    "feature": feature,
                    "family": feature_family(feature),
                    "correct_median": correct[feature].median(),
                    "incorrect_median": incorrect[feature].median(),
                    "effect_size_correct_minus_incorrect": effect,
                    "abs_effect_size": abs(effect),
                    "n_correct": len(correct),
                    "n_incorrect": len(incorrect),
                }
            )
    return pd.DataFrame(rows).sort_values(["class", "abs_effect_size"], ascending=[True, False])


def correct_incorrect_dimensions(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Select one strongest correct/error contrast for each requested diagnostic dimension."""
    selectors = {
        "volume": lambda name: name in {"n_transactions", "n_out", "n_in", "out_share"},
        "amount": lambda name: "amount" in name or name.startswith("fee_"),
        "regularity": lambda name: any(
            token in name for token in ("regular", "interval_cv", "due_score")
        ),
        "recency": lambda name: "days_since" in name or "recency" in name,
        "description/merchant": lambda name: any(
            token in name for token in ("description", "entropy")
        ),
        "periodicity": lambda name: "periodicity_" in name,
        "history_vs_recent": lambda name: any(
            token in name
            for token in (
                "diag_recent",
                "diag_activity_acceleration",
                "transactions_last_",
                "frequency_last_",
            )
        ),
    }
    rows: list[dict[str, Any]] = []
    for label in LABELS:
        class_rows = comparisons[comparisons["class"].eq(label)]
        for dimension, selector in selectors.items():
            eligible = class_rows[class_rows["feature"].map(selector)]
            if eligible.empty:
                continue
            best = eligible.nlargest(1, "abs_effect_size").iloc[0]
            rows.append(
                {
                    "class": label,
                    "dimension": dimension,
                    "top_feature": best["feature"],
                    "correct_median": best["correct_median"],
                    "incorrect_median": best["incorrect_median"],
                    "effect_size": best["effect_size_correct_minus_incorrect"],
                }
            )
    return pd.DataFrame(rows)


def pair_separability(
    analysis: pd.DataFrame,
    feature_columns: list[str],
    pair_table: pd.DataFrame,
    top_pairs: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_rank, pair in enumerate(pair_table.head(top_pairs).itertuples(index=False), start=1):
        subset = analysis[analysis["actual"].isin([pair.class_a, pair.class_b])]
        target = subset["actual"].eq(pair.class_b).astype(int).to_numpy()
        numeric = subset[feature_columns].replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(numeric.median()).fillna(0.0)
        mi_values = mutual_info_classif(
            numeric.to_numpy(), target, discrete_features=False, random_state=RANDOM_SEED
        )
        for feature, mutual_information in zip(feature_columns, mi_values, strict=True):
            values = numeric[feature]
            if values.nunique() < 2:
                auc = 0.5
            else:
                auc = float(roc_auc_score(target, values))
            separability = max(auc, 1.0 - auc)
            class_a_values = subset.loc[subset["actual"].eq(pair.class_a), feature]
            class_b_values = subset.loc[subset["actual"].eq(pair.class_b), feature]
            rows.append(
                {
                    "pair_rank": pair_rank,
                    "class_a": pair.class_a,
                    "class_b": pair.class_b,
                    "total_confusions": pair.total_confusions,
                    "feature": feature,
                    "family": feature_family(feature),
                    "available_to_selected_v1": selected_by_v1(feature),
                    "analysis_only": feature.startswith("diag_"),
                    "auc_separability": separability,
                    "mutual_information": float(mutual_information),
                    "effect_size_a_minus_b": robust_effect_size(class_a_values, class_b_values),
                    "median_a": class_a_values.median(),
                    "median_b": class_b_values.median(),
                }
            )
    result = pd.DataFrame(rows)
    result["rank_score"] = (
        result["auc_separability"] - 0.5 + result["mutual_information"].clip(upper=1.0)
    )
    return result.sort_values(["pair_rank", "rank_score"], ascending=[True, False])


def pair_text_signals(
    documents: pd.Series,
    actual: pd.Series,
    pair_table: pd.DataFrame,
    top_pairs: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_rank, pair in enumerate(pair_table.head(top_pairs).itertuples(index=False), start=1):
        mask = actual.isin([pair.class_a, pair.class_b])
        pair_documents = documents.loc[mask]
        pair_actual = actual.loc[mask]
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, binary=True)
        design = vectorizer.fit_transform(pair_documents)
        terms = vectorizer.get_feature_names_out()
        in_a = pair_actual.eq(pair.class_a).to_numpy()
        in_b = ~in_a
        coverage_a = np.asarray(design[in_a].mean(axis=0)).ravel()
        coverage_b = np.asarray(design[in_b].mean(axis=0)).ravel()
        difference = coverage_a - coverage_b
        order = np.argsort(np.abs(difference))[::-1][:8]
        for term_index in order:
            rows.append(
                {
                    "pair_rank": pair_rank,
                    "class_a": pair.class_a,
                    "class_b": pair.class_b,
                    "term": terms[term_index],
                    "coverage_a_pct": 100.0 * coverage_a[term_index],
                    "coverage_b_pct": 100.0 * coverage_b[term_index],
                    "coverage_difference_pp": 100.0 * difference[term_index],
                    "favours": pair.class_a if difference[term_index] > 0 else pair.class_b,
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    matrix: pd.DataFrame, class_metrics: pd.DataFrame, figures_directory: Path
) -> None:
    figures_directory.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    for row in range(len(matrix)):
        for column in range(len(matrix.columns)):
            axis.text(column, row, int(matrix.iloc[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Real class")
    axis.set_title("V1 validation confusion matrix")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(figures_directory / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    ordered = class_metrics.sort_values("f1")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered["class"], ordered["f1"], color="#31688e")
    axis.axvline(class_metrics["f1"].mean(), color="#b63679", linestyle="--")
    axis.set_xlim(0, max(0.45, float(ordered["f1"].max()) + 0.05))
    axis.set_xlabel("F1")
    axis.set_title("V1 F1 by class")
    fig.tight_layout()
    fig.savefig(figures_directory / "f1_by_class.png", dpi=160)
    plt.close(fig)


def plot_handoff_diagnostics(
    matrix: pd.DataFrame,
    class_metrics: pd.DataFrame,
    pair_table: pd.DataFrame,
    analysis: pd.DataFrame,
    figures_directory: Path,
) -> None:
    """Create the small set of figures that materially supports the team handoff."""
    plot_results(matrix, class_metrics, figures_directory)

    top_pairs = pair_table.head(8).copy()
    top_pairs["pair"] = top_pairs["class_a"] + " ↔ " + top_pairs["class_b"]
    top_pairs = top_pairs.sort_values("total_confusions")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh(top_pairs["pair"], top_pairs["total_confusions"], color="#3b528b")
    axis.set_xlabel("Errors in both directions")
    axis.set_title("Largest V1 confusion pairs")
    fig.tight_layout()
    fig.savefig(figures_directory / "confusion_pairs.png", dpi=160)
    plt.close(fig)

    weakest = class_metrics.nsmallest(3, "f1")["class"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, label in zip(axes, weakest, strict=True):
        feature = f"family_{label}_recurrence_score"
        subset = analysis[analysis["actual"].eq(label)]
        correct = subset.loc[subset["correct"], feature].to_numpy(dtype=float)
        incorrect = subset.loc[~subset["correct"], feature].to_numpy(dtype=float)
        axis.boxplot([correct, incorrect], tick_labels=["correct", "incorrect"], showfliers=False)
        axis.set_title(f"{label}: family score")
        axis.set_ylabel("recurrence score")
    fig.suptitle("Why V1 understands some clients but not others")
    fig.tight_layout()
    fig.savefig(figures_directory / "correct_vs_incorrect.png", dpi=160)
    plt.close(fig)


def pair_hypothesis(class_a: str, class_b: str, top_features: pd.DataFrame) -> str:
    features = ", ".join(top_features["feature"].head(2).tolist())
    if "none" in {class_a, class_b}:
        family = class_b if class_a == "none" else class_a
        return (
            f"V1 confunde `none` con `{family}` porque cualquier evidencia recurrente y "
            "un lift positivo compiten sin una puerta explícita de 'stream cualificado'; "
            f"además, `none_bias=-1.0` penaliza la abstención. Las mayores separaciones "
            f"observadas aparecen en {features}."
        )
    return (
        "V1 reduce cada familia a una suma de evidencias y pierde identidad, fase y evolución "
        f"del stream. En este par destacan {features}."
    )


def build_opportunities(
    class_metrics: pd.DataFrame,
    error_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    text_signals: pd.DataFrame,
) -> pd.DataFrame:
    none_row = class_metrics.set_index("class").loc["none"]
    top_error = error_table.iloc[0]
    top_pair = pair_table.iloc[0]
    top_text = text_signals.iloc[0]
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "hallazgo": "V1 casi nunca se abstiene correctamente en none",
                "evidencia": (
                    f"F1={none_row['f1']:.3f}, recall={none_row['recall']:.3f}; "
                    f"{int(none_row['support'] - none_row['true_positives'])} FN de none"
                ),
                "clases": "none vs todas",
                "impacto": "muy alto",
                "dificultad": "media",
                "riesgo_overfitting": "medio",
                "responsable": "MODELS / TUNING + FEATURE ENGINEERING",
                "prioridad": "crítica",
                "experimento": "NONE-01: gate/umbral de abstención y margen top1-top2",
            },
            {
                "rank": 2,
                "hallazgo": "La mayor fuga es none hacia cloud",
                "evidencia": (
                    f"{int(top_error['number_errors'])} errores {top_error['real_class']}→"
                    f"{top_error['predicted_class']}; par bidireccional="
                    f"{int(top_pair['total_confusions'])}"
                ),
                "clases": "none / cloud",
                "impacto": "alto",
                "dificultad": "media",
                "riesgo_overfitting": "medio",
                "responsable": "TEMPORAL / RECURRENCIA",
                "prioridad": "crítica",
                "experimento": "TEMP-01: stream candidato, fase y estabilidad reciente",
            },
            {
                "rank": 3,
                "hallazgo": (
                    "MCC específicos separan los cinco pares principales pero V1 los ignora"
                ),
                "evidencia": "Top univariante: MCC 5732/4814/7997/6300/5734 para familia vs none",
                "clases": "cloud/mobile/gym/insurance/software vs none",
                "impacto": "alto",
                "dificultad": "baja-media",
                "riesgo_overfitting": "medio-alto",
                "responsable": "FEATURE ENGINEERING + MODELS / TUNING",
                "prioridad": "alta",
                "experimento": "FEAT-02: MCC del stream candidato con ablación y OOF",
            },
            {
                "rank": 4,
                "hallazgo": "El texto discrimina, pero también aparece en clientes none",
                "evidencia": (
                    f"`{top_text['term']}`: {top_text['coverage_a_pct']:.1f}% en "
                    f"{top_text['class_a']} y {top_text['coverage_b_pct']:.1f}% en "
                    f"{top_text['class_b']}"
                ),
                "clases": "familias vs none",
                "impacto": "medio-alto",
                "dificultad": "baja-media",
                "riesgo_overfitting": "medio",
                "responsable": "TEXT / MERCHANTS",
                "prioridad": "alta",
                "experimento": "TEXT-01: texto por stream + lifecycle; no bolsa global sola",
            },
            {
                "rank": 5,
                "hallazgo": "Falta modelar histórico frente a comportamiento reciente por stream",
                "evidencia": (
                    "V1 agrega recurrencia completa; no incorpora drift/lifecycle "
                    "específico del candidato"
                ),
                "clases": "none / cloud / mobile",
                "impacto": "medio-alto",
                "dificultad": "media",
                "riesgo_overfitting": "bajo-medio",
                "responsable": "TEMPORAL / RECURRENCIA",
                "prioridad": "alta",
                "experimento": "TEMP-02: actividad 30/90d vs historia y stream activo/inactivo",
            },
            {
                "rank": 6,
                "hallazgo": "La V1 elegida descarta 186 agregados y el TF-IDF calculado",
                "evidencia": "La heurística decide con 32 columnas de 218 (4 por familia)",
                "clases": "todas; especialmente music/streaming",
                "impacto": "medio",
                "dificultad": "media",
                "riesgo_overfitting": "medio-alto",
                "responsable": "MODELS / TUNING",
                "prioridad": "media",
                "experimento": "MODEL-01: ranking de candidatos con ablaciones por familia",
            },
            {
                "rank": 7,
                "hallazgo": "Heurística y mejor ML aciertan subconjuntos distintos",
                "evidencia": "199 victorias solo heurística y 213 solo Logistic en validación",
                "clases": "todas",
                "impacto": "medio",
                "dificultad": "media",
                "riesgo_overfitting": "alto si se calibra en un único split",
                "responsable": "MODELS / TUNING + EXPERIMENT TRACKING",
                "prioridad": "media",
                "experimento": "MODEL-02: meta-regla OOF o stacking leakage-safe",
            },
            {
                "rank": 8,
                "hallazgo": "La selección/calibración depende de un único valid de 1.000 clientes",
                "evidencia": (
                    "none_bias, temperatura, modelo y ensemble se eligen sobre el mismo split"
                ),
                "clases": "todas",
                "impacto": "medio",
                "dificultad": "media",
                "riesgo_overfitting": "reduce riesgo",
                "responsable": "EXPERIMENT TRACKING + INTEGRATION / VALIDATION",
                "prioridad": "alta",
                "experimento": "VALID-01: OOF por cliente y estabilidad de F1 por clase",
            },
        ]
    )


def render_pair_sections(
    top_pairs: pd.DataFrame, separability: pd.DataFrame, text_signals: pd.DataFrame
) -> str:
    sections: list[str] = []
    for pair_rank, pair in enumerate(top_pairs.itertuples(index=False), start=1):
        pair_features = separability[separability["pair_rank"].eq(pair_rank)].head(6)
        pair_text = text_signals[text_signals["pair_rank"].eq(pair_rank)].head(6)
        hypothesis = pair_hypothesis(pair.class_a, pair.class_b, pair_features)
        owner = (
            "Temporal / Recurrencia + Features / Modelos"
            if "none" in {pair.class_a, pair.class_b}
            else "Texto / Merchants + Modelos"
        )
        feature_table = pair_features[
            [
                "feature",
                "family",
                "available_to_selected_v1",
                "analysis_only",
                "auc_separability",
                "mutual_information",
                "effect_size_a_minus_b",
                "median_a",
                "median_b",
            ]
        ]
        pair_text_table = markdown_table(
            pair_text[
                [
                    "term",
                    "coverage_a_pct",
                    "coverage_b_pct",
                    "coverage_difference_pp",
                    "favours",
                ]
            ],
            digits=1,
        )
        sections.append(
            f"""### {pair.class_a} vs {pair.class_b} ({pair.total_confusions} errores cruzados)

{markdown_table(feature_table, digits=3)}

Señales de texto por presencia a nivel cliente:

{pair_text_table}

**Hipótesis:** {hypothesis}

**Posible solución:** construir candidatos por stream y añadir evidencia negativa, margen entre
familias, fase/recencia del stream y señales de texto normalizadas; medir F1 de ambas clases.

**Responsable recomendado:** {owner}.
"""
        )
    return "\n".join(sections)


def render_experiments() -> str:
    experiments = [
        (
            "EXP-NONE-01",
            "`none` tiene recall 0.058 y absorbe muy pocas predicciones.",
            "Añadir gate de candidato válido y umbral por margen top1-top2; calibrar OOF.",
            "Macro-F1, F1/recall de none y caída máxima de F1 por familia.",
            "Models / Tuning + Feature Engineering",
        ),
        (
            "EXP-TEMP-01",
            "Las principales confusiones son `none` contra familias recurrentes.",
            (
                "Features de fase, CV de intervalos reciente, última ocurrencia esperada y "
                "stream activo."
            ),
            "Macro-F1 y F1 de none/cloud/mobile; matriz de confusión.",
            "Temporal / Recurrencia",
        ),
        (
            "EXP-TEXT-01",
            "Frases de familia discriminan, pero también aparecen en muchos clientes none.",
            "Normalización + char 3-5 grams condicionados al stream y a su lifecycle.",
            "F1 de music/streaming/mobile y errores cruzados de los pares.",
            "Text / Merchants",
        ),
        (
            "EXP-FEAT-01",
            "La decisión suma evidencia por familia y pierde qué stream la produjo.",
            "Tabla candidate-level: familia, recencia, periodicidad, amount CV, lift y margen.",
            "Macro-F1 y top-2 recall de candidatos.",
            "Feature Engineering",
        ),
        (
            "EXP-FEAT-02",
            "MCC de familia es el mejor separador univariante en los cinco pares principales.",
            "Adjuntar MCC/share al stream candidato y hacer ablación OOF contra V1.",
            "Macro-F1, F1 de none y estabilidad por fold; auditar shortcut del generador.",
            "Feature Engineering + Models / Tuning",
        ),
        (
            "EXP-MODEL-01",
            "Heurística y Logistic tienen 199/213 aciertos exclusivos.",
            "Ranking o stacking OOF con ablaciones; sin ajustar sobre valid oficial.",
            "OOF Macro-F1 medio/desv.; luego una única evaluación valid.",
            "Models / Tuning + Experiment Tracking",
        ),
        (
            "EXP-VALID-01",
            "Modelo, none_bias y ensemble se eligen en el mismo split.",
            "Folds estratificados por cliente en train y registro de F1/confusión por fold.",
            "Estabilidad de Macro-F1 y de F1 por clase.",
            "Experiment Tracking + Integration / Validation",
        ),
    ]
    sections = []
    for identifier, problem, proposal, metric, owner in experiments:
        sections.append(
            f"""### {identifier}

**Problema:** {problem}

**Propuesta:** {proposal}

**Métrica objetivo:** {metric}

**Responsable:** {owner}.
"""
        )
    return "\n".join(sections)


def render_report(
    metrics: dict[str, Any],
    class_metrics: pd.DataFrame,
    matrix: pd.DataFrame,
    error_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    correct_incorrect: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    separability: pd.DataFrame,
    text_signals: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> str:
    hardest = class_metrics.nsmallest(3, "f1")["class"].tolist()
    easiest = class_metrics.nlargest(3, "f1")["class"].tolist()
    distribution = class_metrics[
        ["class", "support", "predicted_count", "prediction_minus_support"]
    ].sort_values("prediction_minus_support")
    top_correct_incorrect = (
        correct_incorrect.groupby("class", sort=False).head(3).reset_index(drop=True)
    )
    catalog_summary = (
        feature_catalog.groupby(["family", "status"]).size().rename("feature_count").reset_index()
    )
    top_pairs = pair_table.head(5)
    class_metrics_table = markdown_table(
        class_metrics[["class", "precision", "recall", "f1", "support", "predicted_count"]],
        digits=3,
    )
    correct_incorrect_table = markdown_table(
        top_correct_incorrect[
            [
                "class",
                "feature",
                "family",
                "correct_median",
                "incorrect_median",
                "effect_size_correct_minus_incorrect",
                "n_correct",
                "n_incorrect",
            ]
        ],
        digits=3,
    )
    dimension_display = dimension_summary.copy()
    dimension_display["signal"] = dimension_display.apply(
        lambda row: f"{row['top_feature']} (d={row['effect_size']:+.2f})", axis=1
    )
    dimension_table = markdown_table(
        dimension_display.pivot(index="class", columns="dimension", values="signal")
        .reindex(LABELS)
        .reset_index()
    )
    findings_table = markdown_table(
        opportunities[["hallazgo", "evidencia", "impacto", "responsable", "experimento"]],
        digits=3,
    )
    ranking_table = markdown_table(
        opportunities[
            [
                "rank",
                "hallazgo",
                "clases",
                "impacto",
                "dificultad",
                "riesgo_overfitting",
                "responsable",
                "prioridad",
            ]
        ],
        digits=3,
    )
    handoff = opportunities[["responsable", "hallazgo", "experimento", "prioridad"]].rename(
        columns={
            "responsable": "team_area",
            "hallazgo": "what_we_found",
            "experimento": "what_they_should_test",
            "prioridad": "priority",
        }
    )
    return f"""# V2 error analysis — UBS Transaction Activity Forecasting

## 1. Executive summary

V1 falla sobre todo por **no reconocer `none`**: de 293 clientes reales de esa clase solo
acierta 17. La heurística seleccionada penaliza explícitamente `none` (`none_bias=-1.0`) y
elige una familia cuando encuentra cualquier evidencia recurrente con lift positivo. Esto
provoca 276 falsos negativos de `none`, incluidos 89 `none→cloud`.

La segunda debilidad es de representación: el pipeline calcula 218 agregados, pero el modelo
ganador toma su decisión con solo 32 columnas (cuatro señales por familia). El TF-IDF, los
agregados temporales y los importes existen para candidatos no seleccionados, pero **no forman
parte de la decisión oficial V1**. La tercera es metodológica: modelo, bias y ensemble se
seleccionan sobre el mismo validation de 1.000 clientes, por lo que cualquier V2 debe usar OOF
en train antes de volver a mirar valid.

Acción inmediata: priorizar un gate explícito de candidato/`none`, features de stream activo
y evaluación OOF. Después, texto normalizado por stream y ranking candidate-level.

## 2. Baseline V1

- Modelo seleccionado: `recurrence_heuristic`.
- Macro-F1: **{metrics["macro_f1"]:.7f}**.
- Accuracy: **{metrics["accuracy"]:.4f}**.
- Clientes: **{int(class_metrics["support"].sum())}**.
- Reejecución reproducible; submission SHA-256 sin cambios:
  `{metrics["submission_sha256"]}`.
- La V1 no se modifica en este trabajo.

## 3. F1 por clase

{class_metrics_table}

- Más difíciles: **{", ".join(hardest)}**.
- Más fáciles (relativamente): **{", ".join(easiest)}**.
- Sobre/infrapredicción (`predicted_count - support`):

{markdown_table(distribution, digits=0)}

## 4. Matriz de confusión

Filas = clase real; columnas = predicción.

{markdown_table(matrix.reset_index(names="real"))}

Artefacto visual reproducible: `outputs/figures/v2_error_analysis/confusion_matrix.png`.

## 5. Principales errores

{markdown_table(error_table.head(15), digits=1)}

Conclusión: no hay un problema simétrico general. El volumen dominante es una **fuga desde
`none` hacia familias positivas**, especialmente `cloud` y `mobile`.

## 6. Pares de clases más problemáticos

Selección automática por suma de errores en ambas direcciones:

{markdown_table(top_pairs, digits=0)}

{render_pair_sections(top_pairs, separability, text_signals)}

## 7. Correctos vs incorrectos

Para cada clase se compararon medianas y effect size robusto (correctos menos incorrectos,
winsorizado 1–99%). Valores absolutos alrededor de 0.2/0.5/0.8 sugieren señal pequeña/media/grande;
no implican causalidad. Las tres diferencias mayores por clase son:

{correct_incorrect_table}

Las tablas completas están en `outputs/metrics/v2_error_analysis/correct_vs_incorrect.csv`.
Los importes mezclados nunca se interpretan sin moneda; las comparaciones monetarias son
diagnósticas y deben validarse por divisa.

Mejor contraste de correctos vs incorrectos por dimensión y clase (`d`: effect size; signo
positivo = mayor en correctos):

{dimension_table}

## 8. Qué está viendo actualmente V1

V1 construye 218 features numéricas, pero el modelo finalmente seleccionado usa solo:

1. `family_*_recurrence_score` (suma de score de stream × lift de descripción);
2. `family_*_occurrences`;
3. `family_*_description_lift`;
4. `family_*_regularity`.

Cada señal se calcula para las ocho clases: 32 columnas efectivas. CatBoost y Logistic sí
reciben otros agregados; Logistic añade word/bigram TF-IDF, pero ambos quedaron por debajo.

**Qué utiliza correctamente:** cuando existe evidencia clara de la familia real, su score de
recurrencia/lift es mucho mayor en aciertos que en errores (por ejemplo, `gym` d=2.06 y
`streaming` d=2.51). La combinación de descripción exacta y recurrencia sí contiene señal.

**Qué utiliza superficialmente:** reduce cada familia a cuatro agregados sumados. MCC, amount,
recencia, periodicidad y composición se calculan, pero la decisión seleccionada no los usa.

**Qué no está representado:** merchant normalizado/ID, lifecycle por stream, conflicto entre
candidatos, secuencia, margen de confianza y una noción de `none` como ausencia de candidato.

{markdown_table(catalog_summary)}

Catálogo completo: `outputs/metrics/v2_error_analysis/feature_catalog.csv`.

## 9. Qué parece estar faltando

- Gate explícito que decida si existe un stream candidato suficientemente convincente antes
  de elegir familia; `none` no debería competir como una familia positiva más.
- Identidad candidate-level: la suma por familia oculta qué descripción/stream aporta evidencia.
- Lifecycle y drift por stream: actividad reciente frente al histórico, stream activo/inactivo,
  fase esperada y pagos omitidos.
- Normalización de merchant/descripcion y robustez a variantes; no existe un merchant ID real.
- Margen top-1/top-2, conflicto entre streams y evidencia negativa.
- Validación OOF para separar mejora real de calibración al único validation oficial.

## 10. Hallazgos por área del equipo

{findings_table}

## 11. Ranking de oportunidades

{ranking_table}

## 12. Experimentos recomendados

{render_experiments()}

## 13. Recomendaciones para cada miembro/área

- **Temporal / Recurrencia:** TEMP-01 y TEMP-02; entregar features por stream, no solo por cliente.
- **Text / Merchants:** TEXT-01; normalización y char n-grams ligados al candidato recurrente.
- **Feature Engineering:** FEAT-01/02; tabla candidate-level, evidencia negativa y MCC del stream.
- **Models / Tuning:** NONE-01 y MODEL-01; gate de `none`, ranking y ablaciones, sin cambiar V1.
- **Error Analysis:** mantener slices por clase/par y comprobar si cada experimento reduce el
  error objetivo.
- **Experiment Tracking:** registrar F1 por clase, pares, semillas/folds y deltas, no solo Macro-F1.
- **Integration / Validation:** preservar valid oficial como prueba final y añadir quality gate
  de regresión V1.

## Team handoff

{markdown_table(handoff)}

## Reproducibilidad y artefactos

Ejecutar desde la raíz:

```powershell
python scripts/analyze_v1_errors.py --config configs/ubs_v1.toml
```

El script reutiliza `outputs/predictions/validation_v1_revalidated.csv`, comprueba IDs y
recalcula todas las métricas. No entrena ni modifica V1. Genera CSV/JSON/PNG ignorados por Git
y vuelve a escribir estos dos informes con resultados agregados, sin client IDs ni datos raw.
"""


def render_summary(
    metrics: dict[str, Any],
    class_metrics: pd.DataFrame,
    error_table: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> str:
    worst = class_metrics.nsmallest(3, "f1")[["class", "f1", "recall"]]
    confusions = error_table.head(3)[
        ["real_class", "predicted_class", "number_errors", "pct_of_class_errors"]
    ]
    findings = opportunities.head(5)[["hallazgo", "responsable", "prioridad"]]
    baseline_line = (
        f"**Macro-F1 {metrics['macro_f1']:.7f} · Accuracy {metrics['accuracy']:.4f} · "
        "1.000 clientes · V1 reproducida sin cambios.**"
    )
    return f"""# V2 error analysis — resumen para reunión

## Baseline

{baseline_line}

## 3 peores clases

{markdown_table(worst, digits=3)}

## 3 principales confusiones

{markdown_table(confusions, digits=1)}

## 5 hallazgos que repartir

{markdown_table(findings)}

## 3 experimentos prioritarios

1. **EXP-NONE-01 — Models + Features:** gate/umbral de `none` y margen top1-top2, calibrado OOF.
2. **EXP-TEMP-01 — Temporal:** fase, CV reciente, recencia esperada y stream activo/inactivo.
3. **EXP-FEAT-02 — Features + Models:** MCC del stream candidato, ablación y auditoría OOF.

## Recomendación inmediata

Atacar primero `none`: solo 17/293 correctos y 89 casos `none→cloud`. Construir el gate de
candidato junto con features de lifecycle; evaluar OOF y exigir que mejore F1 de `none` sin
hundir el F1 de las familias positivas. Después probar MCC y texto dentro del ranking
candidate-level.
"""


def render_ea_experiments() -> str:
    """Return isolated experiment briefs ready to send to the six partner workstreams."""
    experiments = [
        {
            "id": "EA-001",
            "owner": "MODELS / TUNING",
            "problem": "V1 predicts `none` only 68 times although its support is 293.",
            "evidence": (
                "Only 17/293 `none` clients are correct; `none_bias=-1.0` suppresses abstention."
            ),
            "change": (
                "Add a candidate-validity gate using top-1 score, top-1/top-2 margin and negative "
                "evidence; calibrate only with train OOF predictions."
            ),
            "secondary": "F1 and recall of `none`; worst per-family F1 delta.",
            "success": (
                "OOF Macro-F1 improves by at least 0.01 and `none` F1 by at least 0.05, with no "
                "positive-family F1 falling more than 0.03."
            ),
            "risk": "Threshold overfitting to one split and excessive abstention.",
        },
        {
            "id": "EA-002",
            "owner": "TEMPORAL / RECURRENCE",
            "problem": (
                "Historical family evidence remains active even when the stream may have ended."
            ),
            "evidence": (
                "The five largest class pairs are family-vs-`none`; V1 sums full-history streams."
            ),
            "change": (
                "Add candidate-stream phase error, recent interval CV, expected-next-date "
                "residual, missed-cycle count and active/inactive status."
            ),
            "secondary": "F1 for `none`, `cloud`, and `mobile`; family-to-`none` confusion counts.",
            "success": (
                "At least 10% relative reduction in the top two pair errors without lower Macro-F1."
            ),
            "risk": (
                "Cutoff leakage or encoding future transactions when constructing lifecycle labels."
            ),
        },
        {
            "id": "EA-003",
            "owner": "FEATURE ENGINEERING",
            "problem": (
                "The selected heuristic ignores strong category evidence already calculated."
            ),
            "evidence": (
                "MCC 5732/4814/7997/6300/5734 is the strongest univariate separator for the five "
                "largest family-vs-`none` pairs."
            ),
            "change": (
                "Attach MCC count/share, amount stability and recency to each recurring candidate; "
                "ablate MCC alone, then the complete candidate table."
            ),
            "secondary": "F1 of the five affected families and `none`; pair error counts.",
            "success": "OOF Macro-F1 gain of at least 0.01 that remains positive without raw text.",
            "risk": (
                "Synthetic-generator shortcut; MCC must be audited across folds and test-like data."
            ),
        },
        {
            "id": "EA-004",
            "owner": "TEXT / MERCHANTS",
            "problem": (
                "Exact descriptions are brittle, but global text also creates false positives."
            ),
            "evidence": (
                "`cloud access` appears in 83.1% of cloud clients but also 17.7% of "
                "`none`; equivalent overlap exists for the other family phrases."
            ),
            "change": (
                "Normalize merchant-like descriptions and test character 3-5 grams only at "
                "candidate stream level, combined with stream recency/lifecycle."
            ),
            "secondary": (
                "F1 of `music`, `streaming`, and `mobile`; text-driven `none` false positives."
            ),
            "success": "Positive OOF Macro-F1 delta and fewer text-driven `none` false positives.",
            "risk": (
                "Memorizing generator vocabulary or increasing false positives from "
                "historical text."
            ),
        },
        {
            "id": "EA-005",
            "owner": "EXPERIMENT TRACKING",
            "problem": (
                "Model, bias and ensemble are selected on the same official validation split."
            ),
            "evidence": (
                "Current V1 tunes `none_bias`, temperature and model selection on 1,000 "
                "valid clients."
            ),
            "change": (
                "Create fixed stratified client folds inside train; log fold Macro-F1, "
                "per-class F1, confusion pairs, seed and exact feature set."
            ),
            "secondary": "Mean/std of per-class F1 and top-pair error counts across folds.",
            "success": (
                "Every proposed V2 change has OOF deltas and variance before official validation."
            ),
            "risk": "Incorrect fold-level fitting of text/lift features causing leakage.",
        },
        {
            "id": "EA-006",
            "owner": "INTEGRATION / VALIDATION",
            "problem": "V2 components may improve accuracy while regressing rare-class Macro-F1.",
            "evidence": (
                "V1 accuracy and Macro-F1 rank models differently; `none` dominates the "
                "error volume."
            ),
            "change": (
                "Add gates for official Macro-F1, every class F1, top confusion counts, "
                "ID/order validity and deterministic prediction hash."
            ),
            "secondary": "Minimum class F1, submission validity and run-to-run identity.",
            "success": (
                "Integration rejects any unexplained V1 regression and records intended trade-offs."
            ),
            "risk": (
                "Overly rigid gates blocking a justified class trade-off; allow reviewed "
                "exceptions."
            ),
        },
    ]
    sections: list[str] = []
    for experiment in experiments:
        sections.append(
            f"""EXPERIMENT ID:
{experiment["id"]}

OWNER:
{experiment["owner"]}

PROBLEM:
{experiment["problem"]}

EVIDENCE:
{experiment["evidence"]}

CHANGE TO TEST:
{experiment["change"]}

PRIMARY METRIC:
Macro-F1.

SECONDARY METRIC:
{experiment["secondary"]}

SUCCESS CRITERIA:
{experiment["success"]}

RISK:
{experiment["risk"]}
"""
        )
    return "\n---\n\n".join(sections)


def render_handoff_pair_sections(
    pair_table: pd.DataFrame,
    separability: pd.DataFrame,
    text_signals: pd.DataFrame,
) -> str:
    sections: list[str] = []
    for pair_rank, pair in enumerate(pair_table.head(5).itertuples(index=False), start=1):
        pair_features = separability[separability["pair_rank"].eq(pair_rank)].head(5)
        pair_text = text_signals[text_signals["pair_rank"].eq(pair_rank)].head(4)
        hypothesis = pair_hypothesis(pair.class_a, pair.class_b, pair_features)
        owner = "TEMPORAL / RECURRENCE" if pair_rank == 1 else "FEATURE ENGINEERING"
        experiment = "EA-002" if pair_rank == 1 else "EA-003"
        evidence_table = markdown_table(
            pair_features[
                [
                    "feature",
                    "available_to_selected_v1",
                    "auc_separability",
                    "mutual_information",
                    "effect_size_a_minus_b",
                    "median_a",
                    "median_b",
                ]
            ],
            digits=3,
        )
        text_table = markdown_table(
            pair_text[["term", "coverage_a_pct", "coverage_b_pct", "coverage_difference_pp"]],
            digits=1,
        )
        sections.append(
            f"""**Pair {pair_rank}: `{pair.class_a}` vs `{pair.class_b}` — "
            f"{pair.total_confusions} crossed errors ({pair.a_to_b}/{pair.b_to_a})**

### Hypothesis

{hypothesis}

### Evidence

{evidence_table}

Text coverage by client:

{text_table}

### Recommended owner

{owner}

### Recommended experiment

Run **{experiment}** and report Macro-F1, both class F1 values and directional error counts.
"""
        )
    return "\n\n---\n\n".join(sections)


def render_team_handoff(
    metrics: dict[str, Any],
    class_metrics: pd.DataFrame,
    matrix: pd.DataFrame,
    error_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    correct_incorrect: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    separability: pd.DataFrame,
    text_signals: pd.DataFrame,
    opportunities: pd.DataFrame,
    baseline_ref: str,
) -> str:
    class_table = markdown_table(
        class_metrics[["class", "precision", "recall", "f1", "support", "predicted_count"]],
        digits=3,
    )
    profile = class_metrics[["class", "support", "predicted_count", "prediction_minus_support"]]
    confusion_table = error_table.head(10).rename(
        columns={
            "number_errors": "errors",
            "pct_of_class_errors": "percentage",
        }
    )
    confusion_table["severity"] = confusion_table["errors"].map(
        lambda value: (
            "CRITICAL"
            if value >= 50
            else "HIGH"
            if value >= 30
            else "MEDIUM"
            if value >= 20
            else "LOW"
        )
    )
    dimension_display = dimension_summary.copy()
    dimension_display["signal"] = dimension_display.apply(
        lambda row: f"{row['top_feature']} (d={row['effect_size']:+.2f})", axis=1
    )
    dimension_table = markdown_table(
        dimension_display.pivot(index="class", columns="dimension", values="signal")
        .reindex(LABELS)
        .reset_index()
    )
    catalog = (
        feature_catalog.groupby(["family", "status"]).size().rename("feature_count").reset_index()
    )
    finding_table = opportunities.rename(
        columns={
            "prioridad": "priority",
            "hallazgo": "finding",
            "evidencia": "evidence",
            "clases": "affected_classes",
            "responsable": "likely_owner",
            "dificultad": "estimated_difficulty",
            "riesgo_overfitting": "overfitting_risk",
            "experimento": "recommended_experiment",
        }
    )[
        [
            "priority",
            "finding",
            "evidence",
            "affected_classes",
            "likely_owner",
            "estimated_difficulty",
            "overfitting_risk",
            "recommended_experiment",
        ]
    ]
    finding_table["priority"] = finding_table["priority"].str.upper()
    area_rows = pd.DataFrame(
        [
            {
                "team_area": "TEMPORAL / RECURRENCE",
                "finding": "Full-history evidence does not identify ended streams.",
                "evidence": "All top five pairs are family-vs-none.",
                "recommended_action": "EA-002: candidate lifecycle and missed cycles.",
                "priority": "CRITICAL",
            },
            {
                "team_area": "TEXT / MERCHANTS",
                "finding": "Family phrases discriminate but leak into none histories.",
                "evidence": "cloud access coverage: 83.1% cloud vs 17.7% none.",
                "recommended_action": "EA-004: candidate-conditioned normalized text.",
                "priority": "HIGH",
            },
            {
                "team_area": "FEATURE ENGINEERING",
                "finding": "Strong MCC/candidate signals are not used by selected V1.",
                "evidence": "Family MCC is top univariate feature for five main pairs.",
                "recommended_action": "EA-003: candidate table and MCC ablation.",
                "priority": "HIGH",
            },
            {
                "team_area": "MODELS / TUNING",
                "finding": "V1 models none as a positive family instead of abstention.",
                "evidence": "17/293 none correct; none_bias=-1.0.",
                "recommended_action": "EA-001: OOF-calibrated candidate gate.",
                "priority": "CRITICAL",
            },
            {
                "team_area": "EXPERIMENT TRACKING",
                "finding": "Selection and calibration share official validation.",
                "evidence": "Model, temperature and bias use the same 1,000 clients.",
                "recommended_action": "EA-005: fixed client-level OOF protocol.",
                "priority": "HIGH",
            },
            {
                "team_area": "INTEGRATION / VALIDATION",
                "finding": "Accuracy can hide rare-class regression.",
                "evidence": "none F1=0.094 despite accuracy=0.266.",
                "recommended_action": "EA-006: class/pair regression gates.",
                "priority": "HIGH",
            },
        ]
    )
    code_files = pd.DataFrame(
        [
            {
                "path": "scripts/analyze_v1_errors.py",
                "purpose": "Reproduce metrics and all diagnostics/reports.",
                "integration_value": "ANALYSIS ONLY; reusable quality-gate inputs.",
            },
            {
                "path": "reports/handoff/santiago_error_analysis.md",
                "purpose": "Technical handoff for the V2 integration AI.",
                "integration_value": "READ FIRST; no production code.",
            },
            {
                "path": "reports/handoff/santiago_error_analysis_summary.md",
                "purpose": "Under-two-minute team meeting summary.",
                "integration_value": "MEETING AID.",
            },
            {
                "path": "reports/figures/error_analysis/*.png",
                "purpose": "Four compact error visualizations.",
                "integration_value": "DOCUMENTATION ONLY.",
            },
        ]
    )
    main_confusion_table = markdown_table(
        confusion_table[["real_class", "predicted_class", "errors", "percentage", "severity"]],
        digits=1,
    )
    return f"""# Team Handoff — Error Analysis

## 1. Scope

Person 5/7 analyzed where official V1 fails on the 1,000 validation clients and translated
those errors into isolated experiments for the other workstreams. We reproduced V1, inspected
its real implementation, computed class/pair metrics, compared correct vs incorrect clients,
and added analysis-only diagnostics. We did **not** change V1, tune or replace models, add
production features, use test labels, or alter the competitive submission.

## 2. Baseline

- Macro-F1: **{metrics["macro_f1"]:.7f}**.
- Accuracy: **{metrics["accuracy"]:.4f}**.
- Evaluated clients: **{int(class_metrics["support"].sum())}**.
- Baseline reference: `{baseline_ref}` on the active analysis branch; submission SHA-256
  `{metrics["submission_sha256"]}`.
- Protocol: fit supervised transforms on 2,000 train clients, score fixed eight-class Macro-F1
  on 1,000 disjoint validation clients with the official evaluator, keep test labels absent.

{class_table}

## 3. Error profile

- Weakest: `none` (F1 0.094), `music` (0.239), `streaming` (0.247).
- Relatively easiest: `mobile` (0.364), `gym` (0.340), `insurance` (0.324).
- Strongly overpredicted: `cloud` (+141) and `mobile` (+105).
- Strongly underpredicted: `none` (-225) and `music` (-27).

{markdown_table(profile.sort_values("prediction_minus_support"))}

## 4. Main confusion pairs

{main_confusion_table}

## 5. Pair-by-pair diagnosis

{render_handoff_pair_sections(pair_table, separability, text_signals)}

## 6. Correct vs incorrect clients

V1 understands clients when the real family's recurrence score, description lift and amount
stability are clearly present. Effect sizes are large for correct `gym` (family score d=2.06),
`insurance` (family descriptions d=2.39), `music` (description lift d=2.03), and `streaming`
(family score d=2.51). It fails when evidence is absent, stale, split across candidates or when
historical family text exists for a future `none` client. Correct `none` clients are unusual:
they have explicit learned `none`-description evidence (median score 1.595 versus 0 in errors),
showing that V1 does not treat `none` as absence of a valid future candidate.

Strongest correct/error contrast by requested dimension (`d`: correct minus incorrect):

{dimension_table}

Full reproducible table: `outputs/metrics/v2_error_analysis/correct_vs_incorrect.csv`.

## 7. What V1 currently sees

The feature builder computes 218 numeric aggregates, but selected V1 is
`recurrence_heuristic` and decides with 32 values: for each of eight labels it uses only
`family_*_recurrence_score`, `family_*_occurrences`, `family_*_description_lift`, and
`family_*_regularity`. Logistic sees global word/bigram TF-IDF and CatBoost sees the numeric
table, but neither was selected. Therefore computed amount, MCC, recency, periodicity and
composition features are not part of the official V1 decision.

{markdown_table(catalog)}

## 8. What V1 is missing

- A candidate-validity/abstention gate: `none` is incorrectly modeled as another positive family.
- Candidate identity: family sums hide which exact stream generated the evidence.
- Stream lifecycle: recent vs historical cadence, missed cycles, expected phase and inactive
  streams.
- Candidate-linked MCC, amount stability, recency and direction, despite being available in data.
- Merchant normalization and robust text tied to a recurring stream rather than a client text bag.
- Conflict/margin between top candidates and calibrated uncertainty.
- OOF evidence separating real improvement from tuning to official validation.

## 9. Main findings

{markdown_table(finding_table)}

## 10. Recommendations by team area

{markdown_table(area_rows)}

## 11. Experiments recommended

{render_ea_experiments()}

## 12. What NOT to do

- Do not assume more global TF-IDF solves the problem; family phrases also occur in `none`
  histories.
- Do not tune gates, class thresholds or ensemble weights directly on the official validation again.
- Do not use raw `client_id`, test labels, post-cutoff rows, or validation-fitted description lifts.
- Do not aggregate monetary values across currencies without a documented conversion or
  per-currency split.
- Do not promote MCC without OOF/stability checks; it may be a synthetic-generator shortcut.
- Do not interpret mutual information/effect size as causal or integrate every diagnostic feature.
- Do not replace V1 before an isolated experiment passes class-level and pair-level regression
  gates.

## 13. Code generated

{markdown_table(code_files)}

Generated CSV/JSON diagnostics under `outputs/metrics/v2_error_analysis/` remain ignored and
contain the complete tables. No dataset, client-level report, model artifact, or secret is tracked.

## 14. Commits worth reviewing

No commit was created in this environment. All files are **ANALYSIS ONLY** until reviewed. The
recommended future commit classification is `ANALYSIS ONLY`; cherry-pick production features
only from the owner experiments after independent validation.

## 15. Dependencies and conflicts

- Temporal/recurrence may independently create lifecycle features; reuse their canonical names.
- Text/merchant work may normalize descriptions; this analysis should consume, not duplicate, it.
- Feature engineering may create candidate tables; MCC must attach to that shared interface.
- Models/tuning owns the gate/ranker; avoid parallel threshold implementations.
- Tracking must fit description lifts/text inside each fold to prevent leakage.
- Integration should retain V1 artifacts and this report as regression references.

## 16. Recommended V2 priorities

1. **CRITICAL:** EA-001 candidate-validity gate for `none`, calibrated OOF.
2. **CRITICAL:** EA-002 stream lifecycle/phase features.
3. **HIGH:** EA-003 candidate-level MCC and recurrence evidence with shortcut audit.
4. **HIGH:** EA-005 client-level OOF protocol before official validation.
5. **HIGH:** EA-004 candidate-conditioned text/merchant normalization.
6. **HIGH:** EA-006 per-class and pair-level integration gates.

## 17. Executive summary for V2 integration AI

- V1 is reproducible at Macro-F1 0.2710243 and accuracy 0.2660.
- Its dominant failure is `none`: 17/293 correct, F1 0.094, 276 false negatives.
- `none→cloud` (89), `none→mobile` (43), and `none→insurance` (33) dominate errors.
- Cloud and mobile are overpredicted by 141 and 105 clients; none is underpredicted by 225.
- V1 selected a recurrence heuristic using only 32 of 218 computed numeric features.
- It models `none` as positive learned-description evidence, not absence of a valid candidate.
- Add an OOF-calibrated candidate-validity gate before choosing a family (EA-001).
- Add stream lifecycle, phase, missed-cycle and active/inactive signals (EA-002).
- Test candidate-linked MCC; it separates all five top pairs but may be a shortcut (EA-003).
- Condition normalized text on the candidate stream and lifecycle, not a global text bag (EA-004).
- Establish fixed client-level OOF tracking before touching official validation (EA-005).
- Preserve per-class F1 and confusion-pair gates during integration (EA-006).
- Do not use test labels, post-cutoff data, client IDs, or validation-fitted transforms.
- Do not assume global text or MCC generalizes without fold stability and ablation evidence.
- Keep V1 unchanged as the regression baseline until isolated experiments pass these gates.
"""


def render_meeting_handoff(
    metrics: dict[str, Any], class_metrics: pd.DataFrame, error_table: pd.DataFrame
) -> str:
    weakest = class_metrics.nsmallest(3, "f1").reset_index(drop=True)
    errors = error_table.head(3).reset_index(drop=True)
    weakest_lines = "\n".join(
        f"{index + 1}. `{row['class']}` — F1 {row['f1']:.3f}, recall {row['recall']:.3f}."
        for index, row in weakest.iterrows()
    )
    error_lines = "\n".join(
        f"{index + 1}. `{row['real_class']}→{row['predicted_class']}` — "
        f"{row['number_errors']} errors."
        for index, row in errors.iterrows()
    )
    return f"""# Error Analysis — Meeting Summary

## Baseline
Macro-F1: **{metrics["macro_f1"]:.7f}**
Accuracy: **{metrics["accuracy"]:.4f}**

## 3 weakest classes

{weakest_lines}

## 3 biggest confusions

{error_lines}

## 5 key findings

1. `none` is the main failure: only 17/293 correct; V1 suppresses abstention.
2. The five largest bidirectional pairs are a positive family versus `none`.
3. Selected V1 uses 32/218 features and ignores strong family MCC signals.
4. Family text is useful but also appears in `none`; text must be tied to active streams.
5. Selection/calibration on one validation split creates material overfitting risk.

## Who should investigate what

| team | task | priority |
| --- | --- | --- |
| MODELS / TUNING | EA-001: OOF candidate-validity gate for `none` | CRITICAL |
| TEMPORAL / RECURRENCE | EA-002: stream lifecycle, phase and missed cycles | CRITICAL |
| FEATURE ENGINEERING | EA-003: candidate MCC/recency/amount table | HIGH |
| TEXT / MERCHANTS | EA-004: candidate-conditioned normalized text | HIGH |
| EXPERIMENT TRACKING | EA-005: fixed client-level OOF protocol | HIGH |
| INTEGRATION / VALIDATION | EA-006: per-class/pair regression gates | HIGH |

## Top 3 recommended experiments

1. **EA-001:** candidate gate for `none`, using score strength and top-1/top-2 margin.
2. **EA-002:** lifecycle/phase features to distinguish active family streams from ended history.
3. **EA-003:** candidate-level MCC ablation with fold-stability and shortcut checks.

## Main risk

Overfitting thresholds, MCC or vocabulary to the single official validation or synthetic generator.

## Recommendation for V2

Fix candidate validity/`none` first, then lifecycle and candidate evidence. Require OOF gains and
per-class regression gates before integrating; keep V1 unchanged as fallback.
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ubs_v1.toml")
    parser.add_argument(
        "--predictions", default="outputs/predictions/validation_v1_revalidated.csv"
    )
    parser.add_argument("--data-directory")
    parser.add_argument("--output-directory", default="outputs/metrics/v2_error_analysis")
    parser.add_argument("--figures-directory", default="reports/figures/error_analysis")
    parser.add_argument("--reports-directory", default="reports/handoff")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    with Path(arguments.config).open("rb") as config_file:
        settings = tomllib.load(config_file)
    data_directory = arguments.data_directory or settings["data"]["directory"]
    prediction_path = Path(arguments.predictions)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing {prediction_path}; run scripts/run_ubs_baseline.py first")

    data = load_ubs_data(data_directory)
    builder = ClientFeatureBuilder().fit(data.train_transactions, data.train_labels)
    v1_features = builder.transform(data.valid_transactions)
    analysis_features, diagnostic_names = add_diagnostic_features(
        v1_features, data.valid_transactions
    )
    labels = data.valid_labels.set_index("client_id")[TARGET_COLUMN].reindex(v1_features.index)
    predictions_frame = pd.read_csv(prediction_path, dtype={"client_id": str})
    if predictions_frame["client_id"].duplicated().any():
        raise ValueError("Validation predictions contain duplicate client IDs")
    predictions = predictions_frame.set_index("client_id")[PREDICTION_COLUMN]
    if set(predictions.index) != set(labels.index):
        raise ValueError("Validation prediction IDs do not match validation labels")
    predictions = predictions.reindex(labels.index)
    metrics = evaluate_predictions(labels, predictions)
    if not np.isclose(metrics["macro_f1"], EXPECTED_MACRO_F1, atol=1e-12) or not np.isclose(
        metrics["accuracy"], EXPECTED_ACCURACY, atol=1e-12
    ):
        raise ValueError(
            "Validation predictions do not reproduce official V1 metrics: "
            f"macro_f1={metrics['macro_f1']}, accuracy={metrics['accuracy']}"
        )
    submission_path = Path(settings["outputs"]["submission"])
    if not submission_path.exists():
        raise FileNotFoundError(f"Missing official V1 submission: {submission_path}")
    metrics["submission_sha256"] = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    actual = labels.rename("actual")
    predicted = predictions.rename("predicted")

    matrix, error_table, pair_table = build_confusion_tables(actual, predicted)
    class_rows = []
    for label in LABELS:
        values = metrics["per_class"][label]
        support = int(actual.eq(label).sum())
        predicted_count = int(predicted.eq(label).sum())
        true_positives = int((actual.eq(label) & predicted.eq(label)).sum())
        class_rows.append(
            {
                "class": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": support,
                "predicted_count": predicted_count,
                "prediction_minus_support": predicted_count - support,
                "true_positives": true_positives,
                "false_negatives": support - true_positives,
                "false_positives": predicted_count - true_positives,
            }
        )
    class_metrics = pd.DataFrame(class_rows)

    analysis = analysis_features.join(actual).join(predicted)
    analysis["correct"] = analysis["actual"].eq(analysis["predicted"])
    feature_columns = analysis_features.columns.tolist()
    correct_incorrect = correct_vs_incorrect(analysis, feature_columns)
    dimension_summary = correct_incorrect_dimensions(correct_incorrect)
    separability = pair_separability(analysis, feature_columns, pair_table)
    documents = build_client_documents(data.valid_transactions, v1_features.index)
    text_signals = pair_text_signals(documents, actual, pair_table)

    feature_catalog = pd.DataFrame(
        {
            "feature": feature_columns,
            "family": [feature_family(feature) for feature in feature_columns],
            "status": [
                "analysis-only diagnostic"
                if feature in diagnostic_names
                else "selected V1 decision"
                if selected_by_v1(feature)
                else "computed but not used by selected V1"
                for feature in feature_columns
            ],
        }
    )
    opportunities = build_opportunities(class_metrics, error_table, pair_table, text_signals)

    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    class_metrics.to_csv(output_directory / "class_metrics.csv", index=False)
    matrix.to_csv(output_directory / "confusion_matrix.csv")
    error_table.to_csv(output_directory / "directional_confusions.csv", index=False)
    pair_table.to_csv(output_directory / "confusion_pairs.csv", index=False)
    correct_incorrect.to_csv(output_directory / "correct_vs_incorrect.csv", index=False)
    dimension_summary.to_csv(output_directory / "correct_vs_incorrect_dimensions.csv", index=False)
    separability.to_csv(output_directory / "pair_separability.csv", index=False)
    text_signals.to_csv(output_directory / "pair_text_signals.csv", index=False)
    feature_catalog.to_csv(output_directory / "feature_catalog.csv", index=False)
    opportunities.to_csv(output_directory / "opportunities.csv", index=False)
    save_json(
        output_directory / "summary.json",
        {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "submission_sha256": metrics["submission_sha256"],
            "validation_clients": len(labels),
            "v1_feature_count": len(v1_features.columns),
            "selected_v1_feature_count": int(
                sum(selected_by_v1(feature) for feature in v1_features.columns)
            ),
            "diagnostic_feature_count": len(diagnostic_names),
            "hardest_classes": class_metrics.nsmallest(3, "f1")["class"].tolist(),
            "top_directional_confusions": error_table.head(5).to_dict(orient="records"),
            "top_pairs": pair_table.head(5).to_dict(orient="records"),
        },
    )
    plot_handoff_diagnostics(
        matrix,
        class_metrics,
        pair_table,
        analysis,
        Path(arguments.figures_directory),
    )

    reports_directory = Path(arguments.reports_directory)
    reports_directory.mkdir(parents=True, exist_ok=True)
    baseline_ref = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = render_team_handoff(
        metrics,
        class_metrics,
        matrix,
        error_table,
        pair_table,
        correct_incorrect,
        dimension_summary,
        feature_catalog,
        separability,
        text_signals,
        opportunities,
        baseline_ref,
    )
    summary = render_meeting_handoff(metrics, class_metrics, error_table)
    report_path = reports_directory / "santiago_error_analysis.md"
    summary_path = reports_directory / "santiago_error_analysis_summary.md"
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            {
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "reports": [
                    str(report_path),
                    str(summary_path),
                ],
                "output_directory": str(output_directory),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
