"""Leakage-safe pseudo-cutoff supervision for recurrent transaction streams.

The experiment creates point-in-time candidate sets from train histories, evaluates
them out of fold with client-isolated folds, freezes one ranker, and only then scores
the frozen strategy on the official validation partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from transaction_forecasting.ubs.data import CUTOFF, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.features import ClientFeatureBuilder
from transaction_forecasting.ubs.models import RecurrenceHeuristic
from transaction_forecasting.ubs.temporal_features import apply_temporal_blocks, temporal_streams

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/metrics/v3_discovery/jaime_pseudocutoffs"
DEFAULT_REPORT = ROOT / "reports/handoff/v3_discovery/jaime_pseudocutoffs.md"
NONE_STREAM = "__none__"

NUMERIC_FEATURES = [
    "is_none",
    "candidate_count",
    "history_event_count",
    "history_length_days",
    "count",
    "recency_days",
    "stream_span_days",
    "last_gap_days",
    "median_gap_days",
    "gap_mad_days",
    "gap_std_days",
    "amount_mean",
    "amount_median",
    "amount_cv",
    "weekday_concentration",
    "monthday_concentration",
    "phase_sin",
    "phase_cos",
    "cutoff_doy_sin",
    "cutoff_doy_cos",
    "predicted_next_days",
]
CATEGORICAL_FEATURES = ["mcc_mode", "type_mode", "direction_mode"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class Config:
    horizon_days: int = 90
    min_history_days: int = 120
    min_history_events: int = 12
    cutoff_spacing_days: int = 60
    min_stream_events: int = 2
    folds: int = 5
    seed: int = 42


def _mode(values: pd.Series) -> str:
    modes = values.astype(str).mode()
    return str(modes.iloc[0]) if len(modes) else "missing"


def _concentration(values: np.ndarray, period: float) -> float:
    if len(values) < 2:
        return 0.0
    return float(abs(np.exp(2j * np.pi * values / period).mean()))


def _next_due_days(recency: float, median_gap: float) -> float:
    if not np.isfinite(median_gap) or median_gap <= 0:
        return np.nan
    cycles = max(1, math.ceil(recency / median_gap - 1e-12))
    return max(cycles * median_gap - recency, 0.0)


def candidate_rows_at_cutoff(
    client_id: str,
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: Config,
    future: pd.DataFrame | None = None,
    pseudo_id: str | None = None,
) -> list[dict[str, object]]:
    """Build features from ``history`` only and optional labels from ``future`` only."""
    if history.empty or history.timestamp.ge(cutoff).any():
        raise ValueError("History must be non-empty and strictly before cutoff")
    if future is not None:
        end = cutoff + pd.Timedelta(days=config.horizon_days)
        if future.timestamp.lt(cutoff).any() or future.timestamp.ge(end).any():
            raise ValueError("Future labels must lie inside the complete forecast window")
    key = pseudo_id or f"{client_id}|{cutoff.isoformat()}"
    repeated = [
        (description, group.sort_values("timestamp", kind="stable"))
        for description, group in history.groupby("description", sort=True)
        if len(group) >= config.min_stream_events
    ]
    future_first: dict[str, pd.Timestamp] = {}
    if future is not None and not future.empty:
        future_first = future.groupby("description").timestamp.min().to_dict()
    eligible_next = {
        description: timestamp
        for description, timestamp in future_first.items()
        if any(description == candidate for candidate, _ in repeated)
    }
    winner = min(
        eligible_next, key=lambda value: (eligible_next[value], value), default=NONE_STREAM
    )
    first_history = history.timestamp.min()
    history_days = (cutoff - first_history).total_seconds() / 86400
    cutoff_angle = 2 * np.pi * (cutoff.dayofyear - 1) / 365.25
    rows: list[dict[str, object]] = []
    candidate_count = len(repeated)
    for description, group in repeated:
        times = pd.DatetimeIndex(group.timestamp.drop_duplicates().sort_values())
        gaps = np.diff(times.as_unit("ns").asi8) / 86_400_000_000_000
        median_gap = float(np.median(gaps)) if len(gaps) else np.nan
        recency = (cutoff - times[-1]).total_seconds() / 86400
        amounts = pd.to_numeric(group.amount, errors="coerce").astype(float)
        amount_mean = float(amounts.mean())
        amount_std = float(amounts.std(ddof=0))
        phase = (recency / median_gap) % 1 if np.isfinite(median_gap) and median_gap > 0 else 0
        next_timestamp = future_first.get(description)
        time_to_next = (
            (next_timestamp - cutoff).total_seconds() / 86400
            if next_timestamp is not None
            else np.nan
        )
        rows.append(
            {
                "pseudo_id": key,
                "client_id": str(client_id),
                "cutoff": cutoff,
                "stream": str(description),
                "is_winner": int(description == winner) if future is not None else np.nan,
                "occurs_90d": int(next_timestamp is not None) if future is not None else np.nan,
                "time_to_next_days": time_to_next,
                "is_none": 0.0,
                "candidate_count": candidate_count,
                "history_event_count": len(history),
                "history_length_days": history_days,
                "count": len(group),
                "recency_days": recency,
                "stream_span_days": (times[-1] - times[0]).total_seconds() / 86400,
                "last_gap_days": float(gaps[-1]) if len(gaps) else np.nan,
                "median_gap_days": median_gap,
                "gap_mad_days": (
                    float(np.median(np.abs(gaps - median_gap))) if len(gaps) >= 2 else np.nan
                ),
                "gap_std_days": float(np.std(gaps)) if len(gaps) >= 2 else np.nan,
                "amount_mean": amount_mean,
                "amount_median": float(amounts.median()),
                "amount_cv": amount_std / max(abs(amount_mean), 1e-6),
                "weekday_concentration": _concentration(times.dayofweek.to_numpy(), 7),
                "monthday_concentration": _concentration((times.day - 1).to_numpy(), 31),
                "phase_sin": math.sin(2 * math.pi * phase),
                "phase_cos": math.cos(2 * math.pi * phase),
                "cutoff_doy_sin": math.sin(cutoff_angle),
                "cutoff_doy_cos": math.cos(cutoff_angle),
                "predicted_next_days": _next_due_days(recency, median_gap),
                "mcc_mode": _mode(group.mcc),
                "type_mode": _mode(group.type),
                "direction_mode": _mode(group.direction),
            }
        )
    none_row = {
        "pseudo_id": key,
        "client_id": str(client_id),
        "cutoff": cutoff,
        "stream": NONE_STREAM,
        "is_winner": int(winner == NONE_STREAM) if future is not None else np.nan,
        "occurs_90d": int(winner == NONE_STREAM) if future is not None else np.nan,
        "time_to_next_days": np.nan,
        "is_none": 1.0,
        "candidate_count": candidate_count,
        "history_event_count": len(history),
        "history_length_days": history_days,
        **{
            name: 0.0
            for name in NUMERIC_FEATURES
            if name
            not in {"is_none", "candidate_count", "history_event_count", "history_length_days"}
        },
        "mcc_mode": NONE_STREAM,
        "type_mode": NONE_STREAM,
        "direction_mode": NONE_STREAM,
    }
    rows.append(none_row)
    return rows


def pseudo_cutoffs_for_client(transactions: pd.DataFrame, config: Config) -> list[pd.Timestamp]:
    """Select separated cutoffs with enough past and a globally observable future."""
    ordered = transactions.sort_values("timestamp", kind="stable")
    earliest = ordered.timestamp.min() + pd.Timedelta(days=config.min_history_days)
    latest = CUTOFF - pd.Timedelta(days=config.horizon_days)
    if earliest > latest:
        return []
    candidate = earliest.ceil("D")
    cutoffs: list[pd.Timestamp] = []
    while candidate <= latest:
        history = ordered.loc[ordered.timestamp.lt(candidate)]
        if len(history) >= config.min_history_events:
            cutoffs.append(candidate)
        candidate += pd.Timedelta(days=config.cutoff_spacing_days)
    return cutoffs


def generate_pseudo_examples(transactions: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Generate candidate rows without ever passing future data into feature construction."""
    required = {"client_id", "timestamp", "description", "amount", "mcc", "type", "direction"}
    if missing := required.difference(transactions.columns):
        raise ValueError(f"Missing transaction columns: {sorted(missing)}")
    if transactions.timestamp.ge(CUTOFF).any():
        raise ValueError("Pseudo-cutoff source must end before the official cutoff")
    rows: list[dict[str, object]] = []
    for client_id, client in transactions.groupby("client_id", sort=True):
        client = client.sort_values("timestamp", kind="stable")
        for cutoff in pseudo_cutoffs_for_client(client, config):
            end = cutoff + pd.Timedelta(days=config.horizon_days)
            history = client.loc[client.timestamp.lt(cutoff)]
            future = client.loc[client.timestamp.ge(cutoff) & client.timestamp.lt(end)]
            rows.extend(
                candidate_rows_at_cutoff(str(client_id), history, cutoff, config, future=future)
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No pseudo-cutoffs satisfy the configuration")
    winners = result.groupby("pseudo_id").is_winner.sum()
    if not winners.eq(1).all():
        raise RuntimeError("Every pseudo-cutoff must have exactly one winner")
    return result.sort_values(["pseudo_id", "stream"], kind="stable").reset_index(drop=True)


def official_candidate_rows(transactions: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for client_id, history in transactions.groupby("client_id", sort=True):
        rows.extend(candidate_rows_at_cutoff(str(client_id), history, CUTOFF, config))
    return (
        pd.DataFrame(rows)
        .sort_values(["pseudo_id", "stream"], kind="stable")
        .reset_index(drop=True)
    )


def client_folds(frame: pd.DataFrame, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    clients = np.array(sorted(frame.client_id.unique()))
    if len(clients) < folds:
        raise ValueError("Fewer clients than folds")
    result = []
    for train, heldout in KFold(folds, shuffle=True, random_state=seed).split(clients):
        fit_clients, heldout_clients = set(clients[train]), set(clients[heldout])
        fit_rows = frame.index[frame.client_id.isin(fit_clients)].to_numpy()
        heldout_rows = frame.index[frame.client_id.isin(heldout_clients)].to_numpy()
        if fit_clients & heldout_clients:
            raise RuntimeError("Client leakage across folds")
        result.append((fit_rows, heldout_rows))
    return result


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_models(seed: int) -> dict[str, tuple[Pipeline, Pipeline]]:
    return {
        "logistic_pointwise": (
            Pipeline(
                [
                    ("features", _preprocessor(scale=True)),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            Pipeline([("features", _preprocessor(scale=True)), ("model", Ridge(alpha=10.0))]),
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    ("features", _preprocessor(scale=False)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.06,
                            max_iter=120,
                            max_leaf_nodes=15,
                            l2_regularization=1.0,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            Pipeline(
                [
                    ("features", _preprocessor(scale=False)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            learning_rate=0.06,
                            max_iter=120,
                            max_leaf_nodes=15,
                            l2_regularization=1.0,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    }


def _rule_scores(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    due = frame.predicted_next_days.to_numpy(dtype=float)
    score = np.where(np.isfinite(due) & (due < horizon), 1.0 - due / horizon, -1.0)
    return np.where(frame.stream.eq(NONE_STREAM), 0.0, score)


def _ranking_metrics(frame: pd.DataFrame, score: np.ndarray, predicted_time: np.ndarray) -> dict:
    scored = frame[["pseudo_id", "stream", "is_winner", "candidate_count"]].copy()
    scored["score"] = score
    scored["predicted_time"] = predicted_time
    ranked = scored.sort_values(
        ["pseudo_id", "score", "stream"], ascending=[True, False, True], kind="stable"
    )
    top1 = ranked.groupby("pseudo_id", sort=False).head(1).set_index("pseudo_id")
    top2 = ranked.groupby("pseudo_id", sort=False).head(2)
    truth = frame.loc[frame.is_winner.eq(1), ["pseudo_id", "stream", "candidate_count"]].set_index(
        "pseudo_id"
    )
    predicted = top1.stream.reindex(truth.index)
    actual = truth.stream
    top2_hit = (
        top2.assign(hit=top2.stream.eq(top2.pseudo_id.map(actual)))
        .groupby("pseudo_id")
        .hit.any()
        .reindex(truth.index)
    )
    actual_none = actual.eq(NONE_STREAM)
    predicted_none = predicted.eq(NONE_STREAM)
    buckets = pd.cut(
        truth.candidate_count,
        bins=[-1, 0, 1, 2, 4, np.inf],
        labels=["0", "1", "2", "3-4", "5+"],
    )
    by_candidates = {}
    for bucket in buckets.cat.categories:
        selected = buckets.eq(bucket)
        by_candidates[str(bucket)] = {
            "pseudo_cutoffs": int(selected.sum()),
            "top1_accuracy": float(predicted[selected].eq(actual[selected]).mean()),
        }
    real = frame.stream.ne(NONE_STREAM) & frame.occurs_90d.eq(1)
    time_mae = (
        float(mean_absolute_error(frame.loc[real, "time_to_next_days"], predicted_time[real]))
        if real.any()
        else None
    )
    return {
        "top1_next_stream_accuracy": float(predicted.eq(actual).mean()),
        "top2_coverage": float(top2_hit.mean()),
        "none_recall": float(predicted_none[actual_none].mean()) if actual_none.any() else None,
        "binary_f1_occurs_90d": float(f1_score(~actual_none, ~predicted_none, zero_division=0)),
        "time_to_next_mae_days": time_mae,
        "by_candidate_count": by_candidates,
    }


def evaluate_oof(frame: pd.DataFrame, config: Config) -> tuple[dict, pd.DataFrame, str]:
    started = time.perf_counter()
    scores = {"next_date_rule": _rule_scores(frame, config.horizon_days)}
    predicted_times = {
        "next_date_rule": frame.predicted_next_days.fillna(config.horizon_days).to_numpy()
    }
    models = make_models(config.seed)
    folds = client_folds(frame, config.folds, config.seed)
    for name, (classifier_template, regressor_template) in models.items():
        oof_score = np.full(len(frame), np.nan)
        oof_time = np.full(len(frame), np.nan)
        for fit_rows, heldout_rows in folds:
            train, heldout = frame.loc[fit_rows], frame.loc[heldout_rows]
            classifier = clone(classifier_template).fit(
                train[MODEL_FEATURES], train.is_winner.astype(int)
            )
            oof_score[heldout_rows] = classifier.predict_proba(heldout[MODEL_FEATURES])[:, 1]
            timed_train = train.stream.ne(NONE_STREAM) & train.occurs_90d.eq(1)
            regressor = clone(regressor_template).fit(
                train.loc[timed_train, MODEL_FEATURES], train.loc[timed_train, "time_to_next_days"]
            )
            real_heldout = heldout.stream.ne(NONE_STREAM)
            oof_time[heldout_rows[real_heldout]] = regressor.predict(
                heldout.loc[real_heldout, MODEL_FEATURES]
            )
        if np.isnan(oof_score).any():
            raise RuntimeError(f"Missing OOF scores for {name}")
        scores[name], predicted_times[name] = oof_score, oof_time
    metrics = {
        name: _ranking_metrics(frame, value, predicted_times[name])
        for name, value in scores.items()
    }
    learned = max(
        models,
        key=lambda name: (
            metrics[name]["top1_next_stream_accuracy"],
            metrics[name]["top2_coverage"],
            name == "logistic_pointwise",
        ),
    )
    prediction_rows = frame[["pseudo_id", "client_id", "cutoff", "stream", "is_winner"]].copy()
    for name in scores:
        prediction_rows[f"score_{name}"] = scores[name]
        prediction_rows[f"time_{name}"] = predicted_times[name]
    metrics["seconds"] = time.perf_counter() - started
    return metrics, prediction_rows, learned


def fit_ranker(frame: pd.DataFrame, model_name: str, config: Config) -> tuple[Pipeline, Pipeline]:
    classifier, regressor = make_models(config.seed)[model_name]
    classifier.fit(frame[MODEL_FEATURES], frame.is_winner.astype(int))
    timed = frame.stream.ne(NONE_STREAM) & frame.occurs_90d.eq(1)
    regressor.fit(frame.loc[timed, MODEL_FEATURES], frame.loc[timed, "time_to_next_days"])
    return classifier, regressor


def choose_streams(frame: pd.DataFrame, scores: np.ndarray) -> pd.Series:
    ranked = frame[["pseudo_id", "client_id", "stream"]].copy()
    ranked["score"] = scores
    top = (
        ranked.sort_values(
            ["pseudo_id", "score", "stream"], ascending=[True, False, True], kind="stable"
        )
        .groupby("pseudo_id", sort=False)
        .head(1)
    )
    return top.set_index("client_id").stream


def family_mapping(builder: ClientFeatureBuilder) -> pd.Series:
    if builder.description_lift_.empty:
        return pd.Series(dtype=object)
    best = builder.description_lift_.idxmax(axis=1)
    return best.where(builder.description_lift_.max(axis=1).gt(0), "none")


def map_streams_to_family(streams: pd.Series, mapping: pd.Series) -> pd.Series:
    prediction = streams.map(mapping).fillna("none")
    prediction.loc[streams.eq(NONE_STREAM)] = "none"
    return prediction


def official_validation(
    train_transactions: pd.DataFrame,
    train_labels: pd.DataFrame,
    valid_transactions: pd.DataFrame,
    valid_labels: pd.DataFrame,
    pseudo: pd.DataFrame,
    selected: str,
    config: Config,
) -> tuple[dict, pd.DataFrame]:
    classifier, _ = fit_ranker(pseudo, selected, config)
    candidates = official_candidate_rows(valid_transactions, config)
    learned_scores = classifier.predict_proba(candidates[MODEL_FEATURES])[:, 1]
    rule_scores = _rule_scores(candidates, config.horizon_days)
    learned_stream = choose_streams(candidates, learned_scores)
    rule_stream = choose_streams(candidates, rule_scores)

    builder = ClientFeatureBuilder().fit(train_transactions, train_labels)
    mapping = family_mapping(builder)
    target = valid_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    learned_family = map_streams_to_family(learned_stream, mapping).reindex(
        target.index, fill_value="none"
    )
    rule_family = map_streams_to_family(rule_stream, mapping).reindex(
        target.index, fill_value="none"
    )

    base = builder.transform(valid_transactions).reindex(target.index)
    streams = temporal_streams(valid_transactions)
    temporal = apply_temporal_blocks(base, streams, builder.description_lift_, ("periodicity",))
    heuristic_family = pd.Series(
        RecurrenceHeuristic(none_bias=-1.0, temperature=1.0).predict(temporal), index=temporal.index
    ).reindex(target.index)
    predictions = pd.DataFrame(
        {
            "target": target,
            "v2_temporal_heuristic": heuristic_family,
            "next_date_rule": rule_family,
            "pseudo_cutoff_ranker": learned_family,
            "ranked_stream": learned_stream.reindex(target.index),
        }
    )
    metrics = {
        "v2_temporal_heuristic": evaluate_predictions(target, heuristic_family),
        "next_date_rule": evaluate_predictions(target, rule_family),
        "pseudo_cutoff_ranker": evaluate_predictions(target, learned_family),
    }
    return metrics, predictions


def _json_default(value):
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(summary: dict, path: Path) -> None:
    oof = summary["oof_metrics"]
    official = summary["official_validation"]
    selected = summary["selected_ranker"]
    learned_gain = (
        official["pseudo_cutoff_ranker"]["macro_f1"] - official["next_date_rule"]["macro_f1"]
    )
    oof_gain = (
        oof[selected]["top1_next_stream_accuracy"]
        - oof["next_date_rule"]["top1_next_stream_accuracy"]
    )
    if learned_gain >= 0.02 and oof_gain >= 0.02:
        conclusion = "PSEUDO-CUTOFF SUPERVISION IS HIGH VALUE"
    elif learned_gain > 0 or oof_gain >= 0.01:
        conclusion = "PSEUDO-CUTOFFS HELP MODERATELY"
    else:
        conclusion = "PSEUDO-CUTOFFS DO NOT JUSTIFY COMPLEXITY"
    summary["conclusion"] = conclusion
    distribution = summary["pseudo_cutoff_distribution"]
    lines = [
        "# Jaime — supervisión temporal con pseudo-cutoffs",
        "",
        "## Resultado",
        "",
        f"**{conclusion}**",
        "",
        f"Se generaron **{summary['pseudo_cutoffs']:,} pseudo-cutoffs** y "
        f"**{summary['candidate_rows']:,} filas candidato** para "
        f"**{summary['clients_with_pseudo_cutoffs']:,} clientes**. El ranker congelado por OOF fue "
        f"`{selected}`.",
        "",
        "## Configuración y distribución temporal",
        "",
        f"Cutoffs por cliente: media {distribution['per_client_mean']:.2f}, mediana "
        f"{distribution['per_client_median']:.1f}, rango "
        f"{distribution['per_client_min']}–{distribution['per_client_max']}. Fechas: "
        f"{distribution['first_cutoff']} a {distribution['last_cutoff']}.",
        "",
        "Se exigen 120 días y 12 eventos de historia, separación de 60 días, dos "
        "apariciones previas por stream y 90 días futuros completamente observables.",
        "",
        "## Métricas OOF agrupadas por cliente",
        "",
        "| modelo | top-1 | top-2 | recall none | F1 ocurre 90d | MAE próxima ocurrencia (días) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("next_date_rule", "logistic_pointwise", "hist_gradient_boosting"):
        row = oof[name]
        lines.append(
            f"| {name} | {row['top1_next_stream_accuracy']:.4f} | "
            f"{row['top2_coverage']:.4f} | {row['none_recall']:.4f} | "
            f"{row['binary_f1_occurs_90d']:.4f} | {row['time_to_next_mae_days']:.2f} |"
        )
    lines.extend(
        [
            "",
            "### Distribución de pseudo-cutoffs por mes",
            "",
            "| mes | pseudo-cutoffs |",
            "| --- | ---: |",
        ]
    )
    for month, count in distribution["by_month"].items():
        lines.append(f"| {month} | {count:,} |")
    lines.extend(
        [
            "",
            "El modelo logístico y el boosting son formulaciones pointwise de ranking: cada "
            "conjunto contiene los streams recurrentes y una fila `none`, y se ordena por la "
            "probabilidad de ser el primer stream que reaparece.",
            "",
            "### Top-1 del modelo seleccionado por número de candidatos",
            "",
            "| candidatos | pseudo-cutoffs | top-1 |",
            "| --- | ---: | ---: |",
        ]
    )
    for bucket, row in oof[selected]["by_candidate_count"].items():
        lines.append(f"| {bucket} | {row['pseudo_cutoffs']:,} | {row['top1_accuracy']:.4f} |")
    lines.extend(
        [
            "",
            "## Aplicación al cutoff oficial y posible efecto en Macro-F1",
            "",
            "La selección del ranker usa únicamente OOF de TRAIN. Después se ajusta en todos los "
            "pseudo-ejemplos de TRAIN y se aplica a VALID en 2026-01-01. La familia se obtiene "
            "mediante el mapping fijo `ClientFeatureBuilder.description_lift_`, aprendido solo con "
            "TRAIN; no se optimiza la clasificación de familia.",
            "",
            "| estrategia | Macro-F1 VALID | accuracy VALID |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in ("v2_temporal_heuristic", "next_date_rule", "pseudo_cutoff_ranker"):
        row = official[name]
        lines.append(f"| {name} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} |")
    lines.extend(
        [
            "",
            f"Cambio del ranking aprendido frente a la regla simple: **{learned_gain:+.4f} "
            "Macro-F1**. Este diagnóstico no es un holdout independiente: VALID ya fue reutilizado "
            "en V2, aunque no interviene en la selección del ranker de este experimento.",
            "",
            "## Análisis de leakage",
            "",
            "- Features: solo transacciones con `timestamp < T`.",
            "- Labels: solo primeras ocurrencias en `[T, T+90d)` y únicamente después "
            "de construir features.",
            "- Observabilidad: todo cutoff termina como máximo en 2025-10-03, dejando "
            "el horizonte completo antes de 2026-01-01.",
            "- Split: cinco folds por `client_id`; todos los cutoffs y candidatos de un "
            "cliente permanecen juntos.",
            "- Aplicación oficial: ranker y mapping se ajustan en TRAIN; VALID solo se "
            "usa para el diagnóstico final.",
            "- Tests: invariancia de features ante cambios futuros, límites del horizonte "
            "y aislamiento de clientes.",
            "",
            "## Coste computacional",
            "",
            f"Tiempo total: **{summary['seconds']:.1f} s**; generación: "
            f"{summary['generation_seconds']:.1f} s; OOF: {oof['seconds']:.1f} s. "
            "El experimento usa modelos tabulares pequeños y no usa deep learning.",
            "",
            "## Reproducibilidad",
            "",
            f"Commit base: `{summary['base_commit']}`. Semilla: `{summary['config']['seed']}`. "
            f"Dataset TRAIN SHA-256: `{summary['train_sha256']}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/ubs_2026")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    started = time.perf_counter()
    config = Config()
    data = load_ubs_data(args.data_dir)

    generation_started = time.perf_counter()
    pseudo = generate_pseudo_examples(data.train_transactions, config)
    generation_seconds = time.perf_counter() - generation_started
    oof_metrics, oof_rows, selected = evaluate_oof(pseudo, config)
    official_metrics, official_predictions = official_validation(
        data.train_transactions,
        data.train_labels,
        data.valid_transactions,
        data.valid_labels,
        pseudo,
        selected,
        config,
    )

    cutoff_table = pseudo[["pseudo_id", "client_id", "cutoff", "candidate_count"]].drop_duplicates()
    per_client = cutoff_table.groupby("client_id").size()
    summary = {
        "experiment": "jaime_pseudocutoffs",
        "base_commit": "5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749",
        "config": asdict(config),
        "selected_ranker": selected,
        "selection_rule": "highest client-grouped OOF top-1, then top-2; logistic wins exact ties",
        "pseudo_cutoffs": len(cutoff_table),
        "candidate_rows": len(pseudo),
        "real_candidate_rows": int(pseudo.stream.ne(NONE_STREAM).sum()),
        "clients_with_pseudo_cutoffs": int(cutoff_table.client_id.nunique()),
        "pseudo_cutoff_distribution": {
            "per_client_mean": float(per_client.mean()),
            "per_client_median": float(per_client.median()),
            "per_client_min": int(per_client.min()),
            "per_client_max": int(per_client.max()),
            "first_cutoff": cutoff_table.cutoff.min().isoformat(),
            "last_cutoff": cutoff_table.cutoff.max().isoformat(),
            "by_month": cutoff_table.cutoff.dt.strftime("%Y-%m")
            .value_counts()
            .sort_index()
            .to_dict(),
        },
        "oof_metrics": oof_metrics,
        "official_validation": official_metrics,
        "generation_seconds": generation_seconds,
        "seconds": time.perf_counter() - started,
        "train_sha256": _digest(args.data_dir / "train_transactions.jsonl"),
    }
    write_report(summary, args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_rows.to_csv(args.output_dir / "oof_candidate_predictions.csv", index=False)
    official_predictions.to_csv(args.output_dir / "official_validation_predictions.csv")
    cutoff_table.to_csv(args.output_dir / "pseudo_cutoff_distribution.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=_json_default, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
