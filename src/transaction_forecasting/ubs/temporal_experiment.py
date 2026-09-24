"""Train-client temporal ablations using the existing loader, mapping and scorer."""

from __future__ import annotations

import hashlib
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN, UBSData
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.features import ClientFeatureBuilder
from transaction_forecasting.ubs.models import RecurrenceHeuristic
from transaction_forecasting.ubs.temporal_features import apply_temporal_blocks, temporal_streams


def _subset(frame: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    return frame.loc[frame.client_id.isin(clients)]


def _digest(clients: pd.Index) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, clients))).encode()).hexdigest()


def client_partitions(target: pd.Series, folds: int, seed: int, fraction: float):
    """Yield fit/calibration/outer-fit/held-out IDs, never splitting one client."""
    if not target.index.is_unique:
        raise ValueError("Client targets must be unique")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (development, heldout) in enumerate(splitter.split(target.index, target)):
        outer = target.index[development]
        fit, calibration = train_test_split(
            outer, test_size=fraction, stratify=target.loc[outer], random_state=seed + fold
        )
        yield pd.Index(fit), pd.Index(calibration), outer, target.index[heldout]


def _features(builder, transactions, clients):
    result = builder.transform(_subset(transactions, clients)).reindex(clients)
    if not result.index.equals(clients) or result.isna().any().any():
        raise ValueError("Missing or misaligned client features")
    # Existing V1 omits the family columns when no repeated streams exist at all.
    for label in LABELS:
        for suffix in ("recurrence_score", "occurrences", "description_lift", "regularity"):
            column = f"family_{label}_{suffix}"
            if column not in result:
                result[column] = 0.0
    return result


def _calibrate(transactions, labels, fit, calibration, streams, variants):
    builder = ClientFeatureBuilder().fit(_subset(transactions, fit), _subset(labels, fit))
    base = _features(builder, transactions, calibration)
    times = streams.loc[streams.index.get_level_values("client_id").isin(calibration)]
    target = labels.set_index("client_id")[TARGET_COLUMN].reindex(calibration)
    return {
        name: RecurrenceHeuristic().tune(
            apply_temporal_blocks(base, times, builder.description_lift_, tuple(blocks)), target
        )
        for name, blocks in variants.items()
    }


def historical_date_diagnostics(transactions, cutoff, horizon):
    """Future-only same-stream labels, NOT the unprovided historical official families."""
    end = cutoff + pd.Timedelta(days=horizon)
    if end > CUTOFF:
        raise ValueError("Historical horizon is not fully observed")
    history = transactions.loc[transactions.timestamp < cutoff]
    future = transactions.loc[transactions.timestamp.ge(cutoff) & transactions.timestamp.lt(end)]
    streams = temporal_streams(history, cutoff, horizon)
    if streams.empty:
        return {"cutoff": str(cutoff), "historical_streams": 0}
    next_event = future.groupby(["client_id", "description"]).timestamp.min().reindex(streams.index)
    known = streams.expected_date_known
    observed = next_event.notna()
    eligible = known & observed
    median_date = cutoff + pd.to_timedelta(
        streams.interval_median - streams.days_since_last, unit="D"
    )
    errors = {}
    for name, dates in (
        ("v1_median_date", median_date),
        ("v2_calendar_date", streams.expected_next_date),
    ):
        error = (dates[eligible] - next_event[eligible]).dt.total_seconds().abs() / 86400
        errors[name] = {
            "mae_days": float(error.mean()) if len(error) else None,
            "median_ae_days": float(error.median()) if len(error) else None,
        }
    predicted = streams.expected_next_within_horizon
    return {
        "cutoff": str(cutoff),
        "horizon_end_exclusive": str(end),
        "label_kind": "next observed same-description event; not official recurrence labels",
        "historical_streams": len(streams),
        "forecastable_streams": int(known.sum()),
        "future_observed_streams": int(observed.sum()),
        "date_error_pairs": int(eligible.sum()),
        "new_future_streams": len(future.groupby(["client_id", "description"]))
        - int(observed.sum()),
        "calendar_adjusted_streams": int(streams.calendar_month_used.sum()),
        "binary_horizon_counts": {
            "tp": int((predicted & observed).sum()),
            "fp": int((predicted & ~observed).sum()),
            "fn": int((~predicted & observed).sum()),
            "tn": int((~predicted & ~observed).sum()),
        },
        "date_errors_conditional_on_observation": errors,
    }


def describe_temporal_data(transactions, target, streams):
    """Aggregate diagnostics only; client target is not a label for its past events."""
    columns = [
        "unique_event_count",
        "number_of_intervals",
        "interval_mean",
        "interval_median",
        "interval_std",
        "interval_mad",
        "interval_cv",
        "days_since_last",
        "days_until_expected_next",
        "weekday_concentration",
        "monthday_concentration",
        "events_30d",
        "events_60d",
        "events_90d",
        "events_180d",
        "recent_to_history_rate",
    ]
    by_client = streams[columns].groupby(level="client_id").median()
    by_client["total_events"] = streams.transaction_count.groupby(level="client_id").sum()
    by_client["streams"] = streams.groupby(level="client_id").size()
    by_client["target"] = target.reindex(by_client.index)
    return {
        "clients": len(target),
        "transactions": len(transactions),
        "streams": len(streams),
        "one_event": int(streams.unique_event_count.eq(1).sum()),
        "two_events": int(streams.unique_event_count.eq(2).sum()),
        "three_plus": int(streams.unique_event_count.ge(3).sum()),
        "duplicate_timestamps": int(streams.duplicate_timestamp_count.sum()),
        "post_cutoff": int(transactions.timestamp.ge(CUTOFF).sum()),
        "quantiles": streams[columns].quantile([0.1, 0.5, 0.9]).to_dict(),
        "by_client_target_medians_not_event_families": by_client.groupby("target")
        .median()
        .to_dict("index"),
        "supported_cycle_closeness_ge_0_5": {
            str(cycle): int(streams[f"cycle_{cycle}_closeness"].ge(0.5).sum())
            for cycle in (7, 14, 28, 30, 31, 90, 180, 365)
        },
        "calendar_adjusted_streams": int(streams.calendar_month_used.sum()),
        "expected_in_horizon_streams": int(streams.expected_next_within_horizon.sum()),
    }


def compare_temporal(data: UBSData, config: dict) -> dict:
    """Evaluate frozen blocks; promote only variants beating V1 on every outer fold."""
    started = time.perf_counter()
    if pd.Timestamp(config["cutoff"]) != CUTOFF:
        raise ValueError("Official labels apply only to the official cutoff")
    if config["horizon_days"] != 90:
        raise ValueError("Official labels require a 90-day horizon")
    transactions, labels = data.train_transactions, data.train_labels
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    variants = config["variants"]
    if variants.get("v1") != []:
        raise ValueError("The unchanged V1 control is required")
    streams = temporal_streams(transactions, CUTOFF, config["horizon_days"])
    scores = {name: [] for name in variants}
    oof = {name: pd.Series(index=target.index, dtype=object) for name in variants}
    partitions = []
    for fold, (fit, calibration, outer, heldout) in enumerate(
        client_partitions(target, config["folds"], config["seed"], config["calibration_fraction"])
    ):
        fold_started = time.perf_counter()
        assert not set(fit) & set(calibration) and not set(outer) & set(heldout)
        models = _calibrate(transactions, labels, fit, calibration, streams, variants)
        builder = ClientFeatureBuilder().fit(_subset(transactions, outer), _subset(labels, outer))
        base = _features(builder, transactions, heldout)
        times = streams.loc[streams.index.get_level_values("client_id").isin(heldout)]
        for name, blocks in variants.items():
            features = apply_temporal_blocks(base, times, builder.description_lift_, tuple(blocks))
            prediction = pd.Series(models[name].predict(features), index=features.index)
            oof[name].loc[heldout] = prediction
            metrics = evaluate_predictions(target.loc[heldout], prediction)
            scores[name].append(
                {
                    "fold": fold,
                    "none_bias": models[name].none_bias,
                    "temperature": models[name].temperature,
                    **metrics,
                }
            )
        partitions.append(
            {
                "fold": fold,
                "seconds": time.perf_counter() - fold_started,
                **{
                    name: {"count": len(ids), "sha256": _digest(ids)}
                    for name, ids in (
                        ("fit", fit),
                        ("calibration", calibration),
                        ("outer_fit", outer),
                        ("heldout", heldout),
                    )
                },
            }
        )
        print(f"Completed temporal fold {fold + 1}/{config['folds']}", flush=True)
    summary = {}
    baseline = np.array([row["macro_f1"] for row in scores["v1"]])
    for name, results in scores.items():
        values = np.array([row["macro_f1"] for row in results])
        if oof[name].isna().any():
            raise ValueError("Missing out-of-fold predictions")
        summary[name] = {
            "mean_macro_f1": float(values.mean()),
            "std_macro_f1": float(values.std(ddof=1)),
            "delta_vs_v1_by_fold": (values - baseline).tolist(),
            "beats_v1_every_fold": bool(np.all(values > baseline)),
            "pooled_oof": evaluate_predictions(target, oof[name]),
            "folds": results,
        }
    eligible = [name for name in variants if summary[name]["beats_v1_every_fold"]]
    selected = max(eligible, key=lambda name: summary[name]["mean_macro_f1"]) if eligible else "v1"
    # Decision is fixed before reading official validation labels. Validation was already
    # repeatedly consulted by V1 and is therefore explicitly NOT an independent holdout.
    final_variants = {
        name: variants[name]
        for name in dict.fromkeys(("v1", selected, "combined"))
        if name in variants
    }
    fit, calibration = train_test_split(
        target.index,
        test_size=config["calibration_fraction"],
        stratify=target,
        random_state=config["seed"],
    )
    models = _calibrate(
        transactions, labels, pd.Index(fit), pd.Index(calibration), streams, final_variants
    )
    builder = ClientFeatureBuilder().fit(transactions, labels)
    valid_target = data.valid_labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    base = _features(builder, data.valid_transactions, valid_target.index)
    valid_streams = temporal_streams(data.valid_transactions, CUTOFF, config["horizon_days"])
    official = {}
    for name, blocks in final_variants.items():
        features = apply_temporal_blocks(
            base, valid_streams, builder.description_lift_, tuple(blocks)
        )
        prediction = pd.Series(models[name].predict(features), index=features.index)
        official[name] = {
            "none_bias": models[name].none_bias,
            **evaluate_predictions(valid_target, prediction),
        }
    return {
        "config": config,
        "train_temporal_analysis": describe_temporal_data(transactions, target, streams),
        "label_order": list(LABELS),
        "selected": selected,
        "selection_rule": "mean gain and positive gain in every outer fold; otherwise keep V1",
        "internal_selection": summary,
        "partitions": partitions,
        "official_validation_independent": False,
        "official_validation_diagnostic": official,
        "historical_diagnostics": [
            historical_date_diagnostics(transactions, pd.Timestamp(cut), config["horizon_days"])
            for cut in config["historical_cutoffs"]
        ],
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost")
        },
        "seconds": time.perf_counter() - started,
    }
