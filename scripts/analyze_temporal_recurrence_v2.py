"""Reproduce temporal handoff evidence without writing a submission or changing V1.

Run from the repository root with PYTHONPATH=src. --render-only redraws figures
from the lightweight aggregate evidence; it does not train or consult validation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
import tomllib
import warnings
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import run_ubs_baseline as baseline_runner

from transaction_forecasting.ubs import temporal_experiment as experiment
from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.features import ClientFeatureBuilder

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

WORK = Path("outputs/metrics/carles_temporal_handoff")
EVIDENCE = Path("reports/handoff/carles_temporal_evidence.json")
FIGURES = Path("reports/figures/temporal_recurrence")


def with_support(metrics):
    """Add row-sum support to existing metrics, without recomputing a score."""
    matrix = np.asarray(metrics["confusion_matrix"])
    for number, label in enumerate(LABELS):
        metrics["per_class"][label]["support"] = int(matrix[number].sum())
    return metrics


def reproduce_baseline():
    """Execute the original runner; serialize/validate its submission only in memory."""
    config = baseline_runner.load_settings("configs/ubs_v1.toml")
    config["outputs"] = {
        "metrics_directory": str(WORK / "v1"),
        "submission": str(WORK / "IN_MEMORY_ONLY.csv"),
    }
    virtual = Path(config["outputs"]["submission"]).resolve()
    if virtual.exists():
        raise ValueError("The in-memory sentinel path must not exist")
    original_write, original_read = pd.DataFrame.to_csv, pd.read_csv
    buffer = io.StringIO()

    def write(frame, destination=None, *args, **kwargs):
        if isinstance(destination, str | Path) and Path(destination).resolve() == virtual:
            return original_write(frame, buffer, *args, **kwargs)
        return original_write(frame, destination, *args, **kwargs)

    def read(source, *args, **kwargs):
        if isinstance(source, str | Path) and Path(source).resolve() == virtual:
            return original_read(io.StringIO(buffer.getvalue()), *args, **kwargs)
        return original_read(source, *args, **kwargs)

    WORK.mkdir(parents=True, exist_ok=True)
    with (
        patch.object(baseline_runner, "load_settings", return_value=config),
        patch.object(sys, "argv", ["run_ubs_baseline.py", "--config", "configs/ubs_v1.toml"]),
        patch.object(pd.DataFrame, "to_csv", write),
        patch.object(pd, "read_csv", read),
        (WORK / "v1_stdout.txt").open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
    ):
        baseline_runner.main()
    assert buffer.getvalue() and not virtual.exists()
    summary = json.loads((WORK / "v1/summary.json").read_text())
    metrics = json.loads((WORK / "v1/validation_metrics.json").read_text())[summary["best_model"]]
    return {
        "metrics": with_support(metrics),
        "config": config,
        "seconds": summary["pipeline_seconds"],
        "model": summary["best_model"],
        "validated_in_memory_submission_rows": summary["submission_rows"],
        "submission_written": False,
    }


def paired_errors(frame):
    correct_v1, correct_v2 = frame.v1.eq(frame.actual), frame.combined.eq(frame.actual)
    return {
        "clients": len(frame),
        "corrected": int((~correct_v1 & correct_v2).sum()),
        "spoiled": int((correct_v1 & ~correct_v2).sum()),
        "both_wrong": int((~correct_v1 & ~correct_v2).sum()),
        "both_correct": int((correct_v1 & correct_v2).sum()),
        "prediction_changed": int(frame.v1.ne(frame.combined).sum()),
        "none_to_family": int((frame.v1.eq("none") & frame.combined.ne("none")).sum()),
        "family_to_none": int((frame.v1.ne("none") & frame.combined.eq("none")).sum()),
        "family_to_family": int(
            (frame.v1.ne("none") & frame.combined.ne("none") & frame.v1.ne(frame.combined)).sum()
        ),
    }


def dataset_details(transactions, target, streams):
    numeric = streams.select_dtypes(include="number")
    client = numeric.groupby(level="client_id").median()
    client["singleton_share"] = streams.unique_event_count.eq(1).groupby(level="client_id").mean()
    client["events"] = streams.transaction_count.groupby(level="client_id").sum()
    labels = pd.DataFrame({"client_id": target.index, TARGET_COLUMN: target.to_numpy()})
    mapping = ClientFeatureBuilder().fit(transactions, labels).description_lift_
    inferred = mapping.idxmax(axis=1).where(mapping.max(axis=1).gt(0), "unmapped")
    family = inferred.reindex(streams.index.get_level_values("description")).fillna("unmapped")
    family.index = streams.index
    dates = streams.expected_next_date.dropna().sort_values()
    details = {
        "client_counts_one_two_three_plus_events": {
            "one": int(client.events.eq(1).sum()),
            "two": int(client.events.eq(2).sum()),
            "three_plus": int(client.events.ge(3).sum()),
        },
        "stream_numeric_quantiles": numeric.quantile([0.1, 0.5, 0.9]).to_dict(),
        "by_client_target_median": client.groupby(target.reindex(client.index))
        .median()
        .to_dict("index"),
        "by_inferred_family_stream_median": numeric.groupby(family).median().to_dict("index"),
        "inferred_family_stream_counts": family.value_counts().to_dict(),
        "family_warning": (
            "Mapping fitted on all train for descriptive analysis only; "
            "not event truth or independent evidence"
        ),
        "streams_with_gap_below_1d": int(streams.interval_min.lt(1).sum()),
        "streams_with_gap_above_180d": int(streams.interval_max.gt(180).sum()),
        "streams_with_cv_ge_1": int(streams.interval_cv.ge(1).sum()),
        "overdue_streams": int(streams.overdue_days.gt(0).sum()),
        "expected_date_quantiles": {
            str(q): str(dates.iloc[int(q * (len(dates) - 1))]) for q in (0.1, 0.5, 0.9)
        },
    }
    return details, client


def measure():
    started = time.perf_counter()
    baseline = reproduce_baseline()
    print(f"V1 reproduced: {baseline['metrics']['macro_f1']:.9f}", flush=True)
    if not np.isclose(baseline["metrics"]["macro_f1"], 0.2710243, atol=5e-8, rtol=0):
        raise RuntimeError(
            "Unexpected V1 score: inspect outputs/metrics/carles_temporal_handoff/v1"
        )
    data = load_ubs_data("data/raw/ubs_2026")
    config = tomllib.loads(Path("configs/ubs_v2_temporal.toml").read_text())
    captured, cached_streams = [], {}
    original_streams = experiment.temporal_streams

    def observe(target, prediction):
        captured.append((target.copy(), prediction.reindex(target.index).copy()))
        return evaluate_predictions(target, prediction)

    def streams(frame, cutoff=CUTOFF, horizon=90):
        result = original_streams(frame, cutoff, horizon)
        if cutoff == CUTOFF and frame.client_id.nunique() == len(data.train_labels):
            cached_streams["train"] = result
        return result

    with (
        patch.object(experiment, "evaluate_predictions", observe),
        patch.object(experiment, "temporal_streams", streams),
    ):
        comparison = experiment.compare_temporal(data, config)
    variants = list(config["variants"])
    fold_calls = [(target, pred) for target, pred in captured if len(target) == 400]
    assert len(fold_calls) == config["folds"] * len(variants)
    pairs = []
    for fold in range(config["folds"]):
        group = fold_calls[fold * len(variants) : (fold + 1) * len(variants)]
        target = group[0][0]
        assert all(t.index.equals(target.index) and t.equals(target) for t, _ in group)
        pairs.append(
            pd.DataFrame(
                {
                    "actual": target,
                    "v1": group[variants.index("v1")][1],
                    "combined": group[variants.index("combined")][1],
                    "fold": fold,
                }
            )
        )
    paired = pd.concat(pairs)
    assert paired.index.is_unique and len(paired) == len(data.train_labels)
    target = data.train_labels.set_index("client_id")[TARGET_COLUMN]
    details, client = dataset_details(data.train_transactions, target, cached_streams["train"])
    paired = paired.join(client[["singleton_share", "interval_cv", "days_since_last"]])
    # Diagnostic strata fixed before inspecting outcomes; they overlap and are not causal.
    strata = {
        "singleton_share_ge_half": paired.singleton_share.ge(0.5),
        "median_cv_ge_0_75": paired.interval_cv.ge(0.75),
        "median_recency_gt_90d": paired.days_since_last.gt(90),
    }
    paired.to_csv(WORK / "private_oof_error_pairs.csv", index_label="client_id")
    results = comparison["internal_selection"]
    for result in results.values():
        for metric in [result["pooled_oof"], *result["folds"]]:
            with_support(metric)
        values = [fold["macro_f1"] for fold in result["folds"]]
        result["min_macro_f1"], result["max_macro_f1"] = min(values), max(values)
        result["mean_accuracy"] = float(np.mean([fold["accuracy"] for fold in result["folds"]]))
        result["delta_macro_f1_mean"] = result["mean_macro_f1"] - results["v1"]["mean_macro_f1"]
        result["delta_accuracy_mean"] = result["mean_accuracy"] - results["v1"]["mean_accuracy"]
    for metrics in comparison["official_validation_diagnostic"].values():
        with_support(metrics)
    previous = json.loads(Path("docs/results/ubs_v2_temporal_results.json").read_text())
    reproducible = all(
        results[name]["folds"]
        == [with_support(fold) for fold in previous["internal_selection"][name]["folds"]]
        for name in variants
    )
    return {
        "baseline": baseline,
        "comparison": comparison,
        "identical_to_previous_fold_metrics": reproducible,
        "dataset_extended": details,
        "paired_oof": {
            "all": paired_errors(paired),
            "by_class": {
                label: paired_errors(paired.loc[paired.actual.eq(label)]) for label in LABELS
            },
            "strata": {name: paired_errors(paired.loc[mask]) for name, mask in strata.items()},
        },
        "seconds_total": time.perf_counter() - started,
        "commit_analyzed": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path("scripts/run_ubs_baseline.py"),
                *Path("src/transaction_forecasting/ubs").glob("*.py"),
            ]
        },
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path("data/raw/ubs_2026").glob("*")
            if path.is_file()
        },
    }


def figures(evidence):
    """Export small standalone plots of aggregates, never client data."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = evidence["comparison"]["internal_selection"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name in ("v1", "combined"):
        axes[0].plot(
            range(1, 6), [f["macro_f1"] for f in results[name]["folds"]], marker="o", label=name
        )
        axes[1].plot(
            LABELS,
            [results[name]["pooled_oof"]["per_class"][label]["f1-score"] for label in LABELS],
            marker="o",
            label=name,
        )
    axes[0].set(
        xlabel="Client fold", ylabel="Macro-F1 (8 classes)", title="Internal selection, seed 42"
    )
    axes[1].set(ylabel="Pooled OOF F1", title="Class impact")
    axes[1].tick_params(axis="x", rotation=45)
    for axis in axes:
        axis.legend()
    fig.savefig(FIGURES / "folds_and_class_f1.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    names = list(results)
    axes[0].bar(
        names,
        [results[n]["mean_macro_f1"] for n in names],
        yerr=[results[n]["std_macro_f1"] for n in names],
        capsize=3,
    )
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set(
        ylabel="Mean Macro-F1 ± sample SD", title="Frozen ablations (not significance intervals)"
    )
    oof = results["v1"]["pooled_oof"]
    supports = [oof["per_class"][label]["support"] for label in LABELS]
    axes[1].bar(
        ["Observed", "V1", "Combined"],
        [
            supports[-1],
            oof["prediction_distribution"]["none"],
            results["combined"]["pooled_oof"]["prediction_distribution"]["none"],
        ],
    )
    axes[1].set(ylabel="Clients (out of 2,000)", title="None frequency, internal OOF")
    fig.savefig(FIGURES / "ablations_and_none.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    vmax = max(
        np.max(results[name]["pooled_oof"]["confusion_matrix"]) for name in ("v1", "combined")
    )
    for axis, name in zip(axes, ("v1", "combined"), strict=True):
        matrix = np.asarray(results[name]["pooled_oof"]["confusion_matrix"])
        axis.imshow(matrix, vmin=0, vmax=vmax, cmap="Blues")
        axis.set(
            xticks=range(8),
            yticks=range(8),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Predicted",
            ylabel="Actual",
            title=f"{name}: internal OOF",
        )
        axis.tick_params(axis="x", rotation=60)
        for i in range(8):
            for j in range(8):
                axis.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[i, j] > vmax / 2 else "black",
                )
    fig.savefig(FIGURES / "confusion_oof.png", dpi=120)
    plt.close(fig)
    medians = evidence["dataset_extended"]["by_client_target_median"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for axis, name in zip(
        axes.flat, ("interval_median", "days_since_last", "interval_cv"), strict=False
    ):
        axis.bar(LABELS, [medians[label][name] for label in LABELS])
        axis.set_title(f"Median client median: {name}")
        axis.tick_params(axis="x", rotation=45)
    cycles = evidence["comparison"]["train_temporal_analysis"]["supported_cycle_closeness_ge_0_5"]
    axes[1, 1].bar(list(cycles), list(cycles.values()))
    axes[1, 1].set(
        title="Cycle evidence ≥0.5, ≥2 intervals (overlapping)",
        xlabel="Cycle days",
        ylabel="Streams",
    )
    fig.savefig(FIGURES / "dataset_patterns.png", dpi=120)
    plt.close(fig)


def json_ready(value):
    """Represent missing descriptive statistics as null without rounding metrics."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        evidence = json.loads(EVIDENCE.read_text())
    else:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", RuntimeWarning)
            evidence = measure()
        evidence["runtime_warnings"] = sorted(
            {str(w.message) for w in recorded if issubclass(w.category, RuntimeWarning)}
        )
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        evidence = json_ready(evidence)
        EVIDENCE.write_text(
            json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    figures(evidence)
    print(f"Evidence: {EVIDENCE}; figures: {FIGURES}")


if __name__ == "__main__":
    main()
