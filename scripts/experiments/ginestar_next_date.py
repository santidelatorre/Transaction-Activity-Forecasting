"""Reproduce the frozen, train-only next-date discovery experiment.

Run from repository root with PYTHONPATH=src. Outputs contain local stream data
and must remain ignored. No test partition is read, including in the V2 diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import CUTOFF, read_labels, read_transactions
from transaction_forecasting.ubs.next_date import (
    DAY,
    METHODS,
    eligible,
    forecast,
    periodicity,
    ranking_metrics,
    ranking_rows,
    temporal_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/metrics/v3_discovery/ginestar_next_date"
CUTOFFS = ("2025-04-01", "2025-07-01", "2025-10-01")
BASE = "5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749"


def build_cases(transactions, cutoffs, *, evaluate=True):
    """Future dates are accessed only after past-only eligibility and forecasting."""
    streams = [
        (client, description, np.unique(group.timestamp.astype("int64") // DAY))
        for (client, description), group in transactions.groupby(
            ["client_id", "description"], sort=True
        )
    ]
    rows = []
    counts = []
    for cutoff in cutoffs:
        day = pd.Timestamp(cutoff, tz="UTC").value // DAY
        if evaluate and day + 90 > CUTOFF.value // DAY:
            raise ValueError("Evaluation requires a complete pre-official 90-day horizon")
        count = 0
        for client, description, full in streams:
            past = full[full < day]
            if not eligible(past):
                continue
            predictions, occurs = forecast(past, day)
            gaps = np.diff(past)
            median = float(np.median(gaps))
            mad = float(np.median(abs(gaps - median)))
            base = {
                "cutoff": cutoff,
                "client_id": client,
                "description": description,
                "n_history": len(past),
                "last": int(past[-1]),
                "median_gap": median,
                "gap_mad": mad,
                "relative_mad": mad / median,
                "periodicity": periodicity(past),
                "jitter": "low" if mad <= 2 else "medium" if mad <= 7 else "high",
            }
            # Only targets below this line inspect dates after the pseudo-cutoff.
            future = full[(full >= day) & (full < day + 90)] if evaluate else []
            base["actual"] = float(future[0]) if len(future) else np.nan
            for method in METHODS:
                rows.append(
                    {
                        **base,
                        "method": method,
                        "predicted": predictions[method],
                        "occurs": bool(occurs[method]),
                    }
                )
            count += 1
        counts.append({"cutoff": cutoff, "streams": count})
        print(f"{cutoff}: {count} candidate streams", flush=True)
    return pd.DataFrame(rows), counts


def summarize(cases):
    records, ranks = [], []
    for (split, method), group in cases.groupby(["split", "method"]):
        rank = ranking_rows(group)
        rank["method"], rank["split"] = method, split
        ranks.append(rank)
        records.append(
            {"split": split, "method": method, **temporal_metrics(group), **ranking_metrics(rank)}
        )
    return pd.DataFrame(records), pd.concat(ranks, ignore_index=True)


def diagnostics(cases, ranks, selected):
    """Post-freeze diagnostics only; never feed results back into selection."""
    none = []
    for (split, method), group in cases.groupby(["split", "method"]):
        none.append(
            {
                "split": split,
                "method": method,
                **ranking_metrics(ranking_rows(group, min_candidates=1)),
            }
        )
    pd.DataFrame(none).to_csv(OUT / "none_all_candidates.csv", index=False)
    holdout = ranks.loc[ranks.split.eq("holdout") & ~ranks.actual_none]
    pivot = holdout.pivot(index="client_id", columns="method", values="top1")
    differences = (pivot[selected] - pivot["median"]).to_numpy()
    rng = np.random.default_rng(42)
    boots = [rng.choice(differences, len(differences), replace=True).mean() for _ in range(2000)]
    info = {
        "selected_minus_median_top1": float(differences.mean()),
        "paired_client_bootstrap_95ci": np.quantile(boots, [0.025, 0.975]).tolist(),
    }
    (OUT / "uncertainty.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    # Local examples retain stream identifiers for audit; report uses anonymous cases.
    examples = cases.loc[
        cases.split.eq("holdout") & cases.method.eq(selected) & cases.actual.notna()
    ].copy()
    examples["absolute_error"] = abs(examples.predicted - examples.actual)
    examples = examples.sort_values(["absolute_error", "client_id", "description"])
    pd.concat([examples.head(5), examples.tail(5)]).to_csv(OUT / "examples.csv", index=False)


def v2_diagnostic(data_dir, method, official):
    """One frozen temporal-factor replacement; all mapping/model/weights stay fixed."""
    from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
    from transaction_forecasting.ubs.evaluation import evaluate_predictions
    from transaction_forecasting.ubs.models import RecurrenceHeuristic
    from transaction_forecasting.ubs.temporal_features import (
        CYCLES,
        apply_temporal_blocks,
        temporal_streams,
    )
    from transaction_forecasting.ubs.v2 import IntegratedV2Model

    train = read_transactions(data_dir / "train_transactions.jsonl")
    valid = read_transactions(data_dir / "valid_transactions.jsonl")
    labels = read_labels(data_dir / "train_labels.csv")
    model = IntegratedV2Model().fit(train, labels)
    components = model.predict_components(valid)
    base = model.mapping_.transform(valid)
    streams = temporal_streams(valid)
    projected = official.loc[official.method.eq(method)].set_index(["client_id", "description"])
    common = streams.index.intersection(projected.index)
    selected = projected.loc[common]
    wait = selected.predicted - CUTOFF.value // DAY
    factor = np.exp(-wait.clip(lower=0) / 90) * selected.occurs
    # Replace ONLY the temporal closeness of eligible streams. Support, family
    # mapping, baseline scores, none bias, CatBoost, and 75/25 weights are identical.
    for cycle in CYCLES:
        streams.loc[common, f"cycle_{cycle}_closeness"] = factor
    changed = apply_temporal_blocks(
        base, streams, model.mapping_.description_lift_, ("periodicity",)
    )
    heuristic = RecurrenceHeuristic(none_bias=-1.0, temperature=1.0).predict_proba(changed)
    blended = 0.75 * components["history"].to_numpy() + 0.25 * heuristic
    prediction = pd.Series(np.asarray(LABELS)[blended.argmax(axis=1)], index=base.index)
    # First and only access to validation labels, after all predictions are fixed.
    truth = read_labels(data_dir / "valid_labels.csv").set_index("client_id")[TARGET_COLUMN]
    result = {
        "method": method,
        "eligible_streams_changed": len(common),
        "before": evaluate_predictions(truth, components["blend"].idxmax(axis=1)),
        "after": evaluate_predictions(truth, prediction),
    }
    pd.DataFrame({"before": components["blend"].idxmax(axis=1), "after": prediction}).to_csv(
        OUT / "v2_valid_predictions.csv"
    )
    (OUT / "v2_diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "V2 Macro-F1:", result["before"]["macro_f1"], "->", result["after"]["macro_f1"], flush=True
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/ubs_2026")
    parser.add_argument("--v2-diagnostic", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train_transactions.jsonl"
    train = read_transactions(train_path)
    cases, counts = build_cases(train, CUTOFFS)
    cases["split"] = np.where(cases.cutoff.eq(CUTOFFS[-1]), "holdout", "development")
    summary, ranks = summarize(cases)
    # Ranking is the primary research question. Freeze before held-out assessment
    # and before official validation. Tie-break by MAE then fixed method name.
    selected = (
        summary.loc[summary.split.eq("development")]
        .sort_values(["top1", "mae", "method"], ascending=[False, True, True])
        .iloc[0]
        .method
    )
    manifest = {
        "base": BASE,
        "cutoffs": counts,
        "selection": "development top1, then MAE",
        "selected": selected,
        "development": list(CUTOFFS[:2]),
        "holdout": CUTOFFS[-1],
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "methods": list(METHODS),
        "min_dates": 3,
        "min_span_days": 14,
        "min_median_gap": 3,
        "ewma_alpha": 0.4,
        "trim_fraction": 0.2,
        "inactive_periods": 2.5,
        "v2_rule": "eligible closeness := occurs * exp(-wait_days / 90); rest frozen",
    }
    (OUT / "frozen_method.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary.to_csv(OUT / "metrics.csv", index=False)
    cases.to_csv(OUT / "pseudo_cutoff_forecasts.csv.gz", index=False)
    ranks.to_csv(OUT / "ranking_cases.csv", index=False)
    segments = []
    for (split, method, kind), group in cases.groupby(["split", "method", "periodicity"]):
        segments.append(
            {"split": split, "method": method, "periodicity": kind, **temporal_metrics(group)}
        )
    pd.DataFrame(segments).to_csv(OUT / "periodicity_metrics.csv", index=False)
    segments = []
    for (split, method, kind), group in ranks.groupby(["split", "method", "periodicity"]):
        segments.append(
            {"split": split, "method": method, "periodicity": kind, **ranking_metrics(group)}
        )
    pd.DataFrame(segments).to_csv(OUT / "ranking_periodicity.csv", index=False)
    jitter = []
    for (split, method, band), group in cases.groupby(["split", "method", "jitter"]):
        jitter.append({"split": split, "method": method, "jitter": band, **temporal_metrics(group)})
    pd.DataFrame(jitter).to_csv(OUT / "jitter_metrics.csv", index=False)
    diagnostics(cases, ranks, selected)
    print(summary.to_string(index=False), flush=True)
    print("FROZEN METHOD:", selected, flush=True)
    # All official projections are produced after freeze; validation histories only.
    valid = read_transactions(args.data_dir / "valid_transactions.jsonl")
    official, _ = build_cases(valid, ("2026-01-01",), evaluate=False)
    official = official.loc[official.method.eq(selected)].copy()
    official["predicted_date"] = pd.to_datetime(official.predicted, unit="D", utc=True)
    official.to_csv(OUT / "official_valid_forecasts.csv", index=False)
    if args.v2_diagnostic:
        v2_diagnostic(args.data_dir, selected, official)


if __name__ == "__main__":
    main()
