"""Independent, deterministic UBS stream baseline on the official validation split.

Run after restoring the six official files under data/raw/ubs_2026. V2 artifacts
are optional: paired analysis uses hard predictions; weighted ensembles require
explicit, comparable probability files for both systems.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from transaction_forecasting.evaluation.official import (
    HORIZON_DAYS,
    LABELS,
    PREDICTION,
    TARGET,
    score_predictions,
)
from transaction_forecasting.ubs.data import CUTOFF, load_ubs_data

DATA_FILES = (
    "train_transactions.jsonl",
    "train_labels.csv",
    "valid_transactions.jsonl",
    "valid_labels.csv",
    "test_transactions.jsonl",
    "sample_submission.csv",
)
V2_REFERENCE = 0.391549456
FAMILIES = tuple(label for label in LABELS if label != "none")


@dataclass(frozen=True)
class Rules:
    """Fixed, interpretable thresholds; no validation search."""

    minimum_events: int = 3
    minimum_mapping_clients: int = 3
    minimum_family_share: float = 0.5
    maximum_relative_mad: float = 0.35
    maximum_missed_intervals: float = 1.0
    high_confidence_events: int = 4
    high_confidence_family_share: float = 0.8
    high_confidence_relative_mad: float = 0.15


def check_inputs(data_dir: Path) -> None:
    """Fail before loading or writing anything, listing every missing input."""
    missing = [str(data_dir / name) for name in DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing official UBS inputs:\n  " + "\n  ".join(missing))


def validate_history(transactions: pd.DataFrame) -> None:
    required = {"client_id", "description", "timestamp"}
    absent = required.difference(transactions.columns)
    if absent:
        raise ValueError(f"Missing history columns: {sorted(absent)}")
    if transactions[list(required)].isna().any().any():
        raise ValueError("Null stream keys or timestamps")
    dates = pd.to_datetime(transactions["timestamp"], utc=True, errors="raise")
    if dates.ge(CUTOFF).any():
        raise ValueError("Transactions at or after the prediction cutoff are forbidden")


def fit_family_map(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Count TRAIN clients with each exact description by their client target.

    The target is a client-level next-family label, so this is a weak association,
    not a transaction-level merchant annotation. Each client contributes once
    per description. None contributes to the denominator, never as a stream family.
    """
    validate_history(transactions)
    target = labels.set_index("client_id")[TARGET]
    if not target.index.is_unique or set(target.index) != set(transactions["client_id"]):
        raise ValueError("Mapping fit requires exactly the labelled TRAIN clients")
    if not target.isin(LABELS).all():
        raise ValueError("Unknown TRAIN target class")
    presence = transactions[["client_id", "description"]].drop_duplicates().copy()
    presence[TARGET] = presence["client_id"].map(target)
    counts = pd.crosstab(presence["description"], presence[TARGET]).reindex(
        columns=LABELS, fill_value=0
    )
    family_counts = counts.loc[:, FAMILIES]
    best = family_counts.idxmax(axis=1)  # official order resolves ties
    best_count = family_counts.max(axis=1)
    support = counts.sum(axis=1)
    return pd.DataFrame(
        {
            "family": best,
            "mapping_clients": support.astype(int),
            "family_share": best_count.div(support).astype(float),
        },
        index=counts.index,
    )


def build_streams(transactions: pd.DataFrame, family_map: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact client-description streams using unique UTC event times."""
    validate_history(transactions)
    rows = []
    for (client_id, description), group in transactions.groupby(
        ["client_id", "description"], sort=True
    ):
        times = pd.DatetimeIndex(
            pd.to_datetime(group["timestamp"], utc=True).unique()
        ).sort_values()
        if len(times) < 2:
            continue
        gaps = np.diff(times.asi8) / 86_400_000_000_000
        interval = float(np.median(gaps))
        if interval <= 0:
            continue
        relative_mad = float(np.median(np.abs(gaps - interval)) / interval)
        initial_next = times[-1] + pd.Timedelta(days=interval)
        overdue_days = max(0.0, (CUTOFF - initial_next).total_seconds() / 86400)
        # Advance old streams to the first future occurrence, but retain how
        # many cycles were missed for the fixed stale-stream rejection rule.
        missed = max(0, int(np.ceil((CUTOFF - initial_next) / pd.Timedelta(days=interval))))
        next_date = initial_next + pd.Timedelta(days=missed * interval)
        rows.append(
            {
                "client_id": client_id,
                "description": description,
                "event_count": len(times),
                "interval_days": interval,
                "relative_mad": relative_mad,
                "overdue_days": overdue_days,
                "next_date": next_date,
                "days_until_next": (next_date - CUTOFF).total_seconds() / 86400,
            }
        )
    columns = [
        "client_id",
        "description",
        "event_count",
        "interval_days",
        "relative_mad",
        "overdue_days",
        "next_date",
        "days_until_next",
    ]
    streams = pd.DataFrame(rows, columns=columns)
    return streams.join(family_map, on="description")


def select_candidates(
    streams: pd.DataFrame, client_ids: pd.Index, variant: str, rules: Rules
) -> tuple[pd.Series, pd.DataFrame, int]:
    """Return predictions, chosen candidates, and clients without an eligible row.

    A: mapping rank; B: add chronological rank; C: add fixed recurrence/none
    gate; D: add 90-day horizon. Every variant covers the same client IDs.
    """
    if variant not in "ABCD" or len(variant) != 1:
        raise ValueError("Variant must be A, B, C, or D")
    if not client_ids.is_unique:
        raise ValueError("Client IDs must be unique")
    eligible = streams.loc[
        streams["family"].isin(FAMILIES)
        & streams["mapping_clients"].ge(rules.minimum_mapping_clients)
        & streams["family_share"].gt(0)
    ].copy()
    if variant in "CD":
        eligible = eligible.loc[
            eligible["event_count"].ge(rules.minimum_events)
            & eligible["family_share"].ge(rules.minimum_family_share)
            & eligible["relative_mad"].le(rules.maximum_relative_mad)
            & eligible["overdue_days"].le(
                eligible["interval_days"] * rules.maximum_missed_intervals
            )
        ]
    if variant == "D":
        eligible = eligible.loc[
            eligible["days_until_next"].ge(0) & eligible["days_until_next"].lt(HORIZON_DAYS)
        ]
    if variant == "A":
        order = ["client_id", "family_share", "mapping_clients", "event_count", "description"]
        ascending = [True, False, False, False, True]
    else:
        order = ["client_id", "next_date", "family_share", "description"]
        ascending = [True, True, False, True]
    chosen = eligible.sort_values(order, ascending=ascending, kind="stable").drop_duplicates(
        "client_id"
    )
    if not set(chosen["client_id"]).issubset(set(client_ids)):
        raise ValueError("Stream table contains unexpected clients")
    prediction = pd.Series("none", index=client_ids, name=PREDICTION, dtype=str)
    prediction.loc[chosen["client_id"]] = chosen["family"].to_numpy()
    return prediction, chosen, len(client_ids) - len(chosen)


def prediction_frame(prediction: pd.Series) -> pd.DataFrame:
    return prediction.rename_axis("client_id").reset_index()


def read_predictions(path: Path, client_ids: pd.Index) -> pd.Series:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != ("client_id", PREDICTION):
        raise ValueError(f"{path}: expected client_id,{PREDICTION}")
    if frame["client_id"].duplicated().any() or set(frame["client_id"]) != set(client_ids):
        raise ValueError(f"{path}: client IDs differ from official VALID")
    result = frame.set_index("client_id")[PREDICTION].reindex(client_ids)
    if not result.isin(LABELS).all():
        raise ValueError(f"{path}: invalid class predictions")
    return result


def complementarity(truth: pd.Series, v2: pd.Series, stream: pd.Series) -> dict:
    """Paired correctness counts overall and for each true class."""
    if not truth.index.equals(v2.index) or not truth.index.equals(stream.index):
        raise ValueError("Complementarity requires identical ordered VALID clients")

    def counts(mask: pd.Series) -> dict:
        a, b = v2.loc[mask].eq(truth.loc[mask]), stream.loc[mask].eq(truth.loc[mask])
        n = len(a)
        values = {
            "v2_only": int((a & ~b).sum()),
            "stream_only": int((~a & b).sum()),
            "both": int((a & b).sum()),
            "neither": int((~a & ~b).sum()),
        }
        return {
            "clients": n,
            "counts": values,
            "percentages": {k: 100 * v / n if n else 0.0 for k, v in values.items()},
        }

    return {
        "overall": counts(pd.Series(True, index=truth.index)),
        "per_true_class": {label: counts(truth.eq(label)) for label in LABELS},
    }


def high_confidence_override(
    v2: pd.Series, stream: pd.Series, chosen: pd.DataFrame, rules: Rules
) -> pd.Series:
    """One predeclared override: repeated, regular, strongly mapped stream."""
    qualified = chosen.loc[
        chosen["event_count"].ge(rules.high_confidence_events)
        & chosen["family_share"].ge(rules.high_confidence_family_share)
        & chosen["relative_mad"].le(rules.high_confidence_relative_mad)
        & chosen["overdue_days"].eq(0)
    ]
    result = v2.copy()
    result.loc[qualified["client_id"]] = stream.loc[qualified["client_id"]].to_numpy()
    return result


def read_probabilities(path: Path, client_ids: pd.Index) -> pd.DataFrame:
    """Read actual eight-class probabilities; never synthesize them from labels."""
    frame = pd.read_csv(path, dtype={"client_id": str})
    if tuple(frame.columns) != ("client_id", *LABELS):
        raise ValueError(f"{path}: expected client_id followed by official class order")
    if frame["client_id"].duplicated().any() or set(frame["client_id"]) != set(client_ids):
        raise ValueError(f"{path}: client IDs differ from official VALID")
    values = frame.set_index("client_id").reindex(client_ids).astype(float)
    array = values.to_numpy()
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError(f"{path}: invalid probabilities")
    if not np.allclose(array.sum(axis=1), 1, atol=1e-6):
        raise ValueError(f"{path}: probability rows must sum to one")
    return values


def predefined_ensembles(
    v2: pd.Series,
    stream: pd.Series,
    chosen: pd.DataFrame,
    rules: Rules,
    v2_probabilities: pd.DataFrame | None = None,
    stream_probabilities: pd.DataFrame | None = None,
) -> dict[str, pd.Series]:
    """Fixed options only; weighted blends need two genuine probability tables."""
    options = {
        "v2": v2,
        "stream": stream,
        "high_confidence_override": high_confidence_override(v2, stream, chosen, rules),
    }
    if (v2_probabilities is None) != (stream_probabilities is None):
        raise ValueError("Weighted blends require both comparable probability tables")
    if v2_probabilities is not None and stream_probabilities is not None:
        if not v2_probabilities.index.equals(v2.index) or not stream_probabilities.index.equals(
            v2.index
        ):
            raise ValueError("Probability clients must match the paired predictions")
        for v2_weight in (0.75, 0.5):
            scores = v2_weight * v2_probabilities + (1 - v2_weight) * stream_probabilities
            options[f"{int(v2_weight * 100)}v2_{int((1 - v2_weight) * 100)}stream"] = scores.idxmax(
                axis=1
            ).rename(PREDICTION)
    return options


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v3_discovery/laura_stream_system")
    )
    parser.add_argument("--v2-predictions", type=Path)
    parser.add_argument("--v2-probabilities", type=Path)
    parser.add_argument("--stream-probabilities", type=Path)
    args = parser.parse_args(argv)
    check_inputs(args.data_dir)
    for optional in (args.v2_predictions, args.v2_probabilities, args.stream_probabilities):
        if optional is not None and not optional.is_file():
            raise FileNotFoundError(f"Missing optional comparison input: {optional}")
    if (args.v2_probabilities is None) != (args.stream_probabilities is None):
        raise ValueError("Both probability files are required for weighted ensembles")
    if args.v2_probabilities is not None and args.v2_predictions is None:
        raise ValueError("V2 hard predictions are required for paired comparison")

    data = load_ubs_data(args.data_dir)
    rules = Rules()
    family_map = fit_family_map(data.train_transactions, data.train_labels)
    valid_ids = pd.Index(data.valid_labels["client_id"], name="client_id")
    streams = build_streams(data.valid_transactions, family_map)
    results = {}
    ablation_predictions = pd.DataFrame({"client_id": valid_ids})
    final_prediction = None
    final_chosen = None
    for variant in "ABCD":
        prediction, chosen, no_candidate = select_candidates(streams, valid_ids, variant, rules)
        metrics = score_predictions(data.valid_labels, prediction_frame(prediction))
        results[variant] = {"metrics": metrics, "clients_without_candidate": no_candidate}
        ablation_predictions[variant] = prediction.to_numpy()
        if variant == "D":
            final_prediction, final_chosen = prediction, chosen
    assert final_prediction is not None and final_chosen is not None

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    family_map.rename_axis("description").reset_index().to_csv(
        output / "train_family_mapping.csv", index=False
    )
    ablation_predictions.to_csv(output / "ablation_predictions.csv", index=False)
    prediction_frame(final_prediction).to_csv(output / "validation_predictions.csv", index=False)
    final_chosen.to_csv(output / "chosen_candidates.csv", index=False)
    write_json(output / "ablations.json", results)
    summary = {
        "rules": asdict(rules),
        "official_cutoff": str(CUTOFF),
        "horizon_days": HORIZON_DAYS,
        "v2_supplied_reference_macro_f1": V2_REFERENCE,
        "v2_reference_verified": False,
        "ablation_definitions": {
            "A": "mapped repeated stream ranked by family share; no temporal rank",
            "B": "A with earliest predicted date rank",
            "C": "B with fixed recurrence, regularity, freshness and family gates",
            "D": "C with strict 90-day horizon filter",
        },
    }
    if args.v2_predictions is not None:
        truth = data.valid_labels.set_index("client_id")[TARGET].reindex(valid_ids)
        v2 = read_predictions(args.v2_predictions, valid_ids)
        v2_metrics = score_predictions(data.valid_labels, prediction_frame(v2))
        paired = complementarity(truth, v2, final_prediction)
        v2_probs = (
            read_probabilities(args.v2_probabilities, valid_ids)
            if args.v2_probabilities is not None
            else None
        )
        stream_probs = (
            read_probabilities(args.stream_probabilities, valid_ids)
            if args.stream_probabilities is not None
            else None
        )
        if v2_probs is not None and not v2_probs.idxmax(axis=1).eq(v2).all():
            raise ValueError("V2 probabilities disagree with V2 hard predictions")
        if stream_probs is not None and not stream_probs.idxmax(axis=1).eq(final_prediction).all():
            raise ValueError("Stream probabilities disagree with D hard predictions")
        ensembles = predefined_ensembles(
            v2, final_prediction, final_chosen, rules, v2_probs, stream_probs
        )
        ensemble_metrics = {
            name: score_predictions(data.valid_labels, prediction_frame(prediction))
            for name, prediction in ensembles.items()
        }
        summary["v2_reference_verified"] = abs(v2_metrics["macro_f1"] - V2_REFERENCE) < 1e-9
        summary["v2_metrics"] = v2_metrics
        summary["weighted_ensembles_available"] = v2_probs is not None
        summary["validation_selection_bias"] = True
        write_json(output / "complementarity.json", paired)
        write_json(output / "ensembles.json", ensemble_metrics)
        errors = pd.DataFrame(
            {
                "client_id": valid_ids,
                "actual": truth.to_numpy(),
                "v2": v2.to_numpy(),
                "stream": final_prediction.to_numpy(),
            }
        )
        errors.loc[errors["v2"].ne(errors["actual"]) | errors["stream"].ne(errors["actual"])].head(
            40
        ).to_csv(output / "representative_errors.csv", index=False)
    write_json(output / "summary.json", summary)
    print(json.dumps({"D_macro_f1": results["D"]["metrics"]["macro_f1"], **summary}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from None
