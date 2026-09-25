"""Computable historical context, never a causal attribution or a future promise."""

from __future__ import annotations

import json

import pandas as pd

from transaction_forecasting.ubs.v3.features import payment_streams

GENERIC_DESCRIPTIONS = frozenset(
    {
        "merchant charge",
        "digital order",
        "card purchase",
        "service payment",
        "monthly plan",
        "subscription charge",
        "digital service",
        "member plan",
        "service fee",
    }
)


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def candidate_streams(history: pd.DataFrame, mapper) -> list[dict]:
    mapped = mapper.transform(history)
    streams = payment_streams(mapped, ["client_id", "description", "currency", "family"])
    streams = streams.loc[streams["count"].ge(2)].copy()
    streams["strong_recurrence"] = (
        streams["count"].ge(3) & streams.regularity.ge(0.75) & streams.amount_cv.le(0.15)
    )
    streams = streams.rename(
        columns={"family": "associated_family", "period": "median_gap_days", "last": "last_seen"}
    )
    # 'due' is an internal heuristic, not a guaranteed payment date; never expose it.
    columns = [
        "description",
        "currency",
        "associated_family",
        "count",
        "last_seen",
        "median_gap_days",
        "gap_std",
        "amount_mean",
        "amount_cv",
        "regularity",
        "score",
        "strong_recurrence",
    ]
    return records(streams.sort_values(["score", "description"], ascending=[False, True])[columns])


def data_quality(history: pd.DataFrame, mapper) -> dict:
    payments = history.loc[history.direction.eq("out") & history.type.eq("card_payment")]
    total = len(payments)
    if not total:
        return {
            "payment_count": 0,
            "generic_share": None,
            "unknown_identity_share": None,
            "degraded": True,
            "definition": "No outbound card payments available",
        }
    generic = payments.description.isin(GENERIC_DESCRIPTIONS)
    unknown = payments.description.map(mapper.mapping_).fillna("unknown").eq("unknown")
    return {
        "payment_count": total,
        "generic_count": int(generic.sum()),
        "generic_share": float(generic.mean()),
        "unknown_identity_count": int(unknown.sum()),
        "unknown_identity_share": float(unknown.mean()),
        "degraded": bool(generic.mean() >= 0.3 or unknown.mean() >= 0.5),
        "generic_descriptions": sorted(GENERIC_DESCRIPTIONS),
        "definition": "Shares of outbound card payments; exact generic-text list and frozen "
        "family-map unknowns. Associations are not verified merchant identities.",
    }


def evidence_summary(history: pd.DataFrame, mapper, family: str) -> dict:
    streams = candidate_streams(history, mapper)
    supporting = [s for s in streams if s["associated_family"] == family and s["strong_recurrence"]]
    credible = [s for s in streams if s["strong_recurrence"]]
    families = sorted({s["associated_family"] for s in credible} - {"unknown"})
    return {
        "transaction_count": len(history),
        "first_seen": history.timestamp.min().isoformat(),
        "last_seen": history.timestamp.max().isoformat(),
        "candidate_count": len(streams),
        "supporting_stream_count": len(supporting),
        "candidate_ambiguity": len(families) > 1,
        "candidate_families": families,
        "data_quality": data_quality(history, mapper),
        "strong_recurrence_rule": "at least 3 observations, regularity >= 0.75, amount CV <= 0.15",
        "scope": "Historical context only; no per-client feature contributions computed",
    }


def select_cases(scores: pd.DataFrame, history: pd.DataFrame, mapper) -> list[dict]:
    """Choose by model margin/history only; accepts no labels of any split."""
    ordered = scores.apply(lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1], axis=1)
    groups = dict(tuple(history.groupby("client_id")))
    cases = []
    for kind, ascending in (("clear", False), ("ambiguous", True)):
        for client in ordered.sort_values(ascending=ascending, kind="stable").index:
            family = scores.loc[client].idxmax()
            evidence = evidence_summary(groups[client], mapper, family)
            if kind == "clear" and (evidence["supporting_stream_count"] == 0 or family == "none"):
                continue
            cases.append(
                {
                    "kind": kind,
                    "client_id": client,
                    "margin": float(ordered[client]),
                    "selection": "Model margin plus pre-cutoff history; no TEST labels",
                }
            )
            break
    if len(cases) != 2:
        raise ValueError("Could not find both real demonstration cases")
    return cases
