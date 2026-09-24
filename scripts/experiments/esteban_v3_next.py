"""Next-merchant temporal model: keyword/high-precision streams → earliest in horizon.

Discovery: with lift-based mapping, earliest projected stream matches true family
only ~33% (VALID). With keyword lexicon mapping, it jumps to ~80% among covered.
Coverage is thinner, so defer to StreamV3Push when no keyword stream is in horizon.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_oracle import POSITIVE_LABELS, build_streams
from transaction_forecasting.ubs.v3_push import StreamV3PushModel

OUT = Path("outputs/metrics/v3_next")

# High-precision description → family (exact match). Built from train lifts + grammar.
EXACT: dict[str, str] = {
    "audio streaming": "music",
    "member pass": "music",
    "media streaming": "streaming",
    "video access": "streaming",
    "urban gym": "gym",
    "fit club": "gym",
    "gym membership": "gym",
    "fitness monthly": "gym",
    "cover plan": "insurance",
    "safe cover": "insurance",
    "policy premium": "insurance",
    "insurance monthly": "insurance",
    "phone contract": "mobile",
    "service bill": "mobile",
    "saas billing": "software",
    "software access": "software",
    "productivity suite": "software",
    "cloud backup": "cloud",
    "cloud access": "cloud",
    "storage plan": "cloud",
    "service plan": "cloud",
}

# Substring tokens — only fire if exactly one family matches
TOKENS: dict[str, tuple[str, ...]] = {
    "cloud": ("cloud backup", "cloud access", "storage plan", "saas "),
    "gym": ("urban gym", "fit club", "gym membership", "fitness monthly", " gym"),
    "insurance": ("cover plan", "safe cover", "policy premium", "insurance monthly"),
    "mobile": ("phone contract", "service bill"),
    "music": ("audio streaming", "member pass"),
    "software": ("saas billing", "software access", "productivity suite"),
    "streaming": ("media streaming", "video access"),
}


def score(yv, pred, results, key):
    m = evaluate_predictions(yv.reindex(pred.index), pred)
    results[key] = {
        "macro_f1": float(m["macro_f1"]),
        "accuracy": float(m["accuracy"]),
        "per_class": {k: float(v["f1-score"]) for k, v in m["per_class"].items()},
        "pred_dist": pred.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
    }
    print(f"{key}: {results[key]['macro_f1']:.4f} acc={results[key]['accuracy']:.3f}", flush=True)
    return results[key]


def map_description(desc: str) -> str | None:
    d = str(desc).lower().strip()
    if d in EXACT:
        return EXACT[d]
    hits = []
    for fam, toks in TOKENS.items():
        if any(t in d for t in toks):
            hits.append(fam)
    if len(hits) == 1:
        return hits[0]
    return None


def precise_streams(streams: pd.DataFrame, *, horizon_days: int = 90) -> pd.DataFrame:
    frame = streams[streams["is_recurrent"]].copy()
    frame["mapped_family"] = frame["description"].map(map_description)
    frame = frame[frame["mapped_family"].notna()].copy()
    if frame.empty:
        return frame
    frame["proj"] = pd.to_datetime(frame["projected_next_date"], utc=True)
    cutoff = pd.Timestamp(CUTOFF)
    horizon = cutoff + pd.Timedelta(days=horizon_days)
    frame = frame[frame["proj"].ge(cutoff) & frame["proj"].le(horizon)]
    frame["days"] = (frame["proj"] - cutoff).dt.total_seconds() / 86400.0
    return frame


def earliest_predictions(
    streams: pd.DataFrame,
    clients: pd.Index,
    *,
    min_appearances: int = 2,
    max_days: float | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    frame = precise_streams(streams)
    if min_appearances > 1 and not frame.empty:
        frame = frame[frame["appearances"] >= min_appearances]
    if max_days is not None and not frame.empty:
        frame = frame[frame["days"] <= max_days]
    pred = pd.Series("none", index=clients, dtype=object)
    meta = pd.DataFrame(
        {
            "has_precise": False,
            "n_precise": 0.0,
            "earliest_days": np.nan,
            "earliest_family": pd.Series("none", index=clients, dtype=object),
            "n_families": 0.0,
        },
        index=clients,
    )
    if frame.empty:
        return pred, meta
    frame = frame.sort_values(["client_id", "days", "appearances"], ascending=[True, True, False])
    for client, group in frame.groupby("client_id", sort=False):
        if client not in pred.index:
            continue
        top = group.iloc[0]
        pred.loc[client] = str(top["mapped_family"])
        meta.at[client, "has_precise"] = True
        meta.at[client, "n_precise"] = float(len(group))
        meta.at[client, "earliest_days"] = float(top["days"])
        meta.at[client, "earliest_family"] = str(top["mapped_family"])
        meta.at[client, "n_families"] = float(group["mapped_family"].nunique())
    return pred, meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results: dict = {}

    print("streams...", flush=True)
    va = build_streams(data.valid_transactions)

    # Pure temporal keyword models
    for min_app in (1, 2, 3):
        for max_days in (None, 30, 45, 60, 90):
            pred, meta = earliest_predictions(
                va, yv.index, min_appearances=min_app, max_days=max_days
            )
            key = f"earliest_a{min_app}_d{max_days}"
            score(yv, pred, results, key)
            covered = int(meta["has_precise"].sum())
            print(f"  covered_clients={covered}", flush=True)

    # Ceiling if we only score covered (oracle none for uncovered positives — diagnostic)
    pred, meta = earliest_predictions(va, yv.index, min_appearances=2, max_days=90)
    ceil = []
    for c, _t in yv.items():
        if meta.at[c, "has_precise"]:
            # if true family anywhere in precise streams, pick true; else earliest
            # diagnostic only
            ceil.append(pred.loc[c])
        else:
            ceil.append("none")
    score(yv, pd.Series(ceil, index=yv.index), results, "precise_only_no_defer")

    # Coverage of true family among precise streams
    frame = precise_streams(va)
    frame = frame[frame["appearances"] >= 2]
    cand = frame.groupby("client_id")["mapped_family"].apply(lambda s: set(s.astype(str)))
    pos_cov = sum(1 for c, t in yv.items() if t != "none" and t in cand.get(c, set()))
    n_pos = int((yv != "none").sum())
    print(f"precise true-family coverage: {pos_cov}/{n_pos}={pos_cov/max(n_pos,1):.3f}", flush=True)
    oracle = []
    for c, t in yv.items():
        fams = cand.get(c, set())
        if t in fams or t == "none":
            oracle.append(t)
        else:
            oracle.append("none")
    score(yv, pd.Series(oracle, index=yv.index), results, "precise_oracle_ceiling")

    print("fitting Push...", flush=True)
    push = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    push_p = push.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    push_pred = push_p.idxmax(axis=1)
    score(yv, push_pred, results, "push")

    # Defer hybrids
    for min_app in (2, 3):
        for max_days in (45, 60, 90):
            temporal, meta = earliest_predictions(
                va, yv.index, min_appearances=min_app, max_days=max_days
            )
            # Always take temporal when present
            out = push_pred.copy()
            out.loc[meta["has_precise"]] = temporal.loc[meta["has_precise"]]
            score(yv, out, results, f"defer_always_a{min_app}_d{max_days}")

            # Take temporal only if unique family or push agrees / push unsure
            for mode in ("unique", "agree", "push_none", "push_weak"):
                out = push_pred.copy()
                mask = meta["has_precise"].copy()
                if mode == "unique":
                    mask &= meta["n_families"].eq(1)
                elif mode == "agree":
                    mask &= temporal.eq(push_pred)
                    # also take unique even if disagree
                    mask |= meta["has_precise"] & meta["n_families"].eq(1)
                elif mode == "push_none":
                    mask &= push_pred.eq("none") | meta["n_families"].eq(1)
                elif mode == "push_weak":
                    mask &= (
                        push_p.max(axis=1).lt(0.4)
                        | meta["n_families"].eq(1)
                        | temporal.eq(push_pred)
                    )
                out.loc[mask] = temporal.loc[mask]
                score(yv, out, results, f"defer_{mode}_a{min_app}_d{max_days}")

    # Soft: boost push logits for earliest family
    temporal, meta = earliest_predictions(va, yv.index, min_appearances=2, max_days=90)
    for alpha in (1.0, 2.0, 3.0, 5.0):
        logits = np.log(np.clip(push_p.to_numpy(), 1e-12, 1.0))
        for i, client in enumerate(yv.index):
            if not meta.at[client, "has_precise"]:
                continue
            fam = temporal.loc[client]
            if fam in POSITIVE_LABELS:
                # stronger boost if closer in time / unique
                boost = alpha
                if meta.at[client, "n_families"] == 1:
                    boost *= 1.5
                days = meta.at[client, "earliest_days"]
                if np.isfinite(days):
                    boost *= float(np.exp(-days / 45.0))
                logits[i, LABELS.index(fam)] += boost
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        proba = w / w.sum(axis=1, keepdims=True)
        pred = pd.Series(np.asarray(LABELS)[proba.argmax(1)], index=yv.index)
        score(yv, pred, results, f"boost_earliest_{alpha}")

    ranking = sorted(((k, v["macro_f1"]) for k, v in results.items()), key=lambda x: -x[1])
    summary = {
        "seconds": perf_counter() - started,
        "precise_coverage": pos_cov / max(n_pos, 1),
        "top20": ranking[:20],
        "results": results,
    }
    (OUT / "next_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"top20": ranking[:20]}, indent=2))
    print(f"done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
