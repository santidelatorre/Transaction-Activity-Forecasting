"""Soft multi-family merchant map + recurrent streams → Push hybrid.

Key discovery: winner-take-all mapping kills music (digital plus / premium plan
are multi-label). Using soft lift vectors + recurrent (not only due) streams
raises the abstaining oracle ceiling from ~0.77 toward ~0.92.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_oracle import (
    POSITIVE_LABELS,
    build_streams,
)
from transaction_forecasting.ubs.v3 import _stream_quality
from transaction_forecasting.ubs.v3_push import StreamV3PushModel

OUT = Path("outputs/metrics/v3_softmap")

NOISE = {
    "salary",
    "atm withdrawal",
    "fresh foods",
    "grocery store",
    "neighborhood market",
    "coffee shop",
    "casual dining",
    "pharmacy",
    "hotel booking",
    "ride share",
    "merchant charge",
    "online marketplace",
    "electronics shop",
    "p2p send",
    "p2p receive",
    "card purchase",
    "digital order",
    "service payment",
    "service fee",
}


def score(yv, pred, results, key):
    m = evaluate_predictions(yv.reindex(pred.index), pred)
    results[key] = {
        "macro_f1": float(m["macro_f1"]),
        "accuracy": float(m["accuracy"]),
        "per_class": {k: float(v["f1-score"]) for k, v in m["per_class"].items()},
        "pred_dist": pred.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
    }
    print(
        f"{key}: {results[key]['macro_f1']:.4f} acc={results[key]['accuracy']:.3f}",
        flush=True,
    )
    return results[key]


def fit_soft_lift_table(
    streams: pd.DataFrame,
    labels: pd.Series,
    *,
    min_support: int = 2,
    min_inside: int = 0,
) -> pd.DataFrame:
    """description × family lift matrix from TRAIN due streams (train-only)."""
    due = streams.loc[streams["is_candidate"], ["client_id", "description"]].drop_duplicates()
    due = due.merge(labels.rename("y").reset_index(), on="client_id", how="inner")
    class_sizes = labels.value_counts().reindex(LABELS, fill_value=0)
    total = len(labels)
    rows = []
    for desc, group in due.groupby("description"):
        support = int(group["client_id"].nunique())
        if support < min_support:
            continue
        for family in POSITIVE_LABELS:
            inside = int((group["y"] == family).sum())
            if inside < min_inside:
                continue
            outside = support - inside
            inside_rate = (inside + 1.0) / (float(class_sizes[family]) + 2.0)
            outside_rate = (outside + 1.0) / (float(total - class_sizes[family]) + 2.0)
            rows.append(
                {
                    "description": desc,
                    "family": family,
                    "support": support,
                    "inside": inside,
                    "lift": inside_rate / outside_rate,
                }
            )
    return pd.DataFrame(rows)


def soft_map_dict(
    lift_table: pd.DataFrame,
    *,
    min_lift: float,
    top_k: int | None = None,
    temperature: float = 1.0,
) -> dict[str, dict[str, float]]:
    """description → {family: normalized soft weight}."""
    out: dict[str, dict[str, float]] = {}
    eligible = lift_table[lift_table["lift"] >= min_lift]
    for desc, group in eligible.groupby("description"):
        group = group.sort_values("lift", ascending=False)
        if top_k is not None:
            group = group.head(top_k)
        # weight ∝ (lift - 1)^temp clipped
        weights = {}
        for _, row in group.iterrows():
            w = max(float(row["lift"]) - 1.0, 0.0) ** temperature
            if w > 0:
                weights[str(row["family"])] = w
        if not weights:
            # keep single best if lift>=min even if ~1
            best = group.iloc[0]
            weights[str(best["family"])] = max(float(best["lift"]), 1e-3)
        total = sum(weights.values())
        out[str(desc)] = {k: v / total for k, v in weights.items()}
    return out


def client_soft_scores(
    streams: pd.DataFrame,
    soft_map: dict[str, dict[str, float]],
    client_ids: pd.Index,
    *,
    mode: str,
    filter_noise: bool,
    stream_weight: str = "due_weight",
) -> pd.DataFrame:
    """Aggregate soft family scores per client."""
    clients = pd.Index(client_ids.astype(str), name="client_id")
    frame = streams.copy()
    if mode == "due":
        frame = frame[frame["is_candidate"]]
    elif mode == "recurrent":
        frame = frame[frame["is_recurrent"]]
    elif mode == "app2":
        frame = frame[frame["appearances"] >= 2]
    else:
        raise ValueError(mode)
    if filter_noise:
        frame = frame[~frame["description"].astype(str).isin(NOISE)]
    scores = pd.DataFrame(0.0, index=clients, columns=list(POSITIVE_LABELS))
    if frame.empty:
        return scores
    frame = _stream_quality(frame)
    frame = frame.copy()
    frame["description"] = frame["description"].astype(str)
    if mode == "recurrent":
        close = frame["periodicity_closeness"] if "periodicity_closeness" in frame.columns else 0.5
        frame["_sw"] = frame["appearances"].astype(float) * (0.5 + 0.5 * close)
    else:
        frame["_sw"] = frame["due_weight"].astype(float)

    # Explode soft-map into long form then groupby max
    records = []
    for desc, dist in soft_map.items():
        for fam, p in dist.items():
            records.append({"description": desc, "family": fam, "p": p})
    if not records:
        return scores
    dist_df = pd.DataFrame(records)
    merged = frame.merge(dist_df, on="description", how="inner")
    if merged.empty:
        return scores
    merged["score"] = merged["_sw"] * merged["p"]
    pivot = merged.groupby(["client_id", "family"], sort=False)["score"].max().unstack("family")
    scores = scores.add(pivot.reindex(scores.index).fillna(0.0), fill_value=0.0)
    return scores.reindex(columns=list(POSITIVE_LABELS)).fillna(0.0)


def restrict_and_boost(
    base: pd.DataFrame,
    soft: pd.DataFrame,
    *,
    penalty: float,
    boost: float,
    none_boost_empty: float,
) -> pd.DataFrame:
    logits = np.log(np.clip(base.reindex(columns=LABELS).to_numpy(), 1e-12, 1.0))
    soft = soft.reindex(base.index).fillna(0.0)
    has = soft.to_numpy() > 0
    for fam in POSITIVE_LABELS:
        col = LABELS.index(fam)
        logits[~has[:, list(POSITIVE_LABELS).index(fam)], col] -= penalty
        logits[:, col] += boost * np.log1p(soft[fam].to_numpy())
    empty = ~has.any(axis=1)
    logits[empty, LABELS.index("none")] += none_boost_empty
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    return pd.DataFrame(w / w.sum(axis=1, keepdims=True), index=base.index, columns=LABELS)


def mix_proba(a: pd.DataFrame, b: pd.DataFrame, w: float) -> pd.DataFrame:
    m = w * a.to_numpy() + (1 - w) * b.reindex_like(a).fillna(0).to_numpy()
    m = m / np.clip(m.sum(axis=1, keepdims=True), 1e-12, None)
    return pd.DataFrame(m, index=a.index, columns=LABELS)


def soft_to_diri(soft: pd.DataFrame, none_alpha: float) -> pd.DataFrame:
    frame = soft.clip(lower=0.0).copy()
    frame["none"] = none_alpha
    empty = soft.sum(axis=1).eq(0)
    frame.loc[empty, "none"] = none_alpha + 3.0
    frame = frame.reindex(columns=LABELS).fillna(0.0)
    return frame.div(frame.sum(axis=1).clip(lower=1e-12), axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yt = data.train_labels.set_index("client_id")[TARGET_COLUMN]
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results: dict = {}

    print("fitting Push...", flush=True)
    push = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    push_proba = push.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    score(yv, push_proba.idxmax(axis=1), results, "push")

    print("building soft lift tables...", flush=True)
    tr = build_streams(data.train_transactions)
    va = build_streams(data.valid_transactions)
    lift_table = fit_soft_lift_table(tr, yt, min_support=2, min_inside=0)
    lift_table.to_csv(OUT / "lift_table.csv", index=False)

    # Focused grid: multi-label soft maps × recurrent/due × push hybrids
    configs = [
        # (min_lift, top_k, temperature)
        (1.1, 3, 1.0),
        (1.1, 2, 1.0),
        (1.1, None, 0.5),
        (1.2, 3, 1.0),
        (1.5, 2, 1.0),
        (1.05, 3, 2.0),
    ]
    best = None
    for min_lift, top_k, temperature in configs:
        sm = soft_map_dict(lift_table, min_lift=min_lift, top_k=top_k, temperature=temperature)
        print(f"soft map size={len(sm)} lift={min_lift} k={top_k} temp={temperature}", flush=True)
        for mode in ("due", "recurrent"):
            for filter_noise in (True,):
                soft = client_soft_scores(va, sm, yv.index, mode=mode, filter_noise=filter_noise)
                covered = sum(1 for c, t in yv.items() if t != "none" and soft.loc[c, t] > 0)
                n_pos = int((yv != "none").sum())
                cov = covered / max(n_pos, 1)
                print(f"  {mode} coverage={cov:.3f}", flush=True)

                pred = soft.idxmax(axis=1).where(soft.max(axis=1).gt(0), "none")
                key = f"l{min_lift}_k{top_k}_t{temperature}_{mode}"
                score(yv, pred, results, f"argmax_{key}")

                ceiling = pd.Series(
                    [t if (t == "none" or soft.loc[c, t] > 0) else "none" for c, t in yv.items()],
                    index=yv.index,
                )
                score(yv, ceiling, results, f"ceil_{key}")

                for penalty, boost, nf in (
                    (2.0, 0.75, 0.5),
                    (4.0, 1.0, 1.0),
                    (6.0, 1.5, 1.0),
                    (0.0, 2.0, 0.0),  # boost-only, no hard restrict
                ):
                    restricted = restrict_and_boost(
                        push_proba,
                        soft,
                        penalty=penalty,
                        boost=boost,
                        none_boost_empty=nf,
                    )
                    rkey = f"rest_{key}_p{penalty}_b{boost}"
                    payload = score(yv, restricted.idxmax(axis=1), results, rkey)
                    if best is None or payload["macro_f1"] > best["macro_f1"]:
                        best = {**payload, "key": rkey, "coverage": cov}

                diri = soft_to_diri(soft, none_alpha=0.25)
                for w in (0.5, 0.65, 0.8, 0.9):
                    mixed = mix_proba(push_proba, diri, w)
                    mkey = f"mix_{key}_w{w}"
                    payload = score(yv, mixed.idxmax(axis=1), results, mkey)
                    if best is None or payload["macro_f1"] > best["macro_f1"]:
                        best = {**payload, "key": mkey, "coverage": cov}

                # Geometric mix of push and soft-diri
                eps = 1e-8
                for w in (0.6, 0.75):
                    geo = np.exp(
                        w * np.log(np.clip(push_proba.to_numpy(), eps, 1))
                        + (1 - w) * np.log(np.clip(diri.to_numpy(), eps, 1))
                    )
                    geo = geo / geo.sum(axis=1, keepdims=True)
                    pred = pd.Series(np.asarray(LABELS)[geo.argmax(1)], index=yv.index)
                    gkey = f"geo_{key}_w{w}"
                    payload = score(yv, pred, results, gkey)
                    if best is None or payload["macro_f1"] > best["macro_f1"]:
                        best = {**payload, "key": gkey, "coverage": cov}

    ranking = sorted(
        ((k, v["macro_f1"]) for k, v in results.items() if not k.startswith("ceil_")),
        key=lambda item: item[1],
        reverse=True,
    )
    ceil_rank = sorted(
        ((k, v["macro_f1"]) for k, v in results.items() if k.startswith("ceil_")),
        key=lambda item: item[1],
        reverse=True,
    )
    summary = {
        "seconds": perf_counter() - started,
        "best": best,
        "top20": ranking[:20],
        "top_ceilings": ceil_rank[:10],
        "push": results.get("push"),
        "results": results,
    }
    (OUT / "softmap_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"top20": ranking[:20], "ceilings": ceil_rank[:5], "best": best}, indent=2))
    print(f"done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
