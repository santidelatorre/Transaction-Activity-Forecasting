"""Slim disambiguation ranker: soft-stream features + Push probs at inference."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_oracle import POSITIVE_LABELS, build_streams
from transaction_forecasting.ubs.v3 import _stream_quality
from transaction_forecasting.ubs.v3_push import StreamV3PushModel

OUT = Path("outputs/metrics/v3_disambig")
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
    }
    print(f"{key}: {results[key]['macro_f1']:.4f}", flush=True)


def fit_lift(streams, labels):
    due = streams.loc[streams["is_candidate"], ["client_id", "description"]].drop_duplicates()
    due = due.merge(labels.rename("y").reset_index(), on="client_id")
    cs, total = labels.value_counts(), len(labels)
    rows = []
    for desc, g in due.groupby("description"):
        support = g["client_id"].nunique()
        if support < 2:
            continue
        for fam in POSITIVE_LABELS:
            inside = int((g["y"] == fam).sum())
            outside = support - inside
            lift = ((inside + 1) / (cs[fam] + 2)) / ((outside + 1) / (total - cs[fam] + 2))
            rows.append({"description": desc, "family": fam, "lift": lift})
    return pd.DataFrame(rows)


def soft_map(lifts, min_lift=1.1):
    out = {}
    for desc, g in lifts[lifts.lift >= min_lift].groupby("description"):
        g = g.sort_values("lift", ascending=False)
        w = {r.family: max(float(r.lift) - 1.0, 1e-6) for r in g.itertuples()}
        s = sum(w.values())
        out[desc] = {k: v / s for k, v in w.items()}
    return out


def soft_and_details(streams, sm, clients):
    frame = streams[streams["is_recurrent"]].copy()
    frame = frame[~frame["description"].astype(str).isin(NOISE)]
    frame = _stream_quality(frame)
    frame["description"] = frame["description"].astype(str)
    records = [
        {"description": d, "family": f, "p": p} for d, dist in sm.items() for f, p in dist.items()
    ]
    soft = pd.DataFrame(0.0, index=clients, columns=list(POSITIVE_LABELS))
    if not records or frame.empty:
        return soft, pd.DataFrame()
    merged = frame.merge(pd.DataFrame(records), on="description")
    merged["score"] = merged["appearances"].astype(float) * merged["p"]
    soft = (
        merged.groupby(["client_id", "family"])["score"]
        .max()
        .unstack("family")
        .reindex(clients)
        .reindex(columns=list(POSITIVE_LABELS), fill_value=0.0)
        .fillna(0.0)
    )
    details = (
        merged.groupby(["client_id", "family"])
        .agg(
            soft_max=("score", "max"),
            soft_sum=("score", "sum"),
            appearances_max=("appearances", "max"),
            n_streams=("description", "count"),
            p_max=("p", "max"),
            days_to_next_min=("days_to_next", "min"),
        )
        .reset_index()
    )
    return soft, details


def pairs(clients, labels, soft, details, proba=None):
    det = details.set_index(["client_id", "family"]) if len(details) else None
    rows = []
    for client in clients:
        true = labels.loc[client] if labels is not None and client in labels.index else None
        cands = [f for f in POSITIVE_LABELS if soft.loc[client, f] > 0]
        n = len(cands)
        soft_rank = soft.loc[client].rank(ascending=False)
        base = (
            proba.loc[client]
            if proba is not None and client in proba.index
            else pd.Series(0.0, index=LABELS)
        )
        base_rank = base.rank(ascending=False)
        for fam in cands + ["none"]:
            if fam == "none":
                feat = dict(
                    soft_max=0.0,
                    soft_sum=0.0,
                    appearances_max=0.0,
                    n_streams=0.0,
                    p_max=0.0,
                    days_to_next_min=999.0,
                    base_proba=float(base.get("none", 0.0)),
                    soft_rank=99.0,
                    base_rank=float(base_rank.get("none", 99)),
                    is_none=1.0,
                    n_cand=float(n),
                )
            else:
                if det is not None and (client, fam) in det.index:
                    d = det.loc[(client, fam)]
                    if isinstance(d, pd.DataFrame):
                        d = d.iloc[0]
                    feat = {
                        k: float(d[k])
                        for k in [
                            "soft_max",
                            "soft_sum",
                            "appearances_max",
                            "n_streams",
                            "p_max",
                            "days_to_next_min",
                        ]
                    }
                else:
                    feat = dict(
                        soft_max=float(soft.loc[client, fam]),
                        soft_sum=float(soft.loc[client, fam]),
                        appearances_max=0.0,
                        n_streams=1.0,
                        p_max=0.0,
                        days_to_next_min=999.0,
                    )
                feat.update(
                    base_proba=float(base.get(fam, 0.0)),
                    soft_rank=float(soft_rank.get(fam, 99)),
                    base_rank=float(base_rank.get(fam, 99)),
                    is_none=0.0,
                    n_cand=float(n),
                )
            for f in list(POSITIVE_LABELS) + ["none"]:
                feat[f"fam_{f}"] = 1.0 if fam == f else 0.0
            feat.update(client_id=client, family=fam, is_true=int(true == fam) if true else 0)
            rows.append(feat)
    return pd.DataFrame(rows)


FEATS = [
    "soft_max",
    "soft_sum",
    "appearances_max",
    "n_streams",
    "p_max",
    "days_to_next_min",
    "base_proba",
    "soft_rank",
    "base_rank",
    "is_none",
    "n_cand",
] + [f"fam_{f}" for f in list(POSITIVE_LABELS) + ["none"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yt = data.train_labels.set_index("client_id")[TARGET_COLUMN]
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}

    print("push...", flush=True)
    push = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    push_p = push.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    score(yv, push_p.idxmax(1), results, "push")

    tr = build_streams(data.train_transactions)
    va = build_streams(data.valid_transactions)
    sm = soft_map(fit_lift(tr, yt), 1.1)
    soft_tr, det_tr = soft_and_details(tr, sm, yt.index)
    soft_va, det_va = soft_and_details(va, sm, yv.index)

    ceil = pd.Series(
        [t if (t == "none" or soft_va.loc[c, t] > 0) else "none" for c, t in yv.items()],
        index=yv.index,
    )
    score(yv, ceil, results, "ceiling")

    # Train pairs WITHOUT base proba (zeros); VALID pairs WITH push proba
    train_pairs = pairs(yt.index, yt, soft_tr, det_tr, proba=None)
    valid_pairs = pairs(yv.index, yv, soft_va, det_va, proba=push_p)
    print(f"pairs train={len(train_pairs)} valid={len(valid_pairs)}", flush=True)

    # Also train a version where we fake base_proba from soft-normalized (stream-only teacher)
    soft_diri = soft_tr.copy()
    soft_diri["none"] = 0.3
    soft_diri = soft_diri.div(soft_diri.sum(1), axis=0).reindex(columns=LABELS).fillna(0)
    train_pairs_soft = pairs(yt.index, yt, soft_tr, det_tr, proba=soft_diri)

    for name, tr_pairs in (("stream_only", train_pairs), ("soft_teacher", train_pairs_soft)):
        X = tr_pairs[FEATS].astype(float)
        y = tr_pairs["is_true"].astype(int)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        clf = LogisticRegression(max_iter=1200, class_weight="balanced", C=0.4, random_state=42)
        clf.fit(Xs, y)

        vp = valid_pairs.copy()
        vp["score"] = clf.predict_proba(sc.transform(vp[FEATS].astype(float)))[:, 1]
        idx = vp.groupby("client_id")["score"].idxmax()
        pred = vp.loc[idx].set_index("client_id")["family"].reindex(yv.index)
        score(yv, pred, results, f"ranker_{name}")

        # blend with push over candidate logits
        for alpha in (0.4, 0.6, 0.8):
            out = []
            for client, g in vp.groupby("client_id"):
                cands = list(g["family"])
                logits = [
                    alpha * float(r.score)
                    + (1 - alpha) * np.log(max(float(push_p.loc[client, r.family]), 1e-12))
                    for r in g.itertuples()
                ]
                out.append((client, cands[int(np.argmax(logits))]))
            pred = pd.Series({c: p for c, p in out}).reindex(yv.index)
            score(yv, pred, results, f"blend_{name}_{alpha}")

        # margin gate
        margins, preds = {}, {}
        for client, g in vp.groupby("client_id"):
            g = g.sort_values("score", ascending=False)
            preds[client] = g.iloc[0]["family"]
            margins[client] = float(g.iloc[0]["score"] - (g.iloc[1]["score"] if len(g) > 1 else 0))
        fallback = push_p.idxmax(1)
        for thr in (0.08, 0.15, 0.25):
            out = fallback.copy()
            take = pd.Series(margins).reindex(out.index).fillna(0).ge(thr)
            out.loc[take] = pd.Series(preds).reindex(out.index).loc[take]
            score(yv, out, results, f"gate_{name}_{thr}")

    ranking = sorted(((k, v["macro_f1"]) for k, v in results.items()), key=lambda x: -x[1])
    (OUT / "disambig_slim_summary.json").write_text(
        json.dumps(
            {"seconds": perf_counter() - started, "top15": ranking[:15], "results": results},
            indent=2,
        )
    )
    print(json.dumps({"top15": ranking[:15]}, indent=2))


if __name__ == "__main__":
    main()
