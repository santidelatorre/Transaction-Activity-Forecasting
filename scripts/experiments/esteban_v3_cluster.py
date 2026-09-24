"""Merchant signature clustering: amount×cadence fingerprints, label transfer.

Wild idea: subscription merchants share a soft signature (median amount bin +
periodicity bucket + MCC). Cluster TRAIN merchants by signature, transfer family
labels to VALID-only merchants in the same cluster, then rank with Push.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_oracle import (
    POSITIVE_LABELS,
    TrainOnlyFamilyMapper,
    build_streams,
)
from transaction_forecasting.ubs.v3 import _stream_quality
from transaction_forecasting.ubs.v3_push import StreamV3PushModel

OUT = Path("outputs/metrics/v3_cluster")
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
}


def score(yv, pred, results, key):
    m = evaluate_predictions(yv.reindex(pred.index), pred)
    results[key] = {
        "macro_f1": float(m["macro_f1"]),
        "accuracy": float(m["accuracy"]),
        "per_class": {k: float(v["f1-score"]) for k, v in m["per_class"].items()},
    }
    print(f"{key}: {results[key]['macro_f1']:.4f}", flush=True)


def stream_signature(streams: pd.DataFrame) -> pd.DataFrame:
    """One row per description with geometric signature features."""
    frame = streams.copy()
    frame = frame[~frame["description"].astype(str).isin(NOISE)]
    if frame.empty:
        return pd.DataFrame()
    # Aggregate across clients for each description
    rows = []
    for desc, g in frame.groupby("description"):
        amt = g["median_amount"].astype(float) if "median_amount" in g else g.get("amount_median")
        if amt is None:
            # fall back
            med_amt = float(g["amount"].median()) if "amount" in g.columns else 0.0
        else:
            med_amt = float(pd.to_numeric(amt, errors="coerce").median())
        gap = float(
            pd.to_numeric(g.get("median_gap_days", g.get("gap_days")), errors="coerce").median()
        )
        if not np.isfinite(gap):
            gap = 30.0
        mcc_mode = str(g["mcc"].astype(str).mode().iloc[0]) if "mcc" in g and len(g) else "0"
        try:
            mcc_num = float(mcc_mode)
        except ValueError:
            mcc_num = 0.0
        rows.append(
            {
                "description": desc,
                "log_amount": np.log1p(abs(med_amt)),
                "log_gap": np.log1p(max(gap, 0.0)),
                "mcc": mcc_num,
                "n_clients": g["client_id"].nunique(),
                "n_streams": len(g),
            }
        )
    return pd.DataFrame(rows).set_index("description")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yt = data.train_labels.set_index("client_id")[TARGET_COLUMN]
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}

    print("push...", flush=True)
    push = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    push_proba = push.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    score(yv, push_proba.idxmax(axis=1), results, "push")

    print("streams + signatures...", flush=True)
    tr = build_streams(data.train_transactions)
    va = build_streams(data.valid_transactions)
    mapper = TrainOnlyFamilyMapper().fit(tr, data.train_labels)
    mapping = mapper.mapping_

    # Description-level family prior from TRAIN: clients with that due desc
    due_tr = tr[tr["is_candidate"]].copy()
    due_tr = due_tr[~due_tr["description"].astype(str).isin(NOISE)]
    desc_family_counts = defaultdict(lambda: defaultdict(float))
    for client, g in due_tr.groupby("client_id"):
        if client not in yt.index:
            continue
        lab = yt.loc[client]
        for desc in g["description"].unique():
            desc_family_counts[desc][lab] += 1.0

    def soft_family(desc):
        if desc in mapping.index and pd.notna(mapping.loc[desc]):
            return str(mapping.loc[desc]), 1.0
        counts = desc_family_counts.get(desc)
        if not counts:
            return None, 0.0
        total = sum(counts.values())
        best = max(counts, key=counts.get)
        if best == "none":
            return None, 0.0
        return best, counts[best] / total

    # Signatures for kNN transfer
    # Need median_amount on streams — check columns
    print("stream cols", list(tr.columns)[:30], flush=True)
    sig_tr = stream_signature(tr)
    sig_va = stream_signature(va)
    # Train descriptions with known soft family
    train_known = []
    train_fams = []
    for desc in sig_tr.index:
        fam, conf = soft_family(desc)
        if fam in POSITIVE_LABELS and conf >= 0.3:
            train_known.append(desc)
            train_fams.append(fam)
    print(f"train known descs for kNN: {len(train_known)}", flush=True)

    feat_cols = ["log_amount", "log_gap", "mcc"]
    Xtr = sig_tr.loc[train_known, feat_cols].to_numpy()
    # standardize
    mu, sd = Xtr.mean(0), Xtr.std(0).clip(min=1e-6)
    Xtr_n = (Xtr - mu) / sd
    nn = NearestNeighbors(n_neighbors=min(7, len(train_known)), metric="euclidean")
    nn.fit(Xtr_n)

    # Map VALID due descriptions via exact or kNN
    due_va = va[va["is_candidate"]].copy()
    due_va = due_va[~due_va["description"].astype(str).isin(NOISE)]
    due_va = _stream_quality(due_va)

    cache = {}
    fams = []
    confs = []
    for desc in due_va["description"].astype(str):
        if desc not in cache:
            fam, conf = soft_family(desc)
            if fam is None and desc in sig_va.index and len(train_known):
                x = ((sig_va.loc[desc, feat_cols].to_numpy() - mu) / sd).reshape(1, -1)
                dists, idxs = nn.kneighbors(x)
                votes = defaultdict(float)
                for d, i in zip(dists[0], idxs[0], strict=True):
                    w = 1.0 / (1.0 + float(d))
                    votes[train_fams[i]] += w
                fam = max(votes, key=votes.get)
                conf = votes[fam] / sum(votes.values())
                # only accept if close enough
                if float(dists[0].mean()) > 1.5:
                    fam, conf = None, 0.0
            cache[desc] = (fam, conf)
        fams.append(cache[desc][0])
        confs.append(cache[desc][1])
    due_va = due_va.copy()
    due_va["mapped_family"] = fams
    due_va["map_conf"] = confs
    due_va = due_va[due_va["mapped_family"].notna()]

    # Coverage
    cand = due_va.groupby("client_id")["mapped_family"].apply(lambda s: set(s.astype(str)))
    pos_cov = sum(1 for c, t in yv.items() if t != "none" and t in cand.get(c, set()))
    n_pos = int((yv != "none").sum())
    print(f"cluster coverage positives: {pos_cov}/{n_pos} = {pos_cov/max(n_pos,1):.3f}", flush=True)
    ceiling = pd.Series(
        [t if (t in cand.get(c, set()) or t == "none") else "none" for c, t in yv.items()],
        index=yv.index,
    )
    score(yv, ceiling, results, "cluster_ceiling")

    # Weight matrix
    weights = pd.DataFrame(0.0, index=yv.index, columns=list(POSITIVE_LABELS))
    for client, g in due_va.groupby("client_id"):
        if client not in weights.index:
            continue
        for fam, gg in g.groupby("mapped_family"):
            if fam not in POSITIVE_LABELS:
                continue
            weights.at[client, fam] = float((gg["due_weight"] * gg["map_conf"]).max())
    pred = weights.idxmax(axis=1).where(weights.max(axis=1).gt(0), "none")
    score(yv, pred, results, "cluster_argmax")

    # Restrict / boost push
    has = weights > 0
    for temp in (0.5, 1.0, 2.0, 3.0):
        logits = np.log(np.clip(push_proba.to_numpy(), 1e-12, 1))
        for fam in POSITIVE_LABELS:
            col = LABELS.index(fam)
            logits[~has[fam].to_numpy(), col] -= 6.0
            logits[:, col] += temp * np.log1p(weights[fam].to_numpy())
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        proba = pd.DataFrame(w / w.sum(axis=1, keepdims=True), index=yv.index, columns=LABELS)
        score(yv, proba.idxmax(axis=1), results, f"push_cluster_t{temp}")

    # Soft mix
    cw = weights.copy()
    cw["none"] = 0.15
    diri = cw.div(cw.sum(axis=1), axis=0).reindex(columns=LABELS).fillna(0)
    for w in (0.7, 0.8, 0.9):
        mix = w * push_proba.to_numpy() + (1 - w) * diri.to_numpy()
        mix = mix / mix.sum(axis=1, keepdims=True)
        pred = pd.Series(np.asarray(LABELS)[mix.argmax(1)], index=yv.index)
        score(yv, pred, results, f"push_cluster_mix_{w}")

    ranking = sorted(((k, v["macro_f1"]) for k, v in results.items()), key=lambda x: -x[1])
    summary = {
        "seconds": perf_counter() - started,
        "coverage": pos_cov / max(n_pos, 1),
        "top15": ranking[:15],
        "results": results,
    }
    (OUT / "cluster_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"top15": ranking[:15], "coverage": summary["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
