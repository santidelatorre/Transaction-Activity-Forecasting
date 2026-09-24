"""Learned disambiguation among soft merchant candidates.

Diagnostic: perfect pick among soft-recurrent candidates ≈ Macro-F1 0.71–0.77.
Blending/restricting Push fails; we need a ranker that chooses among the
candidate set ∪ {none}, falling back to Push only when empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_oracle import POSITIVE_LABELS, build_streams
from transaction_forecasting.ubs.stream_text import StreamTextFamilyModel
from transaction_forecasting.ubs.v2 import IntegratedV2Model
from transaction_forecasting.ubs.v3 import _stream_quality
from transaction_forecasting.ubs.v3_max import StreamV3MaxModel
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
    print(f"{key}: {results[key]['macro_f1']:.4f} acc={results[key]['accuracy']:.3f}", flush=True)
    return results[key]


def fit_lift_table(streams, labels):
    due = streams.loc[streams["is_candidate"], ["client_id", "description"]].drop_duplicates()
    due = due.merge(labels.rename("y").reset_index(), on="client_id")
    cs = labels.value_counts()
    total = len(labels)
    rows = []
    for desc, g in due.groupby("description"):
        support = g["client_id"].nunique()
        if support < 2:
            continue
        for fam in POSITIVE_LABELS:
            inside = int((g["y"] == fam).sum())
            outside = support - inside
            lift = ((inside + 1) / (cs[fam] + 2)) / ((outside + 1) / (total - cs[fam] + 2))
            rows.append(
                {
                    "description": desc,
                    "family": fam,
                    "lift": lift,
                    "inside": inside,
                    "support": support,
                }
            )
    return pd.DataFrame(rows)


def soft_map(lift_table, min_lift=1.1, top_k=None):
    out = {}
    for desc, g in lift_table[lift_table.lift >= min_lift].groupby("description"):
        g = g.sort_values("lift", ascending=False)
        if top_k:
            g = g.head(top_k)
        w = {r.family: max(float(r.lift) - 1.0, 0.0) for r in g.itertuples()}
        if not any(w.values()):
            w = {g.iloc[0].family: 1.0}
        s = sum(w.values()) or 1.0
        out[desc] = {k: v / s for k, v in w.items() if v > 0}
    return out


def soft_scores(streams, sm, clients, mode="recurrent"):
    frame = streams[
        streams["is_recurrent"] if mode == "recurrent" else streams["is_candidate"]
    ].copy()
    frame = frame[~frame["description"].astype(str).isin(NOISE)]
    if frame.empty:
        return pd.DataFrame(0.0, index=clients, columns=list(POSITIVE_LABELS))
    frame = _stream_quality(frame)
    frame["description"] = frame["description"].astype(str)
    frame["_sw"] = (
        frame["appearances"].astype(float) if mode == "recurrent" else frame["due_weight"]
    )
    records = [
        {"description": d, "family": f, "p": p} for d, dist in sm.items() for f, p in dist.items()
    ]
    dist_df = pd.DataFrame(records)
    merged = frame.merge(dist_df, on="description", how="inner")
    if merged.empty:
        return pd.DataFrame(0.0, index=clients, columns=list(POSITIVE_LABELS))
    merged["score"] = merged["_sw"] * merged["p"]
    pivot = merged.groupby(["client_id", "family"])["score"].max().unstack("family")
    out = pivot.reindex(clients).reindex(columns=list(POSITIVE_LABELS), fill_value=0.0).fillna(0.0)
    return out


def family_stream_detail(streams, sm, clients, mode="recurrent"):
    """Per (client, family): due_weight max, appearances max, map mass, n_streams."""
    frame = streams[
        streams["is_recurrent"] if mode == "recurrent" else streams["is_candidate"]
    ].copy()
    frame = frame[~frame["description"].astype(str).isin(NOISE)]
    frame = _stream_quality(frame)
    frame["description"] = frame["description"].astype(str)
    records = [
        {"description": d, "family": f, "p": p} for d, dist in sm.items() for f, p in dist.items()
    ]
    dist_df = pd.DataFrame(records)
    merged = frame.merge(dist_df, on="description", how="inner")
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for (client, fam), g in merged.groupby(["client_id", "family"]):
        rows.append(
            {
                "client_id": client,
                "family": fam,
                "soft_max": float((g["appearances"].astype(float) * g["p"]).max()),
                "soft_sum": float((g["appearances"].astype(float) * g["p"]).sum()),
                "appearances_max": float(g["appearances"].max()),
                "n_streams": float(len(g)),
                "p_max": float(g["p"].max()),
                "days_to_next_min": float(g["days_to_next"].min())
                if "days_to_next" in g
                else 999.0,
            }
        )
    return pd.DataFrame(rows)


FEATURE_BASE = [
    "soft_max",
    "soft_sum",
    "appearances_max",
    "n_streams",
    "p_max",
    "days_to_next_min",
    "base_proba",
    "text_proba",
    "vmax_proba",
    "is_none_slot",
    "n_cand_families",
    "soft_rank",
    "base_rank",
]


def build_pair_frame(clients, labels, details, soft, base_proba, text_proba, vmax_proba):
    """One row per (client, candidate family) + none slot."""
    records = []
    soft = soft.reindex(clients).fillna(0.0)
    detail_idx = details.set_index(["client_id", "family"]) if not details.empty else pd.DataFrame()
    for client in clients:
        true = labels.loc[client] if client in labels.index else None
        cand_fams = [f for f in POSITIVE_LABELS if soft.loc[client, f] > 0]
        n_cand = len(cand_fams)
        soft_vals = soft.loc[client]
        base_vals = (
            base_proba.loc[client] if client in base_proba.index else pd.Series(0, index=LABELS)
        )
        text_vals = (
            text_proba.loc[client]
            if client in text_proba.index
            else pd.Series(0, index=POSITIVE_LABELS)
        )
        vmax_vals = (
            vmax_proba.loc[client] if client in vmax_proba.index else pd.Series(0, index=LABELS)
        )

        slots = list(cand_fams) + ["none"]
        soft_rank = soft_vals.rank(ascending=False)
        base_rank = base_vals.rank(ascending=False)

        for fam in slots:
            if fam == "none":
                feats = {
                    "soft_max": 0.0,
                    "soft_sum": 0.0,
                    "appearances_max": 0.0,
                    "n_streams": 0.0,
                    "p_max": 0.0,
                    "days_to_next_min": 999.0,
                    "base_proba": float(base_vals.get("none", 0.0)),
                    "text_proba": 0.0,
                    "vmax_proba": float(vmax_vals.get("none", 0.0)),
                    "is_none_slot": 1.0,
                    "n_cand_families": float(n_cand),
                    "soft_rank": 99.0,
                    "base_rank": float(base_rank.get("none", 99.0)),
                }
            else:
                if (client, fam) in detail_idx.index:
                    d = detail_idx.loc[(client, fam)]
                    if isinstance(d, pd.DataFrame):
                        d = d.iloc[0]
                    feats = {
                        "soft_max": float(d["soft_max"]),
                        "soft_sum": float(d["soft_sum"]),
                        "appearances_max": float(d["appearances_max"]),
                        "n_streams": float(d["n_streams"]),
                        "p_max": float(d["p_max"]),
                        "days_to_next_min": float(d["days_to_next_min"]),
                    }
                else:
                    feats = {
                        "soft_max": float(soft_vals[fam]),
                        "soft_sum": float(soft_vals[fam]),
                        "appearances_max": 0.0,
                        "n_streams": 1.0,
                        "p_max": 0.0,
                        "days_to_next_min": 999.0,
                    }
                feats.update(
                    {
                        "base_proba": float(base_vals.get(fam, 0.0)),
                        "text_proba": float(text_vals.get(fam, 0.0))
                        if fam in text_vals.index
                        else 0.0,
                        "vmax_proba": float(vmax_vals.get(fam, 0.0)),
                        "is_none_slot": 0.0,
                        "n_cand_families": float(n_cand),
                        "soft_rank": float(soft_rank.get(fam, 99.0)),
                        "base_rank": float(base_rank.get(fam, 99.0)),
                    }
                )
            for f in POSITIVE_LABELS + ("none",):
                feats[f"fam_{f}"] = 1.0 if fam == f else 0.0
            feats["client_id"] = client
            feats["family"] = fam
            feats["is_true"] = int(true == fam) if true is not None else 0
            records.append(feats)
    return pd.DataFrame(records)


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    cols = FEATURE_BASE + [f"fam_{f}" for f in list(POSITIVE_LABELS) + ["none"]]
    return frame[cols].astype(float)


def predict_from_pairs(model, scaler, pair_frame, fallback: pd.Series) -> pd.Series:
    if pair_frame.empty:
        return fallback.copy()
    X = scaler.transform(feature_matrix(pair_frame))
    pair_frame = pair_frame.copy()
    pair_frame["score"] = model.predict_proba(X)[:, 1]
    idx = pair_frame.groupby("client_id")["score"].idxmax()
    pred = pair_frame.loc[idx].set_index("client_id")["family"]
    # empty-candidate clients: already have none slot, so always covered
    out = fallback.copy()
    out.loc[pred.index] = pred
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yt = data.train_labels.set_index("client_id")[TARGET_COLUMN]
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}

    print("fitting base models (V2, Vmax, Push, text)...", flush=True)
    v2 = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    vmax = StreamV3MaxModel(music_alpha=0.60, streaming_alpha=0.05).fit(
        data.train_transactions, data.train_labels
    )
    push = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    text = StreamTextFamilyModel().fit(data.train_transactions, data.train_labels)

    # Components on train (for OOF-ish training we still use in-sample — mild optimism;
    # mitigate with client-level CV on the ranker itself)
    print("streams + soft map...", flush=True)
    tr = build_streams(data.train_transactions)
    va = build_streams(data.valid_transactions)
    lifts = fit_lift_table(tr, yt)
    sm = soft_map(lifts, min_lift=1.1, top_k=None)
    print(f"soft map size {len(sm)}", flush=True)

    soft_tr = soft_scores(tr, sm, yt.index)
    soft_va = soft_scores(va, sm, yv.index)
    det_tr = family_stream_detail(tr, sm, yt.index)
    det_va = family_stream_detail(va, sm, yv.index)

    # Base probs only on VALID (models refuse train-client prediction).
    # For ranker training use soft-stream features only (zeros for model probs).
    v2_va = v2.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    vmax_va = vmax.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    push_va = push.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    text_va, _ = text.predict_scores(data.valid_transactions)
    text_va = text_va.reindex(yv.index).fillna(0.0)
    v2_tr = pd.DataFrame(0.0, index=yt.index, columns=LABELS)
    vmax_tr = pd.DataFrame(0.0, index=yt.index, columns=LABELS)
    text_tr = pd.DataFrame(0.0, index=yt.index, columns=list(POSITIVE_LABELS))

    score(yv, push_va.idxmax(axis=1), results, "push")
    score(yv, vmax_va.idxmax(axis=1), results, "vmax")

    # Ceiling
    ceil = pd.Series(
        [t if (t == "none" or soft_va.loc[c, t] > 0) else "none" for c, t in yv.items()],
        index=yv.index,
    )
    score(yv, ceil, results, "soft_ceiling")

    # Oracle among candidates using push scores
    oracle = []
    for c, t in yv.items():
        cands = [f for f in POSITIVE_LABELS if soft_va.loc[c, f] > 0] + ["none"]
        oracle.append(t if t in cands else max(cands, key=lambda f: push_va.loc[c, f]))
    score(yv, pd.Series(oracle, index=yv.index), results, "oracle_among_soft")

    print("building pair frames...", flush=True)
    train_pairs = build_pair_frame(yt.index, yt, det_tr, soft_tr, v2_tr, text_tr, vmax_tr)
    valid_pairs = build_pair_frame(yv.index, yv, det_va, soft_va, push_va, text_va, vmax_va)
    # Also valid pairs with v2/vmax as base for ablation
    valid_pairs_v2 = build_pair_frame(yv.index, yv, det_va, soft_va, v2_va, text_va, vmax_va)

    print(
        f"train pairs {len(train_pairs)} pos_rate={train_pairs.is_true.mean():.3f}",
        flush=True,
    )

    # Client-grouped CV for ranker
    clients = yt.index.to_numpy()
    y_client = yt.to_numpy()
    oof_scores = pd.Series(0.0, index=train_pairs.index, dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all = feature_matrix(train_pairs)
    # CV models use fold scaler
    for fold, (tr_c, va_c) in enumerate(skf.split(clients, y_client)):
        tr_clients = set(clients[tr_c])
        va_clients = set(clients[va_c])
        tr_mask = train_pairs["client_id"].isin(tr_clients)
        va_mask = train_pairs["client_id"].isin(va_clients)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_all.loc[tr_mask])
        Xva = sc.transform(X_all.loc[va_mask])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42)
        clf.fit(Xtr, train_pairs.loc[tr_mask, "is_true"])
        oof_scores.loc[va_mask] = clf.predict_proba(Xva)[:, 1]
        print(f"  fold {fold} done", flush=True)

    # OOF predictions on train (sanity)
    tmp = train_pairs.copy()
    tmp["score"] = oof_scores
    idx = tmp.groupby("client_id")["score"].idxmax()
    oof_pred = tmp.loc[idx].set_index("client_id")["family"]
    # Can't score OOF on VALID; just report train OOF macro as diagnostic
    m_oof = evaluate_predictions(yt.reindex(oof_pred.index), oof_pred)
    print(f"train OOF ranker MF1={m_oof['macro_f1']:.4f}", flush=True)

    # Fit full ranker
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_all)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=42)
    clf.fit(Xtr, train_pairs["is_true"])

    push_fallback = push_va.idxmax(axis=1)
    pred = predict_from_pairs(clf, sc, valid_pairs, push_fallback)
    score(yv, pred, results, "ranker_push_features")

    pred_v2 = predict_from_pairs(clf, sc, valid_pairs_v2, v2_va.idxmax(axis=1))
    score(yv, pred_v2, results, "ranker_v2_features")

    # Hybrid: if ranker confidence margin high, take ranker, else push
    Xva = sc.transform(feature_matrix(valid_pairs))
    valid_pairs = valid_pairs.copy()
    valid_pairs["score"] = clf.predict_proba(Xva)[:, 1]
    # margin = top1 - top2
    margins = {}
    preds = {}
    for client, g in valid_pairs.groupby("client_id"):
        g = g.sort_values("score", ascending=False)
        preds[client] = g.iloc[0]["family"]
        margins[client] = float(g.iloc[0]["score"] - (g.iloc[1]["score"] if len(g) > 1 else 0))
    rank_pred = pd.Series(preds)
    margin = pd.Series(margins)
    for thr in (0.05, 0.1, 0.15, 0.2, 0.3):
        out = push_fallback.copy()
        take = margin.reindex(out.index).fillna(0).ge(thr)
        out.loc[take] = rank_pred.reindex(out.index).loc[take]
        score(yv, out, results, f"ranker_if_margin_{thr}")

    # Softmax blend of ranker scores with push probs over candidate set
    for alpha in (0.3, 0.5, 0.7, 1.0):
        out = []
        for client in yv.index:
            g = valid_pairs[valid_pairs.client_id == client]
            scores_r = g.set_index("family")["score"]
            # combine with push logprob
            cands = list(scores_r.index)
            logits = []
            for fam in cands:
                logits.append(
                    alpha * float(scores_r[fam])
                    + (1 - alpha) * np.log(max(float(push_va.loc[client, fam]), 1e-12))
                )
            out.append(cands[int(np.argmax(logits))])
        score(yv, pd.Series(out, index=yv.index), results, f"blend_rank_push_{alpha}")

    # When soft has exactly 1 positive family, force it if push not confident none
    single = soft_va.gt(0).sum(axis=1).eq(1)
    forced = push_fallback.copy()
    top_fam = soft_va.idxmax(axis=1)
    for thr in (0.0, 0.2, 0.35):
        out = forced.copy()
        mask = single & push_va["none"].lt(0.55) & soft_va.max(axis=1).ge(thr)
        out.loc[mask] = top_fam.loc[mask]
        score(yv, out, results, f"single_family_force_{thr}")

    ranking = sorted(((k, v["macro_f1"]) for k, v in results.items()), key=lambda x: -x[1])
    summary = {
        "seconds": perf_counter() - started,
        "train_oof_mf1": float(m_oof["macro_f1"]),
        "top20": ranking[:20],
        "results": results,
    }
    (OUT / "disambig_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"top20": ranking[:20]}, indent=2))
    print(f"done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
