"""Fuzzy merchant linking + learned stream ranker toward oracle ceiling.

Hypothesis: many VALID merchants are near-duplicates of TRAIN merchants
(typos / token reorder / shared stems). Link them, expand candidates, then
learn a pairwise stream ranker that picks the winning family among due streams.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_oracle import (
    POSITIVE_LABELS,
    TrainOnlyFamilyMapper,
    build_streams,
)
from transaction_forecasting.ubs.v3 import _stream_quality
from transaction_forecasting.ubs.v3_push import StreamV3PushModel

OUT = Path("outputs/metrics/v3_ranker")

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

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(str(text).lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score(yv, pred, results, key):
    m = evaluate_predictions(yv.reindex(pred.index), pred)
    results[key] = {
        "macro_f1": float(m["macro_f1"]),
        "accuracy": float(m["accuracy"]),
        "per_class": {k: float(v["f1-score"]) for k, v in m["per_class"].items()},
    }
    print(f"{key}: {results[key]['macro_f1']:.4f} acc={results[key]['accuracy']:.3f}", flush=True)


def build_fuzzy_map(
    train_descriptions: list[str],
    mapping: pd.Series,
    *,
    min_jaccard: float = 0.5,
    min_char_cos: float = 0.55,
) -> pd.Series:
    """Legacy stub — use FuzzyFamilyLinker for fuzzy mapping."""
    _ = (train_descriptions, min_jaccard, min_char_cos)
    return mapping.copy()


class FuzzyFamilyLinker:
    def __init__(self, mapping: pd.Series, *, min_jaccard=0.45, min_char_cos=0.50):
        self.exact = mapping.dropna().astype(str)
        self.min_jaccard = min_jaccard
        self.min_char_cos = min_char_cos
        self.descs = list(self.exact.index.astype(str))
        self.fams = self.exact.to_numpy()
        self.tokens = [tokenize(d) for d in self.descs]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.X = self.vectorizer.fit_transform(self.descs)

    def map_one(self, description: str) -> tuple[str | None, float]:
        d = str(description)
        if d in self.exact.index:
            return str(self.exact.loc[d]), 1.0
        tok = tokenize(d)
        # cheap jaccard scan
        best_j, best_i = 0.0, -1
        for i, t in enumerate(self.tokens):
            j = jaccard(tok, t)
            if j > best_j:
                best_j, best_i = j, i
        if best_j >= self.min_jaccard and best_i >= 0:
            return str(self.fams[best_i]), float(best_j)
        # char cosine
        q = self.vectorizer.transform([d])
        sims = cosine_similarity(q, self.X).ravel()
        i = int(sims.argmax())
        if sims[i] >= self.min_char_cos:
            return str(self.fams[i]), float(sims[i])
        return None, 0.0

    def map_series(self, descriptions: pd.Series) -> tuple[pd.Series, pd.Series]:
        fams, confs = [], []
        cache: dict[str, tuple[str | None, float]] = {}
        for d in descriptions.astype(str):
            if d not in cache:
                cache[d] = self.map_one(d)
            f, c = cache[d]
            fams.append(f)
            confs.append(c)
        return pd.Series(fams, index=descriptions.index), pd.Series(confs, index=descriptions.index)


def stream_rows_with_families(
    streams: pd.DataFrame,
    linker: FuzzyFamilyLinker,
    *,
    filter_noise: bool = True,
    due_only: bool = True,
) -> pd.DataFrame:
    frame = streams.copy()
    if due_only:
        frame = frame[frame["is_candidate"]].copy()
    if filter_noise:
        frame = frame[~frame["description"].astype(str).isin(NOISE)].copy()
    if frame.empty:
        return frame
    frame = _stream_quality(frame)
    fams, confs = linker.map_series(frame["description"])
    frame["mapped_family"] = fams
    frame["map_conf"] = confs
    frame = frame[frame["mapped_family"].notna()].copy()
    return frame


def aggregate_family_weights(rows: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=clients, columns=list(POSITIVE_LABELS))
    if rows.empty:
        return weights
    for client, group in rows.groupby("client_id", sort=False):
        if client not in weights.index:
            continue
        for fam, g in group.groupby("mapped_family"):
            if fam not in POSITIVE_LABELS:
                continue
            score_v = float((g["due_weight"] * g["map_conf"]).max())
            weights.at[client, fam] = score_v
    return weights.fillna(0.0)


def build_ranker_training(
    streams: pd.DataFrame,
    labels: pd.Series,
    linker: FuzzyFamilyLinker,
) -> tuple[pd.DataFrame, pd.Series]:
    """One row per (client, candidate family); label=1 if family==true."""
    rows = stream_rows_with_families(streams, linker, filter_noise=True, due_only=True)
    if rows.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    records = []
    for client, group in rows.groupby("client_id", sort=False):
        if client not in labels.index:
            continue
        true = labels.loc[client]
        fam_feats = {}
        for fam, g in group.groupby("mapped_family"):
            if fam not in POSITIVE_LABELS:
                continue
            fam_feats[fam] = {
                "due_weight_max": float(g["due_weight"].max()),
                "due_weight_sum": float(g["due_weight"].sum()),
                "map_conf_max": float(g["map_conf"].max()),
                "n_streams": float(len(g)),
                "days_to_next_min": float(g["days_to_next"].min()),
                "appearances_max": float(g["appearances"].max()) if "appearances" in g else 0.0,
            }
        if not fam_feats:
            continue
        # also a none row
        for fam, feats in fam_feats.items():
            rec = {"client_id": client, "family": fam, **feats}
            rec["is_true"] = int(fam == true)
            records.append(rec)
        # none candidate always present
        records.append(
            {
                "client_id": client,
                "family": "none",
                "due_weight_max": 0.0,
                "due_weight_sum": 0.0,
                "map_conf_max": 0.0,
                "n_streams": 0.0,
                "days_to_next_min": 999.0,
                "appearances_max": 0.0,
                "is_true": int(true == "none"),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame, pd.Series(dtype=int)
    y = frame["is_true"].astype(int)
    return frame, y


FEATURE_COLS = [
    "due_weight_max",
    "due_weight_sum",
    "map_conf_max",
    "n_streams",
    "days_to_next_min",
    "appearances_max",
]


def add_family_onehots(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[FEATURE_COLS].copy()
    for fam in list(POSITIVE_LABELS) + ["none"]:
        out[f"fam_{fam}"] = (frame["family"] == fam).astype(float)
    return out


def predict_ranker(model, streams, clients, linker, push_proba=None):
    rows = stream_rows_with_families(streams, linker, filter_noise=True, due_only=True)
    # Build candidate feature rows per client
    records = []
    for client in clients:
        group = rows[rows["client_id"] == client] if not rows.empty else rows
        fam_feats = {}
        if not group.empty:
            for fam, g in group.groupby("mapped_family"):
                if fam not in POSITIVE_LABELS:
                    continue
                fam_feats[fam] = {
                    "due_weight_max": float(g["due_weight"].max()),
                    "due_weight_sum": float(g["due_weight"].sum()),
                    "map_conf_max": float(g["map_conf"].max()),
                    "n_streams": float(len(g)),
                    "days_to_next_min": float(g["days_to_next"].min()),
                    "appearances_max": float(g["appearances"].max()) if "appearances" in g else 0.0,
                }
        cands = list(fam_feats.keys()) + ["none"]
        for fam in cands:
            if fam == "none":
                feats = {
                    "due_weight_max": 0.0,
                    "due_weight_sum": 0.0,
                    "map_conf_max": 0.0,
                    "n_streams": 0.0,
                    "days_to_next_min": 999.0,
                    "appearances_max": 0.0,
                }
            else:
                feats = fam_feats[fam]
            records.append({"client_id": client, "family": fam, **feats})
    frame = pd.DataFrame(records)
    X = add_family_onehots(frame)
    scores = model.predict_proba(X)[:, 1]
    frame["score"] = scores
    # pick max score per client
    idx = frame.groupby("client_id")["score"].idxmax()
    pred = frame.loc[idx].set_index("client_id")["family"]
    # also soft probs
    pivot = frame.pivot_table(index="client_id", columns="family", values="score", aggfunc="max")
    pivot = pivot.reindex(columns=LABELS, fill_value=0.0).reindex(clients).fillna(0.0)
    # softmax over available
    logits = np.log(np.clip(pivot.to_numpy(), 1e-12, None))
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    proba = pd.DataFrame(w / w.sum(axis=1, keepdims=True), index=clients, columns=LABELS)
    return pred.reindex(clients), proba


def mix(a, b, w=0.5):
    m = w * a.to_numpy() + (1 - w) * b.reindex_like(a).to_numpy()
    m = m / m.sum(axis=1, keepdims=True)
    return pd.DataFrame(m, index=a.index, columns=a.columns)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yt = data.train_labels.set_index("client_id")[TARGET_COLUMN]
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}

    print("fitting Push baseline...", flush=True)
    push = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    push_proba = push.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    score(yv, push_proba.idxmax(axis=1), results, "push")

    print("building fuzzy linker...", flush=True)
    train_streams = build_streams(data.train_transactions)
    valid_streams = build_streams(data.valid_transactions)
    mapper = TrainOnlyFamilyMapper().fit(train_streams, data.train_labels)
    # Expand mapping with loose eligibility
    table = mapper.mapping_table_
    loose = table["support_clients"].ge(2) & table["winning_lift"].ge(1.15)
    mapping = table.loc[loose].set_index("description")["family"].dropna()
    # Also include eligible mapping_
    mapping = pd.concat([mapper.mapping_, mapping]).groupby(level=0).first()

    linker = FuzzyFamilyLinker(mapping, min_jaccard=0.45, min_char_cos=0.50)

    # Coverage diagnostic with fuzzy
    rows_va = stream_rows_with_families(valid_streams, linker)
    cand = rows_va.groupby("client_id")["mapped_family"].apply(lambda s: set(s.astype(str)))
    pos_cov = sum(1 for cid, lab in yv.items() if lab != "none" and lab in cand.get(cid, set()))
    n_pos = int((yv != "none").sum())
    print(
        f"fuzzy due coverage positives: {pos_cov}/{n_pos} = {pos_cov/max(n_pos,1):.3f}", flush=True
    )

    ceiling = []
    for cid, true in yv.items():
        fams = cand.get(cid, set())
        ceiling.append(true if (true in fams or true == "none") else "none")
    score(yv, pd.Series(ceiling, index=yv.index), results, "fuzzy_ceiling")

    # Pure fuzzy argmax weight
    weights = aggregate_family_weights(rows_va, yv.index)
    pred = weights.idxmax(axis=1).where(weights.max(axis=1).gt(0), "none")
    score(yv, pred, results, "fuzzy_argmax")

    # Learned ranker with OOF on train then fit on full train
    print("training stream ranker (OOF)...", flush=True)
    train_frame, train_y = build_ranker_training(train_streams, yt, linker)
    if train_frame.empty:
        print("empty ranker training — abort")
        return
    X_all = add_family_onehots(train_frame)
    # Client-level stratified OOF is hard; use simple LR fit on all train streams
    # (features are stream-derived, labels are client — mild leakage risk within train only)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.8, random_state=42)
    clf.fit(X_all, train_y)
    rank_pred, rank_proba = predict_ranker(clf, valid_streams, yv.index, linker)
    score(yv, rank_pred, results, "stream_ranker")

    for w in (0.4, 0.5, 0.6, 0.7, 0.8):
        mixed = mix(push_proba, rank_proba, w=w)
        score(yv, mixed.idxmax(axis=1), results, f"push_ranker_{w}")

    # Restrict push to fuzzy candidates
    has = weights.reindex(yv.index).fillna(0.0) > 0
    for temp in (1.0, 2.0, 4.0):
        logits = np.log(np.clip(push_proba.to_numpy(), 1e-12, 1))
        for fam in POSITIVE_LABELS:
            col = LABELS.index(fam)
            logits[~has[fam].to_numpy(), col] -= 10.0
            logits[:, col] += temp * np.log1p(weights[fam].reindex(yv.index).fillna(0).to_numpy())
        logits -= logits.max(axis=1, keepdims=True)
        ww = np.exp(logits)
        proba = pd.DataFrame(ww / ww.sum(axis=1, keepdims=True), index=yv.index, columns=LABELS)
        score(yv, proba.idxmax(axis=1), results, f"restrict_fuzzy_t{temp}")

    # Graph label propagation lite: co-occurrence of descriptions across train clients
    print("description co-occurrence propagation...", flush=True)
    # Build desc→family soft from train client labels via streams
    tr_rows = stream_rows_with_families(train_streams, linker, due_only=False, filter_noise=True)
    # For each description, empirical P(family) from clients who have it
    desc_counts = defaultdict(lambda: defaultdict(float))
    for client, group in tr_rows.groupby("client_id"):
        if client not in yt.index:
            continue
        lab = yt.loc[client]
        for desc in group["description"].unique():
            desc_counts[desc][lab] += 1.0
    desc_prior = {}
    for desc, counts in desc_counts.items():
        total = sum(counts.values()) + 1e-9
        desc_prior[desc] = {k: v / total for k, v in counts.items()}

    # Valid: average priors of client's due descriptions
    prop = pd.DataFrame(0.0, index=yv.index, columns=LABELS)
    va_all = valid_streams[valid_streams["is_candidate"]].copy()
    va_all = va_all[~va_all["description"].astype(str).isin(NOISE)]
    for client, group in va_all.groupby("client_id"):
        if client not in prop.index:
            continue
        acc = defaultdict(float)
        n = 0
        for desc in group["description"].astype(str):
            if desc in desc_prior:
                for k, v in desc_prior[desc].items():
                    acc[k] += v
                n += 1
            else:
                fam, conf = linker.map_one(desc)
                if fam:
                    acc[fam] += conf
                    n += 1
        if n == 0:
            acc["none"] = 1.0
            n = 1
        for k, v in acc.items():
            if k in LABELS:
                prop.at[client, k] = v / n
    prop = prop.div(prop.sum(axis=1).clip(lower=1e-12), axis=0)
    score(yv, prop.idxmax(axis=1), results, "desc_propagation")
    for w in (0.5, 0.65, 0.8):
        mixed = mix(push_proba, prop, w=w)
        score(yv, mixed.idxmax(axis=1), results, f"push_prop_{w}")

    ranking = sorted(((k, v["macro_f1"]) for k, v in results.items()), key=lambda x: -x[1])
    summary = {
        "seconds": perf_counter() - started,
        "fuzzy_pos_coverage": pos_cov / max(n_pos, 1),
        "top15": ranking[:15],
        "results": results,
    }
    (OUT / "ranker_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"top15": ranking[:15]}, indent=2))
    print(f"done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
