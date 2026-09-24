"""Wild mathematical paths: description SVD embeddings, OT-like soft assignment, specialists."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.text_v2 import normalize_description
from transaction_forecasting.ubs.v3_push import StreamV3PushModel

OUT = Path("outputs/metrics/v3_math")
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


def client_docs(transactions, client_ids, *, drop_noise=True):
    frame = transactions.copy()
    frame["description"] = frame["description"].map(normalize_description)
    if drop_noise:
        frame = frame[~frame["description"].isin(NOISE)]
    docs = frame.groupby("client_id")["description"].agg(lambda s: " ".join(s.astype(str)))
    return docs.reindex(client_ids, fill_value="")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yt = data.train_labels.set_index("client_id")[TARGET_COLUMN]
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}

    push = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    push_proba = push.predict_components(data.valid_transactions)["blend"].reindex(yv.index)
    score(yv, push_proba.idxmax(axis=1), results, "push")

    # Char TF-IDF + SVD client documents (noise filtered)
    train_docs = client_docs(data.train_transactions, yt.index, drop_noise=True)
    valid_docs = client_docs(data.valid_transactions, yv.index, drop_noise=True)
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000),
        TruncatedSVD(n_components=64, random_state=42),
    )
    Xtr = pipe.fit_transform(train_docs)
    Xva = pipe.transform(valid_docs)
    Xtr = normalize(Xtr)
    Xva = normalize(Xva)

    # Nearest centroid in embedding space
    nc = NearestCentroid()
    nc.fit(Xtr, yt.to_numpy())
    pred_nc = pd.Series(nc.predict(Xva), index=yv.index)
    score(yv, pred_nc, results, "embed_nearest_centroid")

    # Softmax logistic on embeddings
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    lr.fit(Xtr, yt.to_numpy())
    embed_proba = pd.DataFrame(lr.predict_proba(Xva), index=yv.index, columns=lr.classes_)
    embed_proba = embed_proba.reindex(columns=LABELS, fill_value=0.0)
    score(yv, embed_proba.idxmax(axis=1), results, "embed_logistic")

    # Mix with push
    for w in (0.5, 0.65, 0.75, 0.85):
        mix = w * push_proba.to_numpy() + (1 - w) * embed_proba.to_numpy()
        mix = mix / mix.sum(axis=1, keepdims=True)
        pred = pd.Series(np.asarray(LABELS)[mix.argmax(1)], index=yv.index)
        score(yv, pred, results, f"push_embed_{w}")

    # Per-class specialist: binary one-vs-rest on embeddings, then pick max margin vs none
    specialist = {}
    for family in LABELS:
        ybin = yt.eq(family).astype(int)
        clf = LogisticRegression(max_iter=800, class_weight="balanced", C=0.5, random_state=42)
        clf.fit(Xtr, ybin)
        specialist[family] = clf.predict_proba(Xva)[:, 1]
    spec = pd.DataFrame(specialist, index=yv.index)[list(LABELS)]
    spec = spec.div(spec.sum(axis=1), axis=0)
    score(yv, spec.idxmax(axis=1), results, "ovr_specialists")
    for w in (0.5, 0.7):
        mix = w * push_proba.to_numpy() + (1 - w) * spec.to_numpy()
        mix = mix / mix.sum(axis=1, keepdims=True)
        pred = pd.Series(np.asarray(LABELS)[mix.argmax(1)], index=yv.index)
        score(yv, pred, results, f"push_ovr_{w}")

    # Sinkhorn-ish soft assignment: treat push logits as costs, entropy-regularize toward embed
    for eps in (0.3, 0.7, 1.5):
        a = np.log(np.clip(push_proba.to_numpy(), 1e-12, 1))
        b = np.log(np.clip(embed_proba.to_numpy(), 1e-12, 1))
        # Soft geometric interpolant via temperature
        z = (a + eps * b) / (1 + eps)
        z -= z.max(axis=1, keepdims=True)
        w = np.exp(z)
        proba = w / w.sum(axis=1, keepdims=True)
        pred = pd.Series(np.asarray(LABELS)[proba.argmax(1)], index=yv.index)
        score(yv, pred, results, f"sinkhornish_{eps}")

    # Amount fingerprint: median |amount| histogram soft match to class prototypes
    def amount_hist(transactions, clients):
        frame = transactions.copy()
        frame = frame[frame.direction.eq("out")]
        frame["bin"] = pd.cut(
            frame.amount.abs(),
            bins=[0, 5, 10, 15, 25, 50, 100, 250, 1e9],
            labels=False,
            include_lowest=True,
        )
        pivot = (
            frame.pivot_table(index="client_id", columns="bin", values="amount", aggfunc="size")
            .reindex(clients)
            .fillna(0.0)
        )
        pivot = pivot.div(pivot.sum(axis=1).clip(lower=1), axis=0)
        return pivot

    a_tr = amount_hist(data.train_transactions, yt.index)
    a_va = amount_hist(data.valid_transactions, yv.index).reindex(
        columns=a_tr.columns, fill_value=0
    )
    # Class mean histograms
    proto = pd.DataFrame({lab: a_tr.loc[yt.eq(lab)].mean() for lab in LABELS}).T
    # Cosine similarity client↔prototype
    A = normalize(a_va.to_numpy())
    P = normalize(proto.to_numpy())
    sims = A @ P.T
    sims = np.clip(sims, 0, None)
    sims = sims / sims.sum(axis=1, keepdims=True).clip(min=1e-12)
    amount_proba = pd.DataFrame(sims, index=yv.index, columns=LABELS)
    score(yv, amount_proba.idxmax(axis=1), results, "amount_proto")
    for w in (0.85, 0.9):
        mix = w * push_proba.to_numpy() + (1 - w) * amount_proba.to_numpy()
        mix = mix / mix.sum(axis=1, keepdims=True)
        pred = pd.Series(np.asarray(LABELS)[mix.argmax(1)], index=yv.index)
        score(yv, pred, results, f"push_amount_{w}")

    ranking = sorted(((k, v["macro_f1"]) for k, v in results.items()), key=lambda x: -x[1])
    summary = {"seconds": perf_counter() - started, "top15": ranking[:15], "results": results}
    (OUT / "math_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"top15": ranking[:15]}, indent=2))


if __name__ == "__main__":
    main()
