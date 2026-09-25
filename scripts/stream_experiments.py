import argparse
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from ubs_recurrence.data import ROOT, aligned_target, transactions
from ubs_recurrence.evaluation import metrics, record
from ubs_recurrence.streams import extract_streams, family_features
from ubs_recurrence.templates import template_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.035)
    ap.add_argument("--model", choices=["pooled", "direct"], default="pooled")
    ap.add_argument("--filter-background", action="store_true")
    args = ap.parse_args()
    start = time.perf_counter()
    df = transactions("train")
    ids = np.sort(df.client_id.unique())
    y = aligned_target(ids)
    suffix = "_filtered" if args.filter_background else ""
    cache = ROOT / f"data/cache/streams_{args.eps}{suffix}_train.parquet"
    if cache.exists():
        s = pd.read_parquet(cache)
    else:
        s = extract_streams(df, args.eps, args.filter_background)
        s.to_parquet(cache)
    print("Extracted", len(s), "streams", flush=True)
    cache = ROOT / f"data/cache/family_{args.eps}{suffix}_train.parquet"
    if cache.exists():
        X = pd.read_parquet(cache)
    else:
        X = family_features(s, ids)
        X.to_parquet(cache)
    print("Streams", s.shape, "Features", X.shape, flush=True)
    template = template_features(df, ids)
    globalx = X.unstack("family").reindex(ids)
    globalx.columns = ["__".join(c) for c in globalx.columns]
    globalx = pd.concat([globalx, template], axis=1).fillna(-999)
    ylong = np.repeat(y, 7) == np.tile(np.arange(7), len(ids))
    p = np.zeros((len(ids), 8))
    scores = []
    params = {
        "n_estimators": 300,
        "num_leaves": 15,
        "max_depth": -1,
        "learning_rate": 0.035,
        "min_child_samples": 30,
        "colsample_bytree": 0.85,
        "reg_lambda": 3,
        "n_jobs": 4,
        "verbosity": -1,
        "random_state": args.seed,
    }
    for fold, (tr, va) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(ids, y)
    ):
        if args.model == "pooled":
            it = (tr[:, None] * 7 + np.arange(7)).ravel()
            iv = (va[:, None] * 7 + np.arange(7)).ravel()
            model = LGBMClassifier(**params)
            model.fit(X.iloc[it], ylong[it])
            fam = model.predict_proba(X.iloc[iv])[:, 1].reshape(-1, 7)
            none = LGBMClassifier(**params)
            none.fit(globalx.iloc[tr], y[tr] == 7)
            pn = none.predict_proba(globalx.iloc[va])[:, 1]
            p[va, :7] = (
                fam
                / np.maximum(fam.sum(axis=1, keepdims=True), 1e-12)
                * (1 - pn[:, None])
            )
            p[va, 7] = pn
        else:
            model = LGBMClassifier(**params)
            model.fit(globalx.iloc[tr], y[tr])
            p[va] = model.predict_proba(globalx.iloc[va])
        score = metrics(y[va], p[va])["macro_f1"]
        scores.append(score)
        print(args.model, fold, score, flush=True)
    record(
        f"streams_{args.model}_e{args.eps}{suffix}_s{args.seed}",
        ids,
        y,
        p,
        runtime=time.perf_counter() - start,
        fold_scores=scores,
        metadata={
            "hypothesis": "Separate amount-stable latent streams and share recurrence-ranking rules across families",
            "features": "Amount-cluster and broad family recurrence, circular periodicity, relative family ranks, plus template evidence for none",
            "model": args.model + " LightGBM",
            "parameters": params,
            "seed": args.seed,
            "protocol": "5-fold stratified client CV; family rows grouped by client; official holdout sealed",
            "status": "evaluated",
            "conclusion": "Compare pooled candidate ranking and direct multiclass formulation",
        },
    )


if __name__ == "__main__":
    main()
