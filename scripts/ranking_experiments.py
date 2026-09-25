import argparse
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

from ubs_recurrence.data import ROOT, aligned_target
from ubs_recurrence.evaluation import metrics, record
from ubs_recurrence.ranking import ranking_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["binary", "ranker"], default="ranker")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-clocks", action="store_true")
    ap.add_argument("--no-family-id", action="store_true")
    args = ap.parse_args()
    start = time.perf_counter()
    T = pd.read_parquet(ROOT / "data/cache/templates_strict_train.parquet")
    ids = T.index.to_numpy()
    y = aligned_target(ids)
    F = pd.read_parquet(ROOT / "data/cache/family_0.035_filtered_train.parquet")
    X = ranking_features(F, T, ids, clocks=not args.no_clocks)
    if args.no_family_id:
        X = X.drop(columns="family_index")
    X.to_parquet(
        ROOT
        / f"data/cache/ranking_clocks{not args.no_clocks}_id{not args.no_family_id}_train.parquet"
    )
    print("Ranking matrix", X.shape, flush=True)
    target = (np.repeat(y, 8) == np.tile(np.arange(8), len(ids))).astype(int)
    params = {
        "n_estimators": 450,
        "num_leaves": 15,
        "learning_rate": 0.035,
        "min_child_samples": 35,
        "colsample_bytree": 0.9,
        "reg_lambda": 5,
        "n_jobs": 4,
        "verbosity": -1,
        "random_state": args.seed,
    }
    p = np.zeros((len(ids), 8))
    scores = []
    importance = []
    for fold, (tr, va) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(ids, y)
    ):
        it = (tr[:, None] * 8 + np.arange(8)).ravel()
        iv = (va[:, None] * 8 + np.arange(8)).ravel()
        if args.model == "ranker":
            model = LGBMRanker(**params, objective="lambdarank")
            model.fit(X.iloc[it], target[it], group=np.full(len(tr), 8))
            p[va] = softmax(model.predict(X.iloc[iv]).reshape(-1, 8), axis=1)
        else:
            model = LGBMClassifier(**params)
            model.fit(X.iloc[it], target[it])
            q = model.predict_proba(X.iloc[iv])[:, 1].reshape(-1, 8)
            p[va] = q / q.sum(axis=1, keepdims=True)
        scores.append(metrics(y[va], p[va])["macro_f1"])
        importance.append(model.feature_importances_)
        print(args.model, fold, scores[-1], flush=True)
    eid = f"ranking_{args.model}_clocks{not args.no_clocks}_id{not args.no_family_id}_s{args.seed}"
    record(
        eid,
        ids,
        y,
        p,
        runtime=time.perf_counter() - start,
        fold_scores=scores,
        metadata={
            "hypothesis": "Joint family competition with a none candidate and clock-aware next-event phase improves ranking",
            "features": f"Symmetric eight-candidate features, clock={not args.no_clocks}, family identity={not args.no_family_id}",
            "model": args.model,
            "parameters": params,
            "seed": args.seed,
            "protocol": "5-fold stratified client CV; all 8 candidates of a client stay together; official holdout sealed",
            "status": "evaluated",
            "conclusion": "Compare ranking objective, symmetric binary classifier and clock ablation",
        },
    )
    pd.Series(np.mean(importance, axis=0), index=X.columns).sort_values(
        ascending=False
    ).to_csv(ROOT / f"outputs/experiments/{eid}/importance.csv")


if __name__ == "__main__":
    main()
