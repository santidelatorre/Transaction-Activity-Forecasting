import argparse
import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from ubs_recurrence.data import ROOT, aligned_target, transactions
from ubs_recurrence.evaluation import metrics, record
from ubs_recurrence.templates import template_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="strict", choices=["strict", "canonical"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--subset",
        default="all",
        choices=["all", "counts", "diversity", "timing", "amount"],
    )
    args = ap.parse_args()
    start = time.perf_counter()
    d = transactions("train")
    ids = np.sort(d.client_id.unique())
    y = aligned_target(ids)
    X = template_features(d, ids, args.mode)
    X.to_parquet(ROOT / f"data/cache/templates_{args.mode}_train.parquet")
    if args.subset == "counts":
        X = X[[c for c in X if not any(s in c for s in ["age", "amount"])]]
    elif args.subset == "diversity":
        X = X[
            [
                c
                for c in X
                if any(s in c for s in ["diversity", "repeated", "_sum", "_max"])
            ]
        ]
    elif args.subset == "timing":
        X = X[[c for c in X if "age" in c]]
    elif args.subset == "amount":
        X = X[[c for c in X if "amount" in c]]
    p = np.zeros((len(ids), 8))
    scores = []
    for fold, (tr, va) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=args.seed).split(ids, y)
    ):
        m = CatBoostClassifier(
            iterations=600,
            depth=5,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=args.seed,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
        )
        m.fit(X.iloc[tr], y[tr])
        p[va] = m.predict_proba(X.iloc[va])
        score = metrics(y[va], p[va])["macro_f1"]
        scores.append(score)
        print(args.mode, args.subset, fold, score, flush=True)
    record(
        f"templates_{args.mode}_{args.subset}_s{args.seed}",
        ids,
        y,
        p,
        runtime=time.perf_counter() - start,
        fold_scores=scores,
        metadata={
            "hypothesis": "Within-family template diversity distinguishes recurring target streams from distractors",
            "features": f"{args.mode} fixed description/MCC templates; {args.subset}",
            "model": "CatBoostClassifier",
            "parameters": m.get_params(),
            "seed": args.seed,
            "protocol": "5-fold stratified client CV; fixed label-free features; official holdout sealed",
            "status": "evaluated",
            "conclusion": "Compare template structure against text and cadence baselines",
        },
    )


if __name__ == "__main__":
    main()
