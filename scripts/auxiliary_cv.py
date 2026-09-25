"""Diagnostic validation on the explicit auxiliary historical task, not challenge labels."""

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

from ubs_recurrence.data import LABELS, ROOT
from ubs_recurrence.evaluation import metrics, record


def main():
    start = time.perf_counter()
    X = pd.read_parquet(ROOT / "data/cache/auxiliary_2000_ranking.parquet")
    ids = X.index.get_level_values(0).unique().to_numpy()
    y = (
        pd.read_csv(ROOT / "data/cache/auxiliary_2000_labels.csv")
        .set_index("client_id")
        .loc[ids, "auxiliary_target"]
        .map({v: k for k, v in enumerate(LABELS)})
        .to_numpy()
    )
    target = (np.repeat(y, 8) == np.tile(np.arange(8), len(ids))).astype(int)
    p = np.zeros((len(ids), 8))
    scores = []
    params = {
        "n_estimators": 450,
        "num_leaves": 15,
        "learning_rate": 0.035,
        "min_child_samples": 35,
        "colsample_bytree": 0.9,
        "reg_lambda": 5,
        "n_jobs": 3,
        "verbosity": -1,
        "random_state": 42,
        "objective": "lambdarank",
    }
    for fold, (tr, va) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y)
    ):
        it = (tr[:, None] * 8 + np.arange(8)).ravel()
        iv = (va[:, None] * 8 + np.arange(8)).ravel()
        m = LGBMRanker(**params)
        m.fit(X.iloc[it], target[it], group=np.full(len(tr), 8))
        p[va] = softmax(m.predict(X.iloc[iv]).reshape(-1, 8), axis=1)
        scores.append(metrics(y[va], p[va])["macro_f1"])
        print(fold, scores[-1], flush=True)
    record(
        "auxiliary_historical_cv_2000",
        ids,
        y,
        p,
        runtime=time.perf_counter() - start,
        fold_scores=scores,
        split="auxiliary_task_oof_NOT_challenge_validation",
        metadata={
            "hypothesis": "Check predictability of the explicitly reconstructed historical next-event task",
            "features": "Pre-2025-10-01 histories; fixed 30% text masking; later events only construct weak auxiliary targets",
            "model": "LightGBM LambdaRank",
            "parameters": params,
            "seed": 42,
            "protocol": "Five-fold client CV on unlabeled-source auxiliary examples; no challenge labels; no future events in features",
            "status": "diagnostic_only",
            "conclusion": "This score does not measure challenge performance and cannot satisfy the 0.80 objective",
        },
    )


if __name__ == "__main__":
    main()
