import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.special import softmax

from ubs_recurrence.data import LABELS, ROOT, aligned_target
from ubs_recurrence.evaluation import record


def main():
    start = time.perf_counter()
    aux = pd.read_parquet(ROOT / "data/cache/auxiliary_2000_ranking.parquet")
    auxids = aux.index.get_level_values(0).unique().to_numpy()
    ay = (
        pd.read_csv(ROOT / "data/cache/auxiliary_2000_labels.csv")
        .set_index("client_id")
        .loc[auxids, "auxiliary_target"]
        .map({v: k for k, v in enumerate(LABELS)})
        .to_numpy()
    )
    at = (np.repeat(ay, 8) == np.tile(np.arange(8), len(ay))).astype(int)
    aug = [
        pd.read_parquet(ROOT / f"data/cache/auxiliary_2000_{s}_ranking.parquet")
        .reindex(index=aux.index, columns=aux.columns)
        .fillna(-999)
        for s in ["valid_like", "test_like"]
    ]
    X = pd.concat([aux, *aug], ignore_index=True)
    yt = np.tile(at, 3)
    params = {
        "n_estimators": 450,
        "num_leaves": 15,
        "learning_rate": 0.035,
        "min_child_samples": 105,
        "colsample_bytree": 0.9,
        "reg_lambda": 5,
        "n_jobs": 4,
        "verbosity": -1,
        "random_state": 42,
        "objective": "lambdarank",
    }
    m = LGBMRanker(**params)
    m.fit(X, yt, group=np.full(len(yt) // 8, 8))
    joblib.dump(m, ROOT / "data/cache/auxiliary_expert.joblib")
    for scenario in ["original", "valid_like", "test_like"]:
        path = ROOT / (
            "data/cache/ranking_clocksFalse_idTrue_train.parquet"
            if scenario == "original"
            else f"data/cache/ranking_train_noise{scenario}_s2026.parquet"
        )
        z = pd.read_parquet(path).reindex(columns=aux.columns).fillna(-999)
        ids = z.index.get_level_values(0).unique().to_numpy()
        assert not set(ids) & set(auxids)
        p = softmax(m.predict(z).reshape(-1, 8), axis=1)
        y = aligned_target(ids)
        np.save(ROOT / f"data/cache/auxiliary_expert_{scenario}_train.npy", p)
        record(
            "auxiliary_only_expert_" + scenario,
            ids,
            y,
            p,
            runtime=time.perf_counter() - start,
            split="train_clients_independent_auxiliary_model",
            metadata={
                "hypothesis": "An independently pretrained historical-recurrence expert may transfer or provide a complementary signal",
                "features": "Historical auxiliary recurrence features with three noise views",
                "model": "LightGBM ranker trained ONLY on disjoint unlabeled-source clients",
                "parameters": params,
                "seed": 42,
                "protocol": "All 2000 challenge train clients are unseen by this model; challenge labels used only for evaluation",
                "status": "evaluated",
                "conclusion": "Evaluate direct transfer before considering the auxiliary expert as a learned feature",
            },
        )


if __name__ == "__main__":
    main()
