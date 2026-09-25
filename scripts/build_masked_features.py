import argparse
import time

import numpy as np

from ubs_recurrence.augmentation import corrupt_transactions, mask_descriptions
from ubs_recurrence.data import ROOT, transactions
from ubs_recurrence.ranking import ranking_features
from ubs_recurrence.streams import extract_streams, family_features
from ubs_recurrence.templates import template_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split", default="train")
    ap.add_argument("--noise", choices=["valid_like", "test_like"])
    args = ap.parse_args()
    start = time.perf_counter()
    tag = (
        f"{args.split}_noise{args.noise}_s{args.seed}"
        if args.noise
        else f"{args.split}_mask{args.rate}_s{args.seed}"
    )
    d = (
        corrupt_transactions(transactions(args.split), args.noise, args.seed)
        if args.noise
        else mask_descriptions(transactions(args.split), args.rate, args.seed)
    )
    ids = np.sort(d.client_id.unique())
    T = template_features(d, ids)
    T.to_parquet(ROOT / f"data/cache/templates_{tag}.parquet")
    print(tag, "templates ready", flush=True)
    S = extract_streams(d, 0.035, True)
    S.to_parquet(ROOT / f"data/cache/streams_{tag}.parquet")
    print(tag, "streams", len(S), flush=True)
    F = family_features(S, ids)
    F.to_parquet(ROOT / f"data/cache/family_{tag}.parquet")
    X = ranking_features(F, T, ids, clocks=False)
    X.to_parquet(ROOT / f"data/cache/ranking_{tag}.parquet")
    print(tag, "completed", X.shape, "seconds", time.perf_counter() - start, flush=True)


if __name__ == "__main__":
    main()
