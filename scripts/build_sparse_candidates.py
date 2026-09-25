import argparse
import json

import numpy as np

from ubs_recurrence.augmentation import corrupt_transactions
from ubs_recurrence.data import ROOT, transactions
from ubs_recurrence.ranking import ranking_features
from ubs_recurrence.streams import extract_streams, family_features
from ubs_recurrence.templates import template_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario", choices=["original", "valid_like", "test_like"], required=True
    )
    args = ap.parse_args()
    d = transactions("train")
    if args.scenario != "original":
        d = corrupt_transactions(d, args.scenario, 2026)
    ids = np.sort(d.client_id.unique())
    profile = json.loads(
        (ROOT / "data/cache/unlabeled_price_profiles.json").read_text()
    )
    T = template_features(d, ids)
    S = extract_streams(d, 0.035, True, min_count=2)
    F = family_features(S, ids, price_profiles=profile)
    X = ranking_features(F, T, ids, clocks=False)
    X.to_parquet(ROOT / f"data/cache/sparse_ranking_{args.scenario}_train.parquet")
    print(args.scenario, X.shape, "streams", len(S), flush=True)


if __name__ == "__main__":
    main()
