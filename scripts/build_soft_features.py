import argparse
import json
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT
from ubs_recurrence.price_prior import learn_price_profiles
from ubs_recurrence.streams import family_features
from ubs_recurrence.ranking import ranking_features


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tag",required=True);args=ap.parse_args()
    path=ROOT/"data/cache/unlabeled_price_profiles.json"
    profile=json.loads(path.read_text()) if path.exists() else learn_price_profiles()
    if args.tag=="original":
        T=pd.read_parquet(ROOT/"data/cache/templates_strict_train.parquet")
        S=pd.read_parquet(ROOT/"data/cache/streams_0.035_filtered_train.parquet")
    else:
        T=pd.read_parquet(ROOT/f"data/cache/templates_{args.tag}.parquet")
        S=pd.read_parquet(ROOT/f"data/cache/streams_{args.tag}.parquet")
    ids=T.index.to_numpy();F=family_features(S,ids,price_profiles=profile);X=ranking_features(F,T,ids,clocks=False)
    X.to_parquet(ROOT/f"data/cache/soft_ranking_{args.tag}.parquet")
    print(args.tag,"soft candidate features",X.shape,flush=True)


if __name__=="__main__":main()
