import argparse
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT,CUTOFF,transactions
from ubs_recurrence.augmentation import corrupt_transactions
from ubs_recurrence.templates import template_features
from ubs_recurrence.streams import extract_streams,family_features
from ubs_recurrence.ranking import ranking_features


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--severity",required=True,choices=["valid_like","test_like"]);args=ap.parse_args()
    ids=pd.read_csv(ROOT/"data/cache/auxiliary_2000_labels.csv").client_id.to_numpy()
    cutoff=pd.Timestamp("2025-10-01",tz="UTC");d=transactions("unlabeled_pretrain");d=d[d.client_id.isin(ids)&(d.timestamp<cutoff)].copy();d["timestamp"]+=CUTOFF-cutoff
    d=corrupt_transactions(d,args.severity,1183);T=template_features(d,ids);S=extract_streams(d,.035,True);F=family_features(S,ids);X=ranking_features(F,T,ids,clocks=False)
    X.to_parquet(ROOT/f"data/cache/auxiliary_2000_{args.severity}_ranking.parquet");print(args.severity,X.shape,flush=True)


if __name__=="__main__":main()
