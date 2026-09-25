"""Derive explicit self-supervised next-observed-stream labels from pretraining histories.

Auxiliary labels are NOT challenge labels. Future observations are used only
to construct targets; feature histories stop at the historical cutoff.
"""
import argparse
import json
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT,CUTOFF,transactions,LABELS
from ubs_recurrence.features import normalize,PATTERNS,MCC
from ubs_recurrence.streams import amount_components,extract_streams,family_features
from ubs_recurrence.templates import template_features
from ubs_recurrence.ranking import ranking_features
from ubs_recurrence.augmentation import mask_descriptions


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--clients",type=int,default=2000);args=ap.parse_args()
    historical_cutoff=pd.Timestamp("2025-10-01",tz="UTC")
    full=transactions("unlabeled_pretrain");allids=np.sort(full.client_id.unique())
    ids=np.sort(np.random.default_rng(883).choice(allids,size=args.clients,replace=False));full=full[full.client_id.isin(ids)].copy()
    labels=[]
    for cid,g in full.groupby("client_id"):
        candidates=[]
        card=g[(g.type=="card_payment")&(g.direction=="out")].copy()
        card["description"]=card.description.map(normalize)
        for currency,cg in card.groupby("currency"):
            for component in amount_components(cg.amount.to_numpy(),.035):
                s=cg.iloc[component].sort_values("timestamp")
                before=s[s.timestamp<historical_cutoff];after=s[(s.timestamp>=historical_cutoff)&(s.timestamp<historical_cutoff+pd.Timedelta(days=90))]
                if len(before)<3 or len(after)==0:continue
                gaps=before.timestamp.diff().dropna().dt.total_seconds().to_numpy()/86400
                if not 7<=np.median(gaps)<=100 or np.std(gaps)/np.mean(gaps)>.4:continue
                scores=np.array([s.description.str.contains(PATTERNS[f].replace("(","(?:"),regex=True).mean()+.25*(s.mcc==MCC[f]).mean() for f in LABELS[:-1]])
                if scores.max()<=.25:continue
                family=LABELS[int(scores.argmax())]
                candidates.append((after.timestamp.min(),family))
        label=min(candidates)[1] if candidates else "none"
        labels.append({"client_id":cid,"auxiliary_target":label,"historical_cutoff":str(historical_cutoff)})
    tag=f"auxiliary_{args.clients}"
    y=pd.DataFrame(labels).set_index("client_id").reindex(ids);y.to_csv(ROOT/f"data/cache/{tag}_labels.csv")
    print("Auxiliary distribution",y.auxiliary_target.value_counts().to_dict(),flush=True)
    before=full[full.timestamp<historical_cutoff].copy()
    assert before.timestamp.max()<historical_cutoff
    # Shift the origin, preserving time intervals; the shift is 92 days.
    # Calendar weekday changes by one day; therefore this is an approximation whose
    # contribution must be measured. No future event enters features.
    before["timestamp"]=before.timestamp+(CUTOFF-historical_cutoff)
    before=mask_descriptions(before,.3,1183)
    T=template_features(before,ids);S=extract_streams(before,.035,True);F=family_features(S,ids);X=ranking_features(F,T,ids,clocks=False)
    X.to_parquet(ROOT/f"data/cache/{tag}_ranking.parquet")
    (ROOT/f"outputs/{tag}_provenance.json").write_text(json.dumps({"source":"unlabeled_pretrain_transactions.jsonl","historical_cutoff":str(historical_cutoff),"horizon_days":90,"clients":len(ids),"target_definition":"Earliest future observed amount-stable family with >=3 prior events, median interval 7..100 days and gap CV <=0.4","limitations":"Weak labels depend on stream recovery; not official target generation; weekday origin shifts by one day"},indent=2))
    print("Auxiliary features complete",X.shape,flush=True)


if __name__=="__main__":main()
