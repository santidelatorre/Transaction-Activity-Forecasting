import time
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT,LABELS,aligned_target
from ubs_recurrence.evaluation import record


def main():
    x=pd.read_parquet(ROOT/"data/cache/templates_strict_train.parquet")
    ids=x.index.to_numpy();y=aligned_target(ids)
    start=time.perf_counter()
    diversity=x[[f+"__mcc_999_diversity" for f in LABELS]].to_numpy().argmax(axis=1)
    record("rule_template_diversity",ids,y,np.eye(8)[diversity],runtime=time.perf_counter()-start,metadata={"hypothesis":"The most diverse merchant-family template set identifies the target","features":"Exact template diversity with MCC agreement","model":"Deterministic argmax rule","parameters":{},"seed":None,"protocol":"Fixed label-free rule evaluated on all training clients; no fitted parameters","status":"rejected","conclusion":"Template diversity alone is inadequate; ties use fixed label order"})
    s=pd.read_parquet(ROOT/"data/cache/streams_0.035_train.parquet")
    for name in ["most_frequent","most_recent","minimum_next","regularity"]:
        start=time.perf_counter()
        c=s[(s.group_type=="amount")&(s.family_score>.25)&(s.gap_median.between(5,100))&(s.last_age < s.gap_median*1.6)&(s.next_median<90)].copy()
        if name=="most_frequent":c["score"]=-c["count"]
        elif name=="most_recent":c["score"]=c.last_age
        elif name=="minimum_next":c["score"]=c.next_median.clip(lower=0)
        else:c["score"]=c.gap_cv
        pred=c.sort_values("score",kind="stable").drop_duplicates("client_id").set_index("client_id").family.reindex(ids).fillna("none")
        py=pred.map({v:k for k,v in enumerate(LABELS)}).to_numpy()
        record("rule_stream_"+name,ids,y,np.eye(8)[py],runtime=time.perf_counter()-start,metadata={"hypothesis":"An explicit recurrence rule can identify the next family","features":"Amount-neighborhood streams with family evidence and active cadence","model":name,"parameters":{"max_age_period_ratio":1.6,"min_count":3,"period_days":[5,100],"horizon_days":90},"seed":None,"protocol":"Fixed label-free rule; all training clients retained; no fitted parameters","status":"evaluated","conclusion":"Compare frequency, recency, next-date and regularity ranking"})


if __name__=="__main__":main()
