"""Freeze the final candidate using train-side evidence, without new holdout access."""
import json
import time
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT,LABELS,aligned_target
from ubs_recurrence.evaluation import record


def main():
    start=time.perf_counter();results={}
    seeds=[42,17,2026]
    for scenario in ["original","valid_like","test_like"]:
        ps=[]
        for seed in seeds:
            d=pd.read_csv(ROOT/f"outputs/experiments/compact_l15_hierTrue_payments_{scenario}_s{seed}/predictions.csv")
            ids=d.client_id.to_numpy();ps.append(d[["p_"+l for l in LABELS]].to_numpy())
        y=aligned_target(ids);pure=np.mean(ps,axis=0)
        old=pd.read_csv(ROOT/f"outputs/experiments/ensemble_equal3_{scenario}/predictions.csv").set_index("client_id").loc[ids,["p_"+l for l in LABELS]].to_numpy()
        for name,p in [("compact_3seed",pure),("compact_3seed_plus25pct_v1",.75*pure+.25*old)]:
            r=record("selection_"+name+"_"+scenario,ids,y,p,runtime=time.perf_counter()-start,metadata={"hypothesis":"Averaging all three predeclared seeds stabilizes recurrence/refund evidence; compare a fixed legacy-model blend","features":"Compact recurrence and payment context with explicit none detector","model":name,"parameters":{"seeds":seeds,"legacy_weight":.25 if "plus" in name else 0},"seed":seeds,"protocol":"True client OOF in every constituent; seed ensemble includes every tested seed; no additional official validation access","status":"evaluated","conclusion":"Final train-side selection comparison"})
            results[(name,scenario)]=r["macro_f1"]
    gain=np.mean([results[("compact_3seed_plus25pct_v1",s)]-results[("compact_3seed",s)] for s in ["valid_like","test_like"]])
    config={"seeds":seeds,"min_count":3,"hierarchical":True,"payment_context":"all","augmentation":["original","valid_like","test_like"],"decision":"argmax","legacy_blend_mean_stress_gain":float(gain),"selection_rule":"Retain the larger legacy blend only if mean stress macro-F1 improves by at least 0.005; otherwise choose the simpler three-seed compact ensemble.","selected":"compact_3seed_plus25pct_v1" if gain>=.005 else "compact_3seed","holdout_v1_used":"Previous frozen holdout established distribution shift and failed to confirm the tuned decision biases. No per-client holdout errors were used to engineer the final changes.","target_0_80_achieved":False}
    (ROOT/"reports/final_candidate_config.json").write_text(json.dumps(config,indent=2));print(json.dumps(config,indent=2))


if __name__=="__main__":main()
