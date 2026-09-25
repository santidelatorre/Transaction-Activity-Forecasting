"""First, pre-registered official holdout batch; write predictions before opening labels."""
from datetime import datetime,timezone
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from xgboost import XGBRanker
from scipy.special import softmax
from ubs_recurrence.data import ROOT,LABELS,aligned_target
from ubs_recurrence.evaluation import record,source_fingerprint
from ubs_recurrence.decision import apply_bias


def main():
    start=time.perf_counter();config=json.loads((ROOT/"reports/frozen_candidate_v1.json").read_text())
    out=ROOT/"outputs/frozen_v1";out.mkdir(parents=True,exist_ok=False)
    base=pd.read_parquet(ROOT/"data/cache/ranking_clocksFalse_idTrue_train.parquet")
    ids=base.index.get_level_values(0).unique().to_numpy();y=aligned_target(ids)
    ylong=(np.repeat(y,8)==np.tile(np.arange(8),len(ids))).astype(int)
    ps=[]
    param=dict(n_estimators=450,num_leaves=15,learning_rate=.035,min_child_samples=105,colsample_bytree=.9,reg_lambda=5,n_jobs=4,verbosity=-1,random_state=42,objective="lambdarank")
    validation_ids=None
    for kind in ["hard","soft","xgb","control"]:
        soft=kind in {"soft","xgb"}
        if soft:
            train=pd.read_parquet(ROOT/"data/cache/soft_ranking_original.parquet")
            valid=pd.read_parquet(ROOT/"data/cache/soft_ranking_valid_mask0.0_s2026.parquet")
        else:
            train=base
            valid=pd.read_parquet(ROOT/"data/cache/ranking_valid_mask0.0_s2026.parquet")
        valid=valid.reindex(columns=train.columns).fillna(-999)
        validation_ids=valid.index.get_level_values(0).unique().to_numpy()
        if kind!="control":
            prefix="soft_ranking" if soft else "ranking"
            aug=[pd.read_parquet(ROOT/f"data/cache/{prefix}_train_noise{s}_s2026.parquet").reindex(index=train.index,columns=train.columns).fillna(-999) for s in ["valid_like","test_like"]]
            fit=pd.concat([train,*aug],ignore_index=True);yt=np.tile(ylong,3)
        else:fit=train;yt=ylong
        if kind=="xgb":
            model=XGBRanker(n_estimators=500,max_depth=4,learning_rate=.04,min_child_weight=10,subsample=.85,colsample_bytree=.9,reg_lambda=10,n_jobs=4,random_state=42,objective="rank:pairwise",tree_method="hist",device="cuda",lambdarank_pair_method="mean",lambdarank_num_pair_per_sample=4)
            model.fit(fit.astype(np.float32),yt,group=np.full(len(yt)//8,8))
        else:
            kwargs={**param,"min_child_samples":35 if kind=="control" else 105}
            model=LGBMRanker(**kwargs);model.fit(fit,yt,group=np.full(len(yt)//8,8))
        logits=model.predict(valid).reshape(-1,8).astype(float);p=softmax(logits,axis=1)
        joblib.dump({"model":model,"columns":train.columns.tolist(),"kind":kind},out/f"model_{kind}.joblib")
        pred=pd.DataFrame(p,index=validation_ids,columns=LABELS);pred.index.name="client_id";pred.to_csv(out/f"valid_{kind}.csv")
        if kind!="control":ps.append(p)
        else:control=p
        print("Frozen predictions created:",kind,flush=True)
    ensemble=np.mean(ps,axis=0)
    variants={"control":control,"ensemble":ensemble,"calibrated":apply_bias(ensemble,config["bias"])}
    for name,p in variants.items():
        pred=pd.DataFrame(p,index=validation_ids,columns=LABELS);pred.index.name="client_id";pred.to_csv(out/f"valid_frozen_{name}.csv")
    freeze={"timestamp":datetime.now(timezone.utc).isoformat(),"source_sha256":source_fingerprint(),"selection":"Primary candidate is the calibrated equal 3-model robustness ensemble, selected using train OOF stress only. Control and uncalibrated ensemble are predeclared comparators.","files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("valid*.csv")},"config":config}
    (ROOT/"reports/holdout_v1_freeze.json").write_text(json.dumps(freeze,indent=2))
    # This is the FIRST semantic read of official validation labels in this project.
    with (ROOT/"reports/holdout_access_log.jsonl").open("a") as f:
        f.write(json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),"batch":"v1","purpose":"Evaluate frozen finalists after train-only model/decision selection","variants":list(variants),"predictions_frozen_before_labels":True})+"\n")
    target=aligned_target(validation_ids,"valid",allow_holdout=True)
    for name,p in variants.items():
        record("holdout_v1_"+name,validation_ids,target,p,runtime=time.perf_counter()-start,split="official_validation",metadata={"hypothesis":"Train-only robustness selection should transfer to the official distribution shift","features":name,"model":"Frozen v1 "+name,"parameters":config,"seed":42,"protocol":"Official holdout; all model predictions, ensemble weights and decision biases frozen before reading labels","status":"evaluated","conclusion":"First official holdout batch; compare with pre-registered train-side stress expectations"})


if __name__=="__main__":main()
