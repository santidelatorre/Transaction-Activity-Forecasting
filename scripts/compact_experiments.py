import argparse
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker,LGBMClassifier
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from ubs_recurrence.data import ROOT,LABELS,transactions,aligned_target
from ubs_recurrence.compact import global_features,compact_features
from ubs_recurrence.augmentation import corrupt_transactions
from ubs_recurrence.evaluation import metrics,record
from ubs_recurrence.payment_context import payment_context


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=42);ap.add_argument("--leaves",type=int,default=7);ap.add_argument("--hierarchical",action="store_true");ap.add_argument("--payment-context",action="store_true");ap.add_argument("--context-fields",choices=["all","refund","fee","time"],default="all");ap.add_argument("--sparse-candidates",action="store_true");ap.add_argument("--aux-expert",action="store_true");args=ap.parse_args()
    start=time.perf_counter();base=pd.read_parquet(ROOT/"data/cache/soft_ranking_original.parquet");ids=base.index.get_level_values(0).unique().to_numpy();y=aligned_target(ids);df=transactions("train")
    matrices={};globals={}
    for scenario in ["original","valid_like","test_like"]:
        d=df if scenario=="original" else corrupt_transactions(df,scenario,2026)
        globals[scenario]=global_features(d,ids)
        rank=base if scenario=="original" else pd.read_parquet(ROOT/f"data/cache/soft_ranking_train_noise{scenario}_s2026.parquet")
        if args.sparse_candidates:rank=pd.read_parquet(ROOT/f"data/cache/sparse_ranking_{scenario}_train.parquet")
        matrices[scenario]=compact_features(rank,globals[scenario]).reindex(index=base.index).fillna(-999)
        if args.payment_context:
            path=ROOT/f"data/cache/payment_context_{'sparse_' if args.sparse_candidates else ''}{scenario}_train.parquet"
            if path.exists():context=pd.read_parquet(path)
            else:context=payment_context(d,rank);context.to_parquet(path)
            if args.context_fields!="all":
                patterns={"refund":["refund"],"fee":["fee"],"time":["hour","weekend"]}[args.context_fields]
                context=context[[c for c in context if any(s in c for s in patterns)]]
            matrices[scenario]=pd.concat([matrices[scenario],context.reindex(base.index)],axis=1).fillna(-999)
            print("Payment context ready",scenario,flush=True)
        if args.aux_expert:
            prior=np.load(ROOT/f"data/cache/auxiliary_expert_{scenario}_train.npy")
            matrices[scenario]["auxiliary_probability"]=prior.ravel()
            matrices[scenario]["auxiliary_maximum"]=np.repeat(prior.max(axis=1),8)
            matrices[scenario]["auxiliary_none"]=np.repeat(prior[:,7],8)
        matrices[scenario].to_parquet(ROOT/f"data/cache/compact_{scenario}_train.parquet")
    columns=matrices["original"].columns
    matrices={k:v.reindex(columns=columns).fillna(-999) for k,v in matrices.items()}
    target=(np.repeat(y,8)==np.tile(np.arange(8),len(ids))).astype(int)
    p={k:np.zeros((len(ids),8)) for k in matrices};scores={k:[] for k in matrices}
    params=dict(n_estimators=500,num_leaves=args.leaves,learning_rate=.035,min_child_samples=120,reg_lambda=10,colsample_bytree=.95,n_jobs=4,verbosity=-1,random_state=args.seed)
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ids,y)):
        it=(tr[:,None]*8+np.arange(8)).ravel();iv=(va[:,None]*8+np.arange(8)).ravel();X=pd.concat([m.iloc[it] for m in matrices.values()],ignore_index=True);yt=np.tile(target[it],3)
        model=LGBMRanker(**params,objective="lambdarank");model.fit(X,yt,group=np.full(len(yt)//8,8))
        if args.hierarchical:
            # Separate none classifier sees only client-level evidence and symmetric
            # aggregates, not arbitrary family-slot ordering.
            aggregates={}
            for k,m in matrices.items():
                subset=m.drop(columns=["family_index","is_none"],errors="ignore").replace(-999,np.nan)
                agg=subset.groupby(level=0,sort=False).agg(["min","max","mean"]).reindex(ids).fillna(-999)
                agg.columns=["__".join(c) for c in agg.columns];aggregates[k]=agg
            nx=pd.concat([a.iloc[tr] for a in aggregates.values()],ignore_index=True)
            none=LGBMClassifier(**params);none.fit(nx,np.tile(y[tr]==7,3))
        for key,m in matrices.items():
            q=softmax(model.predict(m.iloc[iv]).reshape(-1,8),axis=1)
            if args.hierarchical:
                pn=none.predict_proba(aggregates[key].iloc[va])[:,1]
                q[:,:7]=q[:,:7]/q[:,:7].sum(axis=1,keepdims=True)*(1-pn[:,None]);q[:,7]=pn
            p[key][va]=q;scores[key].append(metrics(y[va],q)["macro_f1"])
        print("fold",fold,{k:round(s[-1],5) for k,s in scores.items()},flush=True)
    for scenario,prob in p.items():
        suffix=("_payments" if args.payment_context else "")+("_"+args.context_fields if args.context_fields!="all" else "")+("_sparse" if args.sparse_candidates else "")+("_auxexpert" if args.aux_expert else "")
        record(f"compact_l{args.leaves}_hier{args.hierarchical}{suffix}_{scenario}_s{args.seed}",ids,y,prob,runtime=time.perf_counter()-start,fold_scores=scores[scenario],metadata={"hypothesis":"A smaller physical recurrence representation and global transaction evidence reduce overfitting; refund patterns may distinguish continuing from terminated streams",
            "features":"Compact amount/cadence/family confidence + global type/count/activity features; three fixed noise views","model":"LightGBM ranker"+(" + binary none" if args.hierarchical else ""),"parameters":params,"seed":args.seed,
            "protocol":"5-fold grouped client OOF; all augmented copies confined to training clients; no official labels used","status":"evaluated","conclusion":"Train-side response to identified generalization weakness; holdout not used for tuning"})


if __name__=="__main__":main()
