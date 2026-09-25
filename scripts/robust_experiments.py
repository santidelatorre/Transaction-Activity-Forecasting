"""Client-grouped CV with artificial masking stress tests and training augmentation."""
import argparse
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from xgboost import XGBRanker
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from ubs_recurrence.data import ROOT,LABELS,aligned_target
from ubs_recurrence.evaluation import metrics,record


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--augment",action="store_true");ap.add_argument("--drop-text-stats",action="store_true");ap.add_argument("--seed",type=int,default=42);ap.add_argument("--noise",action="store_true");ap.add_argument("--auxiliary",type=int,default=0);ap.add_argument("--soft",action="store_true");ap.add_argument("--engine",choices=["lgbm","xgb"],default="lgbm");args=ap.parse_args()
    start=time.perf_counter();base=pd.read_parquet(ROOT/("data/cache/soft_ranking_original.parquet" if args.soft else "data/cache/ranking_clocksFalse_idTrue_train.parquet"))
    ids=base.index.get_level_values(0).unique().to_numpy();y=aligned_target(ids);target=(np.repeat(y,8)==np.tile(np.arange(8),len(ids))).astype(int)
    matrices={"original":base}
    if args.noise:
        for severity in ["valid_like","test_like"]:
            prefix="soft_ranking" if args.soft else "ranking"
            matrices[severity]=pd.read_parquet(ROOT/f"data/cache/{prefix}_train_noise{severity}_s2026.parquet").reindex(index=base.index,columns=base.columns).fillna(-999)
    else:
        for rate in [.3,.5]:matrices[f"mask{rate}"]=pd.read_parquet(ROOT/f"data/cache/ranking_train_mask{rate}_s2026.parquet").reindex(index=base.index,columns=base.columns).fillna(-999)
    if args.drop_text_stats:
        columns=[c for c in base if not any(t in c for t in ["template","description","generic","semantic"])]
        matrices={k:v[columns] for k,v in matrices.items()}
    train_keys=list(matrices) if args.augment else ["original"]
    auxiliary=None
    if args.auxiliary:
        auxiliary=pd.read_parquet(ROOT/f"data/cache/auxiliary_{args.auxiliary}_ranking.parquet").reindex(columns=next(iter(matrices.values())).columns).fillna(-999)
        auxids=auxiliary.index.get_level_values(0).unique()
        assert not set(auxids)&set(ids)
        auxy=pd.read_csv(ROOT/f"data/cache/auxiliary_{args.auxiliary}_labels.csv").set_index("client_id").loc[auxids,"auxiliary_target"].map({v:k for k,v in enumerate(LABELS)}).to_numpy()
        auxt=(np.repeat(auxy,8)==np.tile(np.arange(8),len(auxids))).astype(int)
    probs={k:np.zeros((len(ids),8)) for k in matrices};scores={k:[] for k in matrices}
    params=dict(n_estimators=450,num_leaves=15,learning_rate=.035,min_child_samples=35*(3 if args.augment else 1),colsample_bytree=.9,reg_lambda=5,n_jobs=4,verbosity=-1,random_state=args.seed,objective="lambdarank")
    if args.engine=="xgb":
        params=dict(n_estimators=500,max_depth=4,learning_rate=.04,min_child_weight=10,subsample=.85,colsample_bytree=.9,reg_lambda=10,n_jobs=4,random_state=args.seed,objective="rank:pairwise",tree_method="hist",device="cuda",lambdarank_pair_method="mean",lambdarank_num_pair_per_sample=4)
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ids,y)):
        it=(tr[:,None]*8+np.arange(8)).ravel();iv=(va[:,None]*8+np.arange(8)).ravel()
        X=pd.concat([matrices[k].iloc[it] for k in train_keys],ignore_index=True);yt=np.tile(target[it],len(train_keys))
        weights=np.ones(len(yt))
        if auxiliary is not None:
            X=pd.concat([X,auxiliary],ignore_index=True);yt=np.r_[yt,auxt];weights=np.r_[weights,np.full(len(auxt),.3)]
        if args.engine=="xgb":
            model=XGBRanker(**params);model.fit(X.astype(np.float32),yt,group=np.full(len(yt)//8,8),sample_weight=weights.reshape(-1,8).mean(axis=1))
        else:
            model=LGBMRanker(**params);model.fit(X,yt,group=np.full(len(yt)//8,8),sample_weight=weights)
        for k,matrix in matrices.items():
            probs[k][va]=softmax(model.predict(matrix.iloc[iv]).reshape(-1,8),axis=1);scores[k].append(metrics(y[va],probs[k][va])["macro_f1"])
        print("fold",fold,{k:round(v[-1],5) for k,v in scores.items()},flush=True)
    for key,p in probs.items():
        suffix=("_noise" if args.noise else "")+(f"_aux{args.auxiliary}" if args.auxiliary else "")+("_soft" if args.soft else "")+("_xgb" if args.engine=="xgb" else "")
        record(f"robust_aug{args.augment}_droptext{args.drop_text_stats}{suffix}_{key}_s{args.seed}",ids,y,p,runtime=time.perf_counter()-start,fold_scores=scores[key],metadata={
            "hypothesis":"Training-time description masking and text-statistic ablation improve robustness to the observed train/deployment shift",
            "features":f"Symmetric candidates; train augment={args.augment}; remove text statistics={args.drop_text_stats}; evaluation={key}","model":args.engine+" ranker","parameters":params,"seed":args.seed,
            "augmentation":{"full_noise":args.noise,"auxiliary_clients":args.auxiliary,"auxiliary_weight":.3 if args.auxiliary else 0,"soft_price_candidates":args.soft},
            "protocol":"5-fold stratified client CV; all augmented copies remain inside training folds; fixed feature-only masking seeds; official labels sealed",
            "status":"evaluated","conclusion":"Assess both clean and deployment-like stress OOF; neither is an official holdout result"})


if __name__=="__main__":main()
