"""Client-grouped CV with artificial masking stress tests and training augmentation."""
import argparse
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from ubs_recurrence.data import ROOT,LABELS,aligned_target
from ubs_recurrence.evaluation import metrics,record


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--augment",action="store_true");ap.add_argument("--drop-text-stats",action="store_true");ap.add_argument("--seed",type=int,default=42);args=ap.parse_args()
    start=time.perf_counter();base=pd.read_parquet(ROOT/"data/cache/ranking_clocksFalse_idTrue_train.parquet")
    ids=base.index.get_level_values(0).unique().to_numpy();y=aligned_target(ids);target=(np.repeat(y,8)==np.tile(np.arange(8),len(ids))).astype(int)
    matrices={"original":base}
    for rate in [.3,.5]:matrices[f"mask{rate}"]=pd.read_parquet(ROOT/f"data/cache/ranking_train_mask{rate}_s2026.parquet").reindex(index=base.index,columns=base.columns).fillna(-999)
    if args.drop_text_stats:
        columns=[c for c in base if not any(t in c for t in ["template","description","generic","semantic"])]
        matrices={k:v[columns] for k,v in matrices.items()}
    train_keys=list(matrices) if args.augment else ["original"]
    probs={k:np.zeros((len(ids),8)) for k in matrices};scores={k:[] for k in matrices}
    params=dict(n_estimators=450,num_leaves=15,learning_rate=.035,min_child_samples=35*(3 if args.augment else 1),colsample_bytree=.9,reg_lambda=5,n_jobs=4,verbosity=-1,random_state=args.seed,objective="lambdarank")
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ids,y)):
        it=(tr[:,None]*8+np.arange(8)).ravel();iv=(va[:,None]*8+np.arange(8)).ravel()
        X=pd.concat([matrices[k].iloc[it] for k in train_keys],ignore_index=True);yt=np.tile(target[it],len(train_keys))
        model=LGBMRanker(**params);model.fit(X,yt,group=np.full(len(tr)*len(train_keys),8))
        for k,matrix in matrices.items():
            probs[k][va]=softmax(model.predict(matrix.iloc[iv]).reshape(-1,8),axis=1);scores[k].append(metrics(y[va],probs[k][va])["macro_f1"])
        print("fold",fold,{k:round(v[-1],5) for k,v in scores.items()},flush=True)
    for key,p in probs.items():
        record(f"robust_aug{args.augment}_droptext{args.drop_text_stats}_{key}_s{args.seed}",ids,y,p,runtime=time.perf_counter()-start,fold_scores=scores[key],metadata={
            "hypothesis":"Training-time description masking and text-statistic ablation improve robustness to the observed train/deployment shift",
            "features":f"Symmetric candidates; train augment={args.augment}; remove text statistics={args.drop_text_stats}; evaluation={key}","model":"LightGBM LambdaRank","parameters":params,"seed":args.seed,
            "protocol":"5-fold stratified client CV; all augmented copies remain inside training folds; fixed feature-only masking seeds; official labels sealed",
            "status":"evaluated","conclusion":"Assess both clean and deployment-like stress OOF; neither is an official holdout result"})


if __name__=="__main__":main()
