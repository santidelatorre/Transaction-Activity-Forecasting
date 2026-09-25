"""Execute fixed client-level CV baselines. No official validation labels."""
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from ubs_recurrence.data import ROOT, LABELS, transactions, aligned_target
from ubs_recurrence.features import documents, tabular
from ubs_recurrence.evaluation import record, metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",nargs="+",default=["majority","prior","word","char","enriched","tabular_lr","tabular_cat"])
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    df=transactions("train")
    ids=np.sort(df.client_id.unique())
    y=aligned_target(ids)
    splitter=StratifiedKFold(5,shuffle=True,random_state=args.seed)
    folds=list(splitter.split(ids,y))
    fold_frame=pd.DataFrame({"client_id":ids,"fold":-1})
    for f,(_,va) in enumerate(folds):
        fold_frame.loc[va,"fold"]=f
    fold_frame.to_csv(ROOT/f"outputs/folds_seed{args.seed}.csv",index=False)
    for name in args.models:
        start=time.perf_counter()
        if name.startswith("tabular"):
            path=ROOT/"data/cache/tabular_train.parquet"
            if not path.exists():
                tabular(df,ids).to_parquet(path)
            X=pd.read_parquet(path).reindex(ids)
        else:
            X=documents(df,ids,"enriched" if name=="enriched" else "raw")
        p=np.zeros((len(ids),8))
        fold_scores=[]
        for f,(tr,va) in enumerate(folds):
            if name in ["majority","prior"]:
                model=DummyClassifier(strategy="most_frequent" if name=="majority" else "prior")
                a,b=np.zeros((len(tr),1)),np.zeros((len(va),1))
            elif name in ["word","enriched","char"]:
                vec=TfidfVectorizer(analyzer="char_wb" if name=="char" else "word",ngram_range=(3,5) if name=="char" else (1,2),min_df=2,sublinear_tf=True,max_features=30000)
                model=make_pipeline(vec,LogisticRegression(C=3,max_iter=1200,class_weight="balanced",random_state=args.seed))
                a,b=X[tr],X[va]
            elif name=="tabular_lr":
                model=make_pipeline(SimpleImputer(missing_values=-999,add_indicator=True),StandardScaler(),LogisticRegression(C=.02,class_weight="balanced",max_iter=3000,random_state=args.seed))
                a,b=X.iloc[tr],X.iloc[va]
            elif name=="tabular_cat":
                model=CatBoostClassifier(iterations=550,depth=5,learning_rate=.055,loss_function="MultiClass",random_seed=args.seed,thread_count=6,verbose=False,allow_writing_files=False)
                a,b=X.iloc[tr],X.iloc[va]
            else:
                raise ValueError(name)
            model.fit(a,y[tr])
            p[np.ix_(va,model.classes_.astype(int))]=model.predict_proba(b)
            score=metrics(y[va],p[va])["macro_f1"]
            fold_scores.append(score)
            print(name,"fold",f,round(score,5),flush=True)
        record(f"baseline_{name}_s{args.seed}",ids,y,p,runtime=time.perf_counter()-start,fold_scores=fold_scores,metadata={
            "hypothesis":{"majority":"Measure majority-only floor","prior":"Measure empirical-prior argmax floor","word":"Description templates identify target families","char":"Character fragments recover noisy merchant wording","enriched":"MCC and text combinations disambiguate templates","tabular_lr":"Aggregated timing and family evidence is linearly useful","tabular_cat":"Nonlinear interactions among recurrence and semantic evidence improve prediction"}[name],
            "features":name,"model":type(model).__name__,"parameters":str(model.get_params()),"seed":args.seed,
            "protocol":"5-fold stratified client CV; all estimator fitting inside fold; official holdout sealed",
            "status":"evaluated","conclusion":"Exploratory baseline; compare with error analysis"})


if __name__=="__main__":
    main()
