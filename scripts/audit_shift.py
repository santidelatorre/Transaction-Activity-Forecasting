"""Feature-only adversarial validation. Never reads validation or test labels."""
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency
from ubs_recurrence.data import ROOT,transactions,labels
from ubs_recurrence.templates import template_features


def main():
    matrices={}
    for split in ["train","valid","test"]:
        d=transactions(split);ids=np.sort(d.client_id.unique());x=template_features(d,ids)
        x["transaction_count"]=d.groupby("client_id").size().reindex(ids)
        g=d.groupby("client_id").timestamp
        x["history_days"]=(g.max()-g.min()).dt.total_seconds()/86400
        matrices[split]=x
    report={}
    for a,b in [("train","valid"),("train","test")]:
        X=pd.concat([matrices[a],matrices[b]],ignore_index=True).fillna(-999)
        y=np.r_[np.zeros(len(matrices[a])),np.ones(len(matrices[b]))]
        p=np.zeros(len(y));scores=[];importance=[]
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=931).split(X,y):
            model=LGBMClassifier(n_estimators=100,num_leaves=7,min_child_samples=40,learning_rate=.04,reg_lambda=5,n_jobs=2,verbosity=-1,random_state=931)
            model.fit(X.iloc[tr],y[tr]);p[va]=model.predict_proba(X.iloc[va])[:,1];scores.append(roc_auc_score(y[va],p[va]));importance.append(model.feature_importances_)
        report[a+":"+b]={"oof_auc":roc_auc_score(y,p),"fold_auc":scores,"n_a":len(matrices[a]),"n_b":len(matrices[b]),"top_features":pd.Series(np.mean(importance,axis=0),index=X.columns).sort_values(ascending=False).head(15).to_dict()}
    y=labels();numeric=y.index.str[1:].astype(int)
    ct=pd.crosstab(pd.qcut(numeric,4),y.target_next_recurring_merchant.to_numpy())
    chi,pvalue,dof,_=chi2_contingency(ct)
    report["identifier_audit"]={"quartile_class_chi_square":float(chi),"pvalue":float(pvalue),"dof":dof,"policy":"Identifiers and row ordering excluded regardless of association"}
    (ROOT/"reports/distribution_shift.json").write_text(json.dumps(report,indent=2))
    for key,value in report.items():print(key,value,flush=True)


if __name__=="__main__":main()
