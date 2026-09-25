"""Fixed, interpretable template evidence discovered in the training audit.

These are transaction descriptions, not a learned mapping from client labels.
Ambiguous templates are represented separately with their observed MCC.
"""
import re
import numpy as np
import pandas as pd
from .data import CUTOFF, LABELS
from .features import MCC,GENERIC,normalize

TEMPLATES={
    "cloud":["cloud access","cloud backup","storage plan","service plan"],
    "gym":["gym membership","fitness monthly","fit club","urban gym"],
    "insurance":["insurance monthly","policy premium","safe cover","cover plan"],
    "mobile":["phone contract","service bill","monthly plan","digital plus"],
    "music":["audio streaming","member pass","digital plus","premium plan"],
    "software":["software access","saas billing","productivity suite","premium plan"],
    "streaming":["media streaming","video access","digital plus","premium plan"],
    "none":GENERIC,
}


def canonical(text):
    text=normalize(text)
    text=re.sub(r"^(?:pay|billing|member) ","",text) if text not in {"member pass","member plan"} else text
    for suffix in [" online"," digital"," core"," service"," plus"]:
        if text.endswith(suffix) and text not in {"digital plus","digital service"}:
            text=text[:-len(suffix)]
            break
    return text


def template_features(df,ids,mode="strict"):
    d=df[(df.direction=="out") & (df.type=="card_payment")].copy()
    d["desc"]=d.description.map(canonical if mode=="canonical" else normalize)
    d["age"]=(CUTOFF-d.timestamp).dt.total_seconds()/86400
    cols={}
    for f,templates in TEMPLATES.items():
        for scope in ["any","mcc"]:
            s=d if scope=="any" or f=="none" else d[d.mcc==MCC[f]]
            for window in [999,90]:
                t=s if window==999 else s[s.age<=window]
                counts=pd.crosstab(t.client_id,t.desc).reindex(index=ids,columns=templates,fill_value=0)
                for desc in templates:
                    cols[f"{f}__{scope}_{window}_{desc}"]=counts[desc].to_numpy()
                cols[f"{f}__{scope}_{window}_diversity"]=(counts>0).sum(axis=1).to_numpy()
                cols[f"{f}__{scope}_{window}_repeated"]=(counts>=2).sum(axis=1).to_numpy()
                cols[f"{f}__{scope}_{window}_sum"]=counts.sum(axis=1).to_numpy()
                cols[f"{f}__{scope}_{window}_max"]=counts.max(axis=1).to_numpy()
        for desc in templates:
            s=d[d.desc==desc]
            grouped=s.groupby("client_id")
            for name,values in {
                "amount_mean":grouped.amount.mean(),"amount_std":grouped.amount.std(),
                "first_age":grouped.age.max(),"last_age":grouped.age.min(),
            }.items():
                cols[f"{f}__{desc}_{name}"]=values.reindex(ids).fillna(-999).to_numpy()
    return pd.DataFrame(cols,index=pd.Index(ids,name="client_id"))
