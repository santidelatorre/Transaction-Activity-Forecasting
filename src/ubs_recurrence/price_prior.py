"""Weakly supervised family price ranges learned only from unlabeled histories."""
import json
import numpy as np
from .data import ROOT,LABELS,transactions
from .features import PATTERNS,MCC


def learn_price_profiles(*,use_cache=True):
    d=transactions("unlabeled_pretrain",use_cache=use_cache)
    d=d[(d.type=="card_payment")&(d.direction=="out")]
    result={}
    for f in LABELS[:-1]:
        s=d[d.description.str.contains(PATTERNS[f].replace("(","(?:"),regex=True)&(d.mcc==MCC[f])].amount
        result[f]={"low":float(s.quantile(.005)),"high":float(s.quantile(.995)),"median":float(s.median()),"n_anchor_events":len(s)}
    path=ROOT/"data/cache/unlabeled_price_profiles.json";path.write_text(json.dumps(result,indent=2))
    return result


def price_support(amount, profile):
    """Smooth plausibility, not a calibrated family probability."""
    amount=np.asarray(amount,dtype=float)
    distance=np.maximum(np.log(profile["low"]/np.maximum(amount,.01)),np.log(np.maximum(amount,.01)/profile["high"]))
    return np.exp(-np.maximum(distance,0)*8)
