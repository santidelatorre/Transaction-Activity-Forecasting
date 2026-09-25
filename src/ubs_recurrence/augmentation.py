"""Feature-only description masking to stress the observed deployment shift."""
import numpy as np
import pandas as pd
from .features import GENERIC,normalize
from .templates import TEMPLATES

BACKGROUND=r"\b(?:coffee|casual|dining|grocery|fresh|foods|neighborhood|market|electronics|marketplace|hotel|booking|ride|share|pharmacy)\b"


def mask_descriptions(df, probability, seed):
    if not 0<=probability<=1:raise ValueError("Mask probability must lie in [0,1]")
    out=df.sort_values(["client_id","timestamp"],kind="stable").reset_index(drop=True).copy()
    norm=out.description.map(normalize)
    eligible=(out.type.isin(["card_payment","refund"]))&~norm.str.contains(BACKGROUND,regex=True)&~norm.isin(GENERIC)
    rng=np.random.default_rng(seed)
    replace=eligible.to_numpy()&(rng.random(len(out))<probability)
    out.loc[replace,"description"]=rng.choice(GENERIC,size=replace.sum())
    return out


def corrupt_transactions(df, severity, seed):
    """Stress observed text masking, category confusion and merchant-name noise.

    Rates are fixed from feature-only audit hypotheses, not validation labels.
    This is a simulator for robustness, not a claim to recover the generator.
    """
    if severity not in {"valid_like","test_like"}:raise ValueError(severity)
    rates={"valid_like":(.30,.18,.06,.15),"test_like":(.50,.30,.10,.25)}
    mask_rate,mcc_rate,background_rate,variant_rate=rates[severity]
    out=mask_descriptions(df,mask_rate,seed)
    rng=np.random.default_rng(seed+100)
    normalized=df.sort_values(["client_id","timestamp"],kind="stable").description.map(normalize).reset_index(drop=True)
    is_card=out.type.isin(["card_payment","refund"]).to_numpy()
    background=normalized.str.contains(BACKGROUND,regex=True).to_numpy()
    recurring=is_card&~background
    confusion={"5732":["5734","4814"],"5734":["5732","5812"],"4814":["5734"],"5812":["5734","5411"],"7997":["5812"]}
    mcc=out.mcc.to_numpy(copy=True)
    original_mcc=mcc.copy()
    for source,options in confusion.items():
        ix=np.flatnonzero(recurring&(original_mcc==source)&(rng.random(len(out))<mcc_rate))
        mcc[ix]=rng.choice(options,size=len(ix))
    out["mcc"]=mcc
    merchant_templates=sorted({v for f,vals in TEMPLATES.items() if f!="none" for v in vals})
    ix=np.flatnonzero(is_card&background&(rng.random(len(out))<background_rate))
    out.loc[ix,"description"]=rng.choice(merchant_templates,size=len(ix))
    ix=np.flatnonzero(is_card&(rng.random(len(out))<variant_rate))
    prefix=rng.choice(["pay ","billing ","member ",""],size=len(ix))
    suffix=rng.choice([" online"," core"," digital"," service",""],size=len(ix))
    out.loc[ix,"description"]=prefix+out.loc[ix,"description"].to_numpy()+suffix
    return out
