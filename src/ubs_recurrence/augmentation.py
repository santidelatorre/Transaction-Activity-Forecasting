"""Feature-only description masking to stress the observed deployment shift."""
import numpy as np
import pandas as pd
from .features import GENERIC,normalize

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
