"""Refund, fee and time evidence attached to candidate streams without labels."""
import numpy as np
import pandas as pd
from .data import CUTOFF


def payment_context(df,ranking):
    d=df.copy();d["age"]=(CUTOFF-d.timestamp).dt.total_seconds()/86400
    grouped={cid:g for cid,g in d.groupby("client_id")}
    rows=[]
    for index,row in ranking.iterrows():
        cid,f=index;g=grouped[cid];result={}
        card=g[g.type=="card_payment"];refund=g[g.type=="refund"]
        for prefix in ["amount0","broad0"]:
            mean=row.get(prefix+"_amount_median",-999);last=row.get(prefix+"_last_age",-999);count=row.get(prefix+"_count",-999)
            if mean<=0 or count<2:continue
            closest=card.iloc[int(np.argmin(np.abs(card.age.to_numpy()-last)))]
            currency=closest.currency
            c=card[(card.currency==currency)&(np.abs(np.log(card.amount/mean))<.04)]
            r=refund[(refund.currency==currency)&(np.abs(np.log(refund.amount/mean))<.04)]
            result[prefix+"_refund_count"]=len(r)
            result[prefix+"_refund_ratio"]=len(r)/max(len(c),1)
            result[prefix+"_refund_last_age"]=r.age.min() if len(r) else 999
            result[prefix+"_refund_after_last"]=int(((last-r.age).between(0,10)).any())
            result[prefix+"_refund_count90"]=int((r.age<90).sum())
            result[prefix+"_refund_count30"]=int((r.age<30).sum())
            if len(c):
                result[prefix+"_fee_fraction"]=(c.fee>0).mean()
                result[prefix+"_fee_mean"]=c.fee.mean()
                result[prefix+"_hour_std"]=c.timestamp.dt.hour.std()
                result[prefix+"_weekend_fraction"]=(c.timestamp.dt.dayofweek>=5).mean()
                result[prefix+"_last_fee"]=closest.fee
        rows.append(result)
    return pd.DataFrame(rows,index=ranking.index).fillna(-999)
