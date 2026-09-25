"""Amount-neighborhood recurring processes, inferred without client labels."""
from collections import Counter
import re
import numpy as np
import pandas as pd
from .data import CUTOFF,LABELS
from .features import PATTERNS,MCC,normalize,GENERIC
from .templates import TEMPLATES

FAMILIES=LABELS[:-1]


def amount_components(amounts, eps=.035, min_samples=3):
    """1-D DBSCAN for min_samples=3, without estimator/thread setup overhead."""
    if min_samples!=3:raise ValueError("Only min_samples=3 is supported")
    order=np.argsort(amounts,kind="stable")
    boundaries=np.flatnonzero(np.diff(np.log(np.asarray(amounts)[order]))>eps)+1
    return [c for c in np.split(order,boundaries) if len(c)>=min_samples]


def fast_stats(amount,days,desc,mcc,semantic,dom,dow,currency):
    order=np.argsort(days,kind="stable")
    a=amount[order]; t=days[order]; desc=desc[order]; mcc=mcc[order]; semantic=semantic[order]
    n=len(t); gap=np.diff(t)
    counts=np.array(list(Counter(desc).values()))
    unique_mcc=np.array(list(Counter(mcc).values()))
    r={"count":n,"amount_mean":a.mean(),"amount_std":a.std(),"amount_cv":a.std()/a.mean(),"amount_min":a.min(),"amount_max":a.max(),
       "amount_median":np.median(a),"amount_mad":np.median(np.abs(a-np.median(a)))/np.median(a),
       "first_age":-t.min(),"last_age":-t.max(),"span":np.ptp(t),"unique_description":len(counts),"description_top_fraction":counts.max()/n,
       "description_entropy":float(-np.sum(counts/n*np.log(counts/n))),"unique_mcc":len(unique_mcc),"mcc_top_fraction":unique_mcc.max()/n,
       "generic_fraction":np.isin(desc,GENERIC).mean(),"dom_std":np.std(dom),"dow_std":np.std(dow),"currency_count":len(set(currency))}
    for w in [30,60,90,180]:r[f"count_{w}"]=int((t>=-w).sum())
    for i,f in enumerate(FAMILIES):
        r[f"semantic_{f}"]=semantic[:,i].mean()
        r[f"mcc_{f}"]=(mcc==MCC[f]).mean()
        ct=np.array([(desc==v).sum() for v in TEMPLATES[f]])
        r[f"template_{f}_diversity"]=(ct>0).sum()
        r[f"template_{f}_count"]=ct.sum()
    if len(gap):
        med=np.median(gap); mad=np.median(np.abs(gap-med))
        r.update({"gap_median":med,"gap_mean":gap.mean(),"gap_std":gap.std(),"gap_mad":mad,"gap_min":gap.min(),"gap_max":gap.max(),
                  "gap_cv":gap.std()/max(gap.mean(),1e-6),"gap_last":gap[-1],"next_median":t[-1]+med,"next_mean":t[-1]+gap.mean(),
                  "next_last":t[-1]+gap[-1],"overdue_ratio":-t[-1]/max(med,1),"gap_recent":np.median(gap[-3:]),
                  "next_recent":t[-1]+np.median(gap[-3:]),"amount_change":a[-1]/a[0]-1})
        # Circular phase concentration identifies a recurrent clock despite missed events.
        for period in [7,14,28,30,30.4375,31,60,90,365]:
            residual=np.minimum(np.mod(gap,period),period-np.mod(gap,period))
            r[f"period_error_{period}"]=np.median(residual)
            phase=np.exp(2j*np.pi*t/period).mean()
            r[f"phase_strength_{period}"]=abs(phase)
            r[f"phase_next_{period}"]=np.mod(np.angle(phase)*period/(2*np.pi),period)
        # Fitted clock allows drift; predictions are evidence, not guaranteed dates.
        slope,intercept=np.polyfit(np.arange(n),t,1)
        r["linear_period"]=slope
        r["linear_residual"]=np.sqrt(np.mean((t-(intercept+slope*np.arange(n)))**2))
        r["linear_next"]=intercept+slope*n
    return r


def extract_streams(df, eps=.035, filter_background=False):
    rows=[]
    d=df[(df.type=="card_payment") & (df.direction=="out")].copy()
    d["description"]=d.description.map(normalize)
    if filter_background:
        background=r"\b(?:coffee|casual|dining|grocery|fresh|foods|neighborhood|market|electronics|marketplace|hotel|booking|ride|share|pharmacy)\b"
        d=d[~d.description.str.contains(background,regex=True)].copy()
    days=(d.timestamp-CUTOFF).dt.total_seconds().to_numpy()/86400
    semantic=np.stack([d.description.str.contains(p.replace("(","(?:"),regex=True).to_numpy() for p in PATTERNS.values()],axis=1)
    d["row_number"]=np.arange(len(d))
    amounts=d.amount.to_numpy();desc=d.description.to_numpy();mcc=d.mcc.to_numpy();curr=d.currency.to_numpy()
    dom=d.timestamp.dt.day.to_numpy();dow=d.timestamp.dt.dayofweek.to_numpy()
    for (cid,currency),g in d.groupby(["client_id","currency"],sort=True):
        ix=g.row_number.to_numpy()
        groups=[("amount",ix[c]) for c in amount_components(amounts[ix],eps)]
        # Broader family/MCC grouping retains variable-amount recurring streams.
        for f in FAMILIES:
            choose=semantic[ix,FAMILIES.index(f)] | ((mcc[ix]==MCC[f]) & np.isin(desc[ix],TEMPLATES[f]+GENERIC))
            if choose.sum()>=2:groups.append(("family_"+f,ix[choose]))
        for group_type,index in groups:
            r=fast_stats(amounts[index],days[index],desc[index],mcc[index],semantic[index],dom[index],dow[index],curr[index])
            score=np.array([r[f"semantic_{f}"]+.25*r[f"mcc_{f}"] for f in FAMILIES])
            r.update({"client_id":cid,"group_type":group_type,"currency":currency,"family":FAMILIES[int(score.argmax())],"family_score":score.max(),"eps":eps})
            rows.append(r)
    return pd.DataFrame(rows)


def family_features(streams, ids):
    """One symmetric row per client/family, sharing statistical strength."""
    rows=[]
    st=streams.copy()
    all_numeric=[c for c in st.select_dtypes(include=np.number).columns if not any(s in c for s in ["semantic_","mcc_","template_"]) and c not in ["eps","family_score"]]
    grouped={k:v for k,v in st.groupby("client_id")}
    for cid in ids:
        g=grouped.get(cid,st.iloc[:0])
        for f in FAMILIES:
            row={"client_id":cid,"family":f,"family_index":FAMILIES.index(f)}
            eligible=g[((g.family==f)&(g.group_type=="amount")) | (g.group_type=="family_"+f)].copy()
            row["stream_count"]=len(eligible)
            for prefix,sub in [("amount",eligible[eligible.group_type=="amount"]),("broad",eligible[eligible.group_type=="family_"+f])]:
                sub=sub.copy()
                sub["strength"]=sub["count"]*np.exp(-sub.last_age/60) if len(sub) else 0
                sub=sub.sort_values("strength",ascending=False)
                for rank in range(2):
                    s=sub.iloc[rank] if len(sub)>rank else None
                    if s is not None:
                        row.update({f"{prefix}{rank}_{c}":s[c] for c in all_numeric})
                        row[f"{prefix}{rank}_own_semantic"]=s["semantic_"+f]
                        row[f"{prefix}{rank}_own_mcc"]=s["mcc_"+f]
                        row[f"{prefix}{rank}_own_template_diversity"]=s[f"template_{f}_diversity"]
                        row[f"{prefix}{rank}_own_template_count"]=s[f"template_{f}_count"]
                        row[f"{prefix}{rank}_other_semantic"]=max(s["semantic_"+o] for o in FAMILIES if o!=f)
            rows.append(row)
    x=pd.DataFrame(rows).set_index(["client_id","family"]).fillna(-999)
    # Context for comparisons is computed within client, never across labeled rows.
    for c in ["amount0_next_median","amount0_last_age","amount0_count","amount0_own_template_diversity","broad0_next_median","broad0_count","broad0_own_template_diversity"]:
        if c in x:
            value=x[c].replace(-999,np.nan)
            x[c+"_rank"]=value.groupby(level=0).rank(method="min",ascending=("count" not in c and "diversity" not in c)).fillna(8)
            x[c+"_vs_best"]=value-value.groupby(level=0).transform("min" if "count" not in c and "diversity" not in c else "max")
    return x.fillna(-999)
