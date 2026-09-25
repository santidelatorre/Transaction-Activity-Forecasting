"""Train-feature-only cadence audit; date errors are diagnostic, not challenge scores."""
import json
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT,transactions,CUTOFF
from ubs_recurrence.streams import amount_components
from ubs_recurrence.features import normalize
from ubs_recurrence.augmentation import BACKGROUND


def main():
    d=transactions("train");d=d[(d.type=="card_payment")&(d.direction=="out")].copy();d=d[~d.description.map(normalize).str.contains(BACKGROUND,regex=True)]
    periods=[];errors={"median_interval":[],"calendar_same_day":[],"calendar_next_business_day":[]};weekends=[]
    for _,g in d.groupby(["client_id","currency"]):
        for component in amount_components(g.amount.to_numpy(),.035):
            s=g.iloc[component].sort_values("timestamp");t=s.timestamp.reset_index(drop=True)
            if len(t)<5:continue
            gaps=t.diff().dropna().dt.total_seconds().to_numpy()/86400
            if np.std(gaps)/np.mean(gaps)>.25:continue
            periods.append(float(np.median(gaps)));weekends.extend((t.dt.dayofweek>=5).astype(int).tolist())
            if not 24<=np.median(gaps)<=35:continue
            for end in range(3,len(t)):
                observed=t.iloc[end];last=t.iloc[end-1]
                absolute=last+pd.Timedelta(days=float(np.median(gaps[:end-1])))
                calendar=last+pd.DateOffset(months=1)
                business=calendar
                while business.dayofweek>=5:business+=pd.Timedelta(days=1)
                for name,pred in [("median_interval",absolute),("calendar_same_day",calendar),("calendar_next_business_day",business)]:errors[name].append(abs((observed-pred).total_seconds())/86400)
    nearest={str(p):int(np.sum(np.abs(np.array(periods)-p)<2)) for p in [7,14,28,30,60,90,365]}
    report={"scope":"Training histories only; recurring candidate audit, not task evaluation","regular_candidates_with_at_least_5_events":len(periods),"median_interval_near_period_plus_minus_2_days":nearest,"count_warning":"28 and 30 day bins overlap; annual recurrence cannot be identified by this >=5-event filter in a 420-day history","weekend_event_fraction":float(np.mean(weekends)),"monthly_like_next_observed_date_diagnostics":{k:{"forecast_count":len(v),"mean_absolute_error_days":float(np.mean(v)),"median_absolute_error_days":float(np.median(v))} for k,v in errors.items()},"limitation":"Recovered streams are approximate. These date diagnostics are not merchant-family macro-F1 and do not establish a complete generator."}
    (ROOT/"reports/cadence_audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=="__main__":main()
