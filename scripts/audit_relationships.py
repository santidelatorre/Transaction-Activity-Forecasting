"""Supplementary train-only descriptive relationships; never used to refit the frozen model."""
import json
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT, CUTOFF, TARGET, LABELS, transactions, labels
from ubs_recurrence.features import normalize


def conditional(rows, key, name, out):
    # Each client contributes at most once per condition, avoiding activity weighting.
    unique = rows.drop_duplicates(["client_id", key]).reset_index(drop=True)
    table = pd.crosstab(unique[key], unique[TARGET]).reindex(columns=LABELS, fill_value=0)
    table.to_csv(out / f"{name}_counts.csv")
    table.div(table.sum(axis=1), axis=0).to_csv(out / f"{name}_probabilities.csv")
    return table


def main():
    out = ROOT / "outputs/audit/relationships"
    out.mkdir(parents=True, exist_ok=True)
    d = transactions("train", use_cache=False).join(labels()[TARGET], on="client_id", validate="many_to_one")
    d["normalized_description"] = d.description.map(normalize)
    d["weekday"] = d.timestamp.dt.dayofweek
    d["day_of_month"] = d.timestamp.dt.day
    d["month"] = d.timestamp.dt.month
    d["amount_band"] = pd.cut(d.amount, [0, 5, 10, 20, 40, 80, 160, 320, 1000, np.inf], include_lowest=True).astype(str)
    d["description_mcc"] = d.normalized_description + " | " + d.mcc
    d["type_mcc"] = d.type + " | " + d.mcc
    for key in ["weekday", "day_of_month", "month", "amount_band", "description_mcc", "type_mcc"]:
        conditional(d, key, key, out)
    ngrams = d[["client_id", TARGET, "normalized_description"]].copy()
    ngrams["ngram"] = ngrams.normalized_description.map(
        lambda s: [" ".join(s.split()[i:i+n]) for n in (1, 2, 3) for i in range(len(s.split())-n+1)])
    ngrams = ngrams.explode("ngram").dropna(subset=["ngram"])
    table = conditional(ngrams, "ngram", "word_ngrams", out)
    count = table.sum(axis=1)
    eligible = table[count >= 50]
    dominant = eligible.div(eligible.sum(axis=1), axis=0).max(axis=1).sort_values(ascending=False)
    rows = []
    card = d[(d.type == "card_payment") & (d.direction == "out")]
    for (cid, description), group in card.groupby(["client_id", "normalized_description"]):
        if len(group) < 2:
            continue
        gap = group.timestamp.sort_values().diff().dropna().dt.total_seconds() / 86400
        rows.append({"client_id": cid, "normalized_description": description, TARGET: group[TARGET].iloc[0],
                     "count": len(group), "recency_days": (CUTOFF-group.timestamp.max()).total_seconds()/86400,
                     "span_days": (group.timestamp.max()-group.timestamp.min()).total_seconds()/86400,
                     "median_interval_days": gap.median(), "interval_std_days": gap.std(ddof=0),
                     "amount_mean": group.amount.mean(), "amount_std": group.amount.std(ddof=0)})
    streams = pd.DataFrame(rows)
    streams["interval_cv"] = streams.interval_std_days / streams.median_interval_days.clip(lower=.01)
    streams["occurrence_band"] = pd.cut(streams["count"], [1, 2, 4, 8, 16, np.inf]).astype(str)
    streams["recency_band"] = pd.cut(streams.recency_days, [0, 15, 30, 60, 90, 180, np.inf]).astype(str)
    streams["interval_band"] = pd.cut(streams.median_interval_days, [0, 10, 20, 35, 65, 100, 200, np.inf]).astype(str)
    streams["regularity_band"] = pd.cut(streams.interval_cv, [-.001, .1, .25, .5, 1, np.inf]).astype(str)
    repeated = streams[streams["count"] >= 3]
    conditional(repeated, "normalized_description", "recurring_description", out)
    for key in ["occurrence_band", "recency_band", "interval_band", "regularity_band"]:
        conditional(streams, key, key, out)
    streams.groupby(TARGET)[["count", "recency_days", "span_days", "median_interval_days", "interval_cv", "amount_mean", "amount_std"]].agg(["count", "mean", "median", "std"]).to_csv(out / "stream_statistics_by_target.csv")
    report = {"scope": "Train labels only; descriptive, not predictive validation; produced after model freeze",
              "conditional_unit": "Unique client per condition; different conditions can overlap",
              "repeated_description_streams_at_least_2_events": len(streams),
              "repeated_description_streams_at_least_3_events": len(repeated),
              "conditional_tables": [p.name for p in sorted(out.glob("*.csv"))],
              "most_concentrated_ngrams_minimum_50_clients": [
                  {"ngram": n, "clients": int(count[n]), "dominant_label": eligible.loc[n].idxmax(),
                   "dominant_fraction": float(dominant[n])} for n in dominant.head(12).index],
              "warning": "Conditional association is not a classifier score; background exposure and class priors are confounders."}
    (ROOT / "reports/target_relationships.json").write_text(json.dumps(report, indent=2))
    summary = ROOT / "outputs/audit/summary.json"
    if summary.exists():
        (ROOT / "reports/audit_summary.json").write_bytes(summary.read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
