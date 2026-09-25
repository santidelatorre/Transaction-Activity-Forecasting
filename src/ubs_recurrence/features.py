"""Deterministic label-free feature transforms motivated by the train audit."""

import re

import numpy as np
import pandas as pd

from .data import CUTOFF, LABELS

FAMILIES = LABELS[:-1]
PATTERNS = {
    "cloud": r"\b(cloud|storage|backup)\b|service plan",
    "gym": r"\b(gym|fitness|fit club|urban)\b",
    "insurance": r"\b(insurance|policy|cover)\b",
    "mobile": r"\b(phone|contract)\b|service bill",
    "music": r"\baudio\b|member pass",
    "software": r"\b(software|saas|productivity)\b",
    "streaming": r"\b(media|video)\b",
}
MCC = {
    "cloud": "5732",
    "gym": "7997",
    "insurance": "6300",
    "mobile": "4814",
    "music": "5812",
    "software": "5734",
    "streaming": "5812",
}
GENERIC = ["monthly plan", "digital service", "member plan", "subscription charge"]


def normalize(text):
    text = text.lower()
    for a, b in [
        ("dgtl", "digital"),
        ("mth", "monthly"),
        ("prem ", "premium "),
        ("insur ", "insurance "),
    ]:
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", text)).strip()


def documents(df, ids, kind="raw"):
    d = df.copy()
    if kind == "normalized":
        d["description"] = d.description.map(normalize)
    if kind == "enriched":
        d["description"] = (
            d.description.map(normalize)
            + " mcc"
            + d.mcc
            + " type"
            + d.type
            + " "
            + d.description.map(normalize).str.replace(" ", "_")
            + "_m"
            + d.mcc
        )
    return (
        d.groupby("client_id")
        .description.agg(" ; ".join)
        .reindex(ids)
        .fillna("")
        .to_numpy()
    )


def summarize(g):
    if len(g) == 0:
        return {"count": 0}
    a = g.amount.to_numpy()
    days = (g.timestamp - CUTOFF).dt.total_seconds().to_numpy() / 86400
    gaps = np.diff(np.sort(days))
    desc = g.description.value_counts()
    result = {
        "count": len(g),
        "amount_mean": np.mean(a),
        "amount_std": np.std(a),
        "amount_min": np.min(a),
        "amount_max": np.max(a),
        "amount_median": np.median(a),
        "amount_cv": np.std(a) / max(np.mean(a), 1e-6),
        "first_age": -days.min(),
        "last_age": -days.max(),
        "span": np.ptp(days),
        "unique_description": len(desc),
        "description_top_fraction": desc.iloc[0] / len(g),
        "description_entropy": float(-np.sum((desc / len(g)) * np.log(desc / len(g)))),
        "unique_mcc": g.mcc.nunique(),
        "unique_currency": g.currency.nunique(),
        "fee_fraction": float((g.fee > 0).mean()),
        "dom_std": g.timestamp.dt.day.std(),
        "dow_std": g.timestamp.dt.dayofweek.std(),
        "month_count": g.timestamp.dt.strftime("%Y-%m").nunique(),
    }
    for window in [30, 60, 90, 180]:
        result[f"count_{window}"] = int((days >= -window).sum())
    if len(gaps):
        result.update(
            {
                "gap_median": np.median(gaps),
                "gap_mean": np.mean(gaps),
                "gap_std": np.std(gaps),
                "gap_min": np.min(gaps),
                "gap_max": np.max(gaps),
                "gap_mad": np.median(np.abs(gaps - np.median(gaps))),
                "gap_cv": np.std(gaps) / max(np.mean(gaps), 1e-6),
                "next_median": days.max() + np.median(gaps),
                "gap_last": gaps[-1],
            }
        )
        for period in [7, 14, 28, 30, 31, 60, 90, 365]:
            result[f"period_error_{period}"] = float(np.median(np.abs(gaps - period)))
    return result


def tabular(df, ids):
    d = df.copy()
    d["description"] = d.description.map(normalize)
    for f in FAMILIES:
        d["is_" + f] = d.description.str.contains(PATTERNS[f], regex=True)
    rows = []
    for client_id, g in d.groupby("client_id", sort=True):
        row = {"client_id": client_id}

        def add(prefix, subset, row=row):
            row.update({prefix + "__" + k: v for k, v in summarize(subset).items()})

        add("global", g)
        out = g[(g.direction == "out") & (g.type == "card_payment")]
        add("card", out)
        for col in ["mcc", "type", "currency", "direction"]:
            for value, n in g[col].value_counts().items():
                row[f"global__{col}_{value}"] = n
        for m in sorted(set(MCC.values())):
            add("mcc" + m, out[out.mcc == m])
        add("generic", out[out.description.isin(GENERIC)])
        for f in FAMILIES:
            add("text_" + f, out[out["is_" + f]])
            add("joint_" + f, out[out["is_" + f] & (out.mcc == MCC[f])])
            row[f"text_{f}__refund_count"] = int(
                ((g.direction == "in") & g["is_" + f]).sum()
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .set_index("client_id")
        .reindex(ids)
        .fillna(-999)
        .sort_index(axis=1)
    )
