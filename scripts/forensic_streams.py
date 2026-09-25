"""Audit recurrence at family and amount-cluster level, using train labels only."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from ubs_recurrence.data import ROOT, labels, transactions
from ubs_recurrence.features import (
    FAMILIES,
    GENERIC,
    MCC,
    PATTERNS,
    normalize,
    summarize,
)


def main():
    df = transactions("train")
    df["description"] = df.description.map(normalize)
    y = labels()["target_next_recurring_merchant"]
    rows = []
    exact_rows = []
    for cid, g in df.groupby("client_id"):
        card = g[(g.type == "card_payment") & (g.direction == "out")].copy()
        # Stable amount neighborhoods do not depend on the target or split.
        for currency, cg in card.groupby("currency"):
            cluster = DBSCAN(eps=0.035, min_samples=3).fit_predict(
                np.log(cg.amount.to_numpy()).reshape(-1, 1)
            )
            for c in sorted(set(cluster) - {-1}):
                s = cg.iloc[np.flatnonzero(cluster == c)]
                r = {
                    "client_id": cid,
                    "cluster": int(c),
                    "currency": currency,
                    "target": y.loc[cid],
                    **summarize(s),
                }
                r["descriptions"] = " | ".join(s.description.value_counts().index)
                r["mcc_mode"] = s.mcc.mode().iloc[0]
                r["generic_fraction"] = s.description.isin(GENERIC).mean()
                for f in FAMILIES:
                    r[f"semantic_{f}"] = s.description.str.contains(
                        PATTERNS[f].replace("(", "(?:"), regex=True
                    ).mean()
                    r[f"mcc_{f}"] = (s.mcc == MCC[f]).mean()
                rows.append(r)
        for desc, s in card.groupby("description"):
            if len(s) >= 3:
                exact_rows.append(
                    {
                        "client_id": cid,
                        "description": desc,
                        "target": y.loc[cid],
                        **summarize(s),
                    }
                )
    out = ROOT / "outputs/audit"
    clusters = pd.DataFrame(rows)
    clusters.to_csv(out / "amount_clusters.csv", index=False)
    pd.DataFrame(exact_rows).to_csv(out / "exact_description_streams.csv", index=False)
    # Compare targets against families suggested by the amount-neighborhood evidence.
    scores = np.array(
        [
            clusters[f"semantic_{f}"].to_numpy()
            + 0.25 * clusters[f"mcc_{f}"].to_numpy()
            for f in FAMILIES
        ]
    ).T
    clusters["inferred_family"] = np.array(FAMILIES)[scores.argmax(axis=1)]
    clusters["matches_target"] = clusters.inferred_family == clusters.target
    report = clusters.groupby(["matches_target"])[
        [
            "count",
            "amount_cv",
            "last_age",
            "span",
            "unique_description",
            "description_top_fraction",
            "description_entropy",
            "gap_median",
            "gap_std",
            "gap_cv",
            "generic_fraction",
        ]
    ].agg(["mean", "median"])
    report.to_csv(out / "cluster_target_relationships.csv")
    print(report.to_string())
    print(
        "Median gap rounded distribution",
        clusters.gap_median.round().value_counts().head(20).to_dict(),
    )
    print("Count", len(clusters), "clients", clusters.client_id.nunique())


if __name__ == "__main__":
    main()
