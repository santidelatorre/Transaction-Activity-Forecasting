"""Label-free TRAIN incidence and synthetic edge cases for the source representation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ubs_recurrence.compact import compact_features
from ubs_recurrence.data import CUTOFF, LABELS, transactions
from ubs_recurrence.payment_context import payment_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="paired_01")
    args = ap.parse_args()
    folder = ROOT / "outputs/stream_time_audit" / args.run_name / "features"
    x = pd.read_parquet(folder / "original.parquet")
    streams = pd.read_parquet(folder / "streams.parquet")
    df = transactions("train", use_cache=False)
    index = pd.MultiIndex.from_product([["synthetic"], LABELS], names=["client_id", "family"])
    toy = pd.DataFrame(-999.0, index=index, columns=[p + "_" + c for p in ("amount0", "amount1", "broad0") for c in ("count", "gap_median", "last_age", "next_median", "own_semantic")])
    toy.loc[("synthetic", "cloud"), ["amount0_count", "amount0_gap_median", "amount0_last_age", "amount0_next_median", "amount0_own_semantic"]] = [6, 7, 8, -1, 1]
    compact = compact_features(toy, pd.DataFrame(index=["synthetic"]))
    active = compact.loc[("synthetic", "cloud")]
    probes = {
        "weekly": {"gap": 7, "age": 8, "source_rounded_period": float(active.amount0_period_rounded),
                   "source_next_active": float(active.amount0_next_active), "weekly_modulo_offset_for_comparison": 6,
                   "interpretation": "Cycle mismatch demonstration; no claim that modulo rolling establishes continuation."},
        "sentinel": {"source_global_mean": float(active.amount0_next_active_global_mean),
                     "mean_excluding_999": float(compact.amount0_next_active.replace(999, np.nan).mean()),
                     "interpretation": "Source mean blends timing and number of absent/inactive candidates."},
    }
    # Same timestamp, different currencies, only one matching refund: demonstrate
    # nearest-row tie ambiguity without asserting incidence in the real data.
    sample = pd.DataFrame([
        {"client_id": "synthetic", "type": "card_payment", "timestamp": CUTOFF - pd.Timedelta(days=5), "currency": "usd", "amount": 500, "fee": 0.0},
        {"client_id": "synthetic", "type": "card_payment", "timestamp": CUTOFF - pd.Timedelta(days=5), "currency": "chf", "amount": 10, "fee": 0.0},
        {"client_id": "synthetic", "type": "refund", "timestamp": CUTOFF - pd.Timedelta(days=3), "currency": "chf", "amount": 10, "fee": 0.0},
    ])
    rank = pd.DataFrame({"amount0_amount_median": [10], "amount0_last_age": [5], "amount0_count": [3]}, index=index[:1])
    before = payment_context(sample, rank).iloc[0].amount0_refund_count
    after = payment_context(sample.iloc[[1, 0, 2]], rank).iloc[0].amount0_refund_count
    probes["refund_tie"] = {"first_order_refund_count": int(before), "reordered_refund_count": int(after), "interpretation": "Synthetic provenance edge case; source currency inferred from nearest timestamp."}

    positive = x.index.get_level_values("family") != "none"
    incidence = {"train_clients": int(df.client_id.nunique()), "candidate_rows_including_none": len(x), "streams": len(streams)}
    for prefix in ("amount0", "amount1", "broad0"):
        present = x[prefix + "_count"].ge(2) & positive
        period = x[prefix + "_gap_median"]
        age = x[prefix + "_last_age"]
        due = x[prefix + "_next_active"]
        incidence[prefix] = {
            "positive_candidates_present": int(present.sum()),
            "gap_between_6_and_8_days": int((present & period.between(6, 8)).sum()),
            "stale_over_1_35_observed_cycles": int((present & period.gt(0) & (age > 1.35 * period)).sum()),
            "inactive_or_absent_999": int(due.eq(999).sum()),
            "negative_signed_next_offset": int((present & x[prefix + "_next_median"].lt(0)).sum()),
        }
        if prefix + "_next_active_global_mean" in x:
            valid_due = due.mask(due.eq(999))
            clean = valid_due.groupby(level=0).mean()
            source = x[prefix + "_next_active_global_mean"].groupby(level=0).first()
            comparable = clean.notna()
            delta = source[comparable] - clean[comparable]
            incidence[prefix]["clients_with_active_candidate"] = int(comparable.sum())
            incidence[prefix]["source_minus_available_mean_median_days"] = float(delta.median())
            incidence[prefix]["source_minus_available_mean_max_days"] = float(delta.max())
    card = df[df.type.eq("card_payment")]
    tied = card.groupby(["client_id", "timestamp"]).currency.nunique()
    incidence["different_currency_card_timestamp_ties"] = int(tied.gt(1).sum())
    incidence["clients_with_different_currency_card_timestamp_ties"] = int(tied[tied.gt(1)].index.get_level_values(0).nunique())
    incidence["exact_duplicate_transaction_rows"] = int(df.duplicated().sum())
    report = {"scope": "TRAIN histories only; no labels read", "synthetic_probes": probes, "observed_incidence": incidence,
              "limitation": "Incidence does not establish impact on predictions. Absent-state sentinels may encode useful availability; removing them needs controlled model evaluation."}
    target = ROOT / "reports/stream_time_audit/state_incidence.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
