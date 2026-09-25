"""Weakly supervised family price ranges learned only from unlabeled histories."""

import json

import numpy as np

from .data import LABELS, ROOT, transactions
from .features import MCC, PATTERNS


def learn_price_profiles(*, use_cache=True, data_dir=None):
    d = transactions("unlabeled_pretrain", use_cache=use_cache, data_dir=data_dir)
    d = d[(d.type == "card_payment") & (d.direction == "out")]
    result = {}
    for f in LABELS[:-1]:
        s = d[
            d.description.str.contains(PATTERNS[f].replace("(", "(?:"), regex=True)
            & (d.mcc == MCC[f])
        ].amount
        result[f] = {
            "low": float(s.quantile(0.005)),
            "high": float(s.quantile(0.995)),
            "median": float(s.median()),
            "n_anchor_events": len(s),
        }
    if use_cache:
        path = ROOT / "data/cache/unlabeled_price_profiles.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    return result


def price_support(amount, profile):
    """Smooth plausibility, not a calibrated family probability."""
    amount = np.asarray(amount, dtype=float)
    distance = np.maximum(
        np.log(profile["low"] / np.maximum(amount, 0.01)),
        np.log(np.maximum(amount, 0.01) / profile["high"]),
    )
    return np.exp(-np.maximum(distance, 0) * 8)
