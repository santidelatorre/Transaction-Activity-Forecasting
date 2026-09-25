"""Symmetric family ranking with clock-aware competition and an explicit none row."""

import numpy as np
import pandas as pd

from .data import LABELS
from .templates import TEMPLATES


def ranking_features(family, templates, ids, clocks=True):
    original = family.copy()
    index = pd.MultiIndex.from_product([ids, LABELS], names=["client_id", "family"])
    X = original.reindex(index).fillna(-999)
    X["is_none"] = (X.index.get_level_values("family") == "none").astype(int)
    X["family_index"] = np.tile(np.arange(8), len(ids))
    # Rotation-invariant template counts share family evidence between labels.
    for f in LABELS:
        ix = pd.IndexSlice[:, f]
        for scope in ["any", "mcc"]:
            for w in [999, 90]:
                base = f"{f}__{scope}_{w}_"
                counts = (
                    templates[[base + t for t in TEMPLATES[f]]].reindex(ids).to_numpy()
                )
                ordered = np.sort(counts, axis=1)[:, ::-1]
                for rank in range(4):
                    col = f"own_template_{scope}_{w}_rank{rank}"
                    X.loc[ix, col] = ordered[:, rank]
                for stat in ["diversity", "repeated", "sum", "max"]:
                    X.loc[ix, f"own_template_{scope}_{w}_{stat}"] = (
                        templates[base + stat].reindex(ids).to_numpy()
                    )
    # Global evidence is identical in all eight rows for a client and cannot encode labels.
    for col in [
        c
        for c in templates
        if c.startswith("none__") and not any(t in c for t in ["amount", "age"])
    ]:
        X["global_" + col] = np.repeat(templates[col].reindex(ids).to_numpy(), 8)
    if clocks:
        periods = np.array([7, 14, 28, 30, 30.4375, 31, 60, 90, 365])
        for prefix in ["amount0", "amount1", "broad0", "broad1"]:
            if prefix + "_gap_median" not in X:
                continue
            med = X[prefix + "_gap_median"].to_numpy()
            strength = np.stack(
                [
                    X[f"{prefix}_phase_strength_{p:g}"].to_numpy()
                    if f"{prefix}_phase_strength_{p:g}" in X
                    else X[f"{prefix}_phase_strength_{p}"].to_numpy()
                    for p in periods
                ],
                axis=1,
            )
            # Prefer clocks explaining both phase and observed event density.
            fit = (1 - strength) + 0.8 * np.abs(
                np.log(np.maximum(med[:, None], 1) / periods[None, :])
            )
            best = fit.argmin(axis=1)
            period = periods[best]
            phase = np.stack(
                [
                    X[f"{prefix}_phase_next_{p:g}"].to_numpy()
                    if f"{prefix}_phase_next_{p:g}" in X
                    else X[f"{prefix}_phase_next_{p}"].to_numpy()
                    for p in periods
                ],
                axis=1,
            )[np.arange(len(X)), best]
            age = X[prefix + "_last_age"].to_numpy()
            valid = (med > 0) & (X[prefix + "_count"].to_numpy() >= 3)
            active = valid & (age < period * 1.4)
            X[prefix + "_clock_period"] = np.where(valid, period, -999)
            X[prefix + "_clock_next"] = np.where(valid, phase, 999)
            X[prefix + "_clock_active_next"] = np.where(active, phase, 999)
            X[prefix + "_clock_strength"] = np.where(
                valid, strength[np.arange(len(X)), best], 0
            )
            X[prefix + "_active"] = active.astype(int)
            X[prefix + "_next_modulo"] = np.where(
                valid, np.mod(X[prefix + "_next_median"].to_numpy(), period), 999
            )
    compare = [
        c
        for c in X
        if any(
            t in c
            for t in [
                "clock_active_next",
                "next_modulo",
                "last_age",
                "own_template_mcc_999",
                "_active",
            ]
        )
        and not c.endswith(("_rank", "_vs_best"))
    ]
    compare += [
        c
        for c in [
            "amount0_count",
            "amount0_gap_cv",
            "amount0_linear_next",
            "broad0_next_median",
            "broad0_count",
        ]
        if c in X
    ]
    extra = {}
    for c in compare:
        value = X[c].replace(-999, np.nan)
        grp = value.groupby(level=0, sort=False)
        extra[c + "_competition_min"] = grp.transform("min")
        extra[c + "_competition_max"] = grp.transform("max")
        extra[c + "_competition_mean"] = grp.transform("mean")
        extra[c + "_competition_rank"] = grp.rank(method="min")
        extra[c + "_competition_delta"] = value - grp.transform("min")
    return pd.concat([X, pd.DataFrame(extra, index=X.index)], axis=1).fillna(-999)
