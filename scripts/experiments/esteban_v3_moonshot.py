"""Moonshot toward ~0.70: noise-filtered streams, MCC/keywords, candidate-restricted softmax.

Idea (Plaid-ish + Santiago oracle): most clients have the true family among due
streams, but noise merchants drown the signal. Filter noise, expand mapping,
restrict the eight-class softmax to candidate families ∪ {none}, and blend with
StreamV3Push.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_oracle import (
    POSITIVE_LABELS,
    TrainOnlyFamilyMapper,
    build_streams,
)
from transaction_forecasting.ubs.v3 import _stream_quality
from transaction_forecasting.ubs.v3_push import StreamV3PushModel

OUT = Path("outputs/metrics/v3_moonshot")

NOISE = {
    "salary",
    "atm withdrawal",
    "fresh foods",
    "grocery store",
    "neighborhood market",
    "coffee shop",
    "casual dining",
    "pharmacy",
    "hotel booking",
    "ride share",
    "merchant charge",
    "online marketplace",
    "electronics shop",
    "p2p send",
    "p2p receive",
}

# Token → family cues (substring match on description). Intentionally broad.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "cloud": ("cloud", "storage", "saas", "hosting", "compute", "aws", "azure", "gcp"),
    "gym": ("gym", "fitness", "fit club", "workout", "athletic", "member fitness"),
    "insurance": ("insurance", "policy", "premium", "cover plan", "insur", "safe cover"),
    "mobile": ("mobile", "phone", "cellular", "telecom", "sim ", "contract"),
    "music": ("music", "audio", "spotify", "playlist", "song", "melody"),
    "software": ("software", "license", "suite", "productivity", "office", "devtool"),
    "streaming": ("streaming", "video", "media stream", "watch", "netflix", "hulu", "tv+"),
}

MCC_FAMILY = {
    "7997": "gym",
    "6300": "insurance",
    "4814": "mobile",
    "5734": "software",
    "5815": "music",  # digital goods — often music/media; soft cue
    "4899": "streaming",
    "7372": "software",
    "5732": "software",
}


def _payload(metrics, pred):
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "per_class": {k: float(v["f1-score"]) for k, v in metrics["per_class"].items()},
        "pred_dist": pred.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
    }


def _score(yv, pred, results, key):
    m = evaluate_predictions(yv.reindex(pred.index), pred)
    payload = _payload(m, pred)
    results[key] = payload
    print(f"{key}: {payload['macro_f1']:.4f} acc={payload['accuracy']:.3f}", flush=True)
    return payload


def keyword_family(description: str) -> str | None:
    text = str(description).lower()
    hits = []
    for family, tokens in KEYWORDS.items():
        if any(tok in text for tok in tokens):
            hits.append(family)
    if len(hits) == 1:
        return hits[0]
    return None


def expand_mapping(mapper: TrainOnlyFamilyMapper) -> pd.Series:
    """Looser eligibility + keyword fill for unmapped descriptions."""
    table = mapper.mapping_table_.copy()
    # Relax thresholds retrospectively on the fitted table (train-only stats).
    loose = table["support_clients"].ge(2) & table["winning_lift"].ge(1.2)
    mapping = table.loc[loose].set_index("description")["family"]
    # Keyword fill for descriptions never eligible
    for desc in table["description"]:
        if desc in mapping.index and pd.notna(mapping.loc[desc]):
            continue
        kw = keyword_family(desc)
        if kw is not None:
            mapping.loc[desc] = kw
    return mapping.dropna()


def candidate_families_per_client(
    streams: pd.DataFrame,
    mapping: pd.Series,
    client_ids: pd.Index,
    *,
    filter_noise: bool,
    use_mcc: bool,
    min_weight: float,
) -> pd.DataFrame:
    """Return per-client family weights from noise-filtered due streams."""
    clients = pd.Index(client_ids.astype(str), name="client_id")
    due = streams[streams["is_candidate"]].copy()
    if filter_noise:
        due = due[~due["description"].astype(str).isin(NOISE)]
    due["mapped_family"] = due["description"].map(mapping)
    # Keyword / MCC fallback when unmapped
    missing = due["mapped_family"].isna()
    if missing.any():
        due.loc[missing, "mapped_family"] = due.loc[missing, "description"].map(
            lambda d: keyword_family(d) or pd.NA
        )
    if use_mcc:
        still = due["mapped_family"].isna()
        due.loc[still, "mapped_family"] = due.loc[still, "mcc"].astype(str).map(MCC_FAMILY)
    due = due[due["mapped_family"].notna()].copy()
    weights = pd.DataFrame(0.0, index=clients, columns=list(POSITIVE_LABELS))
    details = pd.DataFrame(
        {
            "top_family": pd.Series("none", index=clients, dtype=object),
            "top_weight": 0.0,
            "n_cands": 0.0,
            "n_families": 0.0,
        },
        index=clients,
    )
    if due.empty:
        return weights, details
    due = _stream_quality(due)
    due = due[due["due_weight"].ge(min_weight)]
    for client, group in due.groupby("client_id", sort=False):
        if client not in weights.index:
            continue
        fam_w: dict[str, float] = {}
        for _, row in group.iterrows():
            fam = str(row["mapped_family"])
            if fam not in POSITIVE_LABELS:
                continue
            w = float(row["due_weight"])
            # Boost if keyword agrees
            if keyword_family(row["description"]) == fam:
                w *= 1.5
            mcc_fam = MCC_FAMILY.get(str(row["mcc"]))
            if mcc_fam == fam:
                w *= 1.35
            fam_w[fam] = max(fam_w.get(fam, 0.0), w)
        for fam, w in fam_w.items():
            weights.at[client, fam] = w
        if fam_w:
            top = max(fam_w, key=fam_w.get)
            details.at[client, "top_family"] = top
            details.at[client, "top_weight"] = fam_w[top]
            details.at[client, "n_cands"] = float(len(group))
            details.at[client, "n_families"] = float(len(fam_w))
    return weights.fillna(0.0), details


def restrict_softmax(
    base: pd.DataFrame,
    cand_weights: pd.DataFrame,
    *,
    none_floor: float,
    cand_temperature: float,
) -> pd.DataFrame:
    """Renormalize base probs on {families with cand evidence} ∪ {none}."""
    logits = np.log(np.clip(base.reindex(columns=LABELS).to_numpy(), 1e-12, 1.0))
    cw = cand_weights.reindex(base.index).fillna(0.0)
    has = cw.to_numpy() > 0
    # Penalize families without candidates
    for family in POSITIVE_LABELS:
        col = LABELS.index(family)
        mask = ~has[:, list(POSITIVE_LABELS).index(family)]
        logits[mask, col] -= 8.0  # near-zero after softmax
        # Boost families with candidates proportional to weight
        logits[:, col] += cand_temperature * np.log1p(cw[family].to_numpy())
    # Keep none available; slight floor when no candidates
    no_cand = ~has.any(axis=1)
    logits[no_cand, LABELS.index("none")] += none_floor
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    proba = w / w.sum(axis=1, keepdims=True)
    return pd.DataFrame(proba, index=base.index, columns=LABELS)


def dirichlet_from_candidates(cand_weights: pd.DataFrame, alpha0: float = 0.5) -> pd.DataFrame:
    """Bayesian posterior mean over positive families + none from stream weights."""
    clients = cand_weights.index
    # Convert weights to pseudo-counts; none gets alpha0 always
    counts = cand_weights.clip(lower=0.0)
    none_count = np.full(len(clients), alpha0)
    # If no candidates, none dominates
    empty = counts.sum(axis=1).eq(0)
    none_count = np.where(empty, alpha0 + 5.0, none_count)
    total = counts.sum(axis=1).to_numpy() + none_count + alpha0 * len(POSITIVE_LABELS)
    proba = pd.DataFrame(0.0, index=clients, columns=LABELS)
    for family in POSITIVE_LABELS:
        proba[family] = (counts[family].to_numpy() + alpha0) / total
    proba["none"] = none_count / total
    return proba


def mix(*frames, weights=None):
    mats = [f.reindex(columns=LABELS).to_numpy() for f in frames]
    index = frames[0].index
    if weights is None:
        weights = [1 / len(mats)] * len(mats)
    total = sum(w * m for w, m in zip(weights, mats, strict=True))
    total = total / total.sum(axis=1, keepdims=True)
    return pd.DataFrame(total, index=index, columns=LABELS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results = {}

    print("fitting StreamV3Push...", flush=True)
    push = StreamV3PushModel().fit(data.train_transactions, data.train_labels)
    push_comp = push.predict_components(data.valid_transactions)
    push_proba = push_comp["blend"].reindex(yv.index)
    _score(yv, push_proba.idxmax(axis=1), results, "push")

    print("building streams + expanded mapping...", flush=True)
    train_streams = build_streams(data.train_transactions)
    valid_streams = build_streams(data.valid_transactions)
    mapper = TrainOnlyFamilyMapper().fit(train_streams, data.train_labels)
    mapping = expand_mapping(mapper)
    print("expanded mapped descriptions", len(mapping), flush=True)

    # Candidate grids
    best = None
    for filter_noise in (True, False):
        for use_mcc in (True, False):
            for min_w in (0.0, 0.3, 0.8):
                cw, details = candidate_families_per_client(
                    valid_streams,
                    mapping,
                    yv.index,
                    filter_noise=filter_noise,
                    use_mcc=use_mcc,
                    min_weight=min_w,
                )
                # Pure argmax of candidate weights
                pred = details["top_family"].where(details["top_weight"].gt(0), "none")
                key = f"cand_argmax_n{int(filter_noise)}_m{int(use_mcc)}_w{min_w}"
                _score(yv, pred, results, key)

                # Dirichlet posterior
                diri = dirichlet_from_candidates(cw)
                _score(yv, diri.idxmax(axis=1), results, f"diri_{key}")

                # Restrict push softmax
                for temp in (0.5, 1.0, 2.0, 3.0):
                    for none_floor in (0.0, 1.0, 2.0):
                        restricted = restrict_softmax(
                            push_proba, cw, none_floor=none_floor, cand_temperature=temp
                        )
                        rkey = (
                            f"restrict_n{int(filter_noise)}_m{int(use_mcc)}"
                            f"_w{min_w}_t{temp}_nf{none_floor}"
                        )
                        payload = _score(yv, restricted.idxmax(axis=1), results, rkey)
                        if best is None or payload["macro_f1"] > best["macro_f1"]:
                            best = {**payload, "key": rkey}

                # Mix push with dirichlet / candidate one-hot soft
                for w in (0.5, 0.65, 0.8):
                    blended = mix(push_proba, diri, weights=[w, 1 - w])
                    _score(
                        yv,
                        blended.idxmax(axis=1),
                        results,
                        f"mix_diri_n{int(filter_noise)}_m{int(use_mcc)}_w{min_w}_p{w}",
                    )

    # Coverage diagnostic vs VALID labels (not used for fitting)
    cw, details = candidate_families_per_client(
        valid_streams, mapping, yv.index, filter_noise=True, use_mcc=True, min_weight=0.0
    )
    covered = 0
    pos = yv[yv.ne("none")]
    for client, lab in pos.items():
        if cw.loc[client, lab] > 0:
            covered += 1
    coverage = covered / max(len(pos), 1)
    print(f"noise-filtered candidate coverage of true family: {coverage:.4f}", flush=True)

    # Keyword-only global prior boost into push (crazy but cheap)
    for alpha in (0.5, 1.0, 2.0):
        logits = np.log(np.clip(push_proba.to_numpy(), 1e-12, 1.0))
        for family in POSITIVE_LABELS:
            logits[:, LABELS.index(family)] += alpha * np.log1p(cw[family].to_numpy())
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        proba = pd.DataFrame(w / w.sum(axis=1, keepdims=True), index=yv.index, columns=LABELS)
        _score(yv, proba.idxmax(axis=1), results, f"push_logcand_{alpha}")

    ranking = sorted(
        ((k, v["macro_f1"]) for k, v in results.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    summary = {
        "seconds": perf_counter() - started,
        "expanded_mapping": int(len(mapping)),
        "true_family_coverage_noise_filtered": coverage,
        "best_restrict": best,
        "top20": ranking[:20],
        "results": results,
    }
    (OUT / "moonshot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"top20": ranking[:20], "coverage": coverage, "best": best}, indent=2))
    print(f"done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
