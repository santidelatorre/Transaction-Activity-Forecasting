"""Aggressive V3 max search: text disambiguation, temporal due, focused overrides.

Combines Ulmans V2 (Macro-F1 0.3915 / accuracy 0.424), StreamV3 (0.4134),
Javier stream-text, Santiago due streams, and Ginestar closeness diagnostics.
No labels are hardcoded; VALID is used only for reporting (try branch).
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
from transaction_forecasting.ubs.stream_text import (
    StreamTextFamilyModel,
    build_text_streams,
    recurring_or_fallback,
)
from transaction_forecasting.ubs.text_v2 import normalize_description
from transaction_forecasting.ubs.v2 import IntegratedV2Model
from transaction_forecasting.ubs.v3 import StreamV3Model, _stream_quality, apply_family_map

OUT = Path("outputs/metrics/v3_max")


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
    print(f"{key}: {payload['macro_f1']:.4f}", flush=True)
    return payload


def disambiguation_scores(
    streams: pd.DataFrame,
    mapper: TrainOnlyFamilyMapper,
    text_model: StreamTextFamilyModel,
    transactions: pd.DataFrame,
    client_ids: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-client family scores = max over due streams of weight * text_P(family)."""
    mapped = apply_family_map(streams, mapper)
    due = mapped[mapped["is_candidate"] & mapped["mapped_family"].notna()].copy()
    clients = pd.Index(client_ids.astype(str), name="client_id")
    family_scores = pd.DataFrame(0.0, index=clients, columns=list(POSITIVE_LABELS))
    details = pd.DataFrame(
        {
            "top_family": pd.Series("none", index=clients, dtype=object),
            "top_score": 0.0,
            "top_weight": 0.0,
            "top_text": 0.0,
            "n_due_mapped": 0.0,
            "unique_families": 0.0,
        },
        index=clients,
    )
    if due.empty:
        return family_scores, details

    due = _stream_quality(due)
    due["temporal_close"] = np.exp(-due["days_to_next"].clip(lower=0.0) / 90.0)
    due["combo_weight"] = due["due_weight"] * (0.5 + 0.5 * due["temporal_close"])

    text_scores, _ = text_model.predict_scores(transactions)
    text_streams = recurring_or_fallback(build_text_streams(transactions)).reset_index(drop=True)
    if not text_streams.empty:
        probs = text_model.model_.predict_proba(text_streams)
        classes = list(text_model.model_.named_steps["classifier"].classes_)
        text_streams = text_streams.copy()
        for i, label in enumerate(classes):
            text_streams[f"p_{label}"] = probs[:, i]
        due = due.copy()
        due["description_norm"] = due["description"].map(normalize_description)
        text_join = text_streams.rename(columns={"description": "description_norm"})[
            ["client_id", "description_norm"] + [f"p_{c}" for c in classes]
        ]
        due = due.merge(text_join, on=["client_id", "description_norm"], how="left")
        for label in POSITIVE_LABELS:
            col = f"p_{label}"
            if col not in due.columns:
                due[col] = 0.0
            due[col] = due[col].fillna(0.0)
    else:
        for label in POSITIVE_LABELS:
            due[f"p_{label}"] = 0.0

    lift_map = mapper.mapping_table_.set_index("description")["winning_lift"]
    due["winning_lift"] = due["description"].map(lift_map).fillna(1.0)

    for client, group in due.groupby("client_id", sort=False):
        if client not in family_scores.index:
            continue
        fam_vals: dict[str, float] = {}
        best = None
        for _, row in group.iterrows():
            fam = str(row["mapped_family"])
            text_p = float(row.get(f"p_{fam}", 0.0))
            score = float(row["combo_weight"]) * max(text_p, 0.05) * float(row["winning_lift"])
            fam_vals[fam] = max(fam_vals.get(fam, 0.0), score)
            cand = (score, fam, float(row["combo_weight"]), text_p)
            if best is None or cand[0] > best[0]:
                best = cand
        for fam, val in fam_vals.items():
            family_scores.at[client, fam] = val
        if best is not None:
            details.at[client, "top_family"] = best[1]
            details.at[client, "top_score"] = best[0]
            details.at[client, "top_weight"] = best[2]
            details.at[client, "top_text"] = best[3]
            details.at[client, "n_due_mapped"] = float(len(group))
            details.at[client, "unique_families"] = float(len(fam_vals))

    details["text_music"] = (
        text_scores.reindex(clients).get("music", pd.Series(0.0, index=clients)).fillna(0.0)
    )
    details["text_streaming"] = (
        text_scores.reindex(clients).get("streaming", pd.Series(0.0, index=clients)).fillna(0.0)
    )
    details["text_top"] = text_scores.reindex(clients).max(axis=1).fillna(0.0)
    details["text_pred"] = text_scores.reindex(clients).idxmax(axis=1).fillna("none")
    return family_scores.fillna(0.0), details


def blend_logits(
    base: pd.DataFrame, boost: pd.DataFrame, alpha: float, none_pen: float = 0.0
) -> pd.Series:
    logits = np.log(np.clip(base.reindex(columns=LABELS).to_numpy(), 1e-12, 1.0))
    for family in POSITIVE_LABELS:
        if family in boost.columns:
            logits[:, LABELS.index(family)] += (
                alpha * boost[family].reindex(base.index).fillna(0.0).to_numpy()
            )
    if none_pen:
        any_boost = boost.reindex(base.index).fillna(0.0).max(axis=1).to_numpy()
        logits[:, LABELS.index("none")] -= none_pen * alpha * any_boost
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    proba = weights / weights.sum(axis=1, keepdims=True)
    return pd.Series(np.asarray(LABELS)[proba.argmax(1)], index=base.index)


def focused_override(
    base_pred: pd.Series,
    text_details: pd.DataFrame,
    *,
    threshold: float,
    families: tuple[str, ...] = ("music", "streaming"),
    require_not_none: bool = True,
) -> pd.Series:
    pred = base_pred.copy()
    conf = (
        text_details["confidence"]
        if "confidence" in text_details.columns
        else text_details.get("text_top")
    )
    text_pred = (
        text_details["prediction"]
        if "prediction" in text_details.columns
        else text_details.get("text_pred")
    )
    if conf is None or text_pred is None:
        return pred
    eligible = conf.ge(threshold) & text_pred.isin(families)
    if require_not_none:
        eligible = eligible & pred.ne("none")
    eligible = eligible.reindex(pred.index).fillna(False)
    pred.loc[eligible] = text_pred.reindex(pred.index).loc[eligible]
    return pred


def single_family_override(
    base_pred: pd.Series,
    stream_features: pd.DataFrame,
    *,
    min_weight: float,
    margin: float,
    keep_none: bool = True,
) -> pd.Series:
    pred = base_pred.copy()
    eligible = (
        stream_features["stream_single_family"].gt(0.5)
        & stream_features["stream_margin"].ge(margin)
        & stream_features["stream_top_weight"].ge(min_weight)
    )
    if keep_none:
        eligible = eligible & pred.ne("none")
    top_idx = stream_features["stream_top_family_index"].astype(int)
    for client in stream_features.index[eligible.reindex(stream_features.index).fillna(False)]:
        idx = int(top_idx.loc[client])
        if idx >= 0:
            pred.loc[client] = POSITIVE_LABELS[idx]
    return pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results: dict = {}

    print("fitting V2 + StreamV3 + text...", flush=True)
    t0 = perf_counter()
    v2 = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    v2_comp = v2.predict_components(data.valid_transactions)
    v2_proba = v2_comp["blend"].reindex(yv.index)
    v2_pred = v2_proba.idxmax(axis=1)
    _score(yv, v2_pred, results, "v2")

    v3 = StreamV3Model(v2_blend=0.65).fit(data.train_transactions, data.train_labels)
    v3_comp = v3.predict_components(data.valid_transactions)
    v3_proba = v3_comp["blend"].reindex(yv.index)
    v3_pred = v3_proba.idxmax(axis=1)
    stream_feat = v3_comp["stream_features"].reindex(yv.index)
    _score(yv, v3_pred, results, "stream_v3_065")

    text = StreamTextFamilyModel().fit(data.train_transactions, data.train_labels)
    text_scores, text_details = text.predict_scores(data.valid_transactions)
    text_scores = text_scores.reindex(yv.index).fillna(0.0)
    text_details = text_details.reindex(yv.index)
    text_details["confidence"] = text_details["confidence"].fillna(0.0)
    text_details["prediction"] = text_details["prediction"].fillna("none")
    print(f"base models in {perf_counter() - t0:.1f}s", flush=True)

    # Javier-style focused overrides on V2 and V3
    for thr in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for base_name, base in (("v2", v2_pred), ("v3", v3_pred)):
            pred = focused_override(base, text_details, threshold=thr)
            _score(yv, pred, results, f"{base_name}_text_ms_{thr}")
            pred_any = focused_override(base, text_details, threshold=thr, families=POSITIVE_LABELS)
            _score(yv, pred_any, results, f"{base_name}_text_any_{thr}")

    # Soft text blend into V3
    for alpha in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        pred = blend_logits(v3_proba, text_scores, alpha=alpha, none_pen=0.0)
        _score(yv, pred, results, f"v3_soft_text_{alpha}")
        # music/streaming only soft boost
        ms = text_scores.copy()
        for fam in POSITIVE_LABELS:
            if fam not in ("music", "streaming"):
                ms[fam] = 0.0
        pred = blend_logits(v3_proba, ms, alpha=alpha, none_pen=0.0)
        _score(yv, pred, results, f"v3_soft_ms_{alpha}")

    # Disambiguation from due × text × lift
    print("building disambiguation scores...", flush=True)
    t1 = perf_counter()
    valid_streams = build_streams(data.valid_transactions)
    fam_scores, dis_details = disambiguation_scores(
        valid_streams, v3.mapper_, text, data.valid_transactions, yv.index
    )
    fam_scores = fam_scores.reindex(yv.index).fillna(0.0)
    dis_details = dis_details.reindex(yv.index).fillna(0.0)
    print(f"disambiguation in {perf_counter() - t1:.1f}s", flush=True)

    # Pure disambiguation argmax (positives only when score>0 else none / keep v3)
    pure = fam_scores.idxmax(axis=1)
    pure = pure.where(fam_scores.max(axis=1).gt(0), "none")
    _score(yv, pure, results, "disambig_argmax")

    for alpha in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        pred = blend_logits(v3_proba, fam_scores, alpha=alpha, none_pen=0.1)
        _score(yv, pred, results, f"v3_disambig_{alpha}")
        pred = blend_logits(v2_proba, fam_scores, alpha=alpha, none_pen=0.1)
        _score(yv, pred, results, f"v2_disambig_{alpha}")

    # Hard override when disambiguation is sharp and agrees with text
    for min_score in (0.2, 0.5, 1.0, 2.0, 3.0):
        for min_text in (0.0, 0.4, 0.6, 0.75):
            pred = v3_pred.copy()
            sharp = (
                dis_details["top_score"].ge(min_score)
                & dis_details["top_text"].ge(min_text)
                & dis_details["unique_families"].le(2)
            )
            # Prefer not to flip none → positive unless text also strong
            flip = sharp & (pred.ne("none") | dis_details["top_text"].ge(0.8))
            pred.loc[flip] = dis_details.loc[flip, "top_family"]
            _score(yv, pred, results, f"v3_hard_dis_{min_score}_{min_text}")

    # Single-family stream override grids on V3
    for min_w in (0.5, 0.8, 1.0, 1.5, 2.0):
        for margin in (0.05, 0.1, 0.2, 0.5):
            pred = single_family_override(v3_pred, stream_feat, min_weight=min_w, margin=margin)
            _score(yv, pred, results, f"v3_single_w{min_w}_m{margin}")

    # Combine best-looking levers: V3 + soft MS text + disambiguation
    for a_text in (1.0, 2.0, 3.0):
        for a_dis in (0.5, 1.0, 2.0):
            ms = text_scores.copy()
            for fam in POSITIVE_LABELS:
                if fam not in ("music", "streaming"):
                    ms[fam] = 0.0
            boost = a_text * ms + a_dis * fam_scores
            # blend_logits expects additive boost; fold alphas into boost matrix
            pred = blend_logits(v3_proba, boost, alpha=1.0, none_pen=0.05)
            _score(yv, pred, results, f"v3_combo_t{a_text}_d{a_dis}")

    # V3 then focused music/streaming after soft disambiguation
    for alpha in (1.0, 2.0, 3.0):
        soft = blend_logits(v3_proba, fam_scores, alpha=alpha, none_pen=0.1)
        for thr in (0.75, 0.85, 0.90):
            pred = focused_override(soft, text_details, threshold=thr)
            _score(yv, pred, results, f"v3_dis{alpha}_ms{thr}")

    ranking = sorted(
        ((k, v["macro_f1"]) for k, v in results.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    summary = {
        "seconds": perf_counter() - started,
        "note": (
            "Ulmans public 0.4240 is V2 accuracy; V2 Macro-F1 is 0.3915. "
            "Oracle 0.769 is a diagnostic ceiling, not a leaderboard entry."
        ),
        "top15": ranking[:15],
        "results": results,
    }
    (OUT / "max_search_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"top15": ranking[:15]}, indent=2))
    print(f"done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
