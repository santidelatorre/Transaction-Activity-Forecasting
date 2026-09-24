"""Push past StreamV3Max 0.4319: Astra identity + hybrids + wild stacks.

Compares Ulmans V2 (0.3915), StreamV3Max (0.4319), Astra V3-A (~0.424),
Astra-style ensemble (~0.430), soft-text overlays, probability stacks,
two-stage none gates, and kitchen-sink CatBoost.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.stream_oracle import POSITIVE_LABELS, build_streams
from transaction_forecasting.ubs.v2 import HistoryFeatureBuilder, IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3 import StreamV3Model, client_stream_features
from transaction_forecasting.ubs.v3_identity_features import (
    cross_fitted_family_features,
    family_features,
)
from transaction_forecasting.ubs.v3_identity_model import IdentityV3Model
from transaction_forecasting.ubs.v3_max import StreamV3MaxModel, soft_family_boost

OUT = Path("outputs/metrics/v3_push")


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
    print(f"{key}: {payload['macro_f1']:.4f}  acc={payload['accuracy']:.3f}", flush=True)
    return payload


def mix_proba(*frames: pd.DataFrame, weights: list[float] | None = None) -> pd.DataFrame:
    aligned = [frame.reindex(columns=LABELS).fillna(0.0) for frame in frames]
    index = aligned[0].index
    for frame in aligned[1:]:
        index = index.intersection(frame.index)
    mats = [frame.reindex(index).to_numpy() for frame in aligned]
    if weights is None:
        weights = [1.0 / len(mats)] * len(mats)
    total = sum(w * m for w, m in zip(weights, mats, strict=True))
    total = total / total.sum(axis=1, keepdims=True)
    return pd.DataFrame(total, index=index, columns=LABELS)


def geometric_mix(*frames: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    aligned = [frame.reindex(columns=LABELS).clip(lower=eps) for frame in frames]
    index = aligned[0].index
    for frame in aligned[1:]:
        index = index.intersection(frame.index)
    log_sum = sum(np.log(frame.reindex(index).to_numpy()) for frame in aligned)
    vals = np.exp(log_sum / len(aligned))
    vals = vals / vals.sum(axis=1, keepdims=True)
    return pd.DataFrame(vals, index=index, columns=LABELS)


def keep_v2_none(v2_pred: pd.Series, candidate: pd.Series) -> pd.Series:
    out = candidate.copy()
    none_mask = v2_pred.reindex(out.index).eq("none")
    out.loc[none_mask.fillna(False)] = "none"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    results: dict = {}

    print("=== base models ===", flush=True)
    v2 = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    v2_comp = v2.predict_components(data.valid_transactions)
    v2_proba = v2_comp["blend"].reindex(yv.index)
    v2_pred = v2_proba.idxmax(axis=1)
    _score(yv, v2_pred, results, "v2")

    v3 = StreamV3Model(v2_blend=0.65).fit(data.train_transactions, data.train_labels)
    v3_comp = v3.predict_components(data.valid_transactions)
    v3_proba = v3_comp["blend"].reindex(yv.index)
    _score(yv, v3_proba.idxmax(axis=1), results, "stream_v3")

    vmax = StreamV3MaxModel(music_alpha=0.60, streaming_alpha=0.05).fit(
        data.train_transactions, data.train_labels
    )
    vmax_comp = vmax.predict_components(data.valid_transactions)
    vmax_proba = vmax_comp["blend"].reindex(yv.index)
    text_scores = vmax_comp["text_scores"].reindex(yv.index).fillna(0.0)
    _score(yv, vmax_proba.idxmax(axis=1), results, "stream_v3_max")

    print("fitting Astra identity V3-A (cross-fit)...", flush=True)
    t0 = perf_counter()
    ident = IdentityV3Model().fit(data.train_transactions, data.train_labels)
    ident_comp = ident.predict_components(data.valid_transactions)
    print(f"identity fit+predict in {perf_counter() - t0:.1f}s", flush=True)
    a_proba = ident_comp["identity"].reindex(yv.index)
    ens_proba = ident_comp["ensemble_v2_identity"].reindex(yv.index)
    _score(yv, a_proba.idxmax(axis=1), results, "astra_A")
    _score(yv, ens_proba.idxmax(axis=1), results, "astra_ens_50")

    # Soft MS text on Astra arms
    for name, base in (("astra_A", a_proba), ("astra_ens", ens_proba), ("v3", v3_proba)):
        for am, as_ in ((0.6, 0.05), (0.5, 0.1), (0.35, 0.35), (0.7, 0.0), (0.8, 0.05)):
            boosted = soft_family_boost(base, text_scores, alphas={"music": am, "streaming": as_})
            _score(yv, boosted.idxmax(axis=1), results, f"{name}_ms_{am}_{as_}")

    # Arithmetic / geometric stacks of diverse models
    stack_members = {
        "v2": v2_proba,
        "v3": v3_proba,
        "vmax": vmax_proba,
        "astraA": a_proba,
        "astraEns": ens_proba,
    }
    combos = [
        ("vmax_astraA", ["vmax", "astraA"], None),
        ("vmax_astraEns", ["vmax", "astraEns"], None),
        ("v3_astraA", ["v3", "astraA"], None),
        ("v2_vmax_astraA", ["v2", "vmax", "astraA"], None),
        ("vmax_astraA_w7030", ["vmax", "astraA"], [0.7, 0.3]),
        ("vmax_astraA_w6040", ["vmax", "astraA"], [0.6, 0.4]),
        ("vmax_astraA_w5545", ["vmax", "astraA"], [0.55, 0.45]),
        ("vmax_astraEns_w6040", ["vmax", "astraEns"], [0.6, 0.4]),
        ("triple_eq", ["vmax", "astraA", "v3"], None),
        ("triple_503020", ["vmax", "astraA", "v3"], [0.5, 0.3, 0.2]),
    ]
    for key, names, weights in combos:
        frames = [stack_members[n] for n in names]
        mixed = mix_proba(*frames, weights=weights)
        _score(yv, mixed.idxmax(axis=1), results, f"stack_{key}")
        if weights is None:
            geo = geometric_mix(*frames)
            _score(yv, geo.idxmax(axis=1), results, f"geo_{key}")

    # Two-stage: freeze V2 none, take positive from best positive-oriented models
    for name, proba in (
        ("vmax", vmax_proba),
        ("astraA", a_proba),
        ("stack", mix_proba(vmax_proba, a_proba, weights=[0.6, 0.4])),
    ):
        cand = proba.idxmax(axis=1)
        gated = keep_v2_none(v2_pred, cand)
        _score(yv, gated, results, f"gate_v2none_{name}")

    # Kitchen-sink CatBoost: history + identity + due-stream + text
    print("fitting kitchen-sink CatBoost...", flush=True)
    t1 = perf_counter()
    history_builder = HistoryFeatureBuilder().fit(data.train_transactions)
    train_hist = history_builder.transform(data.train_transactions)
    train_ident = cross_fitted_family_features(data.train_transactions, data.train_labels).filter(
        regex="^identity_"
    )
    train_streams = build_streams(data.train_transactions)
    train_stream = client_stream_features(
        data.train_transactions, v3.mapper_, train_hist.index, streams=train_streams
    )
    # In-sample text features from fitted vmax text model
    from transaction_forecasting.ubs.stream_text import (
        aggregate_stream_probabilities,
        build_text_streams,
        recurring_or_fallback,
    )

    def text_feat(transactions, clients, model):
        streams = recurring_or_fallback(build_text_streams(transactions)).reset_index(drop=True)
        feats = pd.DataFrame(0.0, index=clients, columns=[f"text_{f}" for f in POSITIVE_LABELS])
        if streams.empty:
            return feats
        probs = model.predict_proba(streams)
        classes = model.named_steps["classifier"].classes_
        scores, _ = aggregate_stream_probabilities(streams, probs, classes)
        for family in POSITIVE_LABELS:
            if family in scores.columns:
                feats[f"text_{family}"] = scores[family].reindex(clients).fillna(0.0)
        return feats

    train_text = text_feat(data.train_transactions, train_hist.index, vmax.text_.model_)
    train_matrix = train_hist.join(train_ident).join(train_stream).join(train_text).fillna(0.0)
    y_train = data.train_labels.set_index("client_id")[TARGET_COLUMN].reindex(train_matrix.index)
    sink = make_model()
    sink.iterations = 500
    sink.depth = 5
    sink.learning_rate = 0.05
    sink.fit(train_matrix, y_train)

    valid_hist = history_builder.transform(data.valid_transactions)
    valid_ident = family_features(ident.mapper_.transform(data.valid_transactions)).filter(
        regex="^identity_"
    )
    valid_streams = build_streams(data.valid_transactions)
    valid_stream = client_stream_features(
        data.valid_transactions, v3.mapper_, valid_hist.index, streams=valid_streams
    )
    valid_text = text_feat(data.valid_transactions, valid_hist.index, vmax.text_.model_)
    valid_matrix = (
        valid_hist.join(valid_ident)
        .join(valid_stream)
        .join(valid_text)
        .reindex(columns=train_matrix.columns)
        .fillna(0.0)
    )
    sink_proba = pd.DataFrame(
        sink.predict_proba(valid_matrix), index=valid_matrix.index, columns=LABELS
    ).reindex(yv.index)
    print(f"kitchen-sink in {perf_counter() - t1:.1f}s", flush=True)
    _score(yv, sink_proba.idxmax(axis=1), results, "kitchen_sink")
    for w in (0.3, 0.5, 0.65, 0.75):
        blend = mix_proba(v2_proba, sink_proba, weights=[w, 1 - w])
        _score(yv, blend.idxmax(axis=1), results, f"sink_v2_{w}")
    for w in (0.4, 0.5, 0.6):
        blend = mix_proba(vmax_proba, sink_proba, weights=[w, 1 - w])
        _score(yv, blend.idxmax(axis=1), results, f"sink_vmax_{w}")
    for am, as_ in ((0.6, 0.05), (0.5, 0.0)):
        boosted = soft_family_boost(
            mix_proba(v2_proba, sink_proba, weights=[0.5, 0.5]),
            text_scores,
            alphas={"music": am, "streaming": as_},
        )
        _score(yv, boosted.idxmax(axis=1), results, f"sink_v2_ms_{am}_{as_}")

    # Power mean / temperature exploration (wild but cheap)
    for temp in (0.7, 1.0, 1.5, 2.0):
        logits = np.log(np.clip(vmax_proba.to_numpy(), 1e-12, 1.0)) / temp
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        sharpened = pd.DataFrame(
            w / w.sum(axis=1, keepdims=True), index=vmax_proba.index, columns=LABELS
        )
        mixed = mix_proba(sharpened, a_proba, weights=[0.55, 0.45])
        _score(yv, mixed.idxmax(axis=1), results, f"temp_{temp}_vmax_astra")

    # Margin arbitration: when vmax and astra disagree, pick higher top-margin
    vmax_margin = vmax_proba.max(axis=1) - np.sort(vmax_proba.to_numpy(), axis=1)[:, -2]
    a_margin = a_proba.max(axis=1) - np.sort(a_proba.to_numpy(), axis=1)[:, -2]
    arb = vmax_proba.idxmax(axis=1).copy()
    prefer_a = a_margin.gt(vmax_margin) & a_proba.idxmax(axis=1).ne(arb)
    arb.loc[prefer_a] = a_proba.idxmax(axis=1).loc[prefer_a]
    _score(yv, arb, results, "margin_arb_vmax_astra")

    ranking = sorted(
        ((k, v["macro_f1"]) for k, v in results.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    summary = {
        "seconds": perf_counter() - started,
        "baseline_v3max": 0.4318614926653391,
        "astra_reported_A": 0.424111098,
        "astra_reported_ens": 0.429901160,
        "top20": ranking[:20],
        "results": results,
    }
    (OUT / "push_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"top20": ranking[:20]}, indent=2))
    print(f"done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
