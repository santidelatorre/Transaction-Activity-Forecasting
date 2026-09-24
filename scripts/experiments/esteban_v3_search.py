"""Aggressive V3 search: stream overrides, stream+history CatBoost, blend grid."""

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
from transaction_forecasting.ubs.v2 import IntegratedV2Model
from transaction_forecasting.ubs.v3 import StreamV3Model, client_stream_features

OUT = Path("outputs/metrics/v3_try")


def _payload(metrics, pred):
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "per_class": {k: v["f1-score"] for k, v in metrics["per_class"].items()},
        "pred_dist": pred.value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
    }


def override_v2(v2_pred, stream_features, *, min_weight, margin):
    pred = v2_pred.copy()
    single = stream_features["stream_single_family"].gt(0.5) & stream_features["stream_margin"].ge(
        margin
    )
    eligible = single & stream_features["stream_top_weight"].ge(min_weight)
    top_idx = stream_features["stream_top_family_index"].astype(int)
    for client in stream_features.index[eligible]:
        pred.loc[client] = POSITIVE_LABELS[int(top_idx.loc[client])]
    return pred


def multi_family_argmax(stream_features):
    cols = [f"due_{f}_max_weight" for f in POSITIVE_LABELS]
    weights = stream_features[cols]
    pred = pd.Series("none", index=stream_features.index, dtype=object)
    has = stream_features["stream_top_weight"].gt(0)
    idx = weights.to_numpy().argmax(axis=1)
    for i, _client in enumerate(stream_features.index):
        if has.iloc[i]:
            pred.iloc[i] = POSITIVE_LABELS[int(idx[i])]
    return pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    data = load_ubs_data("data/raw/ubs_2026")
    yv = data.valid_labels.set_index("client_id")[TARGET_COLUMN]

    print("building streams...", flush=True)
    t0 = perf_counter()
    train_streams = build_streams(data.train_transactions)
    valid_streams = build_streams(data.valid_transactions)
    print(f"streams done in {perf_counter() - t0:.1f}s", flush=True)

    mapper = TrainOnlyFamilyMapper().fit(train_streams, data.train_labels)
    print("mapped descriptions", int(mapper.mapping_.shape[0]), flush=True)

    stream_valid = client_stream_features(
        data.valid_transactions, mapper, yv.index, streams=valid_streams
    )
    stream_valid.to_csv(OUT / "valid_stream_features.csv")

    v2 = IntegratedV2Model().fit(data.train_transactions, data.train_labels)
    v2_components = v2.predict_components(data.valid_transactions)
    v2_pred = v2_components["blend"].idxmax(axis=1)
    results = {"v2": _payload(evaluate_predictions(yv.reindex(v2_pred.index), v2_pred), v2_pred)}
    print("V2", results["v2"]["macro_f1"], flush=True)

    # Pure stream argmax
    pure = multi_family_argmax(stream_valid)
    results["stream_argmax"] = _payload(evaluate_predictions(yv.reindex(pure.index), pure), pure)
    print(
        "stream_argmax",
        results["stream_argmax"]["macro_f1"],
        results["stream_argmax"]["pred_dist"],
        flush=True,
    )

    # Override grids
    best_override = None
    for min_w in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
        for margin in (0.0, 0.05, 0.1, 0.2, 0.5):
            pred = override_v2(v2_pred, stream_valid, min_weight=min_w, margin=margin)
            m = evaluate_predictions(yv.reindex(pred.index), pred)
            key = f"override_w{min_w}_m{margin}"
            payload = _payload(m, pred)
            results[key] = payload
            if best_override is None or payload["macro_f1"] > best_override["macro_f1"]:
                best_override = {**payload, "key": key, "min_w": min_w, "margin": margin}
    print("best_override", best_override["key"], best_override["macro_f1"], flush=True)

    # Soft probability blend: add stream family weights into V2 logits
    best_soft = None
    v2_proba = v2_components["blend"].reindex(yv.index)
    for alpha in (0.5, 1.0, 2.0, 3.0, 5.0):
        logits = np.log(np.clip(v2_proba.to_numpy(), 1e-12, 1.0))
        for family in POSITIVE_LABELS:
            col = LABELS.index(family)
            logits[:, col] += alpha * stream_valid[f"due_{family}_max_weight"].to_numpy()
        # none penalty when any mapped due exists
        logits[:, LABELS.index("none")] -= (
            alpha * 0.3 * stream_valid["stream_mapped_candidate_count"].clip(0, 5).to_numpy()
        )
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        proba = weights / weights.sum(axis=1, keepdims=True)
        pred = pd.Series(np.asarray(LABELS)[proba.argmax(1)], index=yv.index)
        m = evaluate_predictions(yv, pred)
        payload = _payload(m, pred)
        results[f"soft_alpha_{alpha}"] = payload
        if best_soft is None or payload["macro_f1"] > best_soft["macro_f1"]:
            best_soft = {**payload, "alpha": alpha}
    print("best_soft", best_soft["alpha"], best_soft["macro_f1"], flush=True)

    # Full StreamV3Model
    print("fitting StreamV3Model...", flush=True)
    t1 = perf_counter()
    model = StreamV3Model(v2_blend=0.45, override_weight=3.0).fit(
        data.train_transactions, data.train_labels
    )
    # Replace mapper/streams already computed — model rebuilds internally.
    pred_v3 = model.predict(data.valid_transactions)
    results["stream_v3"] = _payload(
        evaluate_predictions(yv.reindex(pred_v3.index), pred_v3), pred_v3
    )
    print(
        "stream_v3", results["stream_v3"]["macro_f1"], f"in {perf_counter() - t1:.1f}s", flush=True
    )

    # Blend grid on StreamV3 components
    comps = model.predict_components(data.valid_transactions)
    best_blend = None
    for w in (0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        blend = w * comps["v2"].to_numpy() + (1 - w) * comps["stream_model"].to_numpy()
        # re-apply override lightly
        blend_df = pd.DataFrame(blend, index=comps["v2"].index, columns=LABELS)
        stre = comps["stream_features"]
        eligible = (
            stre["stream_single_family"].gt(0.5)
            & stre["stream_margin"].ge(0.05)
            & stre["stream_top_weight"].ge(0.8)
        )
        top_idx = stre["stream_top_family_index"].astype(int)
        for client in blend_df.index[eligible]:
            fam = POSITIVE_LABELS[int(top_idx.loc[client])]
            blend_df.loc[client, fam] += float(stre.loc[client, "stream_top_weight"]) * 2.0
        vals = blend_df.to_numpy()
        vals = vals / vals.sum(axis=1, keepdims=True)
        pred = pd.Series(np.asarray(LABELS)[vals.argmax(1)], index=blend_df.index)
        m = evaluate_predictions(yv.reindex(pred.index), pred)
        payload = _payload(m, pred)
        results[f"v3_blend_{w}"] = payload
        if best_blend is None or payload["macro_f1"] > best_blend["macro_f1"]:
            best_blend = {**payload, "w": w}
        print(f"  blend {w}", payload["macro_f1"], flush=True)

    ranking = sorted(
        ((k, v["macro_f1"]) for k, v in results.items() if isinstance(v, dict) and "macro_f1" in v),
        key=lambda item: item[1],
        reverse=True,
    )
    summary = {
        "seconds": perf_counter() - started,
        "mapped_descriptions": int(mapper.mapping_.shape[0]),
        "best_override": best_override,
        "best_soft": best_soft,
        "best_blend": best_blend,
        "top10": ranking[:10],
        "results": results,
    }
    (OUT / "search_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(
        json.dumps(
            {
                "top10": ranking[:10],
                "best_override": best_override,
                "best_soft": best_soft,
                "best_blend": best_blend,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
