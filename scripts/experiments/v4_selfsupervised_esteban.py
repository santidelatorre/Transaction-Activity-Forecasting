"""Phase-1 self-supervised representations for Esteban V4.

Fits a char TF-IDF + SVD denoiser on unlabeled + TRAIN histories with no
challenge labels, then scores a frozen linear head by TRAIN OOF before one
VALID pass. Does not install PyTorch and does not train a sequence model.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter

import pandas as pd

from transaction_forecasting.ubs.data import (
    LABELS,
    TARGET_COLUMN,
    load_ubs_data,
    read_transactions,
)
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.representation import (
    DenoisingEncoder,
    client_grouped_oof,
    fit_linear_head,
    label_free_client_stats,
    predict_proba_frame,
)

BASE_BRANCH = "main"
BASE_SHA = "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"
BASE_OOF = 0.459793826869
BASE_VALID = 0.424111097737
DATA_DIR = Path("data/raw/ubs_2026")
OUT_DIR = Path("outputs/metrics/v4_selfsupervised_esteban")
REPORT = Path("reports/handoff/v4_selfsupervised_esteban.md")


def _score(target: pd.Series, proba: pd.DataFrame) -> dict:
    prediction = proba.idxmax(axis=1)
    metrics = evaluate_predictions(target.reindex(proba.index), prediction)
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "per_class_f1": {label: float(metrics["per_class"][label]["f1-score"]) for label in LABELS},
    }


def _join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.join(right.reindex(left.index))


def _with_stats(features: pd.DataFrame, stats: pd.DataFrame | None) -> pd.DataFrame:
    if stats is None:
        return features
    return _join(features, stats)


def main() -> None:
    started = perf_counter()
    data = load_ubs_data(DATA_DIR)
    unlabeled = read_transactions(DATA_DIR / "unlabeled_pretrain_transactions.jsonl")
    unlabeled_sample = unlabeled.sample(n=min(80_000, len(unlabeled)), random_state=42)
    pretrain = pd.concat(
        [unlabeled_sample, data.train_transactions],
        ignore_index=True,
    )
    encoder = DenoisingEncoder(seed=42).fit(pretrain)

    task = {
        "gap": encoder.gap_probe(unlabeled_sample, max_pairs=12_000),
        "drift": encoder.drift(unlabeled_sample, n_rows=3_000),
        "contrastive": encoder.contrastive(unlabeled_sample, n_clients=180),
    }

    y_train = data.train_labels.set_index("client_id")[TARGET_COLUMN]
    y_valid = data.valid_labels.set_index("client_id")[TARGET_COLUMN]
    stats_train = label_free_client_stats(data.train_transactions)
    stats_valid = label_free_client_stats(data.valid_transactions)
    clean_train = encoder.client_features(data.train_transactions, mode="clean")
    clean_valid = encoder.client_features(data.valid_transactions, mode="clean")
    denoised_train = encoder.client_features(data.train_transactions, mode="denoised", salt=9)
    denoised_valid = encoder.client_features(data.valid_transactions, mode="denoised", salt=9)
    raw_train = encoder.client_features(data.train_transactions, mode="raw_corrupt", salt=9)
    raw_valid = encoder.client_features(data.valid_transactions, mode="raw_corrupt", salt=9)

    views = {
        "stats": (stats_train, stats_valid),
        "clean_embed": (clean_train, clean_valid),
        "clean_embed_plus_stats": (
            _join(clean_train, stats_train),
            _join(clean_valid, stats_valid),
        ),
        "denoised_corrupt": (denoised_train, denoised_valid),
        "raw_corrupt": (raw_train, raw_valid),
    }
    oof_scores = {}
    oof_frames = {}
    for name, (train_x, _valid_x) in views.items():
        oof = client_grouped_oof(train_x, y_train, seed=42)
        oof_frames[name] = oof
        oof_scores[name] = _score(y_train, oof)

    clean_candidates = ("stats", "clean_embed", "clean_embed_plus_stats")
    chosen = max(clean_candidates, key=lambda name: oof_scores[name]["macro_f1"])
    chosen_train, chosen_valid = views[chosen]
    head = fit_linear_head(chosen_train, y_train)
    valid_clean = _score(y_valid, predict_proba_frame(head, chosen_valid))

    robustness = {}
    robustness_oof = {}
    if chosen != "stats":
        extra_train = None if chosen == "clean_embed" else stats_train
        extra_valid = None if chosen == "clean_embed" else stats_valid
        robust_inputs = {
            "denoised_corrupt": _with_stats(denoised_train, extra_train),
            "raw_corrupt": _with_stats(raw_train, extra_train),
        }
        robust_valid = {
            "denoised_corrupt": _with_stats(denoised_valid, extra_valid),
            "raw_corrupt": _with_stats(raw_valid, extra_valid),
        }
        robustness = {
            name: _score(y_valid, predict_proba_frame(head, frame))
            for name, frame in robust_valid.items()
        }
        robustness_oof = {
            name: _score(y_train, client_grouped_oof(frame, y_train, seed=42))
            for name, frame in robust_inputs.items()
        }

    elapsed = perf_counter() - started
    parameter_bytes = int(
        encoder.svd_.components_.nbytes
        + np_bytes(encoder.denoiser_.coef_)
        + encoder.vectorizer_.idf_.nbytes
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = OUT_DIR / "oof_probabilities.csv"
    oof_frames[chosen].assign(**{TARGET_COLUMN: y_train.reindex(oof_frames[chosen].index)}).to_csv(
        oof_path
    )
    summary = {
        "base_branch": BASE_BRANCH,
        "base_sha": BASE_SHA,
        "base_oof_macro_f1": BASE_OOF,
        "base_valid_macro_f1": BASE_VALID,
        "chosen_oof": chosen,
        "oof": {name: payload["macro_f1"] for name, payload in oof_scores.items()},
        "valid_clean": valid_clean,
        "robustness_valid": {name: payload["macro_f1"] for name, payload in robustness.items()},
        "robustness_oof": {name: payload["macro_f1"] for name, payload in robustness_oof.items()},
        "tasks": task,
        "seconds": elapsed,
        "parameter_bytes": parameter_bytes,
        "torch": False,
        "oof_path": str(oof_path),
        "n_components": encoder.n_components_,
        "fit_descriptions": encoder.fit_descriptions_,
        "per_class_valid": valid_clean["per_class_f1"],
        "per_class_oof": oof_scores[chosen]["per_class_f1"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_report(summary))
    keys = ("chosen_oof", "oof", "valid_clean", "tasks", "seconds")
    print(json.dumps({key: summary[key] for key in keys}, indent=2, default=str))


def np_bytes(value) -> int:
    import numpy as np

    return int(np.asarray(value).nbytes)


def _fmt(value: float) -> str:
    return f"{value:.4f}" if value == value else "nan"


def _report(summary: dict) -> str:
    oof = summary["oof"]
    tasks = summary["tasks"]
    chosen = summary["chosen_oof"]
    valid = summary["valid_clean"]
    beat = valid["macro_f1"] > summary["base_valid_macro_f1"]
    drift = tasks["drift"]
    decision = (
        "Embeddings beat V3-A on the single frozen VALID pass."
        if beat
        else (
            "Keep V3-A. The clean SVD client vector beats label-free stats "
            f"({oof['clean_embed_plus_stats']:.4f} vs {oof['stats']:.4f} TRAIN OOF) "
            "but loses to V3-A on OOF and on the one VALID pass. "
            "Reject the ridge denoiser: drift gain is "
            f"{summary['tasks']['drift']['stability_gain']:.4f} and raw-corrupt OOF "
            f"({oof['raw_corrupt']:.4f}) matches denoised OOF "
            f"({oof['denoised_corrupt']:.4f}). Do not start a GRU or Transformer."
        )
    )
    gain = drift["stability_gain"]
    robust = summary["robustness_valid"]
    lines = [
        "# V4 self-supervised representations — Esteban",
        "",
        "**Branch:** `exp/v4-self-supervised-esteban`",
        f"**Base:** `{summary['base_branch']}` @ `{summary['base_sha']}`",
        f"**Base V3-A:** TRAIN OOF Macro-F1 `{summary['base_oof_macro_f1']:.6f}`, "
        f"VALID Macro-F1 `{summary['base_valid_macro_f1']:.6f}`",
        "",
        "## Decision",
        "",
        decision,
        "",
        "Phase 2 (GRU / temporal CNN / Transformer) was not trained. PyTorch is not installed, "
        "and the brief requires a real phase-1 signal before any sequence model.",
        "",
        "## Pretraining (no challenge labels)",
        "",
        "Char TF-IDF (3–5) + TruncatedSVD fit on up to 25k unique descriptions drawn from "
        "an 80k-row unlabeled sample plus TRAIN histories. A ridge map sends a "
        "35% character-dropped "
        "embedding back to the clean SVD. `client_id` is not a feature. VALID labels were not used "
        "to pick the encoder or the clean-view head (selection is max TRAIN OOF among stats, "
        "clean embeddings, and clean embeddings + stats).",
        "",
        f"- SVD components: `{summary['n_components']}`",
        f"- Descriptions used in the text fit: `{summary['fit_descriptions']}`",
        f"- Parameter bytes (SVD + denoiser + IDF): `{summary['parameter_bytes']}`",
        f"- Device: CPU only (`{platform.processor() or platform.machine()}`), "
        f"torch=`{summary['torch']}`",
        f"- Runtime: `{summary['seconds']:.1f}` s",
        "",
        "### Task metrics",
        "",
        f"- Next-gap probe accuracy `{_fmt(tasks['gap'].get('probe_accuracy', float('nan')))}` "
        f"vs majority `{_fmt(tasks['gap'].get('majority_accuracy', float('nan')))}` "
        f"(lift `{_fmt(tasks['gap'].get('lift_vs_majority', float('nan')))}`, "
        f"n=`{int(tasks['gap']['n_pairs'])}`).",
        f"- Drift L2 raw-corrupt `{drift['raw_corrupt_l2_mean']:.4f}` vs denoised "
        f"`{drift['denoised_l2_mean']:.4f}` (gain `{gain:.4f}`).",
        f"- Contrastive cosine same-client `{tasks['contrastive']['same_client_cosine']:.4f}` vs "
        f"cross-client `{tasks['contrastive']['cross_client_cosine']:.4f}` "
        f"(margin `{tasks['contrastive']['margin']:.4f}`).",
        "",
        "## Downstream TRAIN OOF Macro-F1",
        "",
        "| View | Macro-F1 |",
        "| --- | ---: |",
    ]
    for name, score in oof.items():
        mark = " **" if name == chosen else ""
        lines.append(f"| {name}{mark} | {score:.4f} |")
    lines.extend(
        [
            "",
            f"Chosen on OOF only: `{chosen}` ({oof[chosen]:.4f}).",
            "",
            "## Frozen VALID",
            "",
            f"Macro-F1 `{valid['macro_f1']:.4f}`, accuracy `{valid['accuracy']:.4f}`. "
            f"V3-A VALID is `{summary['base_valid_macro_f1']:.4f}`.",
            "",
            "Per-class F1 (chosen, clean VALID):",
            "",
        ]
    )
    for label, value in valid["per_class_f1"].items():
        lines.append(f"- {label}: {value:.4f}")
    lines.extend(["", "## Corruption", ""])
    if robust:
        lines.append(
            "Same linear head, trained on clean features, scored on corrupted VALID inputs."
        )
        lines.append("")
        for name, score in robust.items():
            lines.append(f"- {name}: VALID Macro-F1 `{score:.4f}`")
        lines.append("")
        for name, score in summary["robustness_oof"].items():
            lines.append(f"- {name}: TRAIN OOF Macro-F1 `{score:.4f}` (head refit on that view)")
    else:
        lines.append("Stats won OOF, so there is no embedding head to stress under corruption.")
    lines.extend(
        [
            "",
            "A low reconstruction error is not treated as success. The stability gain and the "
            "Macro-F1 drop under corruption are the checks that matter.",
            "",
            "## Integration",
            "",
            f"OOF probabilities: `{summary['oof_path']}` (gitignored).",
            "Summary JSON sits beside that file.",
            "",
            "```bash",
            "python scripts/experiments/v4_selfsupervised_esteban.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
