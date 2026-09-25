"""Frozen V3-A TRAIN-only stress suite. Never reads VALID/TEST labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from transaction_forecasting.ubs.corruption import CorruptionSuite, shift_metrics
from transaction_forecasting.ubs.corruption_evaluation import (
    StressDataset,
    client_folds,
    evaluate_under_corruption,
)
from transaction_forecasting.ubs.data import LABELS, read_labels, read_transactions
from transaction_forecasting.ubs.v3.model import V3Model

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA = "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"
BRANCH = "exp/v4-shift-corruption-ginestar"
DEFAULT_OUT = ROOT / "outputs/metrics/v4_shift_corruption_ginestar"


class BaselineAdapter:
    """Unmodified V3 predictor; expose fixed A plus existing diagnostic components."""

    def fit(self, history, labels):
        self.model = V3Model().fit(history, labels)
        return self

    def predict_proba(self, history):
        components = self.model.predict_components(history)
        return {name: components[name] for name in ("A", "V2", "history_control")}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def markdown_table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def report(result, path):
    cards = result["scorecards"]

    def shift_mean(level, key):
        values = [p[key] for p in result["shift_by_fold"] if p["view"] == level]
        return f"{np.mean(values):.4f}"

    def confidence_mean(level, key):
        values = [
            r["confidence"][key]
            for r in result["fold_results"]
            if r["model"] == "A" and r["view"] == level
        ]
        return f"{np.mean(values):.4f}"

    lines = [
        "# V4 merchant-text corruption — Ginestar",
        "",
        f"Branch `{BRANCH}`. Frozen reference `origin/baseline/v4-frozen` at `{BASE_SHA}`.",
        "V3-A from `scripts/run_ubs_v3.py`; historical VALID Macro-F1 0.424111097737.",
        "That score is provenance only, never a selection threshold. This experiment reads "
        "no VALID/TEST labels.",
        "",
        "## Protocol",
        "",
        "The prompt placeholders were resolved from the fetched frozen baseline ref, which "
        "exactly matched initial HEAD. "
        "The unrelated untracked `scripts/emergency_submission.py` was preserved.",
        "Read V3 discovery/protocol, dataset analysis, mapping, NONE-gate, calibration, "
        "disagreement and seven-way synthesis. "
        "They show that clean OOF gains can fail under transfer; these historical reports "
        "already disclose reused VALID results. "
        "No names or thresholds from those results enter this suite.",
        "",
        "Five outer stratified folds of unique TRAIN clients, sorted by ID, seed 42: "
        "identical to the baseline. "
        "Each model is fitted once per fold on clean histories; all views of a held client "
        "remain in that fold. "
        "V3's five inner folds still cross-fit supervised identity. Corruptions are "
        "inference stress, not data augmentation. "
        "The unchanged V3 model and 75/25 weights are reused. A, V2 and the history "
        "classifier share those fits. "
        "No baseline hyperparameter or corruption level is selected using scores.",
        "",
        "Only descriptions in streams anchored by outgoing card payments change, including "
        "matching transfers/refunds. Every date, amount, MCC, type, currency, "
        "direction, fee, "
        "client and target stays intact. No event is dropped. IDs only key deterministic "
        "randomness/folds. "
        "The generic vocabulary is learned independently inside each outer fit, without labels.",
        "",
        "Parameters fixed before the first model fit; seed 20260925. Family order: generic "
        "→ opaque stream mask → "
        "compatible collision → temporal alias fragmentation → token abbreviation → event dropout.",
        "",
    ]
    config = result["suite"]["levels"]
    headers = [
        "level",
        "dropout",
        "generic",
        "stream_mask",
        "fragmentation",
        "collision",
        "decoration",
        "aliases",
    ]
    lines += [
        markdown_table(
            headers, [[level, *[config[level][key] for key in headers[1:]]] for level in config]
        ),
        "",
        "Dropout uses an explicit missing token on sampled events. Generic masking uses "
        "pairs of TRAIN tokens "
        "occurring in >=3 MCCs and >=10 distinct descriptions (top 12 by breadth); a "
        "neutral sentinel is the fallback. "
        "Opaque masking replaces a whole stream by a unique neutral alias, preserving its "
        "grouping. "
        "Fragmentation adds 2/3/4 temporally contiguous suffix aliases to selected "
        "repeated streams. "
        "Collision shares a synthetic alias only within equal MCC/type/currency/direction. "
        "Decoration abbreviates tokens longer than four characters to three and adds punctuation. "
        "Random draws are SHA-256 keyed and independent across families; composition can "
        "overwrite earlier operations. "
        "Probabilities describe operation selection, not the eventual unique fraction changed.",
        "",
        "## Pooled TRAIN OOF scorecard",
        "",
    ]
    rows = []
    for model, levels in cards.items():
        for level in config:
            v = levels[level]
            rows.append(
                [
                    model,
                    level,
                    f"{v['macro_f1']:.9f}",
                    f"{v['accuracy']:.4f}",
                    f"{v['drop_absolute']:+.6f}",
                    f"{v['drop_relative']:.2%}" if v["drop_relative"] is not None else "NA",
                    f"{v['fold_macro_f1_mean']:.6f} ± {v['fold_macro_f1_std']:.6f}",
                ]
            )
    lines += [
        markdown_table(
            ["model", "view", "Macro-F1", "accuracy", "F1 drop", "relative drop", "fold mean ± SD"],
            rows,
        ),
        "",
        "Drops = clean minus corrupted (negative means improvement). Headline is pooled "
        "fixed-eight-class F1, "
        "not the mean of fold F1. Fold SD is descriptive over five dependent fits.",
        "",
        "## A: class F1, prediction counts and confusion",
        "",
    ]
    for level in config:
        v = cards["A"][level]
        lines += [
            f"### {level}",
            "",
            markdown_table(
                ["class", "F1", "drop vs clean", "predicted", "support"],
                [
                    [
                        c,
                        f"{v['per_class'][c]['f1']:.6f}",
                        f"{v['per_class_f1_drop'][c]:+.6f}",
                        v["per_class"][c]["predicted_count"],
                        v["per_class"][c]["support"],
                    ]
                    for c in LABELS
                ],
            ),
            "",
            "Rows true; columns predicted:",
            "",
            markdown_table(
                ["true", *LABELS],
                [[c, *[v["confusion_true_by_predicted"][c][p] for p in LABELS]] for c in LABELS],
            ),
            "",
            "Fold F1 drops: " + ", ".join(f"{v:+.6f}" for v in v["fold_drop_absolute"]) + ".",
            "",
        ]
    lines += [
        "## Isolated families at medium severity",
        "",
        "These controls use the same fitted models and held clients, with one family enabled. "
        "They explain sensitivity without altering the predefined main suite. Effects are "
        "not additive.",
        "",
        markdown_table(
            ["operation", "A F1", "A drop", "V2 drop", "history drop"],
            [
                [
                    name.removeprefix("only_"),
                    f"{cards['A'][name]['macro_f1']:.6f}",
                    *[f"{cards[model][name]['drop_absolute']:+.6f}" for model in cards],
                ]
                for name in cards["A"]
                if name.startswith("only_")
            ],
        ),
        "",
        "## Label-free shift and score geometry",
        "",
        "Means below are across held folds. Vocabulary reference is each clean fit partition; "
        "external diagnostics use all TRAIN as reference after the suite is frozen. "
        "Generic fraction is a lexical proxy "
        "(only TRAIN broad tokens or synthetic generic/missing/collision masks), not "
        "annotated merchant genericity. "
        "Amount CV uses repeated same-client/description/currency streams. Currency units "
        "are never mixed.",
        "",
    ]
    profile_fields = [
        "changed_payment_fraction",
        "generic_description_fraction",
        "unseen_description_rate",
        "vocabulary_overlap_jaccard",
        "description_entropy_nats",
        "streams_per_client",
        "fragmented_original_stream_fraction",
        "collided_result_stream_fraction",
        "amount_cv_median_same_currency_repeated_streams",
    ]
    lines += [
        markdown_table(
            ["view", *profile_fields],
            [
                [
                    level,
                    *[shift_mean(level, key) for key in profile_fields],
                ]
                for level in config
            ],
        ),
        "",
        "MCC/type/currency distributions are identical across paired views by construction "
        "and recorded in JSON. "
        "Synthetic alias/merge rates use original event alignment; on external inputs only "
        "ordinary stream proxies are available. "
        "Model scores are mixture scores, not calibrated confidence.",
        "",
        markdown_table(
            ["view", "A mean max score", "A entropy nats", "A mean none score"],
            [
                [
                    level,
                    *[
                        confidence_mean(level, key)
                        for key in (
                            "max_probability_mean",
                            "entropy_nats_mean",
                            "none_probability_mean",
                        )
                    ],
                ]
                for level in config
            ],
        ),
        "",
    ]
    if result.get("external_shift"):
        lines += [
            "External input diagnostics were run only after all TRAIN scoring and protocol "
            "freeze; no target labels "
            "or model predictions on VALID/TEST are read. These values never select severity.",
            "",
            markdown_table(
                [
                    "input",
                    "generic proxy",
                    "unseen",
                    "vocab Jaccard",
                    "entropy",
                    "streams/client",
                    "amount CV",
                ],
                [
                    [
                        name,
                        *[
                            f"{p[key]:.4f}"
                            for key in (
                                "generic_description_fraction",
                                "unseen_description_rate",
                                "vocabulary_overlap_jaccard",
                                "description_entropy_nats",
                                "streams_per_client",
                                "amount_cv_median_same_currency_repeated_streams",
                            )
                        ],
                    ]
                    for name, p in result["external_shift"].items()
                ],
            ),
            "",
        ]
    losses = sorted(cards["A"]["severe"]["per_class_f1_drop"].items(), key=lambda item: -item[1])
    worst = max(
        (k for k in cards["A"] if k.startswith("only_")),
        key=lambda k: cards["A"][k]["drop_absolute"],
    )
    lines += [
        "## Conclusions and limitations",
        "",
        f"A's severe Macro-F1 drop is {cards['A']['severe']['drop_absolute']:.6f}. "
        "Most sensitive isolated operation: "
        f"`{worst}` ({cards['A'][worst]['drop_absolute']:+.6f}). Largest severe class F1 losses: "
        + ", ".join(f"{c} ({loss:+.4f})" for c, loss in losses[:3])
        + ".",
        "",
        "A versus V2 contrasts probe additional supervised identity; V2 versus history "
        "probes its mapped heuristic. "
        "History still contains description-grouped recurrence features, so it is not text "
        "independent. "
        "The isolated stream mask preserves recurrence grouping and isolates unfamiliar "
        "identity better than event dropout; "
        "fragmentation/collisions also disturb the recurrence representation even though "
        "true timestamps stay fixed.",
        "",
        "This suite is a paired stress test, not an estimator of the hidden test score and "
        "not proof that synthetic "
        "text changes explain all observed cohort shift. TRAIN labels remain latent truth "
        "after renaming merchants. "
        "Broad tokens can include ambiguous terms and synthetic aliases can be more "
        "unfamiliar than real aliases. "
        "Map-size transfer, semantic changes, missing events, new merchants, and behavior "
        "drift are not simulated. "
        "One corruption seed and five folds do not establish seed robustness. External "
        "genericity and amount dispersion "
        "are imperfect proxies; historic reports informed the task, so no claim of "
        "untouched hypothesis development is made.",
        "",
        "No family was removed after inspecting scores. Probability-one/all-text erasure "
        "and cross-MCC/cross-currency "
        "collisions are excluded from the principal protocol as implausible/destructive "
        "extremes. Even severe preserves "
        "all nontextual evidence and substantial unaltered payment descriptions. Report "
        "both clean and stress scores "
        "for future models; do not retune this suite to make a candidate win.",
        "An interrupted implementation pilot masked only card rows, leaving 14,989 TRAIN "
        "transfer/refund events with matching stream keys visible. It was discarded before "
        "completion. The corrected run masks complete anchored streams. Its probabilities "
        "and seeds are unchanged; pilot outputs and source remain in an ignored archive. "
        "No result in this report comes from that pilot.",
        "",
        "## Reproduction and API",
        "",
        "```powershell",
        "$env:PYTHONPATH='src;.'",
        ".venv/Scripts/python.exe scripts/experiments/v4_shift_corruption.py --external-inputs `",
        "  --output-dir outputs/metrics/v4_shift_corruption_ginestar_repro",
        ".venv/Scripts/python.exe -m pytest",
        ".venv/Scripts/python.exe -m ruff check .",
        ".venv/Scripts/python.exe -m ruff format --check .",
        "```",
        "",
        "```python",
        "dataset = StressDataset(train_history, train_labels)",
        "scorecard = evaluate_under_corruption(model_factory, dataset, CorruptionSuite())",
        "```",
        "",
        "Factories return fresh fit(history, labels) models with indexed client "
        "predictions or official-order probabilities. "
        "Multiple named probability components are supported; no baseline-specific imports "
        "exist in the reusable evaluator. "
        "Raw IDs and local probabilities stay under ignored outputs. summary.json contains "
        "every model/view's eight-class "
        "metrics, confusion, prediction counts, per-fold scores, confidence, shifts, "
        "parameters, hashes and timings.",
        "",
        f"Total evaluation runtime: {result['runtime_seconds']:.1f} seconds; "
        "environment and source/input hashes are in JSON.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/ubs_2026")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--external-inputs", action="store_true")
    args = parser.parse_args()

    def git(*parts):
        return subprocess.check_output(["git", *parts], text=True).strip()

    if git("branch", "--show-current") != BRANCH:
        raise ValueError("Run only on assigned experimental branch")
    protected = [
        "scripts/run_ubs_v3.py",
        "src/transaction_forecasting/evaluation/official.py",
        *[
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in (ROOT / "src/transaction_forecasting/ubs").rglob("*.py")
            if "corruption" not in p.name
        ],
    ]
    if git("diff", BASE_SHA, "--", *protected):
        raise ValueError("Frozen baseline/evaluator source differs from BASE_SHA")
    out = args.output_dir
    if (out / "frozen_protocol.json").exists():
        raise ValueError("Use a fresh output directory; never overwrite a frozen run")
    out.mkdir(parents=True, exist_ok=True)
    train_path, labels_path = (
        args.data_dir / "train_transactions.jsonl",
        args.data_dir / "train_labels.csv",
    )
    dataset = StressDataset(read_transactions(train_path), read_labels(labels_path))
    suite = CorruptionSuite()
    folds = client_folds(dataset)
    sources = [ROOT / p for p in protected] + [
        Path(__file__),
        ROOT / "src/transaction_forecasting/ubs/corruption.py",
        ROOT / "src/transaction_forecasting/ubs/corruption_evaluation.py",
    ]
    provenance = {
        "base_branch": "baseline/v4-frozen",
        "base_sha": BASE_SHA,
        "base_runner": "scripts/run_ubs_v3.py",
        "base_macro_f1_valid_historical": 0.424111097737,
        "expected_clean_oof_A": 0.459793826869,
        "commit": git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost")
        },
        "hashes": {
            str(p.relative_to(ROOT)): digest(p) for p in [*sources, train_path, labels_path]
        },
        "fold_assignment_sha256": hashlib.sha256(folds.to_csv().encode()).hexdigest(),
        "suite": suite.config(),
        "fold_seed": 42,
        "corruption_seed": suite.seed,
        "frozen_before_fit": True,
        "validation_labels_accessed": False,
        "test_labels_accessed": False,
        "selection": "none; fixed stress protocol",
    }
    write_json(out / "frozen_protocol.json", provenance)
    started = perf_counter()
    result = evaluate_under_corruption(BaselineAdapter, dataset, suite, folds=folds, output_dir=out)
    result["provenance"] = provenance
    clean = result["scorecards"]["A"]["clean"]["macro_f1"]
    result["baseline_reproduction"] = {
        "expected": provenance["expected_clean_oof_A"],
        "actual": clean,
        "matches_1e_9": abs(clean - provenance["expected_clean_oof_A"]) < 1e-9,
    }
    result["external_shift"] = {}
    if args.external_inputs:
        reference_suite = CorruptionSuite().fit(dataset.history)
        for split in ("train", "valid", "test"):
            path = args.data_dir / f"{split}_transactions.jsonl"
            history = dataset.history if split == "train" else read_transactions(path)
            if split != "train" and set(history.client_id) & set(dataset.history.client_id):
                raise ValueError("External input overlaps TRAIN")
            result["external_shift"][split] = shift_metrics(
                history, dataset.history, reference_suite
            )
            result["external_shift"][split]["input_sha256"] = digest(path)
    if any(digest(ROOT / path) != expected for path, expected in provenance["hashes"].items()):
        raise RuntimeError("Frozen code or TRAIN input changed during run")
    result["runtime_seconds"] = perf_counter() - started
    result["conclusions"] = {
        "selection_performed": False,
        "severe_macro_f1_drop": {
            model: cards["severe"]["drop_absolute"] for model, cards in result["scorecards"].items()
        },
        "most_sensitive_operation": {
            model: max(
                (key for key in cards if key.startswith("only_")),
                key=lambda key: cards[key]["drop_absolute"],
            )
            for model, cards in result["scorecards"].items()
        },
        "severe_class_sensitivity_descending": {
            model: sorted(
                cards["severe"]["per_class_f1_drop"],
                key=lambda key: -cards["severe"]["per_class_f1_drop"][key],
            )
            for model, cards in result["scorecards"].items()
        },
        "interpretation": "Measured stress sensitivity; no causal claim about real cohort shift.",
    }
    result["limitations"] = [
        "One corruption seed; fold dispersion is descriptive, not seed robustness.",
        "Synthetic alias rates need not match VALID/TEST; severity is not calibrated to them.",
        "Genericity and fragmentation are proxies, not verified real merchant identities.",
        "No map-size, real semantic, missing-event, or behavioral shift is simulated.",
        "History component still uses exact-description recurrence groups.",
        "TRAIN and historical reports have been studied before; this is not an untouched cohort.",
        "Card-row-only pilot was discarded; final run includes whole anchored streams.",
    ]
    write_json(out / "summary.json", result)
    report(result, ROOT / "reports/handoff/v4_shift_corruption_ginestar.md")
    print(
        json.dumps(
            {level: result["scorecards"]["A"][level]["macro_f1"] for level in suite.levels},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
