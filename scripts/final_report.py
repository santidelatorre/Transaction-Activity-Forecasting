"""Generate final evidence exclusively from executed results and frozen predictions."""

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ubs_recurrence.data import (
    LABELS,
    PREDICTION,
    ROOT,
    validate_submission,
)


def rebuild_ledger():
    records = []
    for path in (ROOT / "outputs/experiments").glob("*/metrics.json"):
        records.append(json.loads(path.read_text()))
    records.sort(key=lambda r: (r["timestamp"], r["experiment_id"]))
    assert len({r["experiment_id"] for r in records}) == len(records)
    for r in records:
        name = r["experiment_id"]
        if name.startswith(
            ("final_final_eval_", "selection_compact_3seed_plus25pct_v1_")
        ):
            decision = "accepted_final"
            reason = "Frozen selected configuration and its independent reproduction."
        elif (
            re.fullmatch(
                r"compact_l15_hierTrue_payments_(original|valid_like|test_like)_s(42|17|2026)",
                name,
            )
            or name.startswith(
                ("robust_augTrue_droptextFalse_noise_", "selection_compact_3seed_")
            )
            or name.startswith("ensemble_equal3_")
            and "biasfit" not in name
            or name == "holdout_v1_ensemble"
        ):
            decision = "accepted_component"
            reason = "Constituent of the frozen 75% compact / 25% robustness blend; all three compact seeds retained."
        else:
            decision = "rejected_from_final"
            reason = "Comparison, ablation or diagnostic not retained as a final predictor; see research_decisions.md for group-specific evidence."
        r["final_decision"] = {
            "status": decision,
            "reason": reason,
            "annotation": "Post-research disposition; original executed metrics and run status unchanged.",
        }
    (ROOT / "reports/experiment_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    rows = []
    log = [
        "# Executed experiments\n\nGenerated from immutable experiment result files. `oof_decision_fit_not_independent` rows are decision-fit diagnostics, not validation. Auxiliary-task scores are not challenge scores. Final dispositions are post-research annotations: accepting a component does not make its exploratory score confirmatory.\n\n| ID | Evaluation | Macro-F1 | Accuracy | Seconds | Final disposition | Hypothesis |\n|---|---|---:|---:|---:|---|---|\n"
    ]
    for r in records:
        row = {
            k: r.get(k)
            for k in [
                "experiment_id",
                "timestamp",
                "git_commit",
                "source_sha256",
                "evaluation_split",
                "macro_f1",
                "accuracy",
                "runtime_seconds",
                "hypothesis",
                "model",
                "seed",
                "protocol",
                "status",
                "conclusion",
            ]
        }
        row["features"] = json.dumps(r.get("features"))
        row["parameters"] = json.dumps(r.get("parameters"))
        row["final_decision"] = r["final_decision"]["status"]
        row.update(
            {"f1_" + l: r["classification_report"][l]["f1-score"] for l in LABELS}
        )
        rows.append(row)
        log.append(
            f"| {r['experiment_id']} | {r['evaluation_split']} | {r['macro_f1']:.6f} | {r['accuracy']:.6f} | {r['runtime_seconds']:.1f} | {r['final_decision']['status']} | {r.get('hypothesis', '')} |\n"
        )
    pd.DataFrame(rows).to_csv(ROOT / "outputs/experiment_results.csv", index=False)
    (ROOT / "reports/experiment_log.md").write_text("".join(log))
    return records


def main():
    a = ROOT / "outputs/final_eval_a"
    b = ROOT / "outputs/final_eval_b"
    pa = pd.read_csv(a / "probabilities.csv")
    pb = pd.read_csv(b / "probabilities.csv")
    assert np.array_equal(pa.client_id, pb.client_id)
    probcols = ["p_" + l for l in LABELS]
    maxdiff = float(np.max(np.abs(pa[probcols].to_numpy() - pb[probcols].to_numpy())))
    same = bool(np.array_equal(pa[PREDICTION], pb[PREDICTION]))
    assert same and maxdiff < 1e-7, (same, maxdiff)
    ma = json.loads((a / "metrics.json").read_text())
    mb = json.loads((b / "metrics.json").read_text())
    assert ma["macro_f1"] == mb["macro_f1"]
    source_diff = subprocess.check_output(
        ["git", "diff", "ffc656e", "--", "src/ubs_recurrence"], cwd=ROOT, text=True
    )
    assert not source_diff, (
        "Prediction source differs from frozen/reproduced implementation"
    )
    sa = ROOT / "outputs/final_submission/submission.csv"
    sb = ROOT / "outputs/final_submission_b/submission.csv"
    assert sa.read_bytes() == sb.read_bytes(), "Independent test submission differs"
    reproduction = {
        "two_independent_raw_data_training_runs": True,
        "all_predictions_identical": same,
        "maximum_probability_difference": maxdiff,
        "test_submissions_byte_identical": True,
        "macro_f1_a": ma["macro_f1"],
        "macro_f1_b": mb["macro_f1"],
        "source_sha_a": ma["model_metadata"]["source_sha256"],
        "source_sha_b": mb["model_metadata"]["source_sha256"],
        "prediction_source_commit": "ffc656e",
        "prediction_source_unchanged": True,
        "source_hash_note": "Full source fingerprints also include research/reporting scripts, which were added between runs; the prediction package is unchanged.",
        "train_clients": ma["model_metadata"]["training_clients"],
        "validation_used_for_training": False,
    }
    (ROOT / "reports/reproduction.json").write_text(json.dumps(reproduction, indent=2))
    known = pd.read_csv(ROOT / "outputs/experiments/final_final_eval_a/predictions.csv")
    assert np.array_equal(known.client_id, pa.client_id)
    mapper = {v: i for i, v in enumerate(LABELS)}
    y = known.true_label.map(mapper).to_numpy()
    yp = known.prediction.map(mapper).to_numpy()
    rng = np.random.default_rng(60319)
    draws = []
    for _ in range(2000):
        ix = rng.integers(0, len(y), len(y))
        draws.append(
            f1_score(
                y[ix], yp[ix], labels=np.arange(8), average="macro", zero_division=0
            )
        )
    ci = np.quantile(draws, [0.025, 0.975]).tolist()
    class_rows = []
    for l in LABELS:
        r = ma["classification_report"][l]
        class_rows.append(
            {
                "class": l,
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1-score"],
                "true_count": ma["true_frequency"][l],
                "predicted_count": ma["prediction_frequency"][l],
            }
        )
    pd.DataFrame(class_rows).to_csv(ROOT / "reports/final_per_class.csv", index=False)
    summary = {
        "macro_f1": ma["macro_f1"],
        "accuracy": ma["accuracy"],
        "macro_f1_bootstrap_95_ci": ci,
        "bootstrap": "2000 IID client bootstrap resamples; fixed predictions; seed 60319; conditional on this validation sample/model selection history",
        "clients": len(y),
        "target_0_80_achieved": bool(ma["macro_f1"] >= 0.8),
        "class_metrics": class_rows,
        "reproduction": reproduction,
    }
    (ROOT / "reports/final_metrics.json").write_text(json.dumps(summary, indent=2))
    # Cohorts use the already-frozen model's observed feature evidence.
    feat = pd.read_parquet(a / "evidence_features.parquet")
    n = (
        feat["client_total_count"]
        .groupby(level=0)
        .first()
        .reindex(pa.client_id)
        .to_numpy()
    )
    cohorts = {
        "all": np.ones(len(y), dtype=bool),
        "low_history_under_40_events": n < 40,
        "high_activity_over_100_events": n > 100,
        "true_none": y == 7,
        "true_recurring_family": y < 7,
    }
    cohort_rows = []
    for name, mask in cohorts.items():
        cohort_rows.append(
            {
                "cohort": name,
                "clients": int(mask.sum()),
                "accuracy": float((yp[mask] == y[mask]).mean()) if mask.any() else None,
                "errors": int((yp[mask] != y[mask]).sum()),
            }
        )
    pd.DataFrame(cohort_rows).to_csv(
        ROOT / "reports/final_error_cohorts.csv", index=False
    )
    family_index = pd.MultiIndex.from_arrays(
        [pa.client_id, np.array(LABELS)[y]], names=feat.index.names
    )
    true_evidence = feat.reindex(family_index)
    count = true_evidence.amount0_count.to_numpy()
    age = true_evidence.amount0_last_age.to_numpy()
    cv = true_evidence.amount0_gap_cv.to_numpy()
    extra_cohorts = {
        "family_without_3_event_amount_candidate": (y < 7) & (count < 3),
        "family_regular_candidate": (y < 7) & (count >= 3) & (cv >= 0) & (cv <= 0.25),
        "family_irregular_candidate": (y < 7) & (count >= 3) & (cv > 0.25),
        "family_last_candidate_event_within_45_days": (y < 7)
        & (count >= 3)
        & (age >= 0)
        & (age <= 45),
        "family_last_candidate_event_over_90_days": (y < 7) & (count >= 3) & (age > 90),
    }
    for name, mask in extra_cohorts.items():
        cohort_rows.append(
            {
                "cohort": name,
                "clients": int(mask.sum()),
                "accuracy": float((yp[mask] == y[mask]).mean()) if mask.any() else None,
                "errors": int((yp[mask] != y[mask]).sum()),
            }
        )
    pd.DataFrame(cohort_rows).to_csv(
        ROOT / "reports/final_error_cohorts.csv", index=False
    )
    mistakes = known[known.true_label != known.prediction]
    confusions = (
        mistakes.groupby(["true_label", "prediction"])
        .size()
        .sort_values(ascending=False)
    )
    confusions.rename("count").to_csv(ROOT / "reports/final_confusions.csv")
    error_counts = {
        "true_family_predicted_none": int(((y < 7) & (yp == 7)).sum()),
        "true_none_predicted_family": int(((y == 7) & (yp < 7)).sum()),
        "wrong_recurring_family": int(((y < 7) & (yp < 7) & (y != yp)).sum()),
    }
    ece = sum(
        z["n"] * abs(z["confidence"] - z["accuracy"]) for z in ma["calibration_bins"]
    ) / len(y)
    analysis = [
        "# Frozen-model error analysis\n\nThis analysis was produced after final model selection and did not change the predictor. All cohorts retain their difficult clients. They are descriptive, overlapping subsets, not separate validation claims.\n\n",
        f"There are {len(mistakes)} errors: {error_counts['true_family_predicted_none']} true recurring families predicted none, {error_counts['true_none_predicted_family']} true none clients predicted a family, and {error_counts['wrong_recurring_family']} wrong-family predictions.\n\n",
        f"Ten-bin top-label calibration error is {ece:.4f}; log loss is {ma['log_loss']:.4f}. Normalized rank scores should not be described as guaranteed calibrated probabilities.\n\n",
        "| Cohort | Clients | Accuracy | Errors |\n|---|---:|---:|---:|\n",
    ]
    for r in cohort_rows:
        accuracy = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.4f}"
        analysis.append(
            f"| {r['cohort']} | {r['clients']} | {accuracy} | {r['errors']} |\n"
        )
    analysis.append(
        "\nThe none boundary is a major source of error. Suppressing none would trade missed recurring families against false alerts; the earlier OOF bias experiment failed to improve external validation. The frozen model therefore retains its unadjusted decision rule. Wrong-family errors remain spread across several pairs, rather than one vocabulary confusion explaining the entire gap.\n\nAmount groups are approximate candidates: background payments with similar amounts can distort cadence, while masking/MCC corruption can obscure family identity. These are plausible failure mechanisms supported by the feature-only shift audit and train-side ablations, not proven causal explanations for every wrong client.\n\nThe most frequent confusion pairs are in `final_confusions.csv`. The next research priorities and rejected alternatives are in `research_decisions.md`. No post-hoc fixes were selected from these official-validation errors.\n"
    )
    (ROOT / "reports/final_error_analysis.md").write_text("".join(analysis))
    # Evidence notes are observed features, not invented causal explanations.
    notes = [
        "# Observed evidence behind frozen validation predictions\n\nThese examples were selected after model freeze for explanation only. They are not training examples. Counts and timing come from actual extracted candidate streams. Normalized model scores are not guaranteed calibrated probabilities. Refund associations do not prove cancellation.\n"
    ]
    for label in [
        "cloud",
        "gym",
        "insurance",
        "mobile",
        "music",
        "software",
        "streaming",
        "none",
    ]:
        subset = known[(known.prediction == label) & (known.true_label == label)]
        if subset.empty:
            continue
        row = subset.sort_values("p_" + label, ascending=False).iloc[0]
        cid = row.client_id
        f = feat.loc[(cid, label)]
        notes.append(
            f"\n## {cid}: predicted {label}, observed label {label}\n\nNormalized model score: {row['p_' + label]:.3f}. "
        )
        if label == "none":
            client = feat.loc[cid]
            available = int((client.amount0_count >= 3).sum())
            notes.append(
                f"The extractor represented {available} family candidates with at least three amount-grouped events. The dedicated none detector and competing-family scores selected none; this does not assert there were no recurring-looking payments.\n"
            )
        else:
            prefix = "amount0" if f.get("amount0_count", -999) >= 3 else "broad0"
            count = f.get(prefix + "_count", -999)
            if count > 0:
                notes.append(
                    f"The selected evidence group contains {int(count)} payments, with median interval {f[prefix + '_gap_median']:.1f} days, last payment {f[prefix + '_last_age']:.1f} days before cutoff, and median nominal amount {f[prefix + '_amount_median']:.2f}. "
                )
                nr = f.get(prefix + "_refund_count", -999)
                if nr >= 0:
                    notes.append(
                        f"The amount/currency matching procedure also found {int(nr)} refunds. "
                    )
                notes.append(
                    "These observations are model inputs, not a deterministic guarantee of the next family or date.\n"
                )
            else:
                notes.append(
                    "No qualifying repeated candidate was extracted for this family; this is a sparse-evidence prediction and should not be presented as a reconstructed subscription.\n"
                )
    (ROOT / "reports/explanations.md").write_text("".join(notes))
    cm = np.asarray(ma["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=np.arange(8),
        yticks=np.arange(8),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted family",
        ylabel="True family",
        title=f"Official validation: macro-F1 {ma['macro_f1']:.4f}",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(8):
        for j in range(8):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, label="Clients")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/confusion_matrix.png", dpi=180)
    plt.close(fig)
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    submission = pd.read_csv(ROOT / "outputs/final_submission/submission.csv")
    validate_submission(submission, sample)
    dest = ROOT / "submissions"
    dest.mkdir(exist_ok=True)
    submission.to_csv(dest / "submission.csv", index=False)
    receipt = json.loads(
        (ROOT / "outputs/final_submission/submission_validation.json").read_text()
    )
    assert (
        hashlib.sha256((dest / "submission.csv").read_bytes()).hexdigest()
        == receipt["sha256"]
    )
    validate_submission(pd.read_csv(dest / "submission.csv"), sample)
    (dest / "validation.json").write_text(json.dumps(receipt, indent=2))
    records = rebuild_ledger()
    summary["executed_result_records"] = len(records)
    (ROOT / "reports/final_metrics.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Final results\n\n",
        f"**Official validation macro-F1: {ma['macro_f1']:.6f}; accuracy: {ma['accuracy']:.6f}.** The 0.80 objective was not reached.\n\n",
        f"Client-bootstrap 95% interval for macro-F1: [{ci[0]:.4f}, {ci[1]:.4f}]. This interval does not account for all model-selection uncertainty or hidden-test shift.\n\n",
        f"Two independent raw-data training runs produced identical class predictions. Maximum probability difference: {maxdiff:.3g}. All 1,000 official validation clients were retained; none was used for supervised fitting.\n\n",
        "| Class | Precision | Recall | F1 | True n | Predicted n |\n|---|---:|---:|---:|---:|---:|\n",
    ]
    for r in class_rows:
        lines.append(
            f"| {r['class']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['true_count']} | {r['predicted_count']} |\n"
        )
    lines.append(
        f"\n{len(records)} executed evaluation records are preserved, including clearly marked auxiliary and decision-fit diagnostics. No auxiliary-task result counts toward the challenge target. See [experiment log](experiment_log.md), [reproduction evidence](reproduction.json), [error cohorts](final_error_cohorts.csv), [confusion counts](final_confusions.csv), and [observed explanations](explanations.md).\n"
    )
    lines.append(
        "\nThe committed submission contains exactly the required test IDs and legal labels; its contract was checked before and after writing the CSV. No hidden-test score is available.\n"
    )
    (ROOT / "reports/final_results.md").write_text("".join(lines))
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "pyarrow",
                "catboost",
                "lightgbm",
                "xgboost",
                "joblib",
                "pytest",
                "matplotlib",
            ]
        },
    }
    (ROOT / "reports/environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
