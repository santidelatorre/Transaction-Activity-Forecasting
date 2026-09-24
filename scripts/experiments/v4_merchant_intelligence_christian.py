"""Frozen A/B/C merchant experiment. OOF never opens VALID/TEST labels.

Unsupervised fingerprints use all TRAIN + unlabeled histories (transductive
within TRAIN); all supervised features use nested client cross-fitting.
Run --phase oof, then --phase valid exactly once after reviewing the freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.merchant_intelligence import (
    KEYS,
    FamilyEvidence,
    MerchantIntelligence,
    build_fingerprints,
    config_dict,
    cross_fitted_summaries,
    summary_features,
)
from transaction_forecasting.ubs.v2 import IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3.features import (
    FamilyMap,
    cross_fitted_family_features,
    family_features,
)

BASE_SHA = "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"
BASE_MACRO_F1 = 0.424111097737
ARMS = ("A_baseline", "B_add_summary", "C_replace_identity")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(root):
    paths = [Path(__file__), *Path("src/transaction_forecasting/ubs").rglob("*.py")]
    paths += [
        root / name
        for name in (
            "train_transactions.jsonl",
            "train_labels.csv",
            "unlabeled_pretrain_transactions.jsonl",
        )
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def cards(streams):
    return streams.loc[streams.direction.eq("out") & streams.type.eq("card_payment")].reset_index(
        drop=True
    )


def family_metrics(target, baseline, merchant):
    """Candidate recall from OOF features, including abstention and positive-only recall."""
    positive = [label for label in LABELS if label != "none"]
    baseline = baseline.reindex(target.index)
    merchant = merchant.reindex(target.index)
    tables = {
        "exact_identity": baseline[[f"identity_{label}_count" for label in positive]].set_axis(
            positive, axis=1
        ),
        "merchant": merchant[[f"mi_{label}_max" for label in positive]].set_axis(positive, axis=1),
    }
    output = {}
    for name, scores in tables.items():
        eligible = target.ne("none")
        result = {}
        order = np.argsort(-scores.to_numpy(), axis=1, kind="stable")
        for k in (1, 3, 5):
            hits = [
                truth in [positive[j] for j in row[:k] if scores.iloc[i, j] > 0]
                for i, (truth, row) in enumerate(zip(target, order, strict=True))
            ]
            result[f"positive_candidate_recall_at_{k}"] = float(np.asarray(hits)[eligible].mean())
        result["clients_unknown_fraction"] = float(scores.max(axis=1).eq(0).mean())
        output[name] = result
    return output


def corruption_diagnostics(index, streams, transactions, sample_size=500):
    """Artificial identity checks, not a benchmark with real merchant IDs.

    Reference contains original TRAIN history, so these measure recognition
    after corruption of known fingerprints, not unseen-merchant generalization.
    """
    sample = (
        streams.loc[streams.recurrence_count.ge(3)]
        .sample(
            n=min(sample_size, int(streams.recurrence_count.ge(3).sum())),
            random_state=42,
        )
        .reset_index(drop=True)
    )
    clean = index.retrieve(sample)
    source = transactions.copy()
    source["amount_band"] = np.floor(np.log2(source.amount.abs().clip(1))).astype(int)
    source = source.merge(sample[KEYS].assign(probe_id=np.arange(len(sample))), on=KEYS)
    # Each sampled stream is an isolated probe. This does not test collisions
    # from masking an entire multi-merchant client history into one string.
    source["client_id"] = source.probe_id.map(lambda value: f"probe_{value:06d}")
    source = source.sort_values(["client_id", "timestamp"], kind="stable")
    output = {
        "sample_streams": len(sample),
        "reference_contains_original_history": True,
        "scope": "transaction corruption and reaggregation of isolated known-stream probes",
    }
    for corruption in (
        "description_masking",
        "alias_fragmentation",
        "typo_decorations",
        "generic_replacement",
    ):
        damaged_transactions = source.copy()
        if corruption == "description_masking":
            damaged_transactions["description"] = ""
        elif corruption == "alias_fragmentation":
            parity = damaged_transactions.groupby("client_id").cumcount().mod(2).astype(str)
            damaged_transactions["description"] = "opaque_alias_" + parity
        elif corruption == "typo_decorations":
            damaged_transactions["description"] = damaged_transactions.description.map(
                lambda text: "** " + text[:2] + text[3:] + " #2025"
            )
        else:
            damaged_transactions["description"] = "payment"
        damaged = build_fingerprints(damaged_transactions)
        source_positions = damaged.client_id.str.removeprefix("probe_").astype(int)
        originals = [clean[i] for i in source_positions]
        matches = index.retrieve(damaged)
        jaccard, top1, recovery, graph_recovery = [], [], [], []
        for original, corrupted in zip(originals, matches, strict=True):
            a, b = {row[0] for row in original}, {row[0] for row in corrupted}
            jaccard.append(len(a & b) / max(1, len(a | b)))
            top1.append(bool(original and corrupted and original[0][0] == corrupted[0][0]))
            recovery.append(bool(original and original[0][0] in b))
            graph_recovery.append(
                bool(
                    original
                    and any(index.clusters_[j] == index.clusters_[original[0][0]] for j in b)
                )
            )
        output[corruption] = {
            "neighbor_jaccard": float(np.mean(jaccard)),
            "top1_stability": float(np.mean(top1)),
            "original_top1_recovered_at_k": float(np.mean(recovery)),
            "original_cluster_recovered_at_k": float(np.mean(graph_recovery)),
            "unknown_fraction": float(np.mean([not row for row in matches])),
            "query_fragments": len(damaged),
            "exact_description_recovery": float(
                np.mean(
                    damaged.description.to_numpy()
                    == sample.description.iloc[source_positions].to_numpy()
                )
            ),
        }
    grouped = index.nodes_.groupby("description")
    collision = grouped.apply(
        lambda group: group[BLOCK_COLUMNS].drop_duplicates().shape[0] > 1, include_groups=False
    )
    output["exact_name_incompatible_block_fraction"] = float(collision.mean())
    output["collision_definition"] = (
        "same description spans incompatible currency/direction/type/MCC blocks; "
        "proxy, no merchant truth"
    )
    aliases = index.nodes_.assign(cluster=index.clusters_).groupby("cluster").description.nunique()
    output["cross_alias_node_reduction"] = float((aliases - 1).sum() / len(index.nodes_))
    return output


BLOCK_COLUMNS = ["currency", "direction", "type", "mcc_mode"]


class ControlledModel:
    """Identical frozen V3-A control and exactly two small feature ablations."""

    def fit(self, train, labels, index, streams, neighbors):
        self.index = index
        self.v2 = IntegratedV2Model().fit(train, labels)
        self.exact = FamilyMap().fit(train, labels)
        self.family = FamilyEvidence(index).fit(labels)
        history = self.v2.history_.transform(train)
        identity = cross_fitted_family_features(train, labels).filter(regex="^identity_")
        merchant = cross_fitted_summaries(index, streams, labels, neighbors=neighbors)
        self.train_identity = identity
        self.train_merchant = merchant
        target = labels.set_index("client_id")[TARGET_COLUMN].reindex(history.index)
        matrices = self.matrices(history, identity, merchant)
        self.models = {name: make_model().fit(matrix, target) for name, matrix in matrices.items()}
        return self

    @staticmethod
    def matrices(history, identity, merchant):
        return {
            ARMS[0]: pd.concat([history, identity], axis=1).reindex(history.index),
            ARMS[1]: pd.concat([history, identity, merchant], axis=1).reindex(history.index),
            ARMS[2]: pd.concat([history, merchant], axis=1).reindex(history.index),
        }

    def predict(self, transactions, streams, neighbors):
        base = self.v2.predict_components(transactions)
        heuristic = (base["blend"] - 0.75 * base["history"]) / 0.25
        history = self.v2.history_.transform(transactions)
        identity = family_features(self.exact.transform(transactions)).filter(regex="^identity_")
        posterior = self.family.transform(streams, neighbors)
        merchant = summary_features(streams, neighbors, posterior, history.index)
        matrices = self.matrices(history, identity, merchant)
        prediction = {
            name: 0.75
            * pd.DataFrame(model.predict_proba(matrices[name]), index=history.index, columns=LABELS)
            + 0.25 * heuristic
            for name, model in self.models.items()
        }
        return prediction, identity, merchant, posterior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid"), default="oof")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics/v4_merchant_intelligence_christian"),
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stamps = fingerprint(args.data_dir)
    provenance_path = out / "provenance.json"
    if (
        provenance_path.exists()
        and json.loads(provenance_path.read_text())["fingerprints"] != stamps
    ):
        raise ValueError("Source/data changed; choose a fresh output directory")
    provenance = {
        "fingerprints": stamps,
        "base_branch": "origin/baseline/v4-frozen",
        "base_sha": BASE_SHA,
        "base_valid_macro_f1": BASE_MACRO_F1,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "catboost")
        },
        "unsupervised_fit": "all TRAIN + all unlabeled_pretrain; no VALID/TEST histories",
        "supervised_fit": (
            "five outer stratified TRAIN folds, five inner label-independent client folds"
        ),
        "class_order": list(LABELS),
        "seed": 42,
    }
    write_json(provenance_path, provenance)
    started = perf_counter()
    print("Loading TRAIN only", flush=True)
    train = read_transactions(args.data_dir / "train_transactions.jsonl")
    labels = read_labels(args.data_dir / "train_labels.csv")
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    index_path = out / "merchant_index.joblib"
    if index_path.exists():
        index, streams, neighbors = joblib.load(index_path)
    else:
        print("Building TRAIN + unlabeled fingerprints", flush=True)
        unlabeled = read_transactions(args.data_dir / "unlabeled_pretrain_transactions.jsonl")
        if set(train.client_id) & set(unlabeled.client_id):
            raise ValueError("TRAIN/unlabeled client overlap")
        all_train = build_fingerprints(train)
        all_unlabeled = build_fingerprints(unlabeled)
        index = MerchantIntelligence().fit_fingerprints(
            pd.concat([all_train, all_unlabeled], ignore_index=True)
        )
        print(f"Building graph on {len(index.nodes_)} archetypes", flush=True)
        index.build_graph()
        streams = cards(all_train)
        print(f"Retrieving {len(streams)} TRAIN payment streams", flush=True)
        neighbors = index.retrieve(streams)
        joblib.dump((index, streams, neighbors), index_path, compress=3)
        sparse.save_npz(out / "archetype_embeddings.npz", index.transform(index.nodes_))
        write_json(out / "graph_diagnostics.json", index.graph_diagnostics_)
    if args.phase == "oof":
        if (out / "frozen_selection.json").exists():
            raise ValueError(
                "Already frozen; use saved results, or a fresh directory for a new experiment"
            )
        probabilities = {name: [] for name in ARMS}
        identities, merchants, purity = [], [], []
        splitter = StratifiedKFold(5, shuffle=True, random_state=42)
        for fold, (fit, hold) in enumerate(splitter.split(target.index, target), 1):
            fold_path = out / f"fold_{fold}.joblib"
            if fold_path.exists():
                predictions, identity, merchant, fold_purity = joblib.load(fold_path)
            else:
                print(f"Fitting outer fold {fold}/5", flush=True)
                fit_ids, hold_ids = target.index[fit], target.index[hold]
                fit_rows = np.flatnonzero(streams.client_id.isin(fit_ids))
                hold_rows = np.flatnonzero(streams.client_id.isin(hold_ids))
                model = ControlledModel().fit(
                    train.loc[train.client_id.isin(fit_ids)],
                    labels.loc[labels.client_id.isin(fit_ids)],
                    index,
                    streams.iloc[fit_rows].reset_index(drop=True),
                    [neighbors[i] for i in fit_rows],
                )
                hold_streams = streams.iloc[hold_rows].reset_index(drop=True)
                predictions, identity, merchant, posterior = model.predict(
                    train.loc[train.client_id.isin(hold_ids)],
                    hold_streams,
                    [neighbors[i] for i in hold_rows],
                )
                observed = hold_streams.client_id.map(target)
                weak_correct = pd.Series(
                    np.asarray(LABELS)[posterior.argmax(axis=1)] == observed.to_numpy()
                )
                fold_purity = weak_correct.groupby(hold_streams.client_id).mean().to_list()
                joblib.dump((predictions, identity, merchant, fold_purity), fold_path, compress=3)
            for name in ARMS:
                probabilities[name].append(predictions[name])
            identities.append(identity)
            merchants.append(merchant)
            purity.extend(fold_purity)
            print(
                json.dumps(
                    {
                        "fold": fold,
                        **{
                            name: f1_score(
                                target.reindex(pred.index),
                                pred.idxmax(axis=1),
                                labels=LABELS,
                                average="macro",
                            )
                            for name, pred in predictions.items()
                        },
                    }
                ),
                flush=True,
            )
        report, combined = {}, {}
        for name in ARMS:
            combined[name] = pd.concat(probabilities[name]).reindex(target.index)
            if combined[name].isna().any().any() or not combined[name].index.is_unique:
                raise ValueError("Incomplete OOF probabilities")
            combined[name].to_csv(out / f"oof_{name}_probabilities.csv", index_label="client_id")
            report[name] = evaluate_predictions(target, combined[name].idxmax(axis=1))
        identity = pd.concat(identities).reindex(target.index)
        merchant = pd.concat(merchants).reindex(target.index)
        identity.to_parquet(out / "oof_exact_identity.parquet")
        merchant.to_parquet(out / "oof_merchant_summary.parquet")
        selected = max(ARMS, key=lambda name: report[name]["macro_f1"])
        baseline_correct = combined[ARMS[0]].idxmax(axis=1).eq(target)
        complementary = {}
        for name in ARMS[1:]:
            correct = combined[name].idxmax(axis=1).eq(target)
            complementary[name] = {
                "disagreement_fraction": float(
                    combined[name].idxmax(axis=1).ne(combined[ARMS[0]].idxmax(axis=1)).mean()
                ),
                "baseline_errors_recovered": int((~baseline_correct & correct).sum()),
                "baseline_correct_lost": int((baseline_correct & ~correct).sum()),
                "pair_oracle_accuracy": float((baseline_correct | correct).mean()),
            }
        print("Computing corruption diagnostics", flush=True)
        diagnostics = corruption_diagnostics(index, streams, train)
        diagnostics["family_purity_train_oof_client_balanced_weak_proxy"] = float(np.mean(purity))
        diagnostics["candidate_family"] = family_metrics(target, identity, merchant)
        diagnostics["stream_unknown_fraction"] = float(np.mean([not row for row in neighbors]))
        diagnostics["complementarity"] = complementary
        write_json(out / "diagnostics.json", diagnostics)
        summary = {
            "oof": report,
            "selected": selected,
            "seconds": perf_counter() - started,
            "config": config_dict(index),
            "valid_labels_read": False,
        }
        write_json(out / "summary.json", summary)
        write_json(
            out / "frozen_selection.json",
            {
                "selected": selected,
                "criterion": "maximum TRAIN OOF macro-F1 among predeclared A/B/C; tie prefers A",
                "fingerprints": stamps,
                "valid_labels_used": False,
                "config": config_dict(index),
            },
        )
        print(
            json.dumps(
                {
                    "selected": selected,
                    "macro_f1": {name: report[name]["macro_f1"] for name in ARMS},
                }
            ),
            flush=True,
        )
    else:
        frozen = json.loads((out / "frozen_selection.json").read_text())
        if frozen["fingerprints"] != stamps:
            raise ValueError("Freeze source/data mismatch")
        guard = out / "valid_evaluation_started.json"
        if guard.exists():
            raise ValueError("VALID evaluation already started; no second label access permitted")
        print("Refitting frozen models on TRAIN", flush=True)
        model = ControlledModel().fit(train, labels, index, streams, neighbors)
        valid = read_transactions(args.data_dir / "valid_transactions.jsonl")
        if set(valid.client_id) & set(index.presence_.client_id):
            raise ValueError("VALID overlaps unsupervised reference")
        query = cards(build_fingerprints(valid))
        predictions, _, _, _ = model.predict(valid, query, index.retrieve(query))
        for name, values in predictions.items():
            values.to_csv(out / f"valid_{name}_probabilities.csv", index_label="client_id")
        # Freeze predictions on disk before the sole VALID label read.
        write_json(guard, {"selected": frozen["selected"], "predictions_frozen": True})
        valid_labels = read_labels(args.data_dir / "valid_labels.csv").set_index("client_id")[
            TARGET_COLUMN
        ]
        report = {
            name: evaluate_predictions(valid_labels, values.idxmax(axis=1))
            for name, values in predictions.items()
        }
        write_json(
            out / "valid_results.json",
            {"selected": frozen["selected"], "metrics": report, "selection_changed": False},
        )
        index.save(out / "demo_resolver.joblib")
        joblib.dump(model.family, out / "demo_family_evidence.joblib", compress=3)
        print(
            json.dumps({name: metrics["macro_f1"] for name, metrics in report.items()}), flush=True
        )


if __name__ == "__main__":
    main()
