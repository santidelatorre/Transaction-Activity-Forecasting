"""Client-isolated music/streaming specialist on the frozen V3-A base.

Run TRAIN OOF first. The frozen policy is written before VALID is opened.
Generated metrics stay under ignored outputs/metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN, read_labels, read_transactions
from transaction_forecasting.ubs.evaluation import evaluate_predictions
from transaction_forecasting.ubs.text_v2 import normalize_description
from transaction_forecasting.ubs.v2 import IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3.features import (
    FamilyMap,
    cross_fitted_family_features,
    family_features,
    validate_history,
)
from transaction_forecasting.ubs.v3.model import arm_matrix

SPECIAL = ("music", "streaming")
SEED = 42
FOLDS = 5
THRESHOLDS = (0.65, 0.75, 0.85, 0.90)
MARGINS = (0.10, 0.20)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class IdentityBase:
    """Only the V2 and V3-A arms of the fixed V3 recipe."""

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame) -> IdentityBase:
        validate_history(transactions)
        self.v2 = IntegratedV2Model().fit(transactions, labels)
        self.mapping = FamilyMap().fit(transactions, labels)
        history = self.v2.history_.transform(transactions)
        family = cross_fitted_family_features(transactions, labels)
        target = labels.set_index("client_id")[TARGET_COLUMN].reindex(history.index)
        self.model = make_model().fit(arm_matrix(history, family, None, "A"), target)
        return self

    def predict(self, transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        components = self.v2.predict_components(transactions)
        history = self.v2.history_.transform(transactions)
        family = family_features(self.mapping.transform(transactions))
        raw = pd.DataFrame(
            self.model.predict_proba(arm_matrix(history, family, None, "A")),
            index=history.index,
            columns=LABELS,
        )
        heuristic = (components["blend"] - 0.75 * components["history"]) / 0.25
        a = 0.75 * raw + 0.25 * heuristic
        return components["blend"], a


def client_history_features(transactions: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Unsupervised per-client text and cadence; no family/target assignment."""
    validate_history(transactions)
    clients = pd.Index(sorted(transactions.client_id.unique()), name="client_id")
    frame = transactions.loc[
        transactions.direction.eq("out") & transactions.type.eq("card_payment")
    ].copy()
    frame["description"] = frame.description.map(normalize_description)
    frame = frame.sort_values(["client_id", "description", "timestamp"], kind="stable")
    streams = (
        frame.groupby(["client_id", "description"], sort=False)
        .agg(
            events=("timestamp", "nunique"),
            last=("timestamp", "max"),
            amount_mean=("amount", "mean"),
            amount_std=("amount", "std"),
        )
        .reset_index()
    )
    frame["gap"] = (
        frame.groupby(["client_id", "description"]).timestamp.diff().dt.total_seconds() / 86400
    )
    streams = streams.join(
        frame.groupby(["client_id", "description"]).gap.agg(gap_median="median", gap_std="std"),
        on=["client_id", "description"],
    )
    streams["amount_cv"] = streams.amount_std.fillna(0).div(
        streams.amount_mean.abs().clip(lower=1e-6)
    )
    streams["gap_cv"] = streams.gap_std.fillna(0).div(streams.gap_median.clip(lower=1))
    streams["recency"] = (
        pd.Timestamp("2026-01-01", tz="UTC") - streams["last"]
    ).dt.total_seconds() / 86400
    streams["monthly"] = streams.gap_median.between(25, 35).astype(int)
    streams["stable"] = (
        streams.events.ge(2) & streams.amount_cv.le(0.2) & streams.gap_cv.le(0.3)
    ).astype(int)

    # Repeat names only to convey within-client frequency; cap domination by a noisy stream.
    streams["text"] = streams.apply(
        lambda row: (row.description + " ") * min(int(row.events), 4), axis=1
    )
    documents = streams.groupby("client_id").text.agg(" ".join).reindex(clients, fill_value="")
    recurring = streams.loc[streams.events.ge(2)]
    summary = pd.DataFrame(index=clients)
    summary["payment_streams"] = streams.groupby("client_id").size()
    summary["recurring_streams"] = recurring.groupby("client_id").size()
    summary["recurring_events"] = recurring.groupby("client_id").events.sum()
    summary["monthly_streams"] = recurring.groupby("client_id").monthly.sum()
    summary["stable_streams"] = recurring.groupby("client_id").stable.sum()
    summary["best_amount_cv"] = recurring.groupby("client_id").amount_cv.min()
    summary["best_gap_cv"] = recurring.groupby("client_id").gap_cv.min()
    summary["last_recurring_days"] = recurring.groupby("client_id").recency.min()
    summary = summary.fillna(0).clip(lower=0)
    return documents, summary


class Specialist:
    """Three-class char text classifier with optional recurrence scalars."""

    def __init__(self, recurrence: bool):
        self.recurrence = recurrence

    def fit(self, documents: pd.Series, recurrence: pd.DataFrame, target: pd.Series) -> Specialist:
        if not documents.index.equals(target.index) or not recurrence.index.equals(target.index):
            raise ValueError("Client features and targets must be aligned")
        if not target.index.is_unique:
            raise ValueError("Duplicate training clients")
        self.fit_clients = set(target.index)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=5000
        )
        text = self.vectorizer.fit_transform(documents)
        self.scaler = StandardScaler()
        design = text
        if self.recurrence:
            numeric = self.scaler.fit_transform(recurrence)
            design = hstack([text, csr_matrix(numeric)], format="csr")
        self.model = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=600,
            random_state=SEED,
        ).fit(design, target.where(target.isin(SPECIAL), "other"))
        return self

    def predict(self, documents: pd.Series, recurrence: pd.DataFrame) -> pd.DataFrame:
        if self.fit_clients.intersection(documents.index):
            raise ValueError("Cannot score clients seen by specialist fit")
        if not documents.index.equals(recurrence.index):
            raise ValueError("Client features must be aligned")
        text = self.vectorizer.transform(documents)
        design = text
        if self.recurrence:
            design = hstack([text, csr_matrix(self.scaler.transform(recurrence))], format="csr")
        return pd.DataFrame(
            self.model.predict_proba(design), index=documents.index, columns=self.model.classes_
        ).reindex(columns=["music", "streaming", "other"])


def apply_override(
    base: pd.DataFrame, specialist: pd.DataFrame, threshold: float, margin: float
) -> pd.Series:
    """Only promote a confident specialist class from another positive class."""
    if not base.index.equals(specialist.index):
        raise ValueError("Base and specialist client IDs must align")
    original = base.idxmax(axis=1)
    winner = specialist[list(SPECIAL)].idxmax(axis=1)
    confidence = specialist[list(SPECIAL)].max(axis=1)
    difference = confidence - specialist["other"]
    eligible = ~original.isin([*SPECIAL, "none"]) & confidence.ge(threshold) & difference.ge(margin)
    return original.where(~eligible, winner)


def override_audit(target: pd.Series, base: pd.Series, candidate: pd.Series) -> dict:
    changed = base.ne(candidate)
    truth = target.reindex(base.index)
    return {
        "overrides": int(changed.sum()),
        "to_music": int((changed & candidate.eq("music")).sum()),
        "to_streaming": int((changed & candidate.eq("streaming")).sum()),
        "from_music": int((changed & base.eq("music")).sum()),
        "from_streaming": int((changed & base.eq("streaming")).sum()),
        "correct_overrides": int((changed & candidate.eq(truth)).sum()),
        "incorrect_overrides": int((changed & candidate.ne(truth)).sum()),
        "new_correct": int((changed & base.ne(truth) & candidate.eq(truth)).sum()),
        "lost_correct": int((changed & base.eq(truth) & candidate.ne(truth)).sum()),
    }


def metrics(target: pd.Series, prediction: pd.Series) -> dict:
    return evaluate_predictions(target, prediction)


def select_policy(target: pd.Series, base: pd.DataFrame, scores: dict) -> dict:
    """Small fixed grid, selected entirely on TRAIN OOF; prefer fewer overrides on ties."""
    original = base.idxmax(axis=1)
    baseline = metrics(target, original)["macro_f1"]
    candidates = [
        {
            "arm": "identity_only",
            "threshold": None,
            "margin": None,
            "macro_f1": baseline,
            "overrides": 0,
        }
    ]
    for arm in ("text", "text_recurrence"):
        for threshold in THRESHOLDS:
            for margin in MARGINS:
                prediction = apply_override(base, scores[arm], threshold, margin)
                candidates.append(
                    {
                        "arm": arm,
                        "threshold": threshold,
                        "margin": margin,
                        "macro_f1": metrics(target, prediction)["macro_f1"],
                        "overrides": int(prediction.ne(original).sum()),
                    }
                )
    selected = max(
        candidates,
        key=lambda row: (
            row["macro_f1"],
            -row["overrides"],
            row["margin"] or 0,
            row["threshold"] or 0,
        ),
    )
    return {"selected": selected, "candidates": candidates, "baseline_macro_f1": baseline}


def score_policy(base: pd.DataFrame, scores: dict, policy: dict) -> pd.Series:
    selected = policy["selected"]
    if selected["arm"] == "identity_only":
        return base.idxmax(axis=1)
    return apply_override(base, scores[selected["arm"]], selected["threshold"], selected["margin"])


def run_oof(train: pd.DataFrame, labels: pd.DataFrame, out: Path) -> dict:
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    documents, recurrence = client_history_features(train)
    splitter = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (fit_pos, hold_pos) in enumerate(splitter.split(target.index, target), 1):
        path = out / f"fold_{fold}.csv"
        if path.exists():
            continue
        fit_ids, hold_ids = target.index[fit_pos], target.index[hold_pos]
        fit_tx, hold_tx = (
            train.loc[train.client_id.isin(fit_ids)],
            train.loc[train.client_id.isin(hold_ids)],
        )
        fit_labels = labels.loc[labels.client_id.isin(fit_ids)]
        print(f"TRAIN outer fold {fold}/{FOLDS}", flush=True)
        v2, a = IdentityBase().fit(fit_tx, fit_labels).predict(hold_tx)
        rows = pd.DataFrame(index=hold_ids)
        for name, probabilities in (("v2", v2), ("a", a)):
            for label in LABELS:
                rows[f"{name}_{label}"] = probabilities.reindex(hold_ids)[label]
        for arm, use_recurrence in (("text", False), ("text_recurrence", True)):
            specialist = Specialist(use_recurrence).fit(
                documents.reindex(fit_ids), recurrence.reindex(fit_ids), target.reindex(fit_ids)
            )
            scores = specialist.predict(documents.reindex(hold_ids), recurrence.reindex(hold_ids))
            for label in ("music", "streaming", "other"):
                rows[f"{arm}_{label}"] = scores[label]
        if rows.isna().any().any():
            raise RuntimeError("Incomplete OOF fold")
        rows.to_csv(path, index_label="client_id")
    frames = [
        pd.read_csv(out / f"fold_{fold}.csv", index_col="client_id") for fold in range(1, FOLDS + 1)
    ]
    oof = pd.concat(frames).reindex(target.index)
    if not oof.index.is_unique or oof.isna().any().any():
        raise RuntimeError("Incomplete or duplicated OOF")
    oof.to_csv(out / "oof_scores.csv", index_label="client_id")
    v2, a, scores = unpack(oof)
    policy = select_policy(target, a, scores)
    write_json(out / "frozen_policy.json", policy)
    predictions = {
        "V2": v2.idxmax(axis=1),
        "V3_A": a.idxmax(axis=1),
        "identity_only": a.idxmax(axis=1),
    }
    for arm in ("text", "text_recurrence"):
        arm_policy = max(
            (row for row in policy["candidates"] if row["arm"] == arm),
            key=lambda row: (row["macro_f1"], -row["overrides"]),
        )
        predictions[arm] = apply_override(
            a, scores[arm], arm_policy["threshold"], arm_policy["margin"]
        )
    predictions["selected"] = score_policy(a, scores, policy)
    result = report_metrics(target, predictions)
    result["selected_arm"] = policy["selected"]
    write_json(out / "oof_results.json", result)
    return policy


def unpack(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    v2 = frame[[f"v2_{label}" for label in LABELS]].copy()
    a = frame[[f"a_{label}" for label in LABELS]].copy()
    v2.columns = LABELS
    a.columns = LABELS
    scores = {}
    for arm in ("text", "text_recurrence"):
        item = frame[[f"{arm}_{label}" for label in ("music", "streaming", "other")]].copy()
        item.columns = ["music", "streaming", "other"]
        scores[arm] = item
    return v2, a, scores


def report_metrics(target: pd.Series, predictions: dict[str, pd.Series]) -> dict:
    base = predictions["V3_A"]
    return {
        "metrics": {name: metrics(target, prediction) for name, prediction in predictions.items()},
        "overrides": {
            name: override_audit(target, base, prediction)
            for name, prediction in predictions.items()
            if name not in ("V2", "V3_A")
        },
    }


def run_valid(
    train: pd.DataFrame, labels: pd.DataFrame, data_dir: Path, out: Path, policy: dict
) -> dict:
    valid = read_transactions(data_dir / "valid_transactions.jsonl")
    if set(train.client_id) & set(valid.client_id):
        raise ValueError("TRAIN and VALID clients overlap")
    train_docs, train_recurrence = client_history_features(train)
    valid_docs, valid_recurrence = client_history_features(valid)
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    v2, a = IdentityBase().fit(train, labels).predict(valid)
    rows = pd.DataFrame(index=a.index)
    for name, probabilities in (("v2", v2), ("a", a)):
        for label in LABELS:
            rows[f"{name}_{label}"] = probabilities[label]
    for arm, recurrence in (("text", False), ("text_recurrence", True)):
        specialist = Specialist(recurrence).fit(
            train_docs.reindex(target.index), train_recurrence.reindex(target.index), target
        )
        scores = specialist.predict(valid_docs.reindex(a.index), valid_recurrence.reindex(a.index))
        for label in ("music", "streaming", "other"):
            rows[f"{arm}_{label}"] = scores[label]
    rows.to_csv(out / "valid_scores_frozen.csv", index_label="client_id")
    # This is the first read of official VALID labels in this experiment.
    valid_labels = read_labels(data_dir / "valid_labels.csv")
    truth = valid_labels.set_index("client_id")[TARGET_COLUMN]
    v2, a, scores = unpack(rows)
    predictions = {
        "V2": v2.idxmax(axis=1),
        "V3_A": a.idxmax(axis=1),
        "selected": score_policy(a, scores, policy),
    }
    result = report_metrics(truth, predictions)
    result["selected_arm"] = policy["selected"]
    write_json(out / "valid_results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("oof", "valid", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ubs_2026"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metrics/v3_music_streaming_christian")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train_transactions.jsonl"
    label_path = args.data_dir / "train_labels.csv"
    hashes = {
        "train_transactions": fingerprint(train_path),
        "train_labels": fingerprint(label_path),
    }
    train = read_transactions(train_path)
    labels = read_labels(label_path)
    freeze_path = args.output_dir / "frozen_inputs.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != hashes:
        raise ValueError("TRAIN inputs changed; choose a new output directory")
    write_json(freeze_path, hashes)
    if args.phase in ("oof", "all"):
        run_oof(train, labels, args.output_dir)
    if args.phase in ("valid", "all"):
        policy = json.loads((args.output_dir / "frozen_policy.json").read_text())
        result = run_valid(train, labels, args.data_dir, args.output_dir, policy)
        print(
            json.dumps({name: values["macro_f1"] for name, values in result["metrics"].items()}),
            flush=True,
        )


if __name__ == "__main__":
    main()
