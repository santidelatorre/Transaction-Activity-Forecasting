"""Eight-class pointwise scoring with nested, client-disjoint supervised evidence.

No future-event reconstruction or oracle labels are used. Binary relevance is
the official client target, repeated over all eight candidate families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.v2 import IntegratedV2Model, make_model
from transaction_forecasting.ubs.v3.features import (
    FAMILIES,
    FamilyMap,
    cross_fitted_family_features,
    family_features,
    payment_streams,
    validate_history,
)


def client_target(transactions, labels):
    validate_history(transactions)
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    if not target.index.is_unique or set(target.index) != set(transactions.client_id):
        raise ValueError("Unique labels must exactly match history clients")
    if not target.isin(LABELS).all():
        raise ValueError("Invalid official target")
    return target


def client_folds(target, folds=5):
    if target.value_counts().min() < folds:
        raise ValueError("Too few clients per class for stratified folds")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=42)
    for fit, hold in splitter.split(target.index, target):
        yield target.index[fit], target.index[hold]


class FrozenV3A:
    """Exact promoted A recipe, avoiding fitting unused B/AB diagnostic models."""

    def fit(self, transactions, labels):
        client_target(transactions, labels)
        self.v2_ = IntegratedV2Model().fit(transactions, labels)
        self.mapper_ = FamilyMap().fit(transactions, labels)
        history = self.v2_.history_.transform(transactions)
        family = cross_fitted_family_features(transactions, labels).filter(regex="^identity_")
        matrix = pd.concat([history, family], axis=1).reindex(history.index)
        target = labels.set_index("client_id")[TARGET_COLUMN].reindex(matrix.index)
        self.model_ = make_model().fit(matrix, target)
        return self

    def predict_components(self, transactions):
        v2 = self.v2_.predict_components(transactions)
        history = self.v2_.history_.transform(transactions)
        family = family_features(self.mapper_.transform(transactions)).filter(regex="^identity_")
        matrix = pd.concat([history, family], axis=1).reindex(history.index)
        numeric = pd.DataFrame(
            self.model_.predict_proba(matrix), index=matrix.index, columns=LABELS
        )
        heuristic = (v2["blend"] - 0.75 * v2["history"]) / 0.25
        return {"baseline": 0.75 * numeric + 0.25 * heuristic, "v2": v2["blend"]}


class SoftFamilyMap:
    """Smoothed client-presence log lifts, never event ground-truth probabilities.

    Soft mass is softmax over seven lifts times support/(support+5). Unseen
    descriptions have zero mass and a missing indicator; no class is excluded.
    """

    def fit(self, transactions, labels):
        target = client_target(transactions, labels)
        self.fit_clients_ = frozenset(target.index)
        presence = transactions[["client_id", "description"]].drop_duplicates()
        presence = presence.join(target, on="client_id")
        counts = pd.crosstab(presence.description, presence[TARGET_COLUMN]).reindex(
            columns=LABELS, fill_value=0
        )
        support = counts.sum(axis=1)
        sizes = target.value_counts().reindex(LABELS, fill_value=0)
        self.lift_ = pd.DataFrame(index=counts.index)
        for family in FAMILIES:
            inside = (counts[family] + 1) / (sizes[family] + 2)
            outside = (support - counts[family] + 1) / (len(target) - sizes[family] + 2)
            self.lift_[family] = np.log(inside / outside)
        self.mass_ = pd.DataFrame(
            softmax(self.lift_.to_numpy(), axis=1), index=counts.index, columns=FAMILIES
        ).mul(support / (support + 5), axis=0)
        self.hard_ = self.lift_.idxmax(axis=1).where(
            support.ge(5) & self.lift_.max(axis=1).ge(np.log(1.5)), "unknown"
        )
        return self

    def evidence(self, transactions):
        validate_history(transactions)
        if self.fit_clients_.intersection(transactions.client_id):
            raise ValueError("Self-label mapping: prediction clients must be excluded from fit")
        clients = pd.Index(sorted(transactions.client_id.unique()), name="client_id")
        streams = payment_streams(transactions, ["client_id", "description", "currency"])
        streams["family"] = streams.description.map(self.hard_).fillna("unknown")
        streams["recurrent"] = streams["count"].ge(2).astype(float)
        streams["score"] = streams.score.fillna(0) * streams.recurrent
        streams["regularity"] = streams.regularity.fillna(0) * streams.recurrent
        streams["due_stream"] = (streams.recurrent.eq(1) & streams.due.le(90)).astype(float)
        streams["stability"] = 1 / (1 + streams.amount_cv)
        streams["known"] = streams.description.isin(self.mass_.index).astype(float)
        total = streams.groupby("client_id")["count"].sum().reindex(clients, fill_value=0)
        global_features = pd.DataFrame(index=clients)
        grouped = streams.groupby("client_id")
        global_features["recurrent_stream_count"] = grouped.recurrent.sum()
        global_features["strongest_recurrence"] = grouped.score.max()
        global_features["sum_recurrence"] = grouped.score.sum()
        global_features["no_stream_recurs_proxy"] = np.exp(
            -global_features.sum_recurrence.fillna(0)
        )
        global_features["unknown_stream_share"] = 1 - grouped.known.mean()
        history = transactions.groupby("client_id").timestamp.agg(["min", "max", "size"])
        global_features["history_recency"] = (CUTOFF - history["max"]).dt.total_seconds() / 86400
        global_features["history_span"] = (
            history["max"] - history["min"]
        ).dt.total_seconds() / 86400
        global_features["history_events"] = history["size"]
        blocks = {}
        for family in FAMILIES:
            work = streams.copy()
            work["mass"] = work.description.map(self.mass_[family]).fillna(0)
            work["lift"] = work.description.map(self.lift_[family]).fillna(0).clip(lower=0)
            work["soft_events"] = work.mass * work["count"]
            work["soft_recurrence"] = work.mass * work.score
            work["soft_due"] = work.mass * work.due_stream
            mapped = work.loc[work.family.eq(family)]
            hard = mapped.groupby("client_id")
            recurrent = mapped.loc[mapped.recurrent.eq(1)].groupby("client_id")
            soft = work.groupby("client_id")
            block = pd.DataFrame(index=clients)
            block["mapped_event_count"] = hard["count"].sum()
            block["mapped_share"] = block.mapped_event_count / total.clip(lower=1)
            block["mapped_aliases"] = hard.description.nunique()
            block["strongest_merchant_evidence"] = soft.lift.max()
            block["soft_probability_mass"] = soft.soft_events.sum() / total.clip(lower=1)
            block["soft_max_probability"] = soft.mass.max()
            block["soft_recurrence_sum"] = soft.soft_recurrence.sum()
            block["soft_recurrence_max"] = soft.soft_recurrence.max()
            block["soft_due_streams"] = soft.soft_due.sum()
            block["supporting_streams"] = recurrent.size()
            block["max_recurrence_score"] = recurrent.score.max()
            block["sum_recurrence_score"] = recurrent.score.sum()
            block["best_regularity"] = recurrent.regularity.max()
            block["amount_stability"] = recurrent.stability.max()
            block["recency"] = hard.recency.min().reindex(clients).fillna(730)
            block["earliest_next_event"] = recurrent.due.min().reindex(clients).fillna(730)
            block["due_streams"] = hard.due_stream.sum()
            block = block.fillna(0)
            block["no_mapping"] = block.mapped_event_count.eq(0).astype(float)
            blocks[family] = block
        evidence = pd.DataFrame({f: b.mapped_event_count.gt(0) for f, b in blocks.items()})
        strength = pd.DataFrame({f: b.soft_max_probability for f, b in blocks.items()})
        distribution = strength.div(strength.sum(axis=1).clip(lower=1e-12), axis=0)
        global_features["evidence_family_count"] = evidence.sum(axis=1)
        global_features["no_family_evidence"] = evidence.sum(axis=1).eq(0).astype(float)
        global_features["max_positive_evidence"] = strength.max(axis=1)
        global_features["ambiguity_entropy"] = -(
            distribution * np.log(distribution.clip(lower=1e-12))
        ).sum(axis=1)
        sorted_strength = np.sort(strength.to_numpy(), axis=1)
        global_features["gap_to_second"] = sorted_strength[:, -1] - sorted_strength[:, -2]
        global_features["textual_specificity"] = pd.DataFrame(
            {f: b.strongest_merchant_evidence for f, b in blocks.items()}
        ).max(axis=1)
        global_features = global_features.fillna(0)
        blocks["none"] = pd.DataFrame(0.0, index=clients, columns=blocks[FAMILIES[0]].columns)
        return blocks, global_features


@dataclass(frozen=True)
class SourcePartition:
    fit_clients: frozenset[str]
    prediction_clients: frozenset[str]


@dataclass
class CandidateBatch:
    features: pd.DataFrame
    sources: tuple[SourcePartition, ...]

    def validate(self):
        frame = self.features
        if frame.index.names != ["client_id", "candidate"] or not frame.index.is_unique:
            raise ValueError("Expected unique client/candidate index")
        clients = frame.index.get_level_values("client_id").unique()
        expected = pd.MultiIndex.from_product([clients, LABELS], names=frame.index.names)
        if len(clients) == 0 or not frame.index.equals(expected):
            raise ValueError(
                "Exactly eight candidates/client in deterministic class order required"
            )
        if "client_id" in frame or TARGET_COLUMN in frame or "candidate" in frame:
            raise ValueError("Client ID, target and string candidate cannot be model features")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Candidate scores/features must be finite")
        seen = set()
        for source in self.sources:
            if source.fit_clients & source.prediction_clients:
                raise ValueError("OOF baseline probability only: self-label source detected")
            if seen & source.prediction_clients:
                raise ValueError("Overlapping source prediction partitions")
            seen.update(source.prediction_clients)
        if seen != set(clients):
            raise ValueError("Missing OOF baseline probability provenance")
        return self

    def wide(self, column):
        return self.features[column].unstack("candidate").reindex(columns=LABELS)


def combine_batches(batches):
    frames = pd.concat([batch.validate().features for batch in batches])
    clients = sorted(frames.index.get_level_values("client_id").unique())
    index = pd.MultiIndex.from_product([clients, LABELS], names=["client_id", "candidate"])
    return CandidateBatch(
        frames.reindex(index), tuple(source for batch in batches for source in batch.sources)
    ).validate()


class EvidenceProvider:
    """Fit baseline and soft mapping on precisely the same excluded-client partition."""

    def fit(self, transactions, labels):
        self.fit_clients_ = frozenset(client_target(transactions, labels).index)
        self.baseline_ = FrozenV3A().fit(transactions, labels)
        self.mapper_ = SoftFamilyMap().fit(transactions, labels)
        return self

    def transform(self, transactions):
        if self.fit_clients_.intersection(transactions.client_id):
            raise ValueError("OOF baseline probability only: cannot transform fit clients")
        components = self.baseline_.predict_components(transactions)
        blocks, context = self.mapper_.evidence(transactions)
        clients = context.index
        baseline = components["baseline"].reindex(clients)
        if not np.allclose(baseline.sum(axis=1), 1):
            raise ValueError("Baseline probabilities must sum to one")
        context["baseline_entropy"] = -(baseline * np.log(baseline.clip(lower=1e-12))).sum(axis=1)
        context["baseline_none"] = baseline["none"]
        context["baseline_max"] = baseline.max(axis=1)
        rows = []
        for family in LABELS:
            block = pd.concat([blocks[family], context], axis=1)
            for name, probabilities in components.items():
                block[f"{name}_probability"] = probabilities[family].reindex(clients)
            block["baseline_log_probability"] = np.log(block.baseline_probability.clip(lower=1e-8))
            block["baseline_gap"] = block.baseline_probability - context.baseline_max
            for label in LABELS:
                block[f"class_{label}"] = float(label == family)
                block[f"baseline_log_x_{label}"] = block.baseline_log_probability * float(
                    label == family
                )
            for column in context:
                block[f"none_x_{column}"] = context[column] * float(family == "none")
            block["candidate"] = family
            rows.append(block.set_index("candidate", append=True))
        index = pd.MultiIndex.from_product([clients, LABELS], names=["client_id", "candidate"])
        source = SourcePartition(self.fit_clients_, frozenset(clients))
        return CandidateBatch(pd.concat(rows).reindex(index), (source,)).validate()


def cross_fitted_candidates(transactions, labels, folds=3, provider_factory=EvidenceProvider):
    """Inner OOF within an outer fit partition; never slice globally generated OOF."""
    target = client_target(transactions, labels)
    batches = []
    for fit_ids, hold_ids in client_folds(target, folds):
        model = provider_factory().fit(
            transactions.loc[transactions.client_id.isin(fit_ids)],
            labels.loc[labels.client_id.isin(fit_ids)],
        )
        batches.append(model.transform(transactions.loc[transactions.client_id.isin(hold_ids)]))
    return combine_batches(batches)


class SoftCandidateRanker:
    """Binary relevance logits normalized with within-client softmax, temperature 1.

    Every client's eight rows receive the same inverse-frequency client weight
    computed from its official TRAIN target. No predicted prevalence is tuned.
    Softmax scores are comparable but are not claimed to be calibrated.
    """

    def __init__(self, kind="linear"):
        if kind not in ("linear", "catboost"):
            raise ValueError("Only the two predeclared pointwise scorers are supported")
        self.kind = kind

    def fit(self, batch, target):
        features = batch.validate().features
        clients = features.index.get_level_values("client_id")
        if not target.index.is_unique or set(target.index) != set(clients):
            raise ValueError("Ranker target must exactly match candidate clients")
        if not target.isin(LABELS).all():
            raise ValueError("Invalid target")
        truth = target.reindex(clients).to_numpy()
        relevance = (features.index.get_level_values("candidate") == truth).astype(int)
        weights = len(target) / (len(LABELS) * target.value_counts())
        sample_weight = pd.Series(truth).map(weights).to_numpy()
        self.fit_clients_ = frozenset(target.index)
        self.columns_ = features.columns.tolist()
        if self.kind == "linear":
            self.model_ = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=42)
            )
            self.model_.fit(features, relevance, logisticregression__sample_weight=sample_weight)
        else:
            self.model_ = CatBoostClassifier(
                iterations=300,
                depth=4,
                learning_rate=0.05,
                loss_function="Logloss",
                l2_leaf_reg=5,
                random_seed=42,
                thread_count=4,
                verbose=False,
                allow_writing_files=False,
            )
            self.model_.fit(features, relevance, sample_weight=sample_weight)
        return self

    def predict_scores(self, batch):
        features = batch.validate().features
        clients = features.index.get_level_values("client_id").unique()
        if self.fit_clients_.intersection(clients):
            raise ValueError("Ranker prediction clients must be excluded from fit")
        if features.columns.tolist() != self.columns_:
            raise ValueError("Candidate feature schema changed")
        values = (
            self.model_.decision_function(features)
            if self.kind == "linear"
            else self.model_.predict(features, prediction_type="RawFormulaVal")
        )
        scores = pd.DataFrame(np.asarray(values).reshape(-1, 8), index=clients, columns=LABELS)
        if not np.isfinite(scores.to_numpy()).all():
            raise ValueError("Nonfinite candidate scores")
        return scores

    def predict_proba(self, batch):
        scores = self.predict_scores(batch)
        return pd.DataFrame(softmax(scores, axis=1), index=scores.index, columns=LABELS)


def raw_family_probabilities(batch):
    """Untuned soft evidence comparator; NONE uses a no-recurrence proxy.

    Positive logits = log1p(soft recurrence sum) + maximum positive log lift.
    NONE logit = exp(-sum recurrence), a heuristic proxy, not a calibrated event
    probability. All eight logits receive softmax at temperature one.
    """
    scores = np.log1p(batch.wide("soft_recurrence_sum")) + batch.wide("strongest_merchant_evidence")
    scores["none"] = batch.wide("no_stream_recurs_proxy")["none"]
    return pd.DataFrame(softmax(scores, axis=1), index=scores.index, columns=LABELS)


def degrade_history(transactions, seed=2026):
    """Remove 25% of events independently, retaining at least one per client.

    This is inference-only missing-history stress, not label reconstruction.
    """
    ordered = transactions.sort_values(["client_id", "timestamp", "description"], kind="stable")
    keep = np.random.default_rng(seed).random(len(ordered)) >= 0.25
    keep[~ordered.client_id.duplicated().to_numpy()] = True
    return ordered.loc[keep].reset_index(drop=True)
