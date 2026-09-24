"""Local, interpretable merchant retrieval with client-isolated family evidence.

An archetype is evidence, never a claim of ground-truth merchant identity.
No labels enter fingerprints, vocabularies, retrieval weights or the graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.v3.features import validate_history

KEYS = ["client_id", "description", "currency", "direction", "type", "amount_band"]
BLOCK = ["currency", "direction", "type", "mcc_mode"]
NUMERIC = [
    "median_amount",
    "amount_dispersion",
    "median_gap",
    "gap_dispersion",
    "recurrence_count",
    "frequency",
    "recency",
    "weekly",
    "monthly",
    "annual",
]


@dataclass(frozen=True)
class MerchantConfig:
    """Predeclared weights; no VALID-driven threshold or weight selection."""

    neighbors: int = 8
    candidates: int = 32
    text_weight: float = 0.25
    min_similarity: float = 0.55
    graph_similarity: float = 0.80
    max_amount_ratio: float = 2.0
    max_gap_ratio: float = 1.8
    max_cluster_size: int = 64


def build_fingerprints(transactions: pd.DataFrame) -> pd.DataFrame:
    """Split obvious amount collisions before aggregating exact-description streams.

    Amount bands are coarse factor-two buckets, not currency conversions. MCC
    remains a distribution. Timestamp duplicates do not invent recurrence.
    """
    validate_history(transactions)
    frame = transactions.copy()
    for column in ("currency", "direction", "type", "mcc"):
        frame[column] = frame[column].fillna("unknown").astype(str)
    if not np.isfinite(frame.amount).all():
        raise ValueError("Amounts must be finite")
    frame["absolute_amount"] = frame.amount.abs()
    frame["amount_band"] = np.floor(np.log2(frame.absolute_amount.clip(lower=1))).astype(int)
    frame = frame.sort_values([*KEYS, "timestamp"], kind="stable")
    grouped = frame.groupby(KEYS, sort=True, dropna=False)
    result = grouped.agg(
        median_amount=("absolute_amount", "median"),
        recurrence_count=("timestamp", "nunique"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
    )
    result["amount_low"] = grouped.absolute_amount.quantile(0.25)
    result["amount_high"] = grouped.absolute_amount.quantile(0.75)
    unique = frame.drop_duplicates([*KEYS, "timestamp"]).copy()
    unique["gap"] = unique.groupby(KEYS).timestamp.diff().dt.total_seconds() / 86400
    gaps = unique.groupby(KEYS).gap.agg(median_gap="median", gap_std="std")
    result = result.join(gaps)
    result["amount_dispersion"] = (
        result.amount_high - result.amount_low
    ) / result.median_amount.clip(1)
    result["gap_dispersion"] = result.gap_std.fillna(0) / result.median_gap.clip(1)
    result["recency"] = (CUTOFF - result["last"]).dt.total_seconds() / 86400
    exposure = (CUTOFF - result["first"]).dt.total_seconds().div(86400).clip(1)
    result["frequency"] = result.recurrence_count / exposure * 30
    for name, period in (("weekly", 7), ("monthly", 30.44), ("annual", 365.25)):
        result[name] = np.exp(-(result.median_gap - period).abs() / (period * 0.2))
    mcc = frame.groupby([*KEYS, "mcc"]).size().unstack("mcc", fill_value=0)
    mcc = mcc.div(mcc.sum(axis=1), axis=0)
    result["mcc_mode"] = mcc.idxmax(axis=1)
    result["mcc_distribution"] = [
        {str(k): float(v) for k, v in row.items() if v > 0} for row in mcc.to_dict("records")
    ]
    result[NUMERIC] = result[NUMERIC].fillna(0)
    return result.reset_index().drop(columns=["amount_low", "amount_high", "gap_std"])


def payment_fingerprints(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compact downstream scope: outgoing cards; clients with no cards stay unknown."""
    frame = transactions.loc[
        transactions.direction.eq("out") & transactions.type.eq("card_payment")
    ]
    if frame.empty:
        return build_fingerprints(transactions).iloc[:0]
    return build_fingerprints(frame)


def _numeric(frame):
    values = frame[NUMERIC].to_numpy(dtype=float).copy()
    values[:, :7] = np.log1p(values[:, :7].clip(0))
    return values


class MerchantIntelligence:
    """M1 retrieval + M2 bounded complete-link graph, fitted without any labels.

    Fit accepts precomputed fingerprints to avoid repeatedly aggregating history.
    All label-derived information lives separately in FamilyEvidence.
    """

    def __init__(self, config: MerchantConfig | None = None):
        self.config = config or MerchantConfig()

    def fit(self, transactions, unlabeled_pretrain=None):
        frames = [build_fingerprints(transactions)]
        if unlabeled_pretrain is not None:
            if set(transactions.client_id) & set(unlabeled_pretrain.client_id):
                raise ValueError("TRAIN and unlabeled clients must be disjoint")
            frames.append(build_fingerprints(unlabeled_pretrain))
        return self.fit_fingerprints(pd.concat(frames, ignore_index=True))

    def fit_fingerprints(self, streams):
        if streams.empty:
            raise ValueError("Reference streams must be nonempty")
        frame = streams.copy()
        frame["cadence_band"] = np.floor(np.log2(frame.median_gap.clip(1))).astype(int)
        node_keys = ["description", *BLOCK, "amount_band", "cadence_band"]
        grouped = frame.groupby(node_keys, sort=True, dropna=False)
        self.nodes_ = grouped[NUMERIC].median().reset_index()
        mcc_columns = pd.DataFrame(frame.mcc_distribution.to_list()).fillna(0)
        distribution = pd.concat([frame[node_keys].reset_index(drop=True), mcc_columns], axis=1)
        distribution = distribution.groupby(node_keys, sort=True)[mcc_columns.columns].mean()
        self.nodes_["mcc_distribution"] = [
            {k: float(v) for k, v in row.items() if v > 0}
            for row in distribution.to_dict("records")
        ]
        self.nodes_["node_id"] = np.arange(len(self.nodes_))
        membership = frame.merge(
            self.nodes_[node_keys + ["node_id"]], on=node_keys, validate="many_to_one"
        )
        self.presence_ = (
            membership[["node_id", "client_id"]]
            .drop_duplicates()
            .sort_values(["node_id", "client_id"])
        )
        self.char_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), max_features=12000, sublinear_tf=True
        )
        self.token_ = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", max_features=4000, sublinear_tf=True
        )
        # Sentinel ensures a defined vocabulary even for an entirely masked fit.
        vocabulary = sorted(set(self.nodes_.description.astype(str))) + ["__missing__"]
        self.char_.fit(vocabulary)
        self.token_.fit(vocabulary)
        self.mcc_ = DictVectorizer(sparse=True).fit(self.nodes_.mcc_distribution)
        raw = _numeric(self.nodes_)
        self.scale_ = np.maximum(np.std(raw, axis=0), 0.5)
        self.behavior_ = raw / self.scale_
        self.text_ = self._text(self.nodes_)
        self.mcc_matrix_ = self.mcc_.transform(self.nodes_.mcc_distribution).tocsr()
        self.blocks_ = {}
        for key, indices in self.nodes_.groupby(BLOCK, sort=True).indices.items():
            positions = np.asarray(indices)
            knn = NearestNeighbors(algorithm="kd_tree").fit(self.behavior_[positions])
            self.blocks_[key] = (positions, knn)
        self.clusters_ = np.arange(len(self.nodes_))
        return self

    def _text(self, streams):
        descriptions = streams.description.fillna("").astype(str)
        return normalize(
            sparse.hstack(
                [
                    self.char_.transform(descriptions) * np.sqrt(0.8),
                    self.token_.transform(descriptions) * np.sqrt(0.2),
                ],
                format="csr",
            )
        )

    def transform(self, history_or_stream):
        """Deterministic sparse merchant embeddings; labels are never read."""
        frame = self._as_streams(history_or_stream)
        return sparse.hstack(
            [
                self._text(frame) * np.sqrt(self.config.text_weight),
                self.mcc_.transform(frame.mcc_distribution) * 0.5,
                sparse.csr_matrix(
                    _numeric(frame) / self.scale_ * np.sqrt(1 - self.config.text_weight)
                ),
            ],
            format="csr",
        )

    def fit_transform(self, transactions, unlabeled_pretrain=None):
        return self.fit(transactions, unlabeled_pretrain).transform(transactions)

    @staticmethod
    def _as_streams(frame):
        return (
            frame.reset_index(drop=True) if "median_amount" in frame else build_fingerprints(frame)
        )

    def _compatible(self, query, nodes):
        cfg = self.config
        amount_ratio = (nodes.median_amount + 1) / (query.median_amount + 1)
        gap_ratio = (nodes.median_gap + 1) / (query.median_gap + 1)
        known_gap = (nodes.median_gap > 0) & (query.median_gap > 0)
        compatible = amount_ratio.between(1 / cfg.max_amount_ratio, cfg.max_amount_ratio)
        compatible &= ~known_gap | gap_ratio.between(1 / cfg.max_gap_ratio, cfg.max_gap_ratio)
        for column in BLOCK:
            compatible &= nodes[column].eq(query[column])
        # Dominant MCC blocking plus actual distribution overlap.
        overlap = np.array(
            [
                sum(min(v, distribution.get(k, 0)) for k, v in query.mcc_distribution.items())
                for distribution in nodes.mcc_distribution
            ]
        )
        return compatible.to_numpy() & (overlap >= 0.5)

    def retrieve(self, history_or_stream, *, exclude_nodes=None):
        """Return compatible neighbors from union of text and behavior candidates."""
        frame = self._as_streams(history_or_stream)
        numeric = _numeric(frame) / self.scale_
        text_matrix = self._text(frame)
        results = [None] * len(frame)
        for key, query_positions in frame.groupby(BLOCK, sort=True).indices.items():
            if key not in self.blocks_:
                continue
            reference, knn = self.blocks_[key]
            n = min(self.config.candidates, len(reference))
            behavior_indices = knn.kneighbors(
                numeric[query_positions], n_neighbors=n, return_distance=False
            )
            for start in range(0, len(query_positions), 128):
                positions = query_positions[start : start + 128]
                text_scores = (text_matrix[positions] @ self.text_[reference].T).toarray()
                for local, position in enumerate(positions):
                    ts = text_scores[local]
                    text_candidates = np.argpartition(-ts, n - 1)[:n]
                    candidates = np.unique(
                        reference[np.r_[behavior_indices[start + local], text_candidates]]
                    )
                    if exclude_nodes is not None:
                        candidates = candidates[candidates != exclude_nodes[position]]
                    query = frame.iloc[position]
                    candidates = candidates[self._compatible(query, self.nodes_.iloc[candidates])]
                    if not len(candidates):
                        continue
                    behavior_score = np.exp(
                        -np.mean(np.abs(numeric[position] - self.behavior_[candidates]), axis=1)
                    )
                    lexical = (text_matrix[position] @ self.text_[candidates].T).toarray().ravel()
                    # Missing/OOV text has no negative vote; behavior can stand alone.
                    weight = self.config.text_weight if text_matrix[position].nnz else 0.0
                    score = weight * lexical + (1 - weight) * behavior_score
                    order = np.lexsort((candidates, -score))[: self.config.neighbors]
                    order = order[score[order] >= self.config.min_similarity]
                    results[position] = [
                        (
                            int(candidates[i]),
                            float(score[i]),
                            float(lexical[i]),
                            float(behavior_score[i]),
                        )
                        for i in order
                    ]
        return [result or [] for result in results]

    def build_graph(self):
        """Greedy constrained graph; all cross-cluster pairs must be compatible.

        Complete compatibility and a size cap prevent transitive chaining from
        joining incompatible endpoints through a plausible intermediate alias.
        """
        neighbors = self.retrieve(self.nodes_, exclude_nodes=self.nodes_.node_id.to_numpy())
        edges = sorted(
            {
                (min(i, j), max(i, j), score)
                for i, row in enumerate(neighbors)
                for j, score, _, _ in row
                if score >= self.config.graph_similarity
            },
            key=lambda x: (-x[2], x[0], x[1]),
        )
        parent = np.arange(len(self.nodes_))
        members = {i: [i] for i in parent}

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        accepted = 0
        for left, right, _ in edges:
            a, b = root(left), root(right)
            if a == b or len(members[a]) + len(members[b]) > self.config.max_cluster_size:
                continue
            if not all(
                self._compatible(self.nodes_.iloc[i], self.nodes_.iloc[members[b]]).all()
                for i in members[a]
            ):
                continue
            a, b = min(a, b), max(a, b)
            parent[b] = a
            members[a].extend(members.pop(b))
            accepted += 1
        self.clusters_ = np.array([root(i) for i in range(len(parent))])
        sizes = pd.Series(self.clusters_).value_counts()
        aliases = (
            self.nodes_.assign(cluster=self.clusters_).groupby("cluster").description.nunique()
        )
        self.graph_diagnostics_ = {
            "nodes": len(parent),
            "clusters": len(sizes),
            "accepted_edges": accepted,
            "merged_node_fraction": float(np.mean(pd.Series(self.clusters_).map(sizes).gt(1))),
            "multi_alias_cluster_fraction": float(aliases.gt(1).mean()),
            "size_quantiles": sizes.quantile([0, 0.5, 0.9, 0.99, 1]).to_dict(),
            "size_distribution": sizes.value_counts().sort_index().to_dict(),
            "incompatible_pairs": 0,  # Guaranteed by the all-pairs merge condition.
            "identity_ground_truth_available": False,
        }
        return self.graph_diagnostics_

    def resolve_merchant(self, history_or_stream, family_evidence=None):
        streams = self._as_streams(history_or_stream)
        neighbors = self.retrieve(streams)
        evidence = family_evidence.transform(streams, neighbors) if family_evidence else None
        embedding = self.transform(streams)
        resolved = []
        for i, (row, matches) in enumerate(zip(streams.to_dict("records"), neighbors, strict=True)):
            confidence = identity_confidence(matches)
            vector = embedding.getrow(i)
            resolved.append(
                {
                    "raw_description": row["description"],
                    "merchant_embedding": {
                        "indices": vector.indices.tolist(),
                        "values": vector.data.tolist(),
                        "size": embedding.shape[1],
                    },
                    "nearest_neighbors": [
                        {
                            "node_id": j,
                            "alias": str(self.nodes_.iloc[j].description),
                            "similarity": score,
                            "text_similarity": lexical,
                            "behavior_similarity": behavior,
                        }
                        for j, score, lexical, behavior in matches
                    ],
                    "canonical_cluster": int(self.clusters_[matches[0][0]])
                    if matches and confidence >= 0.5
                    else None,
                    "identity_confidence": confidence,
                    "family_evidence": dict(zip(LABELS, evidence[i], strict=True))
                    if evidence is not None
                    else None,
                    "unknown": not bool(matches),
                }
            )
        return resolved

    def save(self, path):
        joblib.dump(self, Path(path), compress=3)

    @staticmethod
    def load(path):
        """Load only trusted local artifacts (joblib uses pickle)."""
        return joblib.load(Path(path))


def identity_confidence(matches):
    """Uncalibrated strength/ambiguity score, explicitly not identity probability."""
    if not matches:
        return 0.0
    top = matches[0][1]
    margin = top - matches[1][1] if len(matches) > 1 else 0.0
    return float(top * (0.5 + 0.5 * min(1, margin / 0.15)))


class FamilyEvidence:
    """Soft client-target association; a client votes at most once per query.

    Multiple transactions, aliases and neighboring nodes from the same client
    contribute their maximum weight, not repeated labels. These are weak
    client-target posteriors, not observed stream-family ground truth.
    """

    def __init__(self, index):
        self.index = index

    def fit(self, labels):
        target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
        if not target.index.is_unique or not target.isin(LABELS).all():
            raise ValueError("Unique client labels in official class order required")
        if not set(target.index).issubset(set(self.index.presence_.client_id)):
            raise ValueError("Labels must belong to reference clients")
        self.fit_clients_ = set(target.index)
        self.target_ = {client: LABELS.index(label) for client, label in target.items()}
        counts = target.value_counts().reindex(LABELS, fill_value=0).to_numpy(float) + 1
        self.prior_ = counts / counts.sum()
        present = self.index.presence_.loc[self.index.presence_.client_id.isin(target.index)]
        self.clients_by_node_ = present.groupby("node_id").client_id.agg(list).to_dict()
        return self

    def transform(self, streams, neighbors=None):
        if self.fit_clients_.intersection(streams.client_id):
            raise ValueError("Query clients must be excluded from supervised family fit")
        neighbors = self.index.retrieve(streams) if neighbors is None else neighbors
        posterior = np.tile(self.prior_, (len(streams), 1))
        for i, matches in enumerate(neighbors):
            votes = {}
            for node, similarity, _, _ in matches:
                for client in self.clients_by_node_.get(node, []):
                    votes[client] = max(votes.get(client, 0), similarity**3)
            if votes:
                counts = self.prior_ * 3
                for client, weight in votes.items():
                    counts[self.target_[client]] += weight
                posterior[i] = counts / counts.sum()
        return posterior


def summary_features(streams, neighbors, posterior, clients):
    """Small, fixed downstream block: soft family mass/max and uncertainty."""
    frame = pd.DataFrame(index=streams.index)
    frame["client_id"] = streams.client_id.to_numpy()
    frame["mi_confidence"] = [identity_confidence(row) for row in neighbors]
    frame["mi_unknown"] = [float(not row) for row in neighbors]
    frame["mi_entropy"] = -(posterior * np.log(posterior.clip(1e-12))).sum(axis=1) / np.log(
        len(LABELS)
    )
    # Presence per stream, not transaction-weighted family evidence.
    for j, label in enumerate(LABELS):
        frame[f"mi_{label}"] = posterior[:, j] * (1 - frame.mi_unknown)
    aggregate = frame.groupby("client_id").agg(
        {
            **{f"mi_{label}": ["mean", "max"] for label in LABELS},
            "mi_confidence": ["mean"],
            "mi_unknown": ["mean"],
            "mi_entropy": ["mean"],
        }
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    result = aggregate.reindex(pd.Index(clients, name="client_id")).fillna(0)
    no_stream = ~result.index.isin(frame.client_id)
    result.loc[no_stream, "mi_unknown_mean"] = 1
    return result


def cross_fitted_summaries(index, streams, labels, folds=5, neighbors=None):
    """Label-independent client folds; even changing own label cannot change its map."""
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    neighbors = index.retrieve(streams) if neighbors is None else neighbors
    blocks = []
    for fit, hold in KFold(folds, shuffle=True, random_state=42).split(target.index):
        fit_ids, hold_ids = target.index[fit], target.index[hold]
        mapping = FamilyEvidence(index).fit(labels.loc[labels.client_id.isin(fit_ids)])
        positions = np.flatnonzero(streams.client_id.isin(hold_ids))
        query = streams.iloc[positions].reset_index(drop=True)
        matches = [neighbors[i] for i in positions]
        posterior = mapping.transform(query, matches)
        blocks.append(summary_features(query, matches, posterior, hold_ids))
    return pd.concat(blocks).reindex(target.index)


def resolve_merchant(history_or_stream, *, resolver, family_evidence=None):
    """Demo entry point; resolver is an explicitly fitted local reference index."""
    return resolver.resolve_merchant(history_or_stream, family_evidence)


def config_dict(index):
    return asdict(index.config)
