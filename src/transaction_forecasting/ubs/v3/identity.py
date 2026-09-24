"""Client-isolated merchant identity evidence for the V3-A numeric arm."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.text_v2 import normalize_description
from transaction_forecasting.ubs.v3.features import FAMILIES, validate_history


class IdentityMap:
    """Fit description associations on clients disjoint from transform clients."""

    def __init__(self, *, normalized=False, probabilities=False, char_alias=False):
        self.normalized = normalized
        self.probabilities = probabilities
        self.char_alias = char_alias

    def fit(self, transactions: pd.DataFrame, labels: pd.DataFrame):
        validate_history(transactions)
        target = labels.set_index("client_id")[TARGET_COLUMN]
        if not target.index.is_unique or set(target.index) != set(transactions.client_id):
            raise ValueError("Unique labels must exactly match history clients")
        if not target.isin(LABELS).all():
            raise ValueError("Invalid family")
        self.fit_clients_ = set(target.index)
        names = (
            transactions.description.map(normalize_description)
            if self.normalized
            else transactions.description
        )
        presence = pd.DataFrame(
            {"client_id": transactions.client_id, "name": names}
        ).drop_duplicates()
        presence[TARGET_COLUMN] = presence.client_id.map(target)
        counts = pd.crosstab(presence.name, presence[TARGET_COLUMN]).reindex(
            columns=LABELS, fill_value=0
        )
        support = counts.sum(axis=1)
        sizes = target.value_counts().reindex(LABELS, fill_value=0)
        lift = pd.DataFrame(index=counts.index)
        for family in FAMILIES:
            inside = (counts[family] + 1) / (sizes[family] + 2)
            outside = (support - counts[family] + 1) / (len(target) - sizes[family] + 2)
            lift[family] = np.log(inside / outside) if sizes[family] else 0.0
        eligible = support.ge(5) & lift.max(axis=1).ge(np.log(1.5))
        self.mapping_ = lift.idxmax(axis=1).where(eligible, "unknown")
        prior = sizes / len(target)
        self.probability_ = (
            counts[list(FAMILIES)].add(8 * prior[list(FAMILIES)], axis=1).div(support + 8, axis=0)
        )
        self.support_ = support
        self.audit_ = pd.DataFrame(
            {"family": self.mapping_, "clients": support, "log_lift": lift.max(axis=1)}
        )
        if self.char_alias:
            # Char matching is limited to eligible, supervised aliases. Exact keys win.
            known = self.mapping_.loc[self.mapping_.ne("unknown")].index.astype(str)
            self.vectorizer_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            self.alias_matrix_ = self.vectorizer_.fit_transform(known)
            self.alias_names_ = known
        return self

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        validate_history(transactions)
        if self.fit_clients_.intersection(transactions.client_id):
            raise ValueError("Identity features require clients excluded from mapping fit")
        result = transactions.copy()
        result["name"] = (
            result.description.map(normalize_description) if self.normalized else result.description
        )
        result["family"] = result.name.map(self.mapping_).fillna("unknown")
        result["evidence"] = result.name.map(self.support_).fillna(0).astype(float)
        if self.char_alias:
            unseen = result.family.eq("unknown")
            names = pd.Index(result.loc[unseen, "name"].unique())
            if len(names) and len(self.alias_names_):
                similarities = self.vectorizer_.transform(names.astype(str)) @ self.alias_matrix_.T
                best = np.asarray(similarities.argmax(axis=1)).ravel()
                scores = np.asarray(similarities.max(axis=1).toarray()).ravel()
                accepted = scores >= 0.88
                aliases = pd.Series(
                    [
                        self.alias_names_[i] if ok else None
                        for i, ok in zip(best, accepted, strict=True)
                    ],
                    index=names,
                )
                source = result.loc[unseen, "name"].map(aliases)
                result.loc[unseen, "family"] = (
                    source.map(self.mapping_).fillna("unknown").to_numpy()
                )
                result.loc[unseen, "evidence"] = source.map(self.support_).fillna(0).to_numpy()
        if self.probabilities:
            for family in FAMILIES:
                result[f"prob_{family}"] = (
                    result.name.map(self.probability_[family]).fillna(0).to_numpy()
                )
        return result


def identity_features(mapped: pd.DataFrame, *, probabilities=False) -> pd.DataFrame:
    """V3-A identity columns plus compact soft family evidence if requested."""
    clients = pd.Index(sorted(mapped.client_id.unique()), name="client_id")
    outgoing = mapped.loc[mapped.direction.eq("out") & mapped.type.eq("card_payment")]
    total = outgoing.groupby("client_id").size().reindex(clients, fill_value=0).clip(lower=1)
    blocks = []
    for family in FAMILIES:
        rows = outgoing.loc[outgoing.family.eq(family)].groupby("client_id")
        block = pd.DataFrame(index=clients)
        block[f"identity_{family}_count"] = rows.size().reindex(clients, fill_value=0)
        block[f"identity_{family}_share"] = block.iloc[:, 0] / total
        block[f"identity_{family}_aliases"] = rows.name.nunique().reindex(clients, fill_value=0)
        if probabilities:
            weighted = outgoing[f"prob_{family}"]
            block[f"identity_{family}_prob_mean"] = (
                weighted.groupby(outgoing.client_id).mean().reindex(clients, fill_value=0)
            )
            block[f"identity_{family}_prob_max"] = (
                weighted.groupby(outgoing.client_id).max().reindex(clients, fill_value=0)
            )
            recurring = outgoing.loc[outgoing.duplicated(["client_id", "name"], keep=False)]
            block[f"identity_{family}_prob_recurrent"] = (
                recurring[f"prob_{family}"]
                .groupby(recurring.client_id)
                .mean()
                .reindex(clients, fill_value=0)
            )
        blocks.append(block)
    return pd.concat(blocks, axis=1)


def cross_fitted_identity_features(transactions, labels, *, folds=5, **options):
    """Every held-out client is transformed by a map fitted on other clients."""
    target = labels.set_index("client_id")[TARGET_COLUMN].sort_index()
    if not target.index.is_unique or set(target.index) != set(transactions.client_id):
        raise ValueError("Unique labels must exactly match history clients")
    splitter = StratifiedKFold(folds, shuffle=True, random_state=42)
    blocks = []
    for fit_pos, hold_pos in splitter.split(target.index, target):
        fit_ids, hold_ids = target.index[fit_pos], target.index[hold_pos]
        mapper = IdentityMap(**options).fit(
            transactions.loc[transactions.client_id.isin(fit_ids)],
            labels.loc[labels.client_id.isin(fit_ids)],
        )
        mapped = mapper.transform(transactions.loc[transactions.client_id.isin(hold_ids)])
        blocks.append(identity_features(mapped, probabilities=options.get("probabilities", False)))
    result = pd.concat(blocks).reindex(target.index)
    if result.isna().any().any() or not result.index.is_unique:
        raise RuntimeError("Incomplete cross-fitted identity features")
    return result
