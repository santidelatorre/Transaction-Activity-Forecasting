"""Label-free denoising merchant representations (V4, phase 1).

Char TF-IDF + TruncatedSVD fit on unlabeled and TRAIN histories only.
A linear denoiser maps a corrupted description embedding back to the clean one.
Challenge labels are refused. client_id is never a feature. No PyTorch.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.v3.features import validate_history

GAP_BUCKETS = ("le7", "d8_30", "d31_90", "gt90")
AUX_NAMES = ("log_amount", "gap_days", "is_out", "is_card")


def gap_bucket(days: np.ndarray) -> np.ndarray:
    """Map a forward gap in days to a fixed four-bucket code. Not a challenge label."""
    values = np.asarray(days, dtype=float)
    buckets = np.full(len(values), 3, dtype=int)
    buckets[values <= 90] = 2
    buckets[values <= 30] = 1
    buckets[values <= 7] = 0
    return buckets


def corrupt_text(text: str, rng: np.random.Generator, rate: float) -> str:
    """Drop characters so the encoder must recover the merchant from a damaged view."""
    chars = list(str(text))
    if len(chars) <= 3:
        return str(text)
    kept = [char for char in chars if rng.random() > rate]
    if len(kept) < 3:
        kept = chars[:3]
    return "".join(kept)


def label_free_client_stats(transactions: pd.DataFrame) -> pd.DataFrame:
    """Recurrence summaries that do not use challenge labels or client_id as a feature."""
    validate_history(transactions)
    frame = transactions.sort_values(["client_id", "description", "timestamp"], kind="stable")
    grouped = frame.groupby("client_id", sort=True)
    stats = pd.DataFrame(
        {
            "n_tx": grouped.size().astype(float),
            "n_desc": grouped["description"].nunique().astype(float),
            "log_amount_median": np.log1p(grouped["amount"].median().clip(lower=0)),
        }
    )
    payments = frame.loc[frame["direction"].eq("out") & frame["type"].eq("card_payment")]
    due_count: dict[str, float] = {}
    periods: dict[str, float] = {}
    for client, rows in payments.groupby("client_id", sort=True):
        horizons: list[float] = []
        n_due = 0
        for _, stream in rows.groupby("description", sort=False):
            if len(stream) < 2:
                continue
            gaps = stream["timestamp"].diff().dt.total_seconds().dropna() / 86400
            period = float(np.median(gaps.to_numpy()))
            horizons.append(period)
            nxt = stream["timestamp"].max() + pd.Timedelta(days=period)
            if CUTOFF < nxt <= CUTOFF + pd.Timedelta(days=90):
                n_due += 1
        due_count[str(client)] = float(n_due)
        periods[str(client)] = float(np.median(horizons)) if horizons else 0.0
    stats["n_due_90"] = pd.Series(due_count).reindex(stats.index).fillna(0.0)
    stats["median_period"] = pd.Series(periods).reindex(stats.index).fillna(0.0)
    stats["repeat_share"] = 1.0 - (stats["n_desc"] / stats["n_tx"].clip(lower=1))
    return stats


class DenoisingEncoder:
    """Phase-1 encoder: text SVD, linear denoiser, backward time/amount context."""

    def __init__(
        self,
        *,
        n_components: int = 32,
        max_descriptions: int = 25_000,
        max_features: int = 12_000,
        min_df: int = 2,
        corrupt_rate: float = 0.35,
        ridge_alpha: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.max_descriptions = max_descriptions
        self.max_features = max_features
        self.min_df = min_df
        self.corrupt_rate = corrupt_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed

    def fit(
        self, transactions: pd.DataFrame, labels: pd.DataFrame | None = None
    ) -> DenoisingEncoder:
        """Fit text geometry and the denoiser. `labels` is rejected if passed."""
        if labels is not None or TARGET_COLUMN in transactions.columns:
            raise ValueError("Challenge labels must not enter pretraining")
        validate_history(transactions)
        rng = np.random.default_rng(self.seed)
        unique = pd.unique(transactions["description"].astype(str))
        if len(unique) > self.max_descriptions:
            unique = rng.choice(unique, size=self.max_descriptions, replace=False)
        texts = [str(value) for value in unique]
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        matrix = self.vectorizer_.fit_transform(texts)
        components = min(self.n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if components < 2:
            raise ValueError("Not enough description variety to fit an SVD")
        self.n_components_ = int(components)
        self.svd_ = TruncatedSVD(n_components=self.n_components_, random_state=self.seed)
        clean = self.svd_.fit_transform(matrix)
        corrupted = self.svd_.transform(
            self.vectorizer_.transform(self._corrupt_list(texts, salt=1))
        )
        self.denoiser_ = Ridge(alpha=self.ridge_alpha, random_state=self.seed)
        self.denoiser_.fit(corrupted, clean)
        self._fit_aux_scales(transactions)
        self.fit_rows_ = int(len(transactions))
        self.fit_descriptions_ = int(len(texts))
        return self

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> DenoisingEncoder:
        model = joblib.load(path)
        if not isinstance(model, DenoisingEncoder):
            raise TypeError(f"Unexpected object in {path}")
        return model

    def text_embedding(self, descriptions: pd.Series, *, mode: str, salt: int = 0) -> np.ndarray:
        """Return SVD rows. mode is clean, raw_corrupt, or denoised."""
        texts = descriptions.astype(str).tolist()
        if mode != "clean":
            texts = self._corrupt_list(texts, salt=salt)
        embedded = self.svd_.transform(self.vectorizer_.transform(texts))
        if mode == "denoised":
            return self.denoiser_.predict(embedded)
        if mode == "clean" or mode == "raw_corrupt":
            return embedded
        raise ValueError(f"Unknown text mode: {mode}")

    def event_frame(
        self, transactions: pd.DataFrame, *, mode: str = "clean", salt: int = 0
    ) -> pd.DataFrame:
        """Event matrix aligned to `transactions` rows. Backward gaps only."""
        validate_history(transactions)
        text = self.text_embedding(transactions["description"], mode=mode, salt=salt)
        aux = self._aux_matrix(transactions)
        columns = [f"z{i:02d}" for i in range(text.shape[1])] + list(AUX_NAMES)
        frame = pd.DataFrame(np.hstack([text, aux]), index=transactions.index, columns=columns)
        frame.insert(0, "client_id", transactions["client_id"].astype(str).to_numpy())
        return frame

    def client_features(
        self, transactions: pd.DataFrame, *, mode: str = "clean", salt: int = 0
    ) -> pd.DataFrame:
        """Mean event embedding plus mean of each stream's last event. No client_id column."""
        events = self.event_frame(transactions, mode=mode, salt=salt)
        values = events.drop(columns="client_id")
        means = values.groupby(events["client_id"], sort=True).mean()
        order = transactions.sort_values(["client_id", "description", "timestamp"], kind="stable")
        tail_index = order.groupby(["client_id", "description"], sort=False).tail(1).index
        tail_clients = events.loc[tail_index, "client_id"]
        tails = values.loc[tail_index].groupby(tail_clients, sort=True).mean()
        tails.columns = [f"tail_{column}" for column in tails.columns]
        return means.join(tails)

    def drift(self, transactions: pd.DataFrame, *, n_rows: int = 4_000) -> dict[str, float]:
        """L2 distance of corrupted views to the clean embedding. Lower is more stable."""
        validate_history(transactions)
        rng = np.random.default_rng(self.seed)
        take = min(n_rows, len(transactions))
        sample = transactions.iloc[rng.choice(len(transactions), size=take, replace=False)]
        clean = self.text_embedding(sample["description"], mode="clean")
        raw = self.text_embedding(sample["description"], mode="raw_corrupt", salt=7)
        denoised = self.text_embedding(sample["description"], mode="denoised", salt=7)
        raw_l2 = np.linalg.norm(clean - raw, axis=1)
        den_l2 = np.linalg.norm(clean - denoised, axis=1)
        return {
            "n_rows": float(take),
            "raw_corrupt_l2_mean": float(raw_l2.mean()),
            "denoised_l2_mean": float(den_l2.mean()),
            "stability_gain": float(raw_l2.mean() - den_l2.mean()),
        }

    def contrastive(self, transactions: pd.DataFrame, *, n_clients: int = 200) -> dict[str, float]:
        """Same-client corrupt views should be closer than different-client views."""
        validate_history(transactions)
        counts = transactions.groupby("client_id").size()
        eligible = counts[counts >= 3].index.astype(str)
        rng = np.random.default_rng(self.seed)
        if len(eligible) == 0:
            raise ValueError("Need clients with at least three transactions")
        if len(eligible) > n_clients:
            eligible = rng.choice(eligible.to_numpy(), size=n_clients, replace=False)
        subset = transactions[transactions["client_id"].astype(str).isin(set(map(str, eligible)))]
        left = self.client_features(subset, mode="denoised", salt=3)
        right = self.client_features(subset, mode="denoised", salt=4)
        shared = left.index.intersection(right.index)
        a = _l2_normalize(left.loc[shared].to_numpy())
        b = _l2_normalize(right.loc[shared].to_numpy())
        same = np.sum(a * b, axis=1)
        cross = a @ b.T
        np.fill_diagonal(cross, np.nan)
        return {
            "n_clients": float(len(shared)),
            "same_client_cosine": float(np.nanmean(same)),
            "cross_client_cosine": float(np.nanmean(cross)),
            "margin": float(np.nanmean(same) - np.nanmean(cross)),
        }

    def gap_probe(self, transactions: pd.DataFrame, *, max_pairs: int = 20_000) -> dict[str, float]:
        """Predict the next same-description gap bucket from the current event. No labels."""
        validate_history(transactions)
        ordered = transactions.sort_values(["client_id", "description", "timestamp"], kind="stable")
        nxt = ordered.groupby(["client_id", "description"], sort=False)["timestamp"].shift(-1)
        gap_days = (nxt - ordered["timestamp"]).dt.total_seconds() / 86400
        has_next = gap_days.notna()
        if int(has_next.sum()) < 40:
            return {
                "n_pairs": float(has_next.sum()),
                "probe_accuracy": float("nan"),
                "majority": float("nan"),
            }
        pairs = ordered.loc[has_next]
        if len(pairs) > max_pairs:
            pairs = pairs.sample(max_pairs, random_state=self.seed)
            gap_days = gap_days.loc[pairs.index]
        features = self.event_frame(pairs, mode="clean").drop(columns="client_id").to_numpy()
        target = gap_bucket(gap_days.loc[pairs.index].to_numpy())
        majority = float(np.bincount(target).max() / len(target))
        accuracy = _stratified_accuracy(features, target, seed=self.seed)
        return {
            "n_pairs": float(len(target)),
            "majority_accuracy": majority,
            "probe_accuracy": accuracy,
            "lift_vs_majority": accuracy - majority,
        }

    def _corrupt_list(self, texts: list[str], *, salt: int) -> list[str]:
        rng = np.random.default_rng(self.seed + 17 * salt)
        return [corrupt_text(text, rng, self.corrupt_rate) for text in texts]

    def _fit_aux_scales(self, transactions: pd.DataFrame) -> None:
        sample = transactions
        if len(sample) > 50_000:
            sample = transactions.sample(50_000, random_state=self.seed)
        aux = _raw_aux(sample)
        self.aux_mean_ = aux.mean(axis=0)
        self.aux_std_ = aux.std(axis=0)
        self.aux_std_[self.aux_std_ < 1e-6] = 1.0

    def _aux_matrix(self, transactions: pd.DataFrame) -> np.ndarray:
        raw = _raw_aux(transactions)
        return (raw - self.aux_mean_) / self.aux_std_


def client_grouped_oof(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    seed: int = 42,
    n_splits: int = 5,
) -> pd.DataFrame:
    """5-fold probabilities. One row per client, so folds never split a client."""
    if not features.index.is_unique or not target.index.is_unique:
        raise ValueError("Client index must be unique")
    y = target.reindex(features.index)
    if y.isna().any() or not y.isin(LABELS).all():
        raise ValueError("Target must cover every client with the fixed label set")
    y_np = y.to_numpy()
    x = features.to_numpy(dtype=float)
    n_splits = min(n_splits, int(pd.Series(y_np).value_counts().min()))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(features), len(LABELS)), dtype=float)
    for train_idx, valid_idx in folds.split(x, y_np):
        train_ids = set(features.index[train_idx])
        valid_ids = set(features.index[valid_idx])
        if train_ids & valid_ids:
            raise ValueError("Client leakage between downstream folds")
        model = _linear_head()
        model.fit(x[train_idx], y_np[train_idx])
        oof[valid_idx] = _align_proba(model, x[valid_idx])
    return pd.DataFrame(oof, index=features.index, columns=list(LABELS))


def fit_linear_head(features: pd.DataFrame, target: pd.Series):
    """Fit the frozen downstream head on all provided TRAIN rows."""
    y = target.reindex(features.index)
    model = _linear_head()
    model.fit(features.to_numpy(dtype=float), y.to_numpy())
    return model


def predict_proba_frame(model, features: pd.DataFrame) -> pd.DataFrame:
    aligned = _align_proba(model, features.to_numpy(dtype=float))
    return pd.DataFrame(aligned, index=features.index, columns=list(LABELS))


def _linear_head() -> object:
    scaler = StandardScaler()
    clf = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        random_state=42,
    )
    from sklearn.pipeline import Pipeline

    return Pipeline([("scaler", scaler), ("clf", clf)])


def _align_proba(model, rows: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(rows)
    classes = list(model.named_steps["clf"].classes_)
    aligned = np.zeros((len(rows), len(LABELS)), dtype=float)
    for column, label in enumerate(classes):
        aligned[:, list(LABELS).index(label)] = raw[:, column]
    return aligned


def _stratified_accuracy(features: np.ndarray, target: np.ndarray, *, seed: int) -> float:
    counts = np.bincount(target)
    n_splits = int(min(3, counts[counts > 0].min()))
    if n_splits < 2:
        return float("nan")
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    hits = 0
    for train_idx, valid_idx in folds.split(features, target):
        clf = LogisticRegression(max_iter=300, random_state=seed)
        clf.fit(features[train_idx], target[train_idx])
        hits += int((clf.predict(features[valid_idx]) == target[valid_idx]).sum())
    return hits / len(target)


def _raw_aux(transactions: pd.DataFrame) -> np.ndarray:
    ordered = transactions.sort_values(["client_id", "description", "timestamp"], kind="stable")
    grouped = ordered.groupby(["client_id", "description"], sort=False)["timestamp"]
    gap = grouped.diff().dt.total_seconds() / 86400
    gap = gap.reindex(transactions.index).fillna(0.0).clip(0, 180).to_numpy(dtype=float)
    log_amount = np.log1p(transactions["amount"].clip(lower=0).to_numpy(dtype=float))
    is_out = transactions["direction"].astype(str).eq("out").to_numpy(dtype=float)
    is_card = transactions["type"].astype(str).eq("card_payment").to_numpy(dtype=float)
    return np.column_stack([log_amount, gap, is_out, is_card])


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return matrix / norms
