"""The single model boundary: actual frozen inference and verifiable history."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from threading import RLock

import joblib
import numpy as np
import pandas as pd

from transaction_forecasting.product.evidence import (
    candidate_streams,
    data_quality,
    evidence_summary,
    records,
)
from transaction_forecasting.product.provenance import (
    DEFAULT_BUNDLE,
    ROOT,
    ArtifactUnavailable,
    verify_bundle,
)
from transaction_forecasting.ubs.data import (
    LABELS,
    PREDICTION_COLUMN,
    read_transactions,
    validate_submission,
)


class UnknownClient(KeyError):
    """The client is not in the official TEST population."""


class ModelAdapter:
    def __init__(self, bundle: Path = DEFAULT_BUNDLE, data_dir: Path | None = None):
        self.bundle = Path(bundle)
        self.data_dir = Path(data_dir) if data_dir else ROOT / "data/raw/ubs_2026"
        self.manifest = verify_bundle(self.bundle, self.data_dir)
        # Only locally generated, fingerprint-checked artifacts; never accept uploads/pickles.
        self._model = joblib.load(self.bundle / "model.joblib")
        self._transactions = read_transactions(self.data_dir / "test_transactions.jsonl")
        self._histories = dict(tuple(self._transactions.groupby("client_id", sort=True)))
        self._predictions = pd.read_csv(self.bundle / "predictions.csv", dtype=str)
        sample = pd.read_csv(self.data_dir / "sample_submission.csv", dtype=str)
        validate_submission(self._predictions, sample, self._transactions)
        if len(sample) != 1000:
            raise ArtifactUnavailable("Expected exactly 1000 TEST clients")
        self._scores = pd.read_csv(self.bundle / "scores.csv", index_col="client_id")
        if list(self._scores.columns) != list(LABELS) or not self._scores.index.is_unique:
            raise ArtifactUnavailable("Invalid score columns or duplicated clients")
        if set(self._scores.index) != set(sample.client_id):
            raise ArtifactUnavailable("Score client IDs differ from sample")
        values = self._scores.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ArtifactUnavailable("Invalid score values")
        if not np.allclose(values.sum(axis=1), 1):
            raise ArtifactUnavailable("Invalid score totals")
        expected = self._predictions.set_index("client_id")[PREDICTION_COLUMN]
        if not expected.eq(self._scores.idxmax(axis=1).reindex(expected.index)).all():
            raise ArtifactUnavailable("Predictions differ from frozen score winners")
        self._lock = RLock()
        self._cache = {}
        self._metrics = json.loads((self.bundle / "metrics.json").read_text(encoding="utf-8"))
        self._cases = json.loads((self.bundle / "cases.json").read_text(encoding="utf-8"))

    def _history(self, client_id: str) -> pd.DataFrame:
        if client_id not in self._histories:
            raise UnknownClient(client_id)
        return self._histories[client_id]

    def get_model_metadata(self) -> dict:
        return {
            **deepcopy(self.manifest["model"]),
            "fit_scope": self.manifest["fit_scope"],
            "prediction_scope": "TEST",
            "artifact_sha256": self.manifest["artifacts"]["model.joblib"],
            "score_warning": "Model scores are not calibrated probabilities.",
            "agent_backend": "deterministic conditional policy; no LLM configured",
        }

    def get_global_metrics(self) -> dict:
        return deepcopy(self._metrics)

    def get_cases(self) -> list[dict]:
        return deepcopy(self._cases)

    def _predict(self, client_id: str) -> dict:
        history = self._history(client_id)
        with self._lock:
            components = self._model.predict_components(history)
        scores = components["A"].loc[client_id]
        if not np.allclose(scores, self._scores.loc[client_id], rtol=0, atol=1e-10):
            raise ArtifactUnavailable("Live scores differ from the frozen runner artifact")
        family = str(scores.idxmax())
        ranked = scores.sort_values(ascending=False, kind="stable")
        # Recover the two actual terms of the frozen 75/25 blend, not a new model.
        heuristic = (components["V2"] - 0.75 * components["history_control"]) / 0.25
        numeric = (components["A"] - 0.25 * heuristic) / 0.75
        component_choices = {
            "family_identity_model": str(numeric.loc[client_id].idxmax()),
            "periodicity_heuristic": str(heuristic.loc[client_id].idxmax()),
        }
        return {
            "client_id": client_id,
            "predicted_family": family,
            "model_version": self.manifest["model"]["model_version"],
            "base_sha": self.manifest["model"]["base_sha"],
            "horizon_days": 90,
            "cutoff": self.manifest["model"]["cutoff"],
            "scores": {label: float(scores[label]) for label in LABELS},
            "score": float(ranked.iloc[0]),
            "margin": float(ranked.iloc[0] - ranked.iloc[1]),
            "top_alternatives": [
                {"family": label, "score": float(value)}
                for label, value in ranked.iloc[1:4].items()
            ],
            "score_warning": "Scores are not calibrated probabilities or guarantees.",
            "component_choices": component_choices,
            "component_disagreement": len(set(component_choices.values())) > 1,
            "evidence": evidence_summary(history, self._model.mapper_, family),
        }

    def predict_client(self, client_id: str) -> dict:
        with self._lock:
            if client_id not in self._cache:
                self._cache[client_id] = self._predict(client_id)
            return deepcopy(self._cache[client_id])

    def get_client_history(self, client_id: str) -> list[dict]:
        return records(self._history(client_id))

    def get_candidate_streams(self, client_id: str) -> list[dict]:
        return candidate_streams(self._history(client_id), self._model.mapper_)

    def get_data_quality(self, client_id: str) -> dict:
        return data_quality(self._history(client_id), self._model.mapper_)

    def explain_prediction(self, client_id: str) -> dict:
        prediction = self.predict_client(client_id)
        return {
            "prediction": prediction,
            "candidates": self.get_candidate_streams(client_id),
            "history": self.get_client_history(client_id),
            "data_quality": self.get_data_quality(client_id),
            "limitations": [
                "History is context, not a causal explanation of the prediction.",
                "Family associations come from client labels; merchant identity is unverified.",
                "none is a predicted class, not an abstention or a guarantee of no payments.",
            ],
        }
