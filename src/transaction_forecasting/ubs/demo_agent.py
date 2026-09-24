"""Subscription Foresight demo agent — tool-calling spine for jury UX.

Tools (no VALID labels required for prediction/explanation):
  - show_metrics: frozen PushEmbed scoreboard
  - explain_streams: recurring / due stream timeline for a client
  - predict_client: PushEmbed family + probability vector (cached or live)

Leakage: predictions use only pre-cutoff transactions. True labels are never
fed into the model; optional --reveal-label is display-only for internal QA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from transaction_forecasting.ubs.data import CUTOFF, LABELS, TARGET_COLUMN, load_ubs_data
from transaction_forecasting.ubs.stream_oracle import build_streams
from transaction_forecasting.ubs.text_v2 import normalize_description
from transaction_forecasting.ubs.v3_pretrain import NOISE, StreamV3PushEmbedModel

DEFAULT_METRICS = Path("reports/handoff/v3_discovery/esteban_v3_pretrain_summary.json")
DEFAULT_CACHE = Path("outputs/demo")


@dataclass
class ForesightAgent:
    """Small agent with explicit tools for the jury demo path."""

    data_dir: Path = Path("data/raw/ubs_2026")
    cache_dir: Path = DEFAULT_CACHE
    metrics_path: Path = DEFAULT_METRICS
    push_weight: float = 0.85

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data = None
        self._model: StreamV3PushEmbedModel | None = None
        self._pred_cache: pd.DataFrame | None = None

    # --- tools -----------------------------------------------------------------

    def show_metrics(self) -> dict[str, Any]:
        """Return frozen PushEmbed metrics and honesty notes for the pitch."""
        payload: dict[str, Any]
        if self.metrics_path.exists():
            payload = json.loads(self.metrics_path.read_text())
        else:
            payload = {
                "model": "StreamV3PushEmbedModel",
                "push_embed_macro_f1": 0.4570384307914666,
                "push_embed_accuracy": 0.484,
                "v2_macro_f1": 0.3915494559105542,
                "push_macro_f1": 0.44454709747059196,
            }
        payload["honesty"] = {
            "ulmans_board_0.424": "V2 accuracy, not Macro-F1",
            "official_metric": "8-class Macro-F1",
            "oracle_0.77": "diagnostic ceiling (uses labels among candidates), not a submission",
            "cutoff": str(CUTOFF),
        }
        payload["story"] = {
            "product": "Subscription Foresight",
            "one_liner": (
                "Predict which recurring merchant family hits a client next "
                "in the 90-day window after cutoff — bank ops / cashflow lens."
            ),
        }
        return payload

    def explain_streams(self, client_id: str, *, top_n: int = 8) -> dict[str, Any]:
        """Explain recurring / due streams that drive the next-merchant story."""
        data = self._ensure_data()
        cid = str(client_id)
        tx = self._client_transactions(cid)
        if tx.empty:
            return {"client_id": cid, "error": "client not found in train/valid/test packs"}

        streams = build_streams(tx)
        if streams.empty:
            return {"client_id": cid, "streams": [], "note": "no streams built"}

        frame = streams.copy()
        frame["description"] = frame["description"].map(normalize_description)
        recurrent = frame[frame["is_recurrent"]].copy()
        due = frame[frame["is_candidate"]].copy()
        signal = recurrent[~recurrent["description"].isin(NOISE)].copy()
        if signal.empty:
            signal = due[~due["description"].isin(NOISE)].copy()

        rows: list[dict[str, Any]] = []
        if not signal.empty:
            signal = signal.sort_values(["is_candidate", "appearances"], ascending=[False, False])
            for _, row in signal.head(top_n).iterrows():
                rows.append(
                    {
                        "description": str(row["description"]),
                        "appearances": int(row["appearances"]),
                        "is_due_in_90d": bool(row["is_candidate"]),
                        "projected_next": str(row.get("projected_next_date", "")),
                        "gap_median_days": float(row.get("gap_median_days", np.nan))
                        if pd.notna(row.get("gap_median_days", np.nan))
                        else None,
                        "mcc": str(row.get("mcc", "")),
                        "amount_median": float(row.get("amount_median", np.nan))
                        if pd.notna(row.get("amount_median", np.nan))
                        else None,
                    }
                )

        pack = (
            "valid"
            if cid in set(data.valid_labels["client_id"].astype(str))
            else (
                "train" if cid in set(data.train_labels["client_id"].astype(str)) else "unlabelled"
            )
        )
        return {
            "client_id": cid,
            "pack": pack,
            "n_transactions": int(len(tx)),
            "n_recurrent_streams": int(recurrent.shape[0]),
            "n_due_streams": int(due.shape[0]),
            "timeline": rows,
            "narrative": self._timeline_narrative(cid, rows),
        }

    def predict_client(
        self,
        client_id: str,
        *,
        reveal_label: bool = False,
        fit_if_needed: bool = True,
    ) -> dict[str, Any]:
        """Predict next recurring merchant family for one client."""
        data = self._ensure_data()
        cid = str(client_id)
        tx = self._client_transactions(cid)
        if tx.empty:
            return {"client_id": cid, "error": "client not found"}

        # Prefer prediction cache (screenshot-ready, no 7-min fit).
        cached = self._load_pred_cache()
        if cached is not None and cid in cached.index:
            proba = cached.loc[cid, list(LABELS)].astype(float)
            pred = str(proba.idxmax())
            source = "cache"
        else:
            if not fit_if_needed and self._model is None:
                return {
                    "client_id": cid,
                    "error": (
                        "No prediction cache and model not fitted. "
                        "Run: python scripts/demo_subscription_foresight.py warm"
                    ),
                }
            model = self._ensure_model()
            # Model refuses train clients used at fit time.
            if cid in model.fit_clients_:
                return {
                    "client_id": cid,
                    "error": (
                        "Client was in the training fit set; pick a VALID/test client "
                        "or use warm-cache built on VALID."
                    ),
                    "hint": "Try a valid client_id from list_clients --pack valid",
                }
            comps = model.predict_components(tx)
            proba = comps["blend"].loc[cid]
            pred = str(proba.idxmax())
            source = "live_model"

        top = proba.sort_values(ascending=False).head(3)
        out: dict[str, Any] = {
            "client_id": cid,
            "predicted_next_recurring_merchant": pred,
            "confidence": float(proba.max()),
            "top3": [{k: float(v)} for k, v in top.items()],
            "probabilities": {k: float(proba[k]) for k in LABELS},
            "source": source,
            "product_line": "Subscription Foresight",
        }
        if reveal_label:
            labels = pd.concat(
                [
                    data.train_labels.set_index("client_id")[TARGET_COLUMN],
                    data.valid_labels.set_index("client_id")[TARGET_COLUMN],
                ]
            )
            labels.index = labels.index.astype(str)
            if cid in labels.index:
                out["true_label_display_only"] = str(labels.loc[cid])
                out["label_note"] = "Display only — never used at train/predict time in this call"
        return out

    def list_clients(self, *, pack: str = "valid", limit: int = 12) -> dict[str, Any]:
        data = self._ensure_data()
        if pack == "valid":
            ids = list(data.valid_labels["client_id"].astype(str).head(limit))
        elif pack == "train":
            ids = list(data.train_labels["client_id"].astype(str).head(limit))
        else:
            return {"error": "pack must be valid or train"}
        return {"pack": pack, "client_ids": ids}

    def warm(self, *, max_valid: int | None = None) -> dict[str, Any]:
        """Fit PushEmbed on TRAIN and cache VALID probabilities for instant demo."""
        data = self._ensure_data()
        model = StreamV3PushEmbedModel(push_weight=self.push_weight).fit(
            data.train_transactions, data.train_labels
        )
        self._model = model
        valid_tx = data.valid_transactions
        if max_valid is not None:
            keep = set(data.valid_labels["client_id"].astype(str).head(max_valid))
            valid_tx = valid_tx[valid_tx["client_id"].astype(str).isin(keep)]
        comps = model.predict_components(valid_tx)
        blend = comps["blend"].copy()
        blend["prediction"] = blend.idxmax(axis=1)
        parquet_path = self.cache_dir / "valid_push_embed_proba.parquet"
        csv_path = self.cache_dir / "valid_push_embed_proba.csv"
        try:
            blend.to_parquet(parquet_path)
            path = parquet_path
        except (ImportError, ValueError):
            blend.to_csv(csv_path)
            path = csv_path
        self._pred_cache = blend
        meta = {
            "path": str(path),
            "n_clients": int(len(blend)),
            "push_weight": self.push_weight,
            "model": "StreamV3PushEmbedModel",
        }
        (self.cache_dir / "warm_meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    # --- internals -------------------------------------------------------------

    def _ensure_data(self):
        if self._data is None:
            self._data = load_ubs_data(self.data_dir)
        return self._data

    def _ensure_model(self) -> StreamV3PushEmbedModel:
        if self._model is None:
            data = self._ensure_data()
            self._model = StreamV3PushEmbedModel(push_weight=self.push_weight).fit(
                data.train_transactions, data.train_labels
            )
        return self._model

    def _load_pred_cache(self) -> pd.DataFrame | None:
        if self._pred_cache is not None:
            return self._pred_cache
        parquet_path = self.cache_dir / "valid_push_embed_proba.parquet"
        csv_path = self.cache_dir / "valid_push_embed_proba.csv"
        if parquet_path.exists():
            try:
                self._pred_cache = pd.read_parquet(parquet_path)
                return self._pred_cache
            except (ImportError, ValueError):
                pass
        if csv_path.exists():
            self._pred_cache = pd.read_csv(csv_path, index_col=0)
            return self._pred_cache
        return None

    def _client_transactions(self, client_id: str) -> pd.DataFrame:
        data = self._ensure_data()
        frames = [
            data.train_transactions,
            data.valid_transactions,
            data.test_transactions,
        ]
        pieces = []
        for frame in frames:
            hit = frame[frame["client_id"].astype(str).eq(client_id)]
            if not hit.empty:
                pieces.append(hit)
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)

    @staticmethod
    def _timeline_narrative(client_id: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return (
                f"Client {client_id}: no clean recurring merchants above noise — "
                "model may lean toward none or weak positives."
            )
        due = [r for r in rows if r.get("is_due_in_90d")]
        head = due[0] if due else rows[0]
        when = head.get("projected_next") or "unknown date"
        return (
            f"Client {client_id}: strongest recurring signal looks like "
            f"“{head['description']}” (≈{head.get('appearances', '?')} hits). "
            f"Projected next around {when}. "
            "Subscription Foresight maps this geometry + history/identity/pretrain "
            "embeds into one of eight families."
        )


def run_tool(agent: ForesightAgent, name: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch a named tool (agentic depth without a heavy framework)."""
    tools = {
        "show_metrics": agent.show_metrics,
        "explain_streams": agent.explain_streams,
        "predict_client": agent.predict_client,
        "list_clients": agent.list_clients,
        "warm": agent.warm,
    }
    if name not in tools:
        return {"error": f"unknown tool {name}", "available": sorted(tools)}
    return tools[name](**kwargs)
