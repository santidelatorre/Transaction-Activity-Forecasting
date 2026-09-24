from __future__ import annotations

import json

import pandas as pd

from transaction_forecasting.pipeline import run_frame, synthetic_config, synthetic_frame


class RecordingTransformer:
    def __init__(self) -> None:
        self.fit_latest: pd.Timestamp | None = None

    def fit(self, frame: pd.DataFrame) -> RecordingTransformer:
        self.fit_latest = frame["event_time"].max()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()


def test_generic_pipeline_is_temporal_and_writes_jsonl(tmp_path) -> None:
    results_path = tmp_path / "results.jsonl"
    transformer = RecordingTransformer()
    result = run_frame(synthetic_frame(), synthetic_config(results_path), transformer=transformer)
    assert transformer.fit_latest is not None
    assert transformer.fit_latest < synthetic_frame().iloc[-4]["event_time"]
    assert result["rows"] == {"train": 12, "validation": 4, "test": 4}
    assert result["test"]["mae"] >= 0.0
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved["model"] == "mean_regressor"
