"""Final recipe guards: no VALID refit, exact submission and score contract."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.ubs.data import LABELS, PREDICTION_COLUMN, TARGET_COLUMN
from transaction_forecasting.ubs.v4 import predict_probabilities, submission_frame


class FixedModel:
    def predict_components(self, history):
        ids = pd.Index(sorted(history.client_id.unique()), name="client_id")
        scores = pd.DataFrame(0.0, index=ids, columns=LABELS)
        scores["cloud"] = 1.0
        return {"A": scores}


def load_runner():
    path = Path(__file__).parents[1] / "scripts/run_ubs_v4.py"
    spec = importlib.util.spec_from_file_location("v4_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_never_opens_valid_and_preserves_order(tmp_path, monkeypatch):
    runner = load_runner()
    data, out = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    # There are deliberately no VALID files in this test.
    (data / "train_transactions.jsonl").write_text("train placeholder")
    (data / "train_labels.csv").write_text("labels placeholder")
    ids = [f"T{i:04}" for i in range(999, -1, -1)]
    sample = pd.DataFrame({"client_id": ids, PREDICTION_COLUMN: "none"})
    sample.to_csv(data / "sample_submission.csv", index=False)
    train = pd.DataFrame({"client_id": ["train-client"]})
    test = pd.DataFrame({"client_id": ids})
    labels = pd.DataFrame({"client_id": ["train-client"], TARGET_COLUMN: ["gym"]})
    reads = []

    def history(path):
        reads.append(path.name)
        return {"train_transactions.jsonl": train, "test_transactions.jsonl": test}[path.name]

    def read_labels(path):
        assert path.name == "train_labels.csv"
        reads.append(path.name)
        return labels

    def fit(fit_history, fit_labels):
        pd.testing.assert_frame_equal(fit_history, train)
        pd.testing.assert_frame_equal(fit_labels, labels)
        return FixedModel()

    monkeypatch.setattr(runner, "read_transactions", history)
    monkeypatch.setattr(runner, "read_labels", read_labels)
    monkeypatch.setattr(runner, "fit_final_model", fit)
    runner.run("submission", data, out)
    result = pd.read_csv(out / "submission_v4.csv")
    assert result.client_id.tolist() == ids
    assert result[PREDICTION_COLUMN].eq("cloud").all()
    assert len(result) == 1000 and result.client_id.is_unique
    assert not any("valid" in name for name in reads)
    with pytest.raises(ValueError, match="overwrite"):
        runner.run("submission", data, out)


def test_submission_rejects_wrong_count_and_extra_probability_ids():
    history = pd.DataFrame({"client_id": [f"T{i}" for i in range(1000)]})
    scores = predict_probabilities(FixedModel(), history)
    sample = history.assign(**{PREDICTION_COLUMN: "none"})
    with pytest.raises(ValueError, match="1000"):
        submission_frame(scores, sample.iloc[:-1], history)
    wrong = scores.rename(index={"T0": "extra"})
    with pytest.raises(ValueError, match="IDs"):
        submission_frame(wrong, sample, history)


def test_final_prediction_rejects_nan_or_wrong_class_order():
    history = pd.DataFrame({"client_id": ["T1"]})

    class BadModel:
        def predict_components(self, rows):
            scores = FixedModel().predict_components(rows)["A"]
            scores.iloc[0, 0] = np.nan
            return {"A": scores}

    with pytest.raises(ValueError, match="Invalid"):
        predict_probabilities(BadModel(), history)
