"""Checks that candidate comparison preserves the supplied split and label contract."""

import numpy as np
import pandas as pd
import pytest

from transaction_forecasting.evaluation.model_comparison import compare_classifiers


class RecordingClassifier:
    def __init__(self, record: dict[str, np.ndarray]):
        self.record = record

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RecordingClassifier":
        self.record["fit"] = features.copy()
        self.record["target"] = target.copy()
        self.classes_ = np.unique(target)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.record["valid"] = features.copy()
        probabilities = np.zeros((len(features), len(self.classes_)))
        probabilities[:, -1] = 1.0
        return probabilities


def test_same_external_client_split_train_only_imputation_and_all_class_macro_f1() -> None:
    features = pd.DataFrame(
        {"signal": [1000.0, np.nan, 1.0, 3.0, np.nan]},
        index=["valid-2", "train-2", "train-1", "train-3", "valid-1"],
    )
    target = pd.Series(
        ["a", "b", "a", "c", "c"],
        index=features.index,
    )
    first: dict[str, np.ndarray] = {}
    second: dict[str, np.ndarray] = {}
    result = compare_classifiers(
        features,
        target,
        train_ids=["train-1", "train-2", "train-3"],
        validation_ids=["valid-1", "valid-2"],
        labels=("a", "b", "c"),
        candidates={
            "first": lambda: RecordingClassifier(first),
            "second": lambda: RecordingClassifier(second),
        },
    )
    np.testing.assert_array_equal(first["fit"], second["fit"])
    np.testing.assert_array_equal(first["valid"], second["valid"])
    np.testing.assert_array_equal(first["fit"].ravel(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(first["valid"].ravel(), [2.0, 1000.0])
    assert result.predictions["first"].index.tolist() == ["valid-1", "valid-2"]
    assert result.summary.set_index("model").loc["first", "macro_f1"] == pytest.approx(2 / 9)
    assert "equal_probability_ensemble" in result.predictions


def test_rejects_overlapping_client_partitions() -> None:
    features = pd.DataFrame({"signal": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    target = pd.Series(["x", "y", "z"], index=features.index)
    with pytest.raises(ValueError, match="disjoint"):
        compare_classifiers(
            features,
            target,
            train_ids=["a", "b", "c"],
            validation_ids=["a"],
            labels=("x", "y", "z"),
            candidates={"one": lambda: RecordingClassifier({})},
        )
