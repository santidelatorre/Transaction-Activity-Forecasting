"""Check the diagnostic accounting and oracle meaning independently of real data."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def analysis():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_v3_seven_way.py"
    spec = importlib.util.spec_from_file_location("seven_way_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    ids = ["a", "b", "c", "d"]
    truth = pd.Series(["gym", "none", "music", "cloud"], index=ids)
    first = pd.Series(["gym", "gym", "music", "none"], index=ids)
    second = pd.Series(["none", "none", "music", "software"], index=ids)
    return truth, first, second


def test_paired_accounting_includes_both_wrong_and_aligns_clients():
    module = analysis()
    truth, first, second = fixture()
    pair = module.pairwise(truth, first, second.iloc[::-1])
    assert (
        pair["both_correct"] == pair["left_only"] == pair["right_only"] == pair["both_wrong"] == 1
    )
    change = module.changes(truth, first, second)
    assert change["changed"] == 3
    assert change["corrections"] == change["regressions"] == change["wrong_to_wrong"] == 1
    assert sum(row["clients"] for row in change["transitions"]) == 3
    with pytest.raises(ValueError, match="mismatch"):
        module.pairwise(truth, first, second.iloc[:-1])


def test_oracle_cannot_recover_a_label_absent_from_every_model():
    module = analysis()
    truth, first, second = fixture()
    result = module.oracle(truth, {"first": first, "second": second})
    assert result["accuracy_ceiling"] == 0.75
    assert result["recoverable"] == 3 and result["all_fail"] == 1
    assert result["extra_over_first"] == 1
    assert result["all_fail_by_true_class"]["cloud"] == 1
    assert result["macro_f1_loose_upper_bound"] >= result["macro_f1_fixed_fallback"]


def test_bootstrap_equality_and_fast_scorer_match_official_fixed_eight_classes():
    module = analysis()
    truth, first, _ = fixture()
    result = module.bootstrap(truth, first, first.iloc[::-1], repeats=50)
    assert result["delta"] == 0 and result["interval_95"] == [0, 0]
    code = {label: i for i, label in enumerate(module.LABELS)}
    expected = module.evaluate_predictions(truth, first)["macro_f1"]
    actual = module.numeric_score(truth.map(code).to_numpy(), first.map(code).to_numpy())
    np.testing.assert_allclose(actual, expected, atol=1e-15)
