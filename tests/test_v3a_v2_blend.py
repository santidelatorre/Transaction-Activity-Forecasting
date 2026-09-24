import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from transaction_forecasting.evaluation.probability_blend import (
    align_probabilities,
    blend_probabilities,
    select_alpha,
)
from transaction_forecasting.ubs.data import LABELS, TARGET_COLUMN
from transaction_forecasting.ubs.evaluation import evaluate_predictions


def probabilities(clients, offset=0):
    values = np.full((len(clients), 8), 0.05)
    values[np.arange(len(clients)), (np.arange(len(clients)) + offset) % 8] = 0.65
    return pd.DataFrame(values, index=pd.Index(clients, name="client_id"), columns=LABELS)


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/run_v3a_v2_blend.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("blend_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_math_alignment_and_exact_endpoints():
    v2 = probabilities(["002", "001", "003"])
    identity = probabilities(v2.index, 1)
    shuffled = identity.iloc[::-1, ::-1]
    pd.testing.assert_frame_equal(blend_probabilities(v2, shuffled, 0), v2)
    pd.testing.assert_frame_equal(blend_probabilities(v2, shuffled, 1), identity)
    pd.testing.assert_frame_equal(blend_probabilities(v2, shuffled, 0.3), 0.7 * v2 + 0.3 * identity)


@pytest.mark.parametrize("alpha", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_invalid_alpha(alpha):
    values = probabilities(["a", "b"])
    with pytest.raises(ValueError, match="alpha"):
        blend_probabilities(values, values, alpha)


@pytest.mark.parametrize(
    "corruption", ["missing_id", "duplicate_id", "null_id", "class", "nan", "negative", "sum"]
)
def test_rejects_misalignment_and_invalid_probabilities(corruption):
    original = probabilities(["a", "b"])
    bad = original.copy()
    if corruption == "missing_id":
        bad = bad.iloc[:1]
    elif corruption == "duplicate_id":
        bad.index = ["a", "a"]
    elif corruption == "null_id":
        bad.index = ["a", None]
    elif corruption == "class":
        bad = bad.rename(columns={"music": "unknown"})
    elif corruption == "nan":
        bad.iloc[0, 0] = np.nan
    elif corruption == "negative":
        bad.iloc[0, 0] = -0.1
    else:
        bad.iloc[0, 0] = 0.9
    with pytest.raises(ValueError):
        align_probabilities(bad, original.index)


def test_predeclared_tie_break_and_genuine_maximum():
    assert select_alpha({0.0: 0.4, 0.5: 0.40005, 1.0: 0.3}) == 0.0
    assert select_alpha({0.3: 0.5, 0.5: 0.49995, 0.7: 0.5}) == 0.5
    assert select_alpha({0.5: 0.4, 0.9: 0.401}) == 0.9


@pytest.fixture
def experiment(tmp_path, runner):
    data, source, out = (tmp_path / name for name in ("data", "source", "result"))
    data.mkdir()
    source.mkdir()
    clients = [f"T{i:03d}" for i in range(80)]
    labels = pd.DataFrame(
        {"client_id": clients, "cutoff_date": "2026-01-01", TARGET_COLUMN: list(LABELS) * 10}
    )
    labels.to_csv(data / "train_labels.csv", index=False)
    target = labels.set_index("client_id")[TARGET_COLUMN]
    v2, identity = probabilities(clients, 1), probabilities(clients)
    frame = pd.DataFrame({"V2": v2.idxmax(axis=1), "A": identity.idxmax(axis=1)})
    frame.to_csv(source / "oof_predictions.csv")
    for fold, (_, hold) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(clients, target), 1
    ):
        for name, values in (("V2", v2), ("A", identity)):
            values.iloc[hold].to_csv(source / f"fold_{fold}_{name}_probabilities.csv")
        frame.iloc[hold].to_csv(source / f"fold_{fold}_predictions.csv")
    runner.write_json(
        source / "oof_results.json",
        {"metrics": {name: evaluate_predictions(target, frame[name]) for name in frame}},
    )
    for split, ids in (
        ("train", clients),
        ("valid", [f"V{i}" for i in range(8)]),
        ("test", [f"X{i}" for i in range(8)]),
    ):
        tx = pd.DataFrame(
            {
                "client_id": ids,
                "timestamp": "2025-12-01",
                "amount": 10,
                "currency": "chf",
                "description": "merchant",
                "direction": "out",
                "fee": 0,
                "mcc": "1234",
                "type": "card_payment",
            }
        )
        tx.to_json(data / f"{split}_transactions.jsonl", orient="records", lines=True)
        if split == "train":
            continue
        left, right = probabilities(ids, 1), probabilities(ids)
        components = {"V2": left, "A": right, "full": right, "ensemble": (left + right) / 2}
        predictions = pd.DataFrame(
            {name: values.idxmax(axis=1) for name, values in components.items()}
        )
        for name, values in components.items():
            values.to_csv(source / f"{split}_{name}_probabilities.csv")
        if split == "valid":
            predictions.to_csv(source / "validation_predictions.csv")
            pd.DataFrame(
                {"client_id": ids, "cutoff_date": "2026-01-01", TARGET_COLUMN: LABELS}
            ).to_csv(data / "valid_labels.csv", index=False)
        else:
            pd.DataFrame(
                {"client_id": ids[::-1], "predicted_next_recurring_merchant": "none"}
            ).to_csv(data / "sample_submission.csv", index=False)
    paths = [
        runner.ROOT / "scripts/run_ubs_v3.py",
        *(runner.ROOT / "src/transaction_forecasting/ubs").rglob("*.py"),
    ]
    fingerprints = {path.relative_to(runner.ROOT).as_posix(): runner.digest(path) for path in paths}
    fingerprints.update({f"data/raw/{path.name}": runner.digest(path) for path in data.iterdir()})
    for phase in ("oof", "valid", "submission"):
        runner.write_json(
            source / f"{phase}_provenance.json",
            {
                "seed": 42,
                "python": platform.python_version(),
                "versions": {},
                "fingerprints": fingerprints,
            },
        )
    return data, source, out


def test_oof_never_reads_valid_and_freezes_before_evaluation(experiment, runner, monkeypatch):
    data, source, out = experiment
    original = runner.read_labels
    reads = []

    def guarded(path):
        reads.append(path.name)
        if path.name == "valid_labels.csv":
            assert runner.load_frozen(out)["valid_labels_used_for_selection"] is False
            assert (out / "valid_predictions.csv").is_file()
        return original(path)

    monkeypatch.setattr(runner, "read_labels", guarded)
    # Hashing VALID is also prohibited during selection.
    original_digest = runner.digest

    def guarded_digest(path):
        if Path(path).name in ("valid_labels.csv", "valid_transactions.jsonl"):
            assert (out / "frozen_selection.sha256").exists()
        return original_digest(path)

    monkeypatch.setattr(runner, "digest", guarded_digest)
    runner.run_oof(data, source, out)
    assert reads == ["train_labels.csv"]
    frozen_bytes = (out / "frozen_selection.json").read_bytes()
    assert runner.load_frozen(out)["alpha"] == 1.0
    runner.run_valid(data, source, out)
    assert reads == ["train_labels.csv", "valid_labels.csv"]
    assert (out / "frozen_selection.json").read_bytes() == frozen_bytes
    with pytest.raises(FileExistsError):
        runner.run_valid(data, source, out)
    with pytest.raises(FileExistsError):
        runner.run_oof(data, source, out)
    assert (out / "frozen_selection.json").read_bytes() == frozen_bytes


def test_valid_requires_untampered_freeze(experiment, runner, monkeypatch):
    data, source, out = experiment
    with pytest.raises(FileNotFoundError):
        runner.run_valid(data, source, out)
    runner.run_oof(data, source, out)
    frozen = runner.read_json(out / "frozen_selection.json")
    frozen["alpha"] = 0.0
    (out / "frozen_selection.json").write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(
        runner, "read_labels", lambda _: pytest.fail("Labels read before freeze validation")
    )
    with pytest.raises(ValueError, match="modified"):
        runner.run_valid(data, source, out)


def test_cached_probability_tampering_is_rejected(experiment, runner):
    data, source, out = experiment
    runner.run_oof(data, source, out)
    with (source / "fold_1_A_probabilities.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Fingerprint"):
        runner.run_valid(data, source, out)


def test_source_fingerprints_and_protected_outputs(experiment, runner):
    data, source, out = experiment
    (data / "train_labels.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Fingerprint"):
        runner.run_oof(data, source, out)
    for path in (source, source / "child", source.parent, runner.ROOT / "outputs/metrics/ubs_v2"):
        with pytest.raises(ValueError, match="baseline"):
            runner.check_output_directory(path, source)


def test_submission_uses_frozen_alpha_and_sample_order(experiment, runner, monkeypatch):
    data, source, out = experiment
    runner.run_oof(data, source, out)
    runner.run_valid(data, source, out)

    class FakeModel:
        def fit(self, tx, labels):
            assert len(labels) == 88
            assert set(labels.client_id) == set(tx.client_id)
            assert set(labels.client_id.str[0]) == {"T", "V"}
            return self

        def predict_components(self, tx):
            clients = pd.Index(sorted(tx.client_id.unique()), name="client_id")
            return {
                name: runner.load_probabilities(source / f"test_{name}_probabilities.csv", clients)
                for name in ("V2", "A")
            }

    monkeypatch.setattr(runner, "V3Model", FakeModel)
    runner.run_submission(data, source, out)
    checks, passed = runner.inspect_submission(
        out / "submission_v3a_v2_blend.csv", data / "sample_submission.csv"
    )
    assert passed and checks["rows"] == 8 and checks["client_alignment"]
    assert runner.read_json(out / "submission_provenance.json")["alpha"] == 1.0
    with pytest.raises(FileExistsError):
        runner.run_submission(data, source, out)
