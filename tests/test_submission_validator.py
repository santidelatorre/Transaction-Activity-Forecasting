from pathlib import Path

import pandas as pd

from transaction_forecasting.evaluation.official import LABELS, PREDICTION


def _load_validator():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "validate_submission.py"
    spec = importlib.util.spec_from_file_location("submission_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_inspection_accepts_exact_contract_and_rejects_bad_rows(tmp_path):
    validator = _load_validator()
    sample = pd.DataFrame({"client_id": ["C1", "C2"], PREDICTION: ["", ""]})
    submission = pd.DataFrame({"client_id": ["C1", "C2"], PREDICTION: [LABELS[0], LABELS[-1]]})
    sample_path, submission_path = tmp_path / "sample.csv", tmp_path / "submission.csv"
    sample.to_csv(sample_path, index=False)
    submission.to_csv(submission_path, index=False)

    checks, valid = validator.inspect_submission(submission_path, sample_path)
    assert valid
    assert checks["client_alignment"]
    assert checks["distribution"][LABELS[0]] == 1

    submission.loc[1] = ["C1", "invalid"]
    submission.to_csv(submission_path, index=False)
    checks, valid = validator.inspect_submission(submission_path, sample_path)
    assert not valid
    assert checks["duplicate_ids"] == 1
    assert checks["invalid_labels"] == 1
