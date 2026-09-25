import importlib.util
from pathlib import Path

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_rejects_modified_or_missing_completed_artifacts(tmp_path):
    runner = load_runner()
    p = tmp_path / "model.bin"
    p.write_bytes(b"verified model")
    state = {"completed": {"model": {"files": {"model.bin": runner.checksum(p)}}}}
    assert runner.valid_completed_stage(state, "model", tmp_path)
    assert not runner.valid_completed_stage(state, "evaluation", tmp_path)
    p.write_bytes(b"modified")
    with pytest.raises(ValueError):
        runner.valid_completed_stage(state, "model", tmp_path)
    p.unlink()
    with pytest.raises(ValueError):
        runner.valid_completed_stage(state, "model", tmp_path)


@pytest.mark.parametrize("interrupt_evaluation", [False, True])
def test_pipeline_executes_missing_stages_and_resumes_verified_outputs(
    tmp_path, monkeypatch, interrupt_evaluation
):
    runner = load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "src/ubs_recurrence"
    source.mkdir(parents=True)
    (source / "model.py").write_text("# frozen prediction source")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "data_manifest.json").write_text("{}")
    monkeypatch.setattr(
        runner.sys,
        "argv",
        ["run_pipeline.py", "--run-name", "audit_run", "--device", "cpu"],
    )
    calls = []
    attempts = []

    def execute(command, **kwargs):
        if command[1] == "scripts/prepare_data.py":
            calls.append("verify_inputs")
            return
        action = command[5]
        calls.append(action)
        folder = Path(command[command.index("--output") + 1])
        folder.mkdir(parents=True)
        files = {
            "train": ["model.joblib", "metadata.json"],
            "evaluate": ["metrics.json", "probabilities.csv"],
            "submit": [
                "submission.csv",
                "submission_validation.json",
                "probabilities.csv",
            ],
        }[action]
        for filename in files:
            (folder / filename).write_text(action + " completed")
        if action == "evaluate":
            attempts.append(folder.name)
            if interrupt_evaluation and len(attempts) == 1:
                raise RuntimeError("Interrupted evaluation after outputs written")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    if interrupt_evaluation:
        with pytest.raises(RuntimeError, match="Interrupted"):
            runner.main()
        assert calls == ["verify_inputs", "train", "evaluate"]
        calls.clear()
        runner.main()
        assert calls == ["verify_inputs", "evaluate", "submit"]
        assert len(set(attempts)) == 2
        assert list((tmp_path / "outputs/audit_run").glob("evaluation_incomplete_*"))
    else:
        runner.main()
        assert calls == ["verify_inputs", "train", "evaluate", "submit"]
    calls.clear()
    runner.main()
    assert calls == ["verify_inputs"]
    (tmp_path / "outputs/audit_run/model/model.joblib").write_text("changed model")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.main()
