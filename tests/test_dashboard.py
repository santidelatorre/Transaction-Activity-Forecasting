"""Dashboard contract: show measured V3-A evidence without selecting a new model."""

import json

import pytest

from transaction_forecasting.api import dashboard, service


def make_report(root):
    directory = root / "outputs/metrics/ubs_v3"
    directory.mkdir(parents=True)
    matrix = [[0] * 8 for _ in range(8)]
    matrix[0][0], matrix[0][1] = 46, 54
    measured = {"macro_f1": 0.42, "accuracy": 0.46, "confusion_matrix": matrix}
    report = {
        "oof_selected_candidate": "full",
        "metrics": {
            "V2": {**measured, "macro_f1": 0.39},
            "A": measured,
            "full": {**measured, "macro_f1": 0.40},
            "ensemble": {**measured, "macro_f1": 0.44},
        },
    }
    (directory / "valid_results.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "importance_A.csv").write_text(
        "feature,importance\nidentity_music_share,2.5\nmcc_4814_share,4.2\n", encoding="utf-8"
    )
    return directory, report


def test_promoted_arm_not_highest_score_or_historical_oof_selection(tmp_path):
    directory, _ = make_report(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    result = dashboard.snapshot(tmp_path)
    assert result["model_version"] == "V3-A"
    assert result["base_sha"] == dashboard.BASE_SHA
    assert result["metrics"]["macro_f1"] == 0.42
    assert result["metrics"]["accuracy"] == 0.46
    assert result["metrics"]["macro_recall"] == pytest.approx(0.0575)
    assert result["metrics"]["validation_clients"] == 100
    assert result["metrics"]["delta_vs_baseline"] == pytest.approx(0.03)
    assert result["importance"][0] == {"feature": "mcc_4814_share", "importance": 4.2}
    selected = [row for row in result["experiments"] if row["result"] == "selected"]
    assert [row["candidate"] for row in selected] == ["A"]
    assert selected[0]["model_name"] == "V3-A · identidad por familia"
    assert selected[0]["milestone"] == "Features de identidad por familia"
    assert "ajustado en TRAIN" in result["importance_scope"]
    assert "evaluar VALID" in result["importance_scope"]
    assert all(row["recorded_at_utc"] is None for row in result["experiments"])
    assert all(row["order_kind"] == "report" for row in result["experiments"])
    assert result["validation_independent"] is False
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_missing_artifacts_do_not_fabricate_metrics(tmp_path):
    result = dashboard.snapshot(tmp_path)
    assert result["available"] is False
    assert result["metrics"] is None
    assert result["importance"] == result["experiments"] == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad_score", [None, True, "0.8", 1.1, float("nan")])
def test_invalid_scores_are_not_rendered_as_real(tmp_path, bad_score):
    directory, report = make_report(tmp_path)
    report["metrics"]["A"]["macro_f1"] = bad_score
    (directory / "valid_results.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        dashboard.snapshot(tmp_path)


def test_api_uses_explicit_artifact_root_and_paginates(tmp_path, monkeypatch):
    make_report(tmp_path)
    monkeypatch.setenv("RECURRING_ARTIFACT_ROOT", str(tmp_path))
    assert service.get_dashboard()["metrics"]["validation_clients"] == 100
    first = service.get_experiments(limit=2, offset=0)["experiments"]
    second = service.get_experiments(limit=2, offset=2)["experiments"]
    assert [row["candidate"] for row in first + second] == ["V2", "A", "full", "ensemble"]
    assert service.get_experiments(limit=2, offset=4)["experiments"] == []
    assert not (tmp_path / "outputs/experiments").exists()


def test_missing_importance_is_an_explicit_empty_state(tmp_path):
    directory, _ = make_report(tmp_path)
    (directory / "importance_A.csv").unlink()
    result = dashboard.snapshot(tmp_path)
    assert result["available"] is True
    assert result["importance"] == []


def test_invalid_artifacts_are_reported_as_unavailable(tmp_path, monkeypatch):
    directory, _ = make_report(tmp_path)
    (directory / "valid_results.json").write_text("invalid JSON", encoding="utf-8")
    monkeypatch.setenv("RECURRING_ARTIFACT_ROOT", str(tmp_path))
    with pytest.raises(service.ArtifactUnavailable):
        service.get_dashboard()
