"""Dependency-free gate tests, also collected by the existing pytest suite."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "quality_gate.py"
SPEC = importlib.util.spec_from_file_location("quality_gate", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

THRESHOLDS = {
    "macro_f1_warn_drop": 0.005,
    "macro_f1_fail_drop": 0.010,
    "per_class_f1_warn_drop": 0.05,
    "prediction_share_warn_change": 0.10,
    "runtime_warn_ratio": 1.5,
    "file_warn_mib": 5,
    "file_fail_mib": 25,
}


def metrics(score=0.3):
    return {
        "clients": 80,
        "macro_f1": score,
        "per_class": {
            label: {
                "precision": score,
                "recall": score,
                "f1": score,
                "support": 10,
                "predicted_count": 10,
            }
            for label in gate.LABELS
        },
    }


class QualityGateTests(unittest.TestCase):
    def test_worst_result_wins_and_empty_is_not_pass(self):
        self.assertEqual(gate.overall([]), "WARN")
        checks = [gate.check("test", state, "reason") for state in ("PASS", "WARN", "FAIL")]
        self.assertEqual(gate.overall(checks), "FAIL")

    def test_absolute_threshold_boundaries(self):
        for candidate, status in (
            (0.300, "PASS"),
            (0.295, "PASS"),
            (0.294, "WARN"),
            (0.290, "WARN"),
            (0.289, "FAIL"),
        ):
            with self.subTest(candidate=candidate):
                result = gate.compare_metrics(metrics(candidate), metrics(), THRESHOLDS)
                self.assertEqual(result[0]["status"], status)

    def test_invalid_threshold_configuration_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            thresholds = {**THRESHOLDS, "macro_f1_fail_drop": value}
            with self.assertRaises(ValueError):
                gate.validate_thresholds(thresholds)
        with self.assertRaises(ValueError):
            gate.validate_thresholds({**THRESHOLDS, "file_fail_mib": 1})

    def test_per_class_and_distribution_warnings(self):
        candidate = metrics()
        candidate["per_class"]["cloud"].update(f1=0.1, predicted_count=0)
        result = gate.compare_metrics(candidate, metrics(), THRESHOLDS)
        self.assertIn("Per-class regression", [r["name"] for r in result])
        self.assertIn("Prediction distribution", [r["name"] for r in result])

    def test_approximate_baseline_cannot_trigger_blocking_metric_rule(self):
        report = {"checks": [], "metrics": metrics(0.1)}
        gate.compare_baseline(report, None, THRESHOLDS, 0.271)
        self.assertEqual(gate.overall(report["checks"]), "WARN")

    def test_mismatched_protocol_disables_blocking_comparison(self):
        manifest = {"data": {"train": "hash"}, "protocol": {"seed": 42}, "environment": {}}
        baseline = {"metrics": metrics(), "provenance": manifest}
        report = {"checks": [], "metrics": metrics(0.1), "provenance": copy.deepcopy(manifest)}
        report["provenance"]["protocol"]["seed"] = 43
        with patch.object(gate, "artifacts_intact", return_value=True):
            gate.compare_baseline(report, baseline, THRESHOLDS, 0.271)
        self.assertEqual(gate.overall(report["checks"]), "WARN")

    def test_invalid_metrics_rejected(self):
        for value in (float("nan"), float("inf"), -1, 2):
            candidate = metrics(value)
            with self.assertRaises(ValueError):
                gate.validate_metric_record(candidate)
        candidate = metrics()
        candidate["per_class"].pop("none")
        with self.assertRaises(ValueError):
            gate.validate_metric_record(candidate)

    def test_credentials_and_outside_paths_rejected_without_reading(self):
        for name in (
            ".env",
            "nested/.env.local",
            "id_rsa",
            "credentials.json",
            "a.pem",
            "credentials/config.toml",
            ".env/settings.py",
        ):
            self.assertTrue(gate.sensitive(name))
            with self.assertRaises(ValueError):
                gate.local_path(name)
        with self.assertRaises(ValueError):
            gate.local_path("../outside.py")

    def test_hygiene_rejects_tracked_artifacts_using_metadata_only(self):
        paths = [".env", "data/raw/fixture.csv", "large_fixture.bin"]
        metadata = subprocess.CompletedProcess([], 0, stdout=str(26 * 1024**2))
        with (
            patch.object(gate, "git_paths", side_effect=[paths, [], [], []]),
            patch.object(gate.subprocess, "run", return_value=metadata),
            patch.object(gate, "digest") as read_contents,
        ):
            checks, _, sizes = gate.hygiene("origin/main", THRESHOLDS)
            read_contents.assert_not_called()
        self.assertEqual(gate.overall(checks), "FAIL")
        self.assertTrue(any(c["name"] == "Large files" for c in checks))
        self.assertNotIn(".env", [item["path"] for item in sizes])

    def test_deleted_sensitive_filename_is_not_a_new_secret(self):
        metadata = subprocess.CompletedProcess([], 1, stdout="")
        with (
            patch.object(gate, "git_paths", side_effect=[["removed_secret.key"], [], [], []]),
            patch.object(gate.subprocess, "run", return_value=metadata),
        ):
            checks, _, _ = gate.hygiene("origin/main", THRESHOLDS)
        self.assertNotEqual(gate.overall(checks), "FAIL")

    def test_artifact_tampering_detected(self):
        # Keep test fixtures inside the repository, including unittest-only runs.
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent.parent) as directory:
            path = Path(directory) / "summary.json"
            path.write_text("{}")
            saved = {"artifact_hashes": {str(path): gate.digest(path)}}
            self.assertTrue(gate.artifacts_intact(saved))
            path.write_text('{"changed": true}')
            self.assertFalse(gate.artifacts_intact(saved))

    def test_junit_records_failures_without_copying_failure_text(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent.parent) as directory:
            path = Path(directory) / "tests.xml"
            path.write_text(
                '<testsuites><testsuite><testcase classname="test_a" name="ok"/>'
                '<testcase classname="test_a" name="bad"><failure>private output'
                "</failure></testcase></testsuite></testsuites>"
            )
            result = gate.test_results(path)
            self.assertEqual(result["failures"], ["test_a::bad"])
            self.assertNotIn("private output", str(result))

    def test_missing_junit_cannot_silently_pass(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent.parent) as directory:
            report = {"checks": [], "commands": []}
            with (
                patch.object(gate.importlib.util, "find_spec", return_value=object()),
                patch.object(gate, "run_command", return_value=0),
            ):
                gate.run_tests(report, Path(directory), ["tests/test_quality_gate.py"])
            self.assertEqual(gate.overall(report["checks"]), "WARN")
            self.assertNotIn("tests", report)

    def test_malformed_junit_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent.parent) as directory:
            (Path(directory) / "targeted.xml").write_text("<unfinished")
            report = {"checks": [], "commands": []}
            with (
                patch.object(gate.importlib.util, "find_spec", return_value=object()),
                patch.object(gate, "run_command", return_value=0),
            ):
                gate.run_tests(report, Path(directory), ["tests/test_quality_gate.py"])
            self.assertEqual(gate.overall(report["checks"]), "FAIL")

    def test_junit_failure_overrides_successful_process_exit(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent.parent) as directory:
            (Path(directory) / "targeted.xml").write_text(
                '<testsuite><testcase name="bad"><failure/></testcase></testsuite>'
            )
            report = {"checks": [], "commands": []}
            with (
                patch.object(gate.importlib.util, "find_spec", return_value=object()),
                patch.object(gate, "run_command", return_value=0),
            ):
                gate.run_tests(report, Path(directory), ["tests/test_quality_gate.py"])
            self.assertEqual(gate.overall(report["checks"]), "FAIL")

    def test_identifies_new_failures_and_removed_tests(self):
        baseline = {"tests": {"nodes": ["a", "b", "c"], "failures": ["a"]}}
        report = {
            "checks": [],
            "tests": {"nodes": ["a", "b"], "failures": ["a", "b"], "scope": "full"},
        }
        gate.compare_tests(report, baseline)
        self.assertEqual(report["test_changes"], {"new_failures": ["b"], "removed_nodes": ["c"]})
        self.assertEqual(gate.overall(report["checks"]), "FAIL")

    def test_targeted_run_does_not_claim_full_suite_tests_removed(self):
        baseline = {"tests": {"nodes": ["a", "b"], "failures": []}}
        report = {"checks": [], "tests": {"nodes": ["a"], "failures": [], "scope": "targeted"}}
        gate.compare_tests(report, baseline)
        self.assertEqual(report["test_changes"]["removed_nodes"], [])

    def test_main_refuses_main_without_writing(self):
        with (
            patch.object(gate.sys, "version_info", (3, 11)),
            patch.object(gate, "load_toml", return_value={"gate": {}}),
            patch.object(gate, "git", return_value="main"),
            patch.object(gate, "run_tests") as tests,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(gate.main([]), 2)
            tests.assert_not_called()

    def test_default_end_to_end_does_not_train_and_writes_warn_report(self):
        # Exercise real Git inventory and report writing; only missing runtime/tooling
        # are simulated. No pipeline execution or dependency installation.
        output_root = SCRIPT.parent.parent / "outputs" / "metrics"

        with tempfile.TemporaryDirectory(dir=output_root) as directory:
            config = {
                "gate": {
                    "reference": "origin/main",
                    "output_directory": directory,
                    "baseline_report": "",
                    "baseline_macro_f1_approximate": 0.271,
                    "pipeline_config": "configs/ubs_v1.toml",
                    "targeted_tests": [],
                },
                "thresholds": THRESHOLDS,
            }

            real_git = gate.git

            def git_for_test(*args):
                if args == ("branch", "--show-current"):
                    return "feature/test-quality-gate"
                return real_git(*args)

            with (
                patch.object(gate.sys, "version_info", (3, 11)),
                patch.object(gate, "load_toml", return_value=config),
                patch.object(gate, "git", side_effect=git_for_test),
                patch.object(gate, "run_tests"),
                patch.object(gate, "evaluate") as train,
                redirect_stdout(io.StringIO()),
            ):
                code = gate.main([])
                train.assert_not_called()

            self.assertEqual(code, 1)
            report = json.loads(next(Path(directory).glob("*/report.json")).read_text())
            self.assertEqual(report["overall"], "WARN")
            self.assertEqual(report["commands"], [])
            self.assertNotIn("metrics", report)


if __name__ == "__main__":
    unittest.main()
