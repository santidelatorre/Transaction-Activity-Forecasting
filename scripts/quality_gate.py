"""Local, non-merging integration gate. Training requires --evaluate.

The orchestration and its unit tests use the standard library. Evaluation reuses
the installed project dependencies. No checkout, network or package installation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none")
DATA_FILES = (
    "train_transactions.jsonl",
    "train_labels.csv",
    "valid_transactions.jsonl",
    "valid_labels.csv",
    "test_transactions.jsonl",
    "sample_submission.csv",
)
SEVERITY = {"PASS": 0, "WARN": 1, "FAIL": 2}


def check(name, status, reason):
    return {"name": name, "status": status, "reason": reason}


def overall(checks):
    return max((item["status"] for item in checks), key=SEVERITY.get, default="WARN")


def sensitive(path):
    return any(
        name in {".ssh", ".aws", ".gnupg", ".env", "id_rsa", "id_ed25519"}
        or name.startswith(".env.")
        or any(word in name for word in ("credential", "secret", "password", "token"))
        or name.endswith((".pem", ".key", ".p12"))
        for name in (part.lower() for part in Path(path).parts)
    )


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or sensitive(path.relative_to(ROOT)):
        raise ValueError("Path must be inside the repository and not a credential path")
    return path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def git_paths(*args):
    raw = subprocess.check_output(["git", *args], cwd=ROOT)
    return [p.decode() for p in raw.split(b"\0") if p]


def load_toml(path):
    try:
        import tomllib
    except ImportError as exc:
        raise RuntimeError("Python 3.11+ is required; no dependencies were installed") from exc
    with path.open("rb") as stream:
        return tomllib.load(stream)


def validate_thresholds(thresholds):
    for name, value in thresholds.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid nonnegative finite threshold: {name}")
    for warning, blocking in (
        ("macro_f1_warn_drop", "macro_f1_fail_drop"),
        ("file_warn_mib", "file_fail_mib"),
    ):
        if thresholds[blocking] < thresholds[warning]:
            raise ValueError("Blocking threshold must not be lower than warning threshold")


def hygiene(reference, thresholds):
    committed = git_paths("diff", "--name-only", "-z", f"{reference}...HEAD")
    staged = git_paths("diff", "--cached", "--name-only", "-z")
    unstaged = git_paths("diff", "--name-only", "-z")
    untracked = git_paths("ls-files", "--others", "--exclude-standard", "-z")
    changes = {
        "committed": committed,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
    }
    results = []
    sizes = []
    for name in sorted(set(committed + staged + unstaged + untracked)):
        path = ROOT / name
        tracked_change = name in committed or name in staged
        status = "FAIL" if tracked_change else "WARN"
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            results.append(check("Git hygiene", "WARN", f"Symlink not followed: {name}"))
            continue
        generated = (
            (name.startswith(("data/", "outputs/")) and not name.endswith(".gitkeep"))
            or any(
                p
                in {
                    "__pycache__",
                    ".ipynb_checkpoints",
                    ".pytest_cache",
                    ".ruff_cache",
                    "node_modules",
                    ".venv",
                }
                for p in Path(name).parts
            )
            or name.endswith(
                (
                    ".csv",
                    ".parquet",
                    ".jsonl",
                    ".pkl",
                    ".pickle",
                    ".joblib",
                    ".cbm",
                    ".onnx",
                    ".pt",
                    ".h5",
                    ".ckpt",
                    ".pyc",
                    ".tmp",
                    ".log",
                    ".zip",
                )
            )
        )
        # Deleted files are reported, but are not prohibited additions.
        blob_sizes = []
        for revision, names in (("HEAD", committed), ("", staged)):
            if name in names:
                proc = subprocess.run(
                    ["git", "cat-file", "-s", f"{revision}:{name}"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    blob_sizes.append(int(proc.stdout))
        if path.is_file():
            blob_sizes.append(path.stat().st_size)
        if not blob_sizes:
            continue
        if sensitive(name):
            results.append(check("Git hygiene", status, f"Sensitive filename: {name}; not opened"))
            continue
        if generated:
            results.append(check("Git hygiene", status, f"Generated/data/artifact path: {name}"))
        size = max(blob_sizes)
        sizes.append({"path": name, "bytes": size})
        mib = size / 1024**2
        if mib > thresholds["file_fail_mib"]:
            results.append(check("Large files", "FAIL", f"{name}: {mib:.2f} MiB"))
        elif mib > thresholds["file_warn_mib"]:
            results.append(check("Large files", "WARN", f"{name}: {mib:.2f} MiB"))
        if name.endswith(".ipynb"):
            results.append(check("Git hygiene", "WARN", f"Review notebook outputs: {name}"))
    if not results:
        results.append(check("Git hygiene", "PASS", "No prohibited paths or oversized changes"))
    results.append(
        check(
            "PR scope / secrets",
            "WARN",
            "Review listed changes for scope and embedded secrets; filename checks "
            "cannot prove absence of credentials. Credential files are not opened.",
        )
    )
    return results, changes, sizes


def test_results(path):
    root = ET.parse(path).getroot()
    failures, nodes = [], []
    for case in root.iter("testcase"):
        node = case.get("classname", "") + "::" + case.get("name", "")
        nodes.append(node)
        if case.find("failure") is not None or case.find("error") is not None:
            failures.append(node)
    return {
        "nodes": sorted(nodes),
        "failures": sorted(failures),
        "skipped": len(list(root.iter("skipped"))),
    }


def run_command(command, report):
    report["commands"].append(command)
    # Avoid echoing arbitrary subprocess output that could contain sensitive data.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode


def run_tests(report, run_dir, targeted):
    checks = report["checks"]
    if importlib.util.find_spec("pytest") is None:
        checks.append(check("Tests", "WARN", "pytest unavailable; tests not run"))
        return
    for name, paths in (("Targeted tests", targeted), ("Tests", ["tests"])):
        xml = run_dir / ("targeted.xml" if name == "Targeted tests" else "pytest.xml")
        code = run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *paths,
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(run_dir / "tmp"),
                f"--junitxml={xml}",
            ],
            report,
        )
        checks.append(check(name, "PASS" if code == 0 else "FAIL", f"pytest exit code {code}"))
        if xml.exists():
            try:
                report["tests"] = test_results(xml)
            except ET.ParseError:
                checks.append(check("Test evidence", "FAIL", "Malformed pytest XML report"))
                return
            report["tests"]["scope"] = "full" if name == "Tests" else "targeted"
            if report["tests"]["failures"]:
                checks.append(check("Test evidence", "FAIL", "JUnit records failing tests"))
            if not report["tests"]["nodes"]:
                checks.append(check("Test evidence", "WARN", "JUnit contains no test cases"))
            if report["tests"]["skipped"]:
                checks.append(check("Tests", "WARN", "Suite contains skipped tests"))
        else:
            report.pop("tests", None)
            checks.append(check("Test evidence", "WARN", "Requested pytest XML report missing"))
        if code != 0:
            checks.append(check("Test coverage", "WARN", "Stopped after test failure"))
            return
    if importlib.util.find_spec("ruff") is None:
        checks.append(check("Lint/format", "WARN", "Ruff unavailable; not installed automatically"))
        return
    for args in (["check", "--no-cache", "."], ["format", "--check", "--no-cache", "."]):
        code = run_command([sys.executable, "-m", "ruff", *args], report)
        checks.append(
            check(
                "Lint/format",
                "PASS" if code == 0 else "FAIL",
                f"ruff {' '.join(args)}: exit {code}",
            )
        )


def protocol(settings):
    return {key: value for key, value in settings.items() if key != "outputs"}


def provenance(settings):
    data_dir = local_path(settings["data"]["directory"])
    data_hashes = {name: digest(local_path(str(data_dir / name))) for name in DATA_FILES}
    source = {}
    for directory in ("src", "scripts", "configs", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".toml"} and not sensitive(path):
                safe = local_path(str(path))
                source[str(path.relative_to(ROOT))] = digest(safe)
    source["pyproject.toml"] = digest(ROOT / "pyproject.toml")
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "catboost"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "missing"
    return {
        "commit": git("rev-parse", "HEAD"),
        "data": data_hashes,
        "source": source,
        "protocol": protocol(settings),
        "environment": {"python": platform.python_version(), "packages": packages},
        "machine": {
            "system": platform.system(),
            "machine": platform.machine(),
            "node": platform.node(),
            "cpu_count": os.cpu_count(),
        },
    }


def compare_metrics(candidate, baseline, thresholds):
    results = []
    delta = candidate["macro_f1"] - baseline["macro_f1"]
    status = "PASS"
    if delta < -thresholds["macro_f1_fail_drop"] - 1e-12:
        status = "FAIL"
    elif delta < -thresholds["macro_f1_warn_drop"] - 1e-12:
        status = "WARN"
    results.append(
        check(
            "Macro-F1",
            status,
            f"candidate={candidate['macro_f1']:.6f}; "
            f"baseline={baseline['macro_f1']:.6f}; delta={delta:+.6f}",
        )
    )
    for label in LABELS:
        current = candidate["per_class"][label]
        previous = baseline["per_class"][label]
        drop = previous["f1"] - current["f1"]
        if drop > thresholds["per_class_f1_warn_drop"]:
            results.append(check("Per-class regression", "WARN", f"{label}: F1 drop {drop:.6f}"))
        change = abs(
            current["predicted_count"] / candidate["clients"]
            - previous["predicted_count"] / baseline["clients"]
        )
        if change > thresholds["prediction_share_warn_change"]:
            results.append(
                check("Prediction distribution", "WARN", f"{label}: share change {change:.2%}")
            )
    return results


def validate_metric_record(metrics):
    if metrics["clients"] <= 0:
        raise ValueError("Metrics must contain clients")
    for value in [
        metrics["macro_f1"],
        *[
            row[key]
            for row in metrics["per_class"].values()
            for key in ("precision", "recall", "f1")
        ],
    ]:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid metric value")
    if set(metrics["per_class"]) != set(LABELS):
        raise ValueError("Expected all eight classes")
    rows = list(metrics["per_class"].values())
    for key in ("support", "predicted_count"):
        if any(not isinstance(row[key], int) or row[key] < 0 for row in rows):
            raise ValueError("Invalid class count")
        if sum(row[key] for row in rows) != metrics["clients"]:
            raise ValueError("Class counts do not match client count")
    if abs(sum(row["f1"] for row in rows) / 8 - metrics["macro_f1"]) > 1e-10:
        raise ValueError("Macro-F1 does not match per-class F1")


def artifacts_intact(saved):
    hashes = saved.get("artifact_hashes", {})
    return bool(hashes) and all(
        local_path(name).is_file() and digest(local_path(name)) == expected
        for name, expected in hashes.items()
    )


def review_candidate(path, settings, report):
    saved = json.loads(path.read_text())
    if saved.get("provenance") != provenance(settings) or not artifacts_intact(saved):
        report["checks"].append(
            check(
                "Candidate artifacts",
                "WARN",
                "Saved candidate does not match current code/data/environment or artifact hashes",
            )
        )
        return
    validate_metric_record(saved["metrics"])
    artifacts = saved["artifact_hashes"]
    summary = next(local_path(name) for name in artifacts if name.endswith("/summary.json"))
    submission = next(local_path(name) for name in artifacts if name.endswith("/submission.csv"))
    evaluate_artifacts(settings, summary.parent, submission, report, write_predictions=False)
    for key in ("provenance", "artifact_hashes", "runtime_seconds"):
        report[key] = saved[key]
    report["checks"].append(
        check(
            "Candidate artifacts",
            "PASS",
            "Current inputs match saved run; metrics independently rescored",
        )
    )


def evaluate_artifacts(settings, metrics_dir, submission, report, write_predictions=True):
    import pandas as pd

    from transaction_forecasting.evaluation.official import PREDICTION, score_predictions
    from transaction_forecasting.ubs.data import load_ubs_data, validate_submission

    data = load_ubs_data(local_path(settings["data"]["directory"]))
    report["checks"].append(
        check(
            "Data integrity",
            "PASS",
            "Loader accepted cutoff, client separation and label alignment",
        )
    )
    errors = pd.read_csv(
        metrics_dir / "validation_error_analysis.csv", dtype=str, keep_default_na=False
    )
    predictions = errors[["client_id", "model_prediction"]].rename(
        columns={"model_prediction": PREDICTION}
    )
    metrics = score_predictions(data.valid_labels, predictions)
    validate_metric_record(metrics)
    summary = json.loads((metrics_dir / "summary.json").read_text())
    recorded = float(summary["best_macro_f1"])
    if not math.isfinite(recorded) or abs(recorded - metrics["macro_f1"]) > 1e-10:
        raise ValueError("Saved Macro-F1 disagrees with independently scored predictions")
    for label, values in metrics["per_class"].items():
        if values["support"] and not values["predicted_count"]:
            report["checks"].append(
                check("Prediction coverage", "WARN", f"No validation predictions for {label}")
            )
    frame = pd.read_csv(submission, dtype=str, keep_default_na=False)
    validate_submission(frame, data.sample_submission, data.test_transactions)
    report["checks"].append(
        check("Submission", "PASS", "Official schema, classes, IDs, count and order accepted")
    )
    # Publish only after all artifact checks succeed; invalid partial results must
    # not feed a baseline comparison or break strict JSON serialization.
    if write_predictions:
        predictions.to_csv(metrics_dir.parent / "valid_predictions.csv", index=False)
    report["metrics"] = metrics
    return metrics


def evaluate(config_path, settings, run_dir, report):
    checks = report["checks"]
    required = ("pandas", "numpy", "scipy", "sklearn", "catboost")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        checks.append(check("Evaluation", "WARN", "Missing dependencies: " + ", ".join(missing)))
        return
    try:
        before = provenance(settings)
    except FileNotFoundError:
        checks.append(check("Evaluation", "WARN", "Required UBS data files unavailable"))
        return
    metrics_dir = run_dir / "candidate"
    submission = run_dir / "submission.csv"
    original = config_path.read_text()
    replacement = (
        "[outputs]\nmetrics_directory = "
        + json.dumps(str(metrics_dir.relative_to(ROOT)))
        + "\nsubmission = "
        + json.dumps(str(submission.relative_to(ROOT)))
        + "\n"
    )
    generated, count = re.subn(
        r"(?ms)^\[outputs\]\s*\n.*?(?=^\[|\Z)", lambda match: replacement, original
    )
    if count != 1:
        raise ValueError("Expected exactly one outputs table in runner configuration")
    candidate_config = run_dir / "candidate.toml"
    candidate_config.write_text(generated)
    if protocol(load_toml(candidate_config)) != protocol(settings):
        raise ValueError("Generated configuration changed evaluation protocol")
    started = time.perf_counter()
    code = run_command(
        [sys.executable, "scripts/run_ubs_baseline.py", "--config", str(candidate_config)], report
    )
    report["runtime_seconds"] = time.perf_counter() - started
    if code:
        checks.append(check("Pipeline", "FAIL", f"Runner exited {code}; no settings changed"))
        return
    if before != provenance(settings):
        checks.append(check("Provenance", "FAIL", "Inputs changed during evaluation"))
        return
    evaluate_artifacts(settings, metrics_dir, submission, report)
    report["provenance"] = before
    report["artifact_hashes"] = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (
            metrics_dir / "summary.json",
            metrics_dir / "validation_error_analysis.csv",
            submission,
            candidate_config,
        )
    }
    checks.append(check("Reproduction", "PASS", "Fresh outputs rescored; provenance recorded"))


def compare_baseline(report, baseline, thresholds, approximate):
    checks = report["checks"]
    if "metrics" not in report:
        checks.append(check("Performance", "WARN", "No verified candidate evaluation available"))
        return
    candidate = report["metrics"]
    validate_metric_record(candidate)
    if not baseline or "metrics" not in baseline or "provenance" not in baseline:
        checks.append(
            check(
                "Macro-F1",
                "WARN",
                f"candidate={candidate['macro_f1']:.6f}; "
                f"delta vs approximate {approximate:.3f}="
                f"{candidate['macro_f1'] - approximate:+.6f}; baseline unverified",
            )
        )
        return
    validate_metric_record(baseline["metrics"])
    if not artifacts_intact(baseline):
        checks.append(
            check(
                "Baseline",
                "WARN",
                "Baseline artifacts missing or changed; blocking score thresholds not applied",
            )
        )
        return
    old, new = baseline["provenance"], report.get("provenance", {})
    compatible = all(
        old.get(key) == new.get(key) and key in old for key in ("data", "protocol", "environment")
    )
    if not compatible:
        checks.append(
            check(
                "Performance",
                "WARN",
                "Baseline data/protocol/environment mismatch; "
                "blocking score thresholds not applied",
            )
        )
        return
    checks.extend(compare_metrics(candidate, baseline["metrics"], thresholds))
    elapsed = baseline.get("runtime_seconds", 0)
    if elapsed and old.get("machine") == new.get("machine"):
        ratio = report["runtime_seconds"] / elapsed
        status = "WARN" if ratio > thresholds["runtime_warn_ratio"] else "PASS"
        checks.append(check("Runtime", status, f"{ratio:.2f}x baseline; includes final refit"))
    else:
        checks.append(check("Runtime", "WARN", "Comparable baseline timing unavailable"))


def compare_tests(report, baseline):
    if "tests" not in report or not baseline or "tests" not in baseline:
        report["checks"].append(
            check(
                "Test baseline", "WARN", "Cannot distinguish new failures without both test records"
            )
        )
        return
    current, previous = report["tests"], baseline["tests"]
    new = sorted(set(current["failures"]) - set(previous["failures"]))
    removed = (
        sorted(set(previous["nodes"]) - set(current["nodes"]))
        if current.get("scope") == "full"
        else []
    )
    report["test_changes"] = {"new_failures": new, "removed_nodes": removed}
    if new:
        report["checks"].append(check("New test failures", "FAIL", f"{len(new)} new failures"))
    if removed:
        report["checks"].append(check("Test coverage", "WARN", f"{len(removed)} tests absent"))


def write_report(report, run_dir):
    report["overall"] = overall(report["checks"])
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [f"QUALITY GATE: {report['overall']}", ""]
    lines.extend(f"[{c['status']}] {c['name']}: {c['reason']}" for c in report["checks"])
    lines += ["", f"OVERALL: {report['overall']}", "Reasons:"]
    reasons = [c for c in report["checks"] if c["status"] != "PASS"]
    lines.extend(f"- {c['name']}: {c['reason']}" for c in reasons)
    if not reasons:
        lines.append("- No blocking issue detected.")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {run_dir.relative_to(ROOT) / 'report.json'}\n")
    print("\n".join(lines))
    return SEVERITY[report["overall"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quality_gate.toml")
    parser.add_argument("--evaluate", action="store_true", help="Explicitly run the full V1 runner")
    parser.add_argument("--candidate", help="Recheck a prior gate report without training")
    parser.add_argument("--baseline", help="Reviewed V1 gate report inside this repository")
    args = parser.parse_args(argv)
    if args.evaluate and args.candidate:
        parser.error("--evaluate and --candidate are mutually exclusive")
    gate = load_toml(local_path(args.config))
    settings = gate["gate"]
    branch = git("branch", "--show-current")
    if branch == "main" or not branch:
        print("[FAIL] Branch: refusing main or detached HEAD.\nOVERALL: FAIL")
        return 2
    validate_thresholds(gate["thresholds"])
    output = local_path(settings["output_directory"])
    if not output.is_relative_to(ROOT / "outputs" / "metrics"):
        raise ValueError("Gate outputs must be under outputs/metrics")
    run_dir = output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "branch": branch,
        "commit": git("rev-parse", "HEAD"),
        "commands": [],
        "checks": [],
        "thresholds": gate["thresholds"],
    }
    try:
        reference = settings["reference"]
        report["reference_commit"] = git("rev-parse", reference)
        checks, changes, sizes = hygiene(reference, gate["thresholds"])
        report["checks"].extend(checks)
        report["changes"], report["file_sizes"] = changes, sizes
        report["checks"].append(
            check(
                "Reference",
                "WARN",
                "Using local remote-tracking reference; no network fetch performed",
            )
        )
        baseline_path = args.baseline or settings["baseline_report"]
        baseline = json.loads(local_path(baseline_path).read_text()) if baseline_path else None
        run_tests(report, run_dir, settings["targeted_tests"])
        compare_tests(report, baseline)
        report["checks"].append(
            check(
                "Leakage review",
                "WARN",
                "V1 has explicit cutoff/client separation and train-only preprocessing "
                "during scoring. Training description mappings include each training "
                "client's label; validation tunes "
                "selection. Final train+valid refit must never be scored as held-out validation. "
                "Changed feature/model code requires human review; no blanket safety claim.",
            )
        )
        report["checks"].append(check("Memory", "WARN", "Peak memory not measured"))
        config_path = local_path(settings["pipeline_config"])
        if args.evaluate and not any(c["status"] == "FAIL" for c in report["checks"]):
            evaluate(config_path, load_toml(config_path), run_dir, report)
        elif args.candidate:
            try:
                review_candidate(local_path(args.candidate), load_toml(config_path), report)
            except (FileNotFoundError, ModuleNotFoundError):
                report["checks"].append(
                    check(
                        "Candidate artifacts",
                        "WARN",
                        "Data, artifacts or evaluation dependencies missing",
                    )
                )
        else:
            report["checks"].append(
                check(
                    "Evaluation",
                    "WARN",
                    "Training not requested or a blocking check failed; use --evaluate explicitly",
                )
            )
        compare_baseline(
            report, baseline, gate["thresholds"], settings["baseline_macro_f1_approximate"]
        )
        reported = {item["name"] for item in report["checks"]}
        for name in ("Submission", "Data integrity", "Runtime"):
            if name not in reported:
                report["checks"].append(check(name, "WARN", "Required evidence unavailable"))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        StopIteration,
        subprocess.CalledProcessError,
    ) as exc:
        report["checks"].append(
            check(
                "Gate execution", "FAIL", f"{type(exc).__name__}: invalid input or execution failed"
            )
        )
    return write_report(report, run_dir)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    try:
        exit_code = main()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"[FAIL] Configuration: {type(error).__name__}; check local paths and settings.")
        print("OVERALL: FAIL")
        exit_code = 2
    raise SystemExit(exit_code)
