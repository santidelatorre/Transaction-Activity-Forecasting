"""Inventory pinned V4 branches and optionally export isolated source snapshots.

No checkout, merge, fetch, installation or Git metadata writes. Snapshots and
test outputs belong under ignored outputs/metrics, never in the source tree.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from time import perf_counter

BASE_SHA = "051ce64a7cf0ab999f8aacb81fa405d5fa0257cf"
BRANCHES = {
    "ginestar": "exp/v4-shift-corruption-ginestar",
    "javi": "exp/v4-direct-robust-javi",
    "christian": "exp/v4-merchant-intelligence-christian",
    "santiago": "exp/v4-soft-candidate-santiago",
    "laura": "exp/v4-survival-laura",
    "esteban": "exp/v4-self-supervised-esteban",
    "jaime": "product/v4-agent-demo-jaime",
}
TESTS = {
    "ginestar": ["tests/test_corruption.py"],
    "javi": ["tests/test_ubs_direct_robust.py"],
    "christian": ["tests/test_merchant_intelligence.py"],
    "santiago": ["tests/test_soft_candidate_ranker.py"],
    "laura": ["tests/test_ubs_survival.py"],
    "esteban": ["tests/test_ubs_representation.py"],
    "jaime": ["tests/test_product_agent.py", "tests/test_product_integration.py"],
}
ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/metrics/v4_synthesis")
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory, tests = {}, {}
    for name, branch in BRANCHES.items():
        ref = f"origin/{branch}"
        try:
            head = git("rev-parse", "--verify", ref).decode().strip()
        except subprocess.CalledProcessError:
            inventory[name] = {"branch": branch, "head": None, "status": "MISSING/INCOMPLETE"}
            continue
        changed = git("diff", "--name-only", f"{BASE_SHA}...{head}").decode().splitlines()
        commits = git("log", "--format=%H %s", f"{BASE_SHA}..{head}").decode().splitlines()
        inventory[name] = {
            "branch": branch,
            "head": head,
            "merge_base": git("merge-base", BASE_SHA, head).decode().strip(),
            "commits": commits,
            "changed_files": changed,
            "stat": git("diff", "--stat", f"{BASE_SHA}...{head}").decode(),
            "status": "AVAILABLE" if commits else "MISSING/INCOMPLETE",
            "remote_freshness_verified": False,
        }
        snapshot = out / "snapshots" / name
        if args.snapshots:
            with zipfile.ZipFile(io.BytesIO(git("archive", "--format=zip", head))) as archive:
                for member in archive.infolist():
                    destination = (snapshot / member.filename).resolve()
                    if not destination.is_relative_to(snapshot.resolve()):
                        raise ValueError("Unsafe archive path")
                    if not member.is_dir():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(archive.read(member))
        if args.tests:
            temporary = out / "branch_test_tmp"
            temporary.mkdir(exist_ok=True)
            env = os.environ.copy()
            env.update(PYTHONPATH=str(snapshot / "src"), OPENBLAS_NUM_THREADS="1")
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *TESTS[name],
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(temporary / name),
            ]
            started = perf_counter()
            run = subprocess.run(
                command, cwd=snapshot, env=env, capture_output=True, text=True, check=False
            )
            log = run.stdout + run.stderr
            (out / f"tests_{name}.log").write_text(log, encoding="utf-8")
            tests[name] = {
                "command": command,
                "exit_code": run.returncode,
                "seconds": perf_counter() - started,
                "output": log,
            }
            print(name, run.returncode, log[-500:], flush=True)
    (out / "branch_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    if args.tests:
        (out / "branch_tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
