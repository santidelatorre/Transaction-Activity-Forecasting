"""Reproduce pinned research branches in ignored snapshots, without integration.

Only OOF/TRAIN and VALID phases are run. No submission is generated. Each
snapshot uses its own source via PYTHONPATH and the same read-only local inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/metrics/v3_seven_way"
BASE = "df9fe4135ff812add0e2f6e0733706050b813cc2"
EXPERIMENTS = {
    "base": (BASE, "scripts/run_ubs_v3.py", "oof", None),
    "none": (
        "1cd708c2fc09d2b319ba13418060fb4df325d1c2",
        "scripts/run_v3_none_gate.py",
        "oof",
        None,
    ),
    "router": (
        "888d291e790d3a81566deb9961f6255dd87c3421",
        "scripts/run_v3_disagreement_gate.py",
        "oof",
        None,
    ),
    "family": (
        "02807fe27266ff4f343f9fb9eb6cdd9f44bdc160",
        "scripts/experiments/v3_family_mapping_javier.py",
        "oof",
        None,
    ),
    "music": (
        "7bf01c7e4e6755894f4da3facbaf85a8f65cd63f",
        "scripts/experiments/v3_music_streaming_christian.py",
        "oof",
        None,
    ),
    "calibration": (
        "e2e534ebd2ab3edb65104c644aaa5d0017f44050",
        "scripts/run_v3_classwise_calibration.py",
        "train",
        "--probability-dir",
    ),
    "gym": (
        "218bfcac4ed9495cd6367026b08372339b9f1cd6",
        "scripts/run_v3_gym_protection.py",
        "train",
        "--oof-dir",
    ),
    "blend": (
        "db21263dc6585fabf556c589cb1c4a0a9f34a642",
        "scripts/run_v3a_v2_blend.py",
        "oof",
        "--source-dir",
    ),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(name):
    commit, *_ = EXPERIMENTS[name]
    destination = OUT / "snapshots" / name
    marker = destination / ".research_snapshot.json"
    if marker.exists():
        if json.loads(marker.read_text())["commit"] != commit:
            raise ValueError("Snapshot commit mismatch")
        return destination
    if destination.exists():
        raise ValueError("Incomplete snapshot: use a fresh output directory")
    destination.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if not (destination / member.name).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Archive path escapes snapshot")
            if member.issym() or member.islnk():
                raise ValueError("Unexpected source archive link")
        source.extractall(destination, filter="data")
    marker.write_text(json.dumps({"commit": commit}), encoding="utf-8")
    return destination


def run(name):
    if name in ("router", "gym"):
        return replay(name)
    commit, script, first_phase, cache_flag = EXPERIMENTS[name]
    if cache_flag and not (OUT / "runs/base/valid_provenance.json").exists():
        raise ValueError("Complete both base phases before replaying cached experiments")
    snapshot = prepare(name)
    output = OUT / "runs" / name
    audit = OUT / "audit" / name
    audit.mkdir(parents=True, exist_ok=True)
    if name not in ("gym", "blend"):
        output.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data/raw/ubs_2026"
    inputs = {str(p): digest(p) for p in data.iterdir() if p.is_file()}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(snapshot / "src"), str(snapshot / "scripts")])
    environment["PYTHONUTF8"] = "1"
    phases = (first_phase, "valid")
    records = []
    for phase in phases:
        # Early runs kept records with their outputs; preserve those records too.
        legacy_record = output / f"reproduction_{phase}.json"
        record_path = legacy_record if legacy_record.exists() else audit / f"{phase}.json"
        if record_path.exists():
            existing = json.loads(record_path.read_text())
            if existing["returncode"] != 0 or existing["inputs"] != inputs:
                raise ValueError(f"Failed or changed prior run: {record_path}")
            records.append(existing)
            continue
        command = [
            sys.executable,
            "-c",
            f"import runpy; runpy.run_path({script!r}, run_name='__main__')",
            "--phase",
            phase,
            "--data-dir",
            str(data),
            "--output-dir",
            str(output),
        ]
        if cache_flag:
            command.extend([cache_flag, str(OUT / "runs/base")])
        started = perf_counter()
        print(f"{name}: starting {phase}", flush=True)
        log_path = audit / f"{phase}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=snapshot,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = {
            "snapshot_commit": commit,
            "command": command,
            "cwd": str(snapshot),
            "returncode": result.returncode,
            "seconds": perf_counter() - started,
            "inputs": inputs,
            "source": {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob("*.py")},
        }
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"{name}: {phase} exit {result.returncode}, {record['seconds']:.1f}s", flush=True)
        if result.returncode:
            raise RuntimeError(f"See {log_path}")
        if inputs != {str(p): digest(p) for p in data.iterdir() if p.is_file()}:
            raise RuntimeError("Input bytes changed during reproduction")
        records.append(record)
    return records


def replay(name):
    """Execute unchanged branch functions through the portable research adapter."""
    snapshot = prepare(name)
    audit = OUT / "audit" / name
    audit.mkdir(parents=True, exist_ok=True)
    output = OUT / "runs" / f"{name}_replay"
    record_path = audit / "replay.json"
    if record_path.exists():
        raise ValueError("Replay already recorded; inspect the existing evidence")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(snapshot / "src")
    environment["PYTHONUTF8"] = "1"
    command = [
        sys.executable,
        str(ROOT / "scripts/verify_v3_cached_experiments.py"),
        name,
        "--source-dir",
        str(OUT / "runs/base"),
        "--data-dir",
        str(ROOT / "data/raw/ubs_2026"),
        "--output-dir",
        str(output),
    ]
    start = perf_counter()
    with (audit / "replay.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=snapshot,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = {
        "snapshot_commit": EXPERIMENTS[name][0],
        "command": command,
        "cwd": str(snapshot),
        "seconds": perf_counter() - start,
        "returncode": result.returncode,
        "method": (
            "original predictive functions; fresh verified base cache; portable diagnostic driver"
        ),
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(record, flush=True)
    if result.returncode:
        raise RuntimeError(f"Replay failed: see {audit / 'replay.log'}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=tuple(EXPERIMENTS))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        if args.prepare_only:
            print(prepare(name))
        else:
            run(name)


if __name__ == "__main__":
    main()
