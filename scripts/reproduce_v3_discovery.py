"""Run pinned discovery sources in ignored snapshots, without merging branches."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SOURCES = {
    "v2": ("5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749", "scripts/run_ubs_v2.py"),
    "santiago": ("c040ec7", "scripts/experiments/santiago_oracle_streams.py"),
    "ginestar": ("0271694", "scripts/experiments/ginestar_next_date.py"),
    "javier": ("c7075bb", "scripts/experiments/javier_stream_family.py"),
    "christian": ("5f86fd9", "scripts/experiments/christian_merchant_normalization.py"),
    "jaime": ("101ad06", "scripts/experiments/jaime_pseudocutoffs.py"),
    "esteban": ("25f029c", "scripts/experiments/esteban_pretrain.py"),
    "laura": ("581f3b4", "scripts/experiments/laura_stream_system.py"),
}


def copy_verified_input(source: Path, target: Path) -> str:
    """Never record the current input hash while silently running a stale copy."""
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    if not target.exists():
        shutil.copyfile(source, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Snapshot input differs from source: {target}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=SOURCES)
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    revision, entry = SOURCES[args.experiment]
    commit = subprocess.check_output(["git", "rev-parse", revision], cwd=root, text=True).strip()
    directory = root / "outputs/metrics/v3_reproduction" / args.experiment
    directory.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", commit], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle:
            target = directory / member.name
            if not target.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Unsafe archive path")
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.extractfile(member).read())
    data = directory / "data/raw/ubs_2026"
    data.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in sorted((root / "data/raw/ubs_2026").iterdir()):
        if path.is_file():
            # Copies isolate source scripts from the original inputs.
            hashes[path.name] = copy_verified_input(path, data / path.name)
    v2_file = root / "outputs/metrics/v3_reproduction/v2/outputs/metrics/ubs_v2"
    if args.experiment in {"javier", "laura"} and v2_file.exists():
        shutil.copytree(v2_file, directory / "outputs/metrics/ubs_v2", dirs_exist_ok=True)
    if args.experiment == "laura" and not extra:
        extra = ["--v2-predictions", "outputs/metrics/ubs_v2/validation_predictions.csv"]
    if args.experiment == "ginestar" and not extra:
        extra = ["--v2-diagnostic"]
    if args.experiment == "v2" and not extra:
        extra = ["--submission", "outputs/predictions/submission_v2.csv"]
    bootstrap = (
        "import sys,runpy;sys.path[:0]=['src','.'];"
        "entry=sys.argv.pop(1);sys.argv[0]=entry;runpy.run_path(entry,run_name='__main__')"
    )
    command = [sys.executable, "-c", bootstrap, entry, *extra]
    started = time.perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4"}
    print(f"Running {args.experiment} at {commit}; log: {directory / 'run.log'}", flush=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=log)
    manifest = {
        "experiment": args.experiment,
        "commit": commit,
        "command": command,
        "seed": 42,
        "seconds": time.perf_counter() - started,
        "returncode": result.returncode,
        "input_sha256": hashes,
    }
    (directory / "reproduction.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
