"""Run unchanged historical V1 and frozen V2 in isolated output directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = "0199a8b7c8b2c790a9d3447b156f2088708c2864"
V2 = "96409b7940a991fbda5235b85ffeb40652a4087b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    args = parser.parse_args()
    output = ROOT / "outputs/metrics/v1_benchmark" / args.version
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs/metrics/v1_benchmark" / f"source_{args.version}"
    commit = V1 if args.version == "v1" else V2
    if not source.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(source), commit], cwd=ROOT, check=True
        )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != commit:
        raise RuntimeError("Unexpected source commit")
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=source):
        raise RuntimeError("Historical source has local modifications")
    data = source / "data/raw/ubs_2026"
    data.mkdir(parents=True, exist_ok=True)
    for original in (ROOT / "data/raw/ubs_2026").iterdir():
        target = data / original.name
        if original.is_file() and not target.exists():
            shutil.copy2(original, target)
    submission = output / "submission.csv"
    if args.version == "v1":
        config = (source / "configs/ubs_v1.toml").read_text(encoding="utf-8")
        config = config.replace("outputs/metrics/ubs_v1", output.as_posix())
        config = config.replace("outputs/predictions/submission_v1.csv", submission.as_posix())
        config_path = output / "config.toml"
        config_path.write_text(config, encoding="utf-8")
        command = [sys.executable, "scripts/run_ubs_baseline.py", "--config", str(config_path)]
    else:
        command = [
            sys.executable,
            "scripts/run_ubs_v2.py",
            "--metrics-directory",
            str(output),
            "--submission",
            str(submission),
        ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source / "src")
    with (output / "execution.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=source, env=env, stdout=log, stderr=log, check=True)
    manifest = {
        "source_commit": commit,
        "command": command,
        "data_sha256": {p.name: digest(p) for p in sorted(data.iterdir()) if p.is_file()},
        "submission_sha256": digest(submission),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed {args.version}: {output}", flush=True)


if __name__ == "__main__":
    main()
