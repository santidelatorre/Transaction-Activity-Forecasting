"""Resumable stage orchestration for preparation, training, evaluation and submission.

Completed stages are checksum-verified. A fresh run name performs a clean
rebuild. An interrupted training stage restarts from raw data; completed model
training is never repeated when resuming the later stages.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prediction_source_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "src/ubs_recurrence").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def valid_completed_stage(state, name, folder):
    item = state.get("completed", {}).get(name)
    if item is None:
        return False
    for filename, expected in item["files"].items():
        path = folder / filename
        if not path.is_file() or checksum(path) != expected:
            raise ValueError(f"Completed {name} artifact changed: {path}")
    return True


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError(
            "Run name must contain only letters, numbers, underscore or hyphen"
        )
    run = (ROOT / "outputs" / args.run_name).resolve()
    run.mkdir(parents=True, exist_ok=True)
    config = {
        "prediction_source_sha256": prediction_source_hash(),
        "data_manifest_sha256": checksum(ROOT / "reports/data_manifest.json"),
        "device": args.device,
        "seeds": [42, 17, 2026],
        "legacy_weight": 0.25,
        "min_count": 3,
    }
    path = run / "run_state.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"config": config, "completed": {}}
    )
    if state["config"] != config:
        raise ValueError("Source/data/config changed; choose a fresh run name")
    subprocess.run([sys.executable, "scripts/prepare_data.py"], cwd=ROOT, check=True)
    evaluation_folder = run / state.get(
        "evaluation_folder", args.run_name + "_evaluation"
    )
    plan = [
        (
            "model",
            run / "model",
            ["train", "--output", str(run / "model"), "--device", args.device],
            ["model.joblib", "metadata.json"],
        ),
        (
            "evaluation",
            evaluation_folder,
            [
                "evaluate",
                "--model",
                str(run / "model/model.joblib"),
                "--output",
                str(evaluation_folder),
                "--purpose",
                "Frozen pipeline evaluation; resumable run " + args.run_name,
            ],
            ["metrics.json", "probabilities.csv"],
        ),
        (
            "submission",
            run / "submission",
            [
                "submit",
                "--model",
                str(run / "model/model.joblib"),
                "--output",
                str(run / "submission"),
            ],
            ["submission.csv", "submission_validation.json", "probabilities.csv"],
        ),
    ]
    for name, folder, command, files in plan:
        if not folder.resolve().is_relative_to(run) or folder.resolve() == run:
            raise ValueError("Unsafe stage path")
        if valid_completed_stage(state, name, folder):
            print("Verified completed stage:", name, flush=True)
            continue
        if folder.exists():
            if not folder.resolve().is_relative_to(run) or folder.resolve() == run:
                raise ValueError("Unsafe incomplete-stage move")
            preserved = run / f"{name}_incomplete_{time.time_ns()}"
            if not preserved.resolve().is_relative_to(run):
                raise ValueError("Unsafe preserved path")
            folder.rename(preserved)
            if name == "evaluation":
                # A failed evaluation may already have written an immutable
                # experiment record. Preserve it and use a new ID on retry.
                folder = run / f"{args.run_name}_evaluation_retry_{time.time_ns()}"
                state["evaluation_folder"] = folder.name
                command[command.index("--output") + 1] = str(folder)
        save_state(path, state)
        subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "ubs_recurrence.cli", *command],
            cwd=ROOT,
            check=True,
        )
        state["completed"][name] = {
            "files": {filename: checksum(folder / filename) for filename in files}
        }
        save_state(path, state)
    print("All stages complete and checksum-verified:", run)


if __name__ == "__main__":
    main()
