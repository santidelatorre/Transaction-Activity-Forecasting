"""Fingerprint official data, source and environment for integration evidence."""

import hashlib
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = (
    "train_transactions.jsonl",
    "train_labels.csv",
    "valid_transactions.jsonl",
    "valid_labels.csv",
    "test_transactions.jsonl",
    "sample_submission.csv",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    return {
        "data": {name: digest(ROOT / "data/raw/ubs_2026" / name) for name in FILES},
        "source": {
            str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
            for directory in ("src", "scripts", "configs", "tests")
            for p in sorted((ROOT / directory).rglob("*"))
            if p.suffix in {".py", ".toml"}
        },
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "catboost")},
        "python": platform.python_version(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "protocol": "official train=2000 / valid=1000; cutoff=2026-01-01; fixed 8 labels",
        "validation_independent": False,
    }
