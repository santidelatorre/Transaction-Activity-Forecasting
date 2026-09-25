"""Pin model source independently of product commits and reject stale artifacts."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = ROOT / "configs/product_model_lock.json"
DEFAULT_BUNDLE = ROOT / "outputs/demo/v4_train_only"
PACKAGES = ("numpy", "pandas", "scikit-learn", "catboost", "joblib")


class ArtifactUnavailable(RuntimeError):
    """Missing, stale, or incompatible frozen model artifacts."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def model_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def verify_source(root: Path = ROOT, lock: dict | None = None) -> dict:
    lock = model_lock() if lock is None else lock
    for name, expected in lock["source_blobs"].items():
        # Git stores text with LF; Windows checkouts may use CRLF.
        source = (root / name).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
        if actual != expected:
            raise ArtifactUnavailable(f"Frozen model source changed: {name}")
    return lock


def runtime_versions() -> dict:
    return {name: version(name) for name in PACKAGES}


def verify_bundle(directory: Path, data_dir: Path) -> dict:
    try:
        lock = verify_source()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != lock or manifest["fit_scope"] != "TRAIN":
            raise ArtifactUnavailable("Bundle is not the locked V3-A submission model")
        if manifest["versions"] != runtime_versions():
            raise ArtifactUnavailable("Model package versions differ; use the recorded environment")
        for name in ("model.joblib", "predictions.csv", "scores.csv", "metrics.json", "cases.json"):
            if digest(directory / name) != manifest["artifacts"][name]:
                raise ArtifactUnavailable(f"Bundle artifact changed: {name}")
        for name, expected in manifest["data_hashes"].items():
            if digest(data_dir / name) != expected:
                raise ArtifactUnavailable(f"Source data changed: {name}")
        return manifest
    except (OSError, ValueError, KeyError) as error:
        raise ArtifactUnavailable("Run python scripts/prepare_product_demo.py first") from error
