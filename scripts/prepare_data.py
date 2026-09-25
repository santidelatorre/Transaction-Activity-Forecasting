"""Download a pinned official archive, verify its hash, and safely extract data."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "796d5805ec5f8a228a3ec0de36a2b4e6e1b1a1df"
ARCHIVE_HASH = "1afc95470f4e8641601503172be3e698ef9eaf91528d911a6a01a120911634c6"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "reports/data_manifest.json").read_text())
    if all(
        (raw / name).exists() and sha(raw / name) == item["sha256"]
        for name, item in manifest.items()
    ):
        print("All seven raw files match the pinned official hashes.")
        return
    archive = ROOT / "data/downloads/dataset.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = f"https://raw.githubusercontent.com/UBS-AG/Swiss-AI-Weeks/{COMMIT}/hackathons/2026/data/dataset.zip"
        urllib.request.urlretrieve(url, archive)
    if sha(archive) != ARCHIVE_HASH:
        raise ValueError("Official archive hash mismatch")
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != set(manifest):
            raise ValueError("Unexpected archive files")
        for name in z.namelist():
            destination = (raw / name).resolve()
            if not destination.is_relative_to(raw.resolve()):
                raise ValueError("Unsafe extraction path")
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != manifest[name]["sha256"]:
                raise ValueError(name)
            destination.write_bytes(data)
    print("Downloaded and verified seven official data files.")


if __name__ == "__main__":
    main()
