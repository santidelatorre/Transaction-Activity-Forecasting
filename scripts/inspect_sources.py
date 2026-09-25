"""Inspect the locally downloaded official source without reading validation labels."""

import hashlib
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

root = Path(__file__).resolve().parents[1]
archive = root / "data/official/hackathons/2026/data/dataset.zip"
print(
    "ARCHIVE", archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest()
)
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        print(item.filename, item.file_size, item.compress_size)
    target = (root / "data/raw").resolve()
    for item in z.infolist():
        if not (target / item.filename).resolve().is_relative_to(target):
            raise ValueError("Unsafe archive path")
    z.extractall(target)
print("OFFICIAL LICENSE")
print((root / "data/official/LICENSE").read_text(encoding="utf-8"))
soup = BeautifulSoup(
    (root / "data/jury_site.html").read_text(encoding="utf-8"), "html.parser"
)
for a in soup.find_all("a", href=True):
    if any(
        k in (a.get_text(" ", strip=True) + a["href"]).lower()
        for k in ["jury", "criteria", "evaluation", "judg"]
    ):
        print("JURY LINK", a.get_text(" ", strip=True), a["href"])
text = soup.get_text("\n", strip=True)
(root / "data/jury_site.txt").write_text(text, encoding="utf-8")
for line in text.splitlines():
    if any(
        k in line.lower()
        for k in ["jury", "criteria", "evaluation", "agentic", "originality"]
    ):
        print("JURY TEXT", line)
