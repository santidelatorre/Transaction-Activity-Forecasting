"""Structured and append-only experiment result persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def save_experiment(path: str | Path, result: dict[str, Any]) -> Path:
    """Append one JSON record, including a timestamp, and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"recorded_at": datetime.now(UTC).isoformat(), **result}
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    return output_path
