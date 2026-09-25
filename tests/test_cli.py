from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "ubs_recurrence.official", "--help"],
        [sys.executable, "scripts/validate_submission.py", "--help"],
        [sys.executable, "scripts/run_stream_identity_clean.py", "--help"],
        [sys.executable, "scripts/reproduce_stream_identity_submission.py", "--help"],
    ],
)
def test_lightweight_cli_help(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
