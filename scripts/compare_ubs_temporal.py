"""Run the frozen temporal-only comparison: python scripts/compare_ubs_temporal.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

from transaction_forecasting.ubs.data import load_ubs_data
from transaction_forecasting.ubs.temporal_experiment import compare_temporal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ubs_v2_temporal.toml"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    data = load_ubs_data(config["data_directory"])
    result = compare_temporal(data, config)
    result["base_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sources = [args.config, Path(__file__)] + list(
        Path("src/transaction_forecasting/ubs").glob("*.py")
    )
    result["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    result["input_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(config["data_directory"]).glob("*"))
        if path.is_file()
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Selected: {result['selected']}; results: {output}")


if __name__ == "__main__":
    main()
