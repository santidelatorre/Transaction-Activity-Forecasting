"""Typed, dataset-agnostic configuration for the forecasting pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Input contract. Populate these TODO fields when the dataset is available."""

    path: Path | None = None
    format: str | None = None
    required_columns: tuple[str, ...] = ()
    column_types: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitConfig:
    time_column: str = ""
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    embargo_rows: int = 0


@dataclass(frozen=True)
class ModelConfig:
    target_column: str = ""
    baseline: str = "mean_regressor"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    results_path: Path = Path("outputs/metrics/experiments.jsonl")


@dataclass(frozen=True)
class PipelineConfig:
    random_seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    drop_duplicate_rows: bool = True

    def validate_for_run(self) -> None:
        """Reject incomplete TODO values before processing a real dataset."""
        missing = [
            name
            for name, value in {
                "dataset.path": self.dataset.path,
                "dataset.format": self.dataset.format,
                "split.time_column": self.split.time_column,
                "model.target_column": self.model.target_column,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Complete dataset-specific configuration: {', '.join(missing)}")


def load_config(path: str | Path) -> PipelineConfig:
    """Load a TOML config without imposing any source schema."""
    with Path(path).open("rb") as config_file:
        raw = tomllib.load(config_file)
    dataset = raw.get("dataset", {})
    split = raw.get("split", {})
    model = raw.get("model", {})
    experiment = raw.get("experiment", {})
    processing = raw.get("processing", {})
    project = raw.get("project", {})
    return PipelineConfig(
        random_seed=int(project.get("random_seed", 42)),
        dataset=DatasetConfig(
            path=Path(dataset["path"]) if dataset.get("path") else None,
            format=dataset.get("format") or None,
            required_columns=tuple(dataset.get("required_columns", [])),
            column_types=dict(dataset.get("column_types", {})),
        ),
        split=SplitConfig(
            time_column=str(split.get("time_column", "")),
            validation_fraction=float(split.get("validation_fraction", 0.2)),
            test_fraction=float(split.get("test_fraction", 0.2)),
            embargo_rows=int(split.get("embargo_rows", 0)),
        ),
        model=ModelConfig(
            target_column=str(model.get("target_column", "")),
            baseline=str(model.get("baseline", "mean_regressor")),
            parameters=dict(model.get("parameters", {})),
        ),
        experiment=ExperimentConfig(
            results_path=Path(experiment.get("results_path", "outputs/metrics/experiments.jsonl"))
        ),
        drop_duplicate_rows=bool(processing.get("drop_duplicate_rows", True)),
    )
