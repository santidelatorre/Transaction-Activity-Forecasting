"""Validate predictions against a reference ID list and write a submission CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path


def _read_csv(
    path: Path, *, required: set[str], exact_columns: bool
) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Empty CSV: {path}") from error
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate column names in {path}")
        if not required.issubset(header):
            raise ValueError(f"Missing columns in {path}: {sorted(required.difference(header))}")
        if exact_columns and set(header) != required:
            raise ValueError(f"Prediction columns must be exactly {sorted(required)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"Wrong number of columns in {path}, line {line_number}")
            rows.append(row)
    return header, rows


def build_submission(
    predictions: str | Path,
    reference: str | Path,
    output: str | Path,
    *,
    id_column: str,
    label_column: str,
    classes: Sequence[str],
    expected_rows: int | None = None,
    overwrite: bool = False,
) -> int:
    """Write exactly one allowed prediction per reference ID, in reference order."""
    prediction_path, reference_path, output_path = map(Path, (predictions, reference, output))
    if output_path.resolve() in {prediction_path.resolve(), reference_path.resolve()}:
        raise ValueError("Output must differ from predictions and reference")
    if output_path.exists() and not overwrite:
        raise ValueError(f"Output already exists: {output_path}; pass --overwrite to replace it")
    if not id_column or not label_column or id_column == label_column:
        raise ValueError("Specify distinct, nonempty ID and label columns")
    if len(classes) != 8 or len(set(classes)) != 8 or any(not label.strip() for label in classes):
        raise ValueError("Specify exactly 8 distinct, nonempty allowed classes")
    if expected_rows is not None and expected_rows < 1:
        raise ValueError("expected_rows must be positive")

    prediction_header, prediction_rows = _read_csv(
        prediction_path, required={id_column, label_column}, exact_columns=True
    )
    reference_header, reference_rows = _read_csv(
        reference_path, required={id_column}, exact_columns=False
    )
    if not reference_rows:
        raise ValueError("Reference contains no IDs")
    if expected_rows is not None and len(reference_rows) != expected_rows:
        raise ValueError(f"Reference has {len(reference_rows)} rows; expected {expected_rows}")
    if len(prediction_rows) != len(reference_rows):
        raise ValueError(
            f"Prediction row count {len(prediction_rows)} differs from "
            f"reference {len(reference_rows)}"
        )

    reference_id_index = reference_header.index(id_column)
    reference_ids = [row[reference_id_index] for row in reference_rows]
    if any(not client_id.strip() for client_id in reference_ids):
        raise ValueError("Reference contains blank IDs")
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Reference contains duplicate IDs")

    prediction_id_index = prediction_header.index(id_column)
    prediction_label_index = prediction_header.index(label_column)
    prediction_map: dict[str, str] = {}
    allowed = set(classes)
    for row in prediction_rows:
        client_id, label = row[prediction_id_index], row[prediction_label_index]
        if not client_id.strip():
            raise ValueError("Predictions contain blank IDs")
        if client_id in prediction_map:
            raise ValueError(f"Predictions contain duplicate ID: {client_id}")
        if label not in allowed:
            raise ValueError(f"Invalid or missing class for ID {client_id}: {label!r}")
        prediction_map[client_id] = label
    if set(prediction_map) != set(reference_ids):
        missing = sorted(set(reference_ids) - set(prediction_map))
        extra = sorted(set(prediction_map) - set(reference_ids))
        raise ValueError(
            f"Prediction IDs differ from reference: missing={missing[:5]}, extra={extra[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=output_path.parent, suffix=".tmp", delete=False
        ) as target:
            temporary_path = Path(target.name)
            writer = csv.writer(target)
            writer.writerow([id_column, label_column])
            writer.writerows((client_id, prediction_map[client_id]) for client_id in reference_ids)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(reference_ids)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the standalone ``predict`` command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Validate and reorder existing predictions")
    predict.add_argument("--predictions", type=Path, required=True)
    predict.add_argument("--reference", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--id-column", required=True)
    predict.add_argument("--label-column", required=True)
    predict.add_argument("--classes", nargs=8, required=True, metavar="CLASS")
    predict.add_argument("--expected-rows", type=int)
    predict.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        count = build_submission(
            arguments.predictions,
            arguments.reference,
            arguments.output,
            id_column=arguments.id_column,
            label_column=arguments.label_column,
            classes=arguments.classes,
            expected_rows=arguments.expected_rows,
            overwrite=arguments.overwrite,
        )
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    print(f"Wrote {count} rows to {arguments.output}")


if __name__ == "__main__":
    main()
