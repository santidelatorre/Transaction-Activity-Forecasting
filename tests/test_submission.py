from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from transaction_forecasting.submission import build_submission, main

CLASSES = ("cloud", "gym", "insurance", "mobile", "music", "software", "streaming", "none")
ID = "client_id"
LABEL = "predicted_next_recurring_merchant"


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


class SubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.predictions = root / "predictions.csv"
        self.reference = root / "sample_submission.csv"
        self.output = root / "submission.csv"
        write_csv(self.reference, [ID, LABEL], [["C002", "none"], ["C001", "none"]])
        write_csv(self.predictions, [ID, LABEL], [["C001", "gym"], ["C002", "cloud"]])

    def build(self, **overrides: object) -> int:
        options: dict[str, object] = {
            "id_column": ID,
            "label_column": LABEL,
            "classes": CLASSES,
            **overrides,
        }
        return build_submission(self.predictions, self.reference, self.output, **options)

    def test_writes_exact_schema_in_reference_order(self) -> None:
        self.assertEqual(self.build(expected_rows=2), 2)
        with self.output.open("r", encoding="utf-8", newline="") as source:
            self.assertEqual(
                list(csv.reader(source)),
                [[ID, LABEL], ["C002", "cloud"], ["C001", "gym"]],
            )

    def test_rejects_duplicate_prediction_ids_without_output(self) -> None:
        write_csv(self.predictions, [ID, LABEL], [["C001", "gym"], ["C001", "cloud"]])
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_rejects_missing_or_extra_ids(self) -> None:
        write_csv(self.predictions, [ID, LABEL], [["C001", "gym"], ["C003", "cloud"]])
        with self.assertRaisesRegex(ValueError, "missing=.*C002.*extra=.*C003"):
            self.build()

    def test_rejects_unknown_or_blank_class(self) -> None:
        for value in ("unknown", ""):
            write_csv(self.predictions, [ID, LABEL], [["C001", value], ["C002", "cloud"]])
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "Invalid or missing"),
            ):
                self.build()

    def test_rejects_wrong_schema_or_row_count(self) -> None:
        write_csv(self.predictions, [ID, LABEL, "confidence"], [["C001", "gym", "0.5"]])
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.build()
        write_csv(self.predictions, [ID, LABEL], [["C001", "gym"], ["C002", "cloud"]])
        with self.assertRaisesRegex(ValueError, "expected 3"):
            self.build(expected_rows=3)

    def test_rejects_duplicate_reference_ids(self) -> None:
        write_csv(self.reference, [ID], [["C001"], ["C001"]])
        with self.assertRaisesRegex(ValueError, "Reference contains duplicate IDs"):
            self.build()

    def test_requires_explicit_overwrite(self) -> None:
        self.build()
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            self.build()
        self.assertEqual(self.build(overwrite=True), 2)

    def test_cli_predict_command(self) -> None:
        main(
            [
                "predict",
                "--predictions",
                str(self.predictions),
                "--reference",
                str(self.reference),
                "--output",
                str(self.output),
                "--id-column",
                ID,
                "--label-column",
                LABEL,
                "--classes",
                *CLASSES,
                "--expected-rows",
                "2",
            ]
        )
        self.assertTrue(self.output.exists())
