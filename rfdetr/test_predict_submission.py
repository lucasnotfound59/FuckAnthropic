"""Tests for the competition submission schema."""

from __future__ import annotations

import csv
import io
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

try:
    from .predict_submission import (
        format_submission_result,
        validate_submission,
        write_submission,
    )
except ImportError:  # Direct execution: python rfdetr/test_predict_submission.py
    from predict_submission import (
        format_submission_result,
        validate_submission,
        write_submission,
    )


class _NonClosingStringIO(io.StringIO):
    def close(self) -> None:
        """Keep captured CSV available after the writer context exits."""


class SubmissionFormatTests(unittest.TestCase):
    def test_formats_nine_counts_in_competition_order(self) -> None:
        counts = Counter({0: 4, 2: 5, 3: 1, 5: 1, 8: 2})
        self.assertEqual(format_submission_result(counts), "4;0;5;1;0;1;0;0;2")

    def test_formats_empty_predictions_as_nine_zero_counts(self) -> None:
        self.assertEqual(
            format_submission_result(Counter()),
            "0;0;0;0;0;0;0;0;0",
        )

    def test_writes_exact_two_column_schema_and_validates(self) -> None:
        images = [Path("first.jpg"), Path("second.jpg")]
        counts = {
            "first.jpg": Counter({2: 3, 7: 1}),
            "second.jpg": Counter(),
        }
        output = Path("submission.csv")
        buffer = _NonClosingStringIO()
        with (
            patch.object(Path, "mkdir"),
            patch.object(Path, "open", return_value=buffer),
        ):
            write_submission(output, images, counts)
        csv_text = buffer.getvalue()

        with patch.object(Path, "open", return_value=io.StringIO(csv_text)):
            validate_submission(output, [path.name for path in images])
        rows = list(csv.reader(io.StringIO(csv_text)))

        self.assertEqual(rows[0], ["pic_name", "results"])
        self.assertEqual(rows[1], ["first.jpg", "0;0;3;0;0;0;0;1;0"])
        self.assertEqual(rows[2], ["second.jpg", "0;0;0;0;0;0;0;0;0"])
        self.assertNotIn("\r\n", csv_text)

    def test_rejects_unknown_class_id(self) -> None:
        with self.assertRaises(ValueError):
            format_submission_result(Counter({9: 1}))

    def test_validator_rejects_empty_results_field(self) -> None:
        csv_text = "pic_name,results\nempty.jpg,\n"
        with patch.object(Path, "open", return_value=io.StringIO(csv_text)):
            with self.assertRaisesRegex(ValueError, "Null or empty results"):
                validate_submission(Path("submission.csv"), ["empty.jpg"])


if __name__ == "__main__":
    unittest.main()
