"""Regressions for retained failures and immutable source provenance."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from summarize_stored_validation import summarize


class StoredSummary(unittest.TestCase):
    def fixture(self, root):
        (root / "source.md").write_text("x <= y\n")
        return {
            "new_engine_steps": 0,
            "input_reports": [],
            "chapters": [
                {
                    "chapter": 4,
                    "title": "Fixture",
                    "comparisons": [{"id": "bad", "observed": 2, "bound": 1, "passed": False}],
                    "coverage": {
                        "source": "source.md",
                        "source_sha256": hashlib.sha256(b"x <= y\n").hexdigest(),
                        "checked_required_expressions": 0,
                        "required_expressions": 1,
                        "missing_required_expressions": 1,
                        "complete": False,
                        "unbound_evidence": [],
                        "expression_ledger": [
                            {
                                "expression_id": "a",
                                "formula": "x<=y",
                                "status": "not_checked_from_stored_data",
                            }
                        ],
                    },
                }
            ],
        }

    def test_failure_and_missing_expression_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = summarize(self.fixture(root), root / "out", root)
            self.assertEqual(result["comparisons_failed"], 1)
            self.assertFalse(result["complete_required_expressions"])
            self.assertTrue((root / "out" / "missing-estimates.json").exists())

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            (root / "source.md").write_text("changed theorem\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                summarize(report, root / "out", root)

    def test_fresh_simulations_cannot_be_called_stored_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.fixture(root)
            report["new_engine_steps"] = 1
            with self.assertRaisesRegex(ValueError, "zero fresh-step"):
                summarize(report, root / "out", root)


if __name__ == "__main__":
    unittest.main()
