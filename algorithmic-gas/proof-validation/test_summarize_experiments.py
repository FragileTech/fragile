"""Protect honest coverage accounting and independent-dataset counting."""

import json
from pathlib import Path
import tempfile
import unittest

from summarize_experiments import summarize


class ExperimentSummary(unittest.TestCase):
    def dataset(self, root):
        root.mkdir()
        (root / "archive-index.json").write_text(
            json.dumps({"status": "completed", "entries": []})
        )
        chapter = {
            "chapter": 4,
            "title": "Fixture",
            "comparisons": [{"id": "failed", "observed": 2, "bound": 1, "passed": False}],
            "coverage": {
                "required_expressions": 1,
                "checked_required_expressions": 0,
                "unbound_evidence": [],
                "complete": False,
                "expression_ledger": [
                    {"expression_id": "global", "status": "not_checked_from_stored_data"}
                ],
            },
        }
        (root / "report.json").write_text(json.dumps({"config": {}, "chapters": [chapter]}))
        return root

    def test_failures_and_unvalidated_theory_remain_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.dataset(root / "raw")
            result = summarize([dataset], root / "derived")
            self.assertEqual(result["chapters"][0]["failed"], 1)
            self.assertFalse(result["chapters"][0]["all_required_checked"])
            self.assertEqual(len(json.loads((root / "derived/failures.json").read_text())), 1)
            self.assertEqual(
                len(json.loads((root / "derived/unvalidated-expressions.json").read_text())), 1
            )

    def test_same_samples_cannot_be_counted_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.dataset(root / "raw")
            with self.assertRaisesRegex(ValueError, "counted twice"):
                summarize([dataset, dataset], root / "derived")

    def test_incomplete_execution_is_not_a_completed_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.dataset(root / "raw")
            (dataset / "archive-index.json").write_text(
                json.dumps({"status": "recording", "entries": []})
            )
            with self.assertRaisesRegex(ValueError, "Incomplete dataset"):
                summarize([dataset], root / "derived")

    def test_cubature_review_rejects_changed_raw_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.dataset(root / "raw")
            source = root / "cubature-report.json"
            source.write_text("changed source")
            review = root / "review.json"
            review.write_text(
                json.dumps({
                    "derived_review": {
                        "source_report": str(source),
                        "source_report_sha256": "0" * 64,
                    }
                })
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                summarize([dataset], root / "derived", review)


if __name__ == "__main__":
    unittest.main()
