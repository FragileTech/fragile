"""Exact shared-ledger and incomplete-coverage regressions."""

import json
from pathlib import Path
import tempfile
import unittest

from summarize_estimates import summarize


class SharedLedger(unittest.TestCase):
    def test_relational_values_and_failed_hypotheses_survive_summary(self):
        comparison = {
            "id": "actual",
            "observed": 2.0,
            "bound": 1.0,
            "passed": False,
            "source_labels": ["comparison_owner"],
        }
        evidence = {
            "chapter": 1,
            "source_labels": ["formula_owner"],
            "source_formula": "x<=y",
            "inputs": {"fixture": 0},
            "scope": "finite declared fixture",
            "checks": {"set": 0},
            "hypothesis_checks": {"set": 0},
        }
        report = {
            "schema_version": 2,
            "samples": 32,
            "suites": [{"evidence": [evidence]}],
            "fixtures": [{"x": 2}],
            "comparisons": [comparison],
            "comparison_sets": [[0]],
            "coverage": {
                "unbound_evidence": [],
                "chapters": [
                    {
                        "chapter": 1,
                        "source": "retained.md",
                        "source_sha256": "hash",
                        "expressions": [
                            {
                                "id": "missing",
                                "formula": "u<=v",
                                "requires_expression_evidence": True,
                                "disposition": "no_expression_level_evidence",
                            }
                        ],
                    }
                ],
            },
            "summary": {
                "formula_evidence_records": 1,
                "comparisons": 1,
                "comparisons_failed": 1,
                "hypotheses_failed": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            result = summarize(report, destination)
            self.assertFalse(result["complete"])
            self.assertEqual(result["unvalidated_required_expressions"], 1)
            ledger = json.loads((destination / "comparisons.json").read_text())
            self.assertEqual(ledger["fixtures"][0], {"x": 2})
            record = ledger["records"][0]
            actual = ledger["comparisons"][ledger["comparison_sets"][record["checks"]["set"]][0]]
            self.assertEqual(actual, comparison)
            self.assertEqual(record["evidence_source_labels"], ["formula_owner"])
            failures = json.loads((destination / "failed-comparisons.json").read_text())
            self.assertEqual(len(failures), 2)
            self.assertEqual(
                {item["comparison_kind"] for item in failures}, {"checks", "hypothesis_checks"}
            )
            self.assertEqual(failures[0]["evidence_source_labels"], ["formula_owner"])
            self.assertEqual(failures[0]["source_labels"], ["comparison_owner"])


if __name__ == "__main__":
    unittest.main()
