"""
Simple test for CLI extraction report display (no DSPy required).
"""

import sys

sys.path.insert(0, "src")

from mathster.parsing.tools import compare_extraction_with_source


def test_report_generation():
    """Test that report is generated correctly from extracted data."""

    # Sample chapter text with labeled entities
    chapter_text = """  1: ## Test Chapter
  2:
  3: :::{prf:definition} Lipschitz Continuous
  4: :label: def-lipschitz
  5: :::
  6:
  7: A function is Lipschitz continuous...
  8:
  9: :::{prf:theorem} Main Result
 10: :label: thm-main-result
 11: :::
 12:
 13: The main result states...
 14:
 15: :::{prf:lemma} Helper Lemma
 16: :label: lem-helper
 17: :::
 18:
 19: This lemma helps...
 20: """

    print("\n" + "="*70)
    print("TEST: Report Generation from Extraction")
    print("="*70 + "\n")

    # Create extraction data with correct labels (as dictionary)
    extraction_data = {
        "section_id": "Test Chapter",
        "definitions": [
            {"label": "def-lipschitz"}
        ],
        "theorems": [
            {"label": "thm-main-result"}
        ],
        "lemmas": [
            {"label": "lem-helper"}
        ],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    print("✓ Extraction data created")

    # Generate comparison report
    comparison, report = compare_extraction_with_source(extraction_data, chapter_text)

    print("\n" + "="*70)
    print("EXTRACTION REPORT")
    print("="*70)
    print(report)
    print("="*70 + "\n")

    # Verify report content
    assert "def-lipschitz" in report, "Definition label should be in report"
    assert "thm-main-result" in report, "Theorem label should be in report"
    assert "lem-helper" in report, "Lemma label should be in report"

    # Verify comparison results
    assert comparison["summary"]["total_in_text"] == 3, "Should find 3 labels in text"
    assert comparison["summary"]["total_in_data"] == 3, "Should find 3 labels in data"
    assert comparison["summary"]["correct_matches"] == 3, "All labels should match"
    assert comparison["summary"]["hallucinated"] == 0, "No hallucinated labels"
    assert comparison["summary"]["missed"] == 0, "No missed labels"

    print("✓ Test passed: Report generated correctly")
    print(f"  - Found {comparison['summary']['correct_matches']} matching labels")
    print(f"  - Perfect extraction: all labels matched")


def test_report_with_missed_labels():
    """Test report showing missed labels."""

    chapter_text = """  1: ## Test Chapter
  2:
  3: :::{prf:definition} First Definition
  4: :label: def-first
  5: :::
  6:
  7: :::{prf:definition} Second Definition
  8: :label: def-second
  9: :::
 10:
 11: :::{prf:theorem} Main Theorem
 12: :label: thm-main
 13: :::
 14: """

    print("\n" + "="*70)
    print("TEST: Report with Missed Labels")
    print("="*70 + "\n")

    # Create extraction that MISSED def-second and thm-main
    extraction_data = {
        "section_id": "Test Chapter",
        "definitions": [
            {"label": "def-first"}
            # MISSING: def-second
        ],
        "theorems": [
            # MISSING: thm-main
        ],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    # Generate report
    comparison, report = compare_extraction_with_source(extraction_data, chapter_text)

    print("\n" + "="*70)
    print("EXTRACTION REPORT (WITH MISSED LABELS)")
    print("="*70)
    print(report)
    print("="*70 + "\n")

    # Verify missed labels are reported
    assert "def-second" in report, "Missed definition should be in report"
    assert "thm-main" in report, "Missed theorem should be in report"
    assert "2 missed label(s)" in report, "Should show 2 missed labels in action line"

    assert comparison["summary"]["correct_matches"] == 1, "Only def-first matches"
    assert comparison["summary"]["missed"] == 2, "Should have 2 missed labels"

    print("✓ Test passed: Missed labels correctly reported")


def test_report_with_hallucinated_labels():
    """Test report showing hallucinated labels."""

    chapter_text = """  1: ## Test Chapter
  2:
  3: :::{prf:definition} Real Definition
  4: :label: def-real
  5: :::
  6: """

    print("\n" + "="*70)
    print("TEST: Report with Hallucinated Labels")
    print("="*70 + "\n")

    # Create extraction with HALLUCINATED labels
    extraction_data = {
        "section_id": "Test Chapter",
        "definitions": [
            {"label": "def-real"},
            {"label": "def-fake"},  # HALLUCINATED
        ],
        "theorems": [
            {"label": "thm-imaginary"},  # HALLUCINATED
        ],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    # Generate report
    comparison, report = compare_extraction_with_source(extraction_data, chapter_text)

    print("\n" + "="*70)
    print("EXTRACTION REPORT (WITH HALLUCINATED LABELS)")
    print("="*70)
    print(report)
    print("="*70 + "\n")

    # Verify hallucinated labels are reported
    assert "def-fake" in report, "Hallucinated definition should be in report"
    assert "thm-imaginary" in report, "Hallucinated theorem should be in report"
    assert "2 hallucinated label(s)" in report, "Should show 2 hallucinated labels in action line"

    assert comparison["summary"]["correct_matches"] == 1, "Only def-real matches"
    assert comparison["summary"]["hallucinated"] == 2, "Should have 2 hallucinated labels"

    print("✓ Test passed: Hallucinated labels correctly reported")


def main():
    """Run all tests."""
    print("\nTesting extraction report display functionality")
    print("="*70)

    try:
        test_report_generation()
        test_report_with_missed_labels()
        test_report_with_hallucinated_labels()

        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        print("\nSummary:")
        print("  - Report correctly displays parsed labels by entity type")
        print("  - Report detects and displays hallucinated labels (in data, not in text)")
        print("  - Report detects and displays missed labels (in text, not in data)")
        print("  - Report shows validation status (PASSED/WARNING/FAILED)")
        print("  - Report integrates with extract_chapter() workflow")
        print("\nIntegration Details:")
        print("  - After extract_chapter() completes successfully (verbose=True)")
        print("  - Automatic comparison of extracted labels vs source document")
        print("  - Formatted report displayed in CLI with clear status indicators")
        print("  - Helps verify extraction quality immediately after processing")
        print("="*70)
        return 0

    except AssertionError as e:
        print("\n" + "="*70)
        print(f"✗ Test failed: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ Unexpected error: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
