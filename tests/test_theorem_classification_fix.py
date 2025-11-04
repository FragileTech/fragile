"""
Test for theorem-like entity classification fix.

This test verifies that the bug where lemmas/propositions/corollaries were incorrectly
classified in validation reports has been fixed.

Bug scenario:
- lem-mass-conservation-transport appeared as HALLUCINATED in theorems
- lem-mass-conservation-transport appeared as MISSED in lemmas
- Root cause: _extract_labels_from_data() expected separate lists that don't exist

Solution:
- Updated _extract_labels_from_data() to classify by statement_type field
- RawDocumentSection stores all theorem-like entities in single 'theorems' list
- Each entity has statement_type field ("theorem", "lemma", "proposition", "corollary")
"""

import sys


sys.path.insert(0, "src")

from mathster.parsing.tools import _extract_labels_from_data, compare_extraction_with_source


def test_lemma_classification():
    """Test that lemmas are correctly classified when using dictionary format with statement_type."""

    chapter_text = """  1: ## Test Chapter
  2:
  3: :::{prf:theorem} Mass Conservation
  4: :label: thm-mass-conservation
  5: :::
  6:
  7: :::{prf:lemma} Transport Lemma
  8: :label: lem-mass-conservation-transport
  9: :::
 10: """

    print("\n" + "=" * 70)
    print("TEST: Lemma Classification Fix (Dictionary Format)")
    print("=" * 70 + "\n")

    # Create extraction data with theorems list containing both theorem and lemma
    # This simulates what convert_to_raw_document_section() produces
    extraction_data = {
        "section_id": "Test Chapter",
        "definitions": [],
        "theorems": [
            {
                "label": "thm-mass-conservation",
                "statement_type": "theorem",  # This is a theorem
            },
            {
                "label": "lem-mass-conservation-transport",
                "statement_type": "lemma",  # This is a LEMMA in theorems list
            },
        ],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    print("✓ Extraction data created with lemma in theorems list")
    print(f"  - theorems list contains: {[t['label'] for t in extraction_data['theorems']]}")
    print(f"  - statement_types: {[t['statement_type'] for t in extraction_data['theorems']]}")

    # Test _extract_labels_from_data() directly
    labels = _extract_labels_from_data(extraction_data)
    print("\n✓ Extracted labels by type:")
    for entity_type, label_list in labels.items():
        print(f"  - {entity_type}: {label_list}")

    # Verify classification
    assert "theorems" in labels, "Should have theorems key"
    assert "lemmas" in labels, "Should have lemmas key"
    assert "thm-mass-conservation" in labels["theorems"], "Theorem should be in theorems"
    assert "lem-mass-conservation-transport" in labels["lemmas"], "Lemma should be in lemmas"
    assert (
        "lem-mass-conservation-transport" not in labels["theorems"]
    ), "Lemma should NOT be in theorems"

    print("\n✓ Classification correct: lemma separated from theorems")

    # Generate comparison report
    comparison, report = compare_extraction_with_source(extraction_data, chapter_text)

    print("\n" + "=" * 70)
    print("EXTRACTION REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70 + "\n")

    # Verify the bug is fixed
    assert comparison["summary"]["correct_matches"] == 2, "Both entities should match"
    assert comparison["summary"]["hallucinated"] == 0, "No hallucinated labels"
    assert comparison["summary"]["missed"] == 0, "No missed labels"

    assert "thm-mass-conservation" in comparison["theorems"]["found"], "Theorem should be found"
    assert (
        "lem-mass-conservation-transport" in comparison["lemmas"]["found"]
    ), "Lemma should be found"

    # Verify lemma is NOT in wrong categories
    assert "lem-mass-conservation-transport" not in comparison["theorems"].get(
        "found", []
    ), "Lemma should NOT be in theorems found"
    assert "lem-mass-conservation-transport" not in comparison["theorems"].get(
        "missing_from_text", []
    ), "Lemma should NOT be hallucinated in theorems"
    assert "lem-mass-conservation-transport" not in comparison["lemmas"].get(
        "not_extracted", []
    ), "Lemma should NOT be missed in lemmas"

    print("✓ Test passed: Bug is fixed!")
    print("  - Lemma correctly classified as lemma (not as theorem)")
    print("  - No false hallucinations or missed labels")


def test_all_theorem_types():
    """Test that all theorem-like types are correctly classified."""

    chapter_text = """  1: ## Test Chapter
  2: :::{prf:theorem} Main Theorem
  3: :label: thm-main
  4: :::
  5: :::{prf:lemma} Helper Lemma
  6: :label: lem-helper
  7: :::
  8: :::{prf:proposition} Key Proposition
  9: :label: prop-key
 10: :::
 11: :::{prf:corollary} Direct Corollary
 12: :label: cor-direct
 13: :::
 14: """

    print("\n" + "=" * 70)
    print("TEST: All Theorem-Like Types Classification")
    print("=" * 70 + "\n")

    # Create extraction data with all types in theorems list
    extraction_data = {
        "section_id": "Test Chapter",
        "definitions": [],
        "theorems": [
            {"label": "thm-main", "statement_type": "theorem"},
            {"label": "lem-helper", "statement_type": "lemma"},
            {"label": "prop-key", "statement_type": "proposition"},
            {"label": "cor-direct", "statement_type": "corollary"},
        ],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    print("✓ Extraction data created with all theorem-like types in theorems list")

    # Test extraction
    labels = _extract_labels_from_data(extraction_data)
    print("\n✓ Extracted labels by type:")
    for entity_type, label_list in labels.items():
        print(f"  - {entity_type}: {label_list}")

    # Verify all types correctly classified
    assert "thm-main" in labels["theorems"], "Theorem should be in theorems"
    assert "lem-helper" in labels["lemmas"], "Lemma should be in lemmas"
    assert "prop-key" in labels["propositions"], "Proposition should be in propositions"
    assert "cor-direct" in labels["corollaries"], "Corollary should be in corollaries"

    print("\n✓ All types correctly classified")

    # Generate comparison report
    comparison, report = compare_extraction_with_source(extraction_data, chapter_text)

    print("\n" + "=" * 70)
    print("EXTRACTION REPORT (ALL TYPES)")
    print("=" * 70)
    print(report)
    print("=" * 70 + "\n")

    # Verify perfect match
    assert comparison["summary"]["correct_matches"] == 4, "All 4 entities should match"
    assert comparison["summary"]["hallucinated"] == 0, "No hallucinations"
    assert comparison["summary"]["missed"] == 0, "No missed labels"

    print("✓ Test passed: All theorem-like types correctly classified!")


def main():
    """Run all tests."""
    print("\nTesting theorem-like entity classification fix")
    print("=" * 70)

    try:
        test_lemma_classification()
        test_all_theorem_types()

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nSummary:")
        print("  - Lemmas correctly classified (not misidentified as theorems)")
        print("  - All theorem-like types (theorem/lemma/proposition/corollary) work")
        print("  - No false hallucinations or missed labels")
        print("  - Bug fixed: statement_type field now used for classification")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"✗ Test failed: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ Unexpected error: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
