"""
Tests for label comparison and validation functionality.
"""

import sys


sys.path.insert(0, "src")

from mathster.parsing.tools import _extract_labels_from_data, compare_extraction_with_source


def test_perfect_match_dict():
    """Test comparison with perfect match using dictionary input."""
    chapter_text = """
:::{prf:definition} Lipschitz Continuous
:label: def-lipschitz
:::

:::{prf:theorem} Main Result
:label: thm-main
:::
"""

    extracted_data = {
        "section_id": "Test",
        "definitions": [{"label": "def-lipschitz"}],
        "theorems": [{"label": "thm-main"}],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    comparison, report = compare_extraction_with_source(extracted_data, chapter_text)

    # Check summary
    assert comparison["summary"]["total_in_text"] == 2
    assert comparison["summary"]["total_in_data"] == 2
    assert comparison["summary"]["correct_matches"] == 2
    assert comparison["summary"]["hallucinated"] == 0
    assert comparison["summary"]["missed"] == 0

    # Check report
    assert "✓ VALIDATION PASSED: Perfect match between text and data" in report
    assert "def-lipschitz" in report
    assert "thm-main" in report

    print("✓ Test 1 passed: Perfect match with dict input")


def test_hallucinated_labels():
    """Test detection of hallucinated labels (in data but not in text)."""
    chapter_text = """
:::{prf:theorem} Real Theorem
:label: thm-real
:::
"""

    extracted_data = {
        "section_id": "Test",
        "definitions": [],
        "theorems": [
            {"label": "thm-real"},
            {"label": "thm-fake"},  # HALLUCINATED
            {"label": "thm-also-fake"},  # HALLUCINATED
        ],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    comparison, report = compare_extraction_with_source(extracted_data, chapter_text)

    # Check summary
    assert comparison["summary"]["total_in_text"] == 1
    assert comparison["summary"]["total_in_data"] == 3
    assert comparison["summary"]["correct_matches"] == 1
    assert comparison["summary"]["hallucinated"] == 2
    assert comparison["summary"]["missed"] == 0

    # Check detailed comparison
    assert "thm-real" in comparison["theorems"]["found"]
    assert "thm-fake" in comparison["theorems"]["missing_from_text"]
    assert "thm-also-fake" in comparison["theorems"]["missing_from_text"]

    # Check report
    assert "✗ VALIDATION FAILED: Hallucinated labels detected" in report
    assert "thm-fake" in report
    assert "thm-also-fake" in report
    assert "HALLUCINATED (2)" in report

    print("✓ Test 2 passed: Hallucinated labels detected correctly")


def test_missed_labels():
    """Test detection of missed labels (in text but not in data)."""
    chapter_text = """
:::{prf:definition} First Definition
:label: def-first
:::

:::{prf:definition} Second Definition
:label: def-second
:::

:::{prf:definition} Third Definition
:label: def-third
:::
"""

    extracted_data = {
        "section_id": "Test",
        "definitions": [{"label": "def-first"}],  # Missing def-second and def-third
        "theorems": [],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    comparison, report = compare_extraction_with_source(extracted_data, chapter_text)

    # Check summary
    assert comparison["summary"]["total_in_text"] == 3
    assert comparison["summary"]["total_in_data"] == 1
    assert comparison["summary"]["correct_matches"] == 1
    assert comparison["summary"]["hallucinated"] == 0
    assert comparison["summary"]["missed"] == 2

    # Check detailed comparison
    assert "def-first" in comparison["definitions"]["found"]
    assert "def-second" in comparison["definitions"]["not_extracted"]
    assert "def-third" in comparison["definitions"]["not_extracted"]

    # Check report
    assert "⚠ VALIDATION WARNING: Some labels missed in extraction" in report
    assert "def-second" in report
    assert "def-third" in report
    assert "MISSED (2)" in report

    print("✓ Test 3 passed: Missed labels detected correctly")


def test_both_hallucinated_and_missed():
    """Test detection of both hallucinated and missed labels."""
    chapter_text = """
:::{prf:theorem} Real Theorem One
:label: thm-real-1
:::

:::{prf:theorem} Real Theorem Two
:label: thm-real-2
:::
"""

    extracted_data = {
        "section_id": "Test",
        "definitions": [],
        "theorems": [
            {"label": "thm-real-1"},  # Correct
            {"label": "thm-fake"},  # Hallucinated
        ],
        # Missing thm-real-2
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    comparison, report = compare_extraction_with_source(extracted_data, chapter_text)

    # Check summary
    assert comparison["summary"]["total_in_text"] == 2
    assert comparison["summary"]["total_in_data"] == 2
    assert comparison["summary"]["correct_matches"] == 1
    assert comparison["summary"]["hallucinated"] == 1
    assert comparison["summary"]["missed"] == 1

    # Check report
    assert "✗ VALIDATION FAILED: Both hallucinations and missed extractions detected" in report
    assert "Remove 1 hallucinated label(s)" in report
    assert "Re-extract to capture 1 missed label(s)" in report

    print("✓ Test 4 passed: Both hallucinated and missed labels detected")


def test_empty_data():
    """Test with empty extracted data."""
    chapter_text = """
:::{prf:definition} Some Definition
:label: def-something
:::
"""

    extracted_data = {
        "section_id": "Test",
        "definitions": [],
        "theorems": [],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    comparison, _report = compare_extraction_with_source(extracted_data, chapter_text)

    assert comparison["summary"]["total_in_text"] == 1
    assert comparison["summary"]["total_in_data"] == 0
    assert comparison["summary"]["missed"] == 1
    assert "def-something" in comparison["definitions"]["not_extracted"]

    print("✓ Test 5 passed: Empty data handled correctly")


def test_empty_text():
    """Test with empty source text (no labels)."""
    chapter_text = """
# Chapter Title

Some content without any labeled entities.
"""

    extracted_data = {
        "section_id": "Test",
        "definitions": [{"label": "def-fake"}],  # This is hallucinated
        "theorems": [],
        "proofs": [],
        "axioms": [],
        "assumptions": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    comparison, _report = compare_extraction_with_source(extracted_data, chapter_text)

    assert comparison["summary"]["total_in_text"] == 0
    assert comparison["summary"]["total_in_data"] == 1
    assert comparison["summary"]["hallucinated"] == 1
    assert "def-fake" in comparison["definitions"]["missing_from_text"]

    print("✓ Test 6 passed: Empty text handled correctly")


def test_multiple_entity_types():
    """Test comparison across multiple entity types."""
    chapter_text = """
:::{prf:definition} Important Definition
:label: def-important
:::

:::{prf:theorem} Main Theorem
:label: thm-main
:::

:::{prf:lemma} Helper Lemma
:label: lem-helper
:::

:::{prf:remark} Important Note
:label: remark-note
:::

:::{assumption} Bounded Domain
:label: assumption-bounded
:::
"""

    extracted_data = {
        "section_id": "Test",
        "definitions": [{"label": "def-important"}],
        "theorems": [{"label": "thm-main"}],
        "proofs": [],
        "axioms": [],
        "assumptions": [{"label": "assumption-bounded"}],
        "parameters": [],
        "remarks": [{"label": "remark-note"}],
        "citations": [],
    }

    comparison, _report = compare_extraction_with_source(extracted_data, chapter_text)

    # Check all correct
    assert comparison["summary"]["correct_matches"] == 4
    assert comparison["summary"]["hallucinated"] == 0
    assert comparison["summary"]["missed"] == 1  # lem-helper missed

    # Check specific types
    assert "def-important" in comparison["definitions"]["found"]
    assert "thm-main" in comparison["theorems"]["found"]
    assert "lem-helper" in comparison["lemmas"]["not_extracted"]
    assert "remark-note" in comparison["remarks"]["found"]
    assert "assumption-bounded" in comparison["assumptions"]["found"]

    print("✓ Test 7 passed: Multiple entity types compared correctly")


def test_extract_labels_from_dict():
    """Test _extract_labels_from_data with dict input."""
    data = {
        "definitions": [
            {"label": "def-first"},
            {"label": "def-second"},
        ],
        "theorems": [
            {"label": "thm-main"},
        ],
        "proofs": [],
        "axioms": [],
        "assumptions": [
            {"label": "assumption-bounded"},
        ],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    labels = _extract_labels_from_data(data)

    assert "definitions" in labels
    assert len(labels["definitions"]) == 2
    assert "def-first" in labels["definitions"]
    assert "def-second" in labels["definitions"]

    assert "theorems" in labels
    assert "thm-main" in labels["theorems"]

    assert "assumptions" in labels
    assert "assumption-bounded" in labels["assumptions"]

    print("✓ Test 8 passed: Label extraction from dict works correctly")


def test_assumptions_entity_type():
    """Test that assumptions are properly compared."""
    chapter_text = """
:::{assumption} Bounded Domain
:label: assumption-bounded-domain
:::

:::{assumption} Lipschitz Regularity
:label: assumption-lipschitz-regularity
:::
"""

    extracted_data = {
        "section_id": "Test",
        "definitions": [],
        "theorems": [],
        "proofs": [],
        "axioms": [],
        "assumptions": [
            {"label": "assumption-bounded-domain"},
            {"label": "assumption-lipschitz-regularity"},
        ],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    comparison, _report = compare_extraction_with_source(extracted_data, chapter_text)

    assert comparison["summary"]["correct_matches"] == 2
    assert comparison["summary"]["hallucinated"] == 0
    assert comparison["summary"]["missed"] == 0

    assert "assumption-bounded-domain" in comparison["assumptions"]["found"]
    assert "assumption-lipschitz-regularity" in comparison["assumptions"]["found"]

    print("✓ Test 9 passed: Assumptions properly compared")


def main():
    """Run all tests."""
    print("Testing label comparison and validation functionality")
    print("=" * 70)

    try:
        test_perfect_match_dict()
        test_hallucinated_labels()
        test_missed_labels()
        test_both_hallucinated_and_missed()
        test_empty_data()
        test_empty_text()
        test_multiple_entity_types()
        test_extract_labels_from_dict()
        test_assumptions_entity_type()

        print()
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  - Perfect match detection works")
        print("  - Hallucinated labels detected (in data, not in text)")
        print("  - Missed labels detected (in text, not in data)")
        print("  - Both types of errors detected simultaneously")
        print("  - Edge cases handled (empty data/text)")
        print("  - Multiple entity types compared correctly")
        print("  - Dict input format works")
        print("  - New assumption entity type supported")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"✗ Test failed: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
