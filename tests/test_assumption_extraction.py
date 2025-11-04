"""
Tests for Assumption entity extraction in DSPy workflow.
"""

import sys


sys.path.insert(0, "src")

from mathster.parsing.extract_workflow import (
    AssumptionExtraction,
    ChapterExtraction,
    convert_to_raw_document_section,
    validate_extraction,
)


def test_assumption_extraction_basic():
    """Test that AssumptionExtraction can be created with minimal fields."""
    assumption = AssumptionExtraction(
        label="assumption-bounded-domain",
        line_start=10,
        line_end=15,
    )

    assert assumption.label == "assumption-bounded-domain"
    assert assumption.line_start == 10
    assert assumption.line_end == 15
    print("✓ Test 1 passed: AssumptionExtraction basic creation")


def test_assumption_in_chapter_extraction():
    """Test that assumptions can be included in ChapterExtraction."""
    extraction = ChapterExtraction(
        section_id="Test Section",
        definitions=[],
        theorems=[],
        proofs=[],
        axioms=[],
        assumptions=[
            AssumptionExtraction(
                label="assumption-bounded-domain",
                line_start=10,
                line_end=15,
            ),
            AssumptionExtraction(
                label="assumption-lipschitz-regularity",
                line_start=20,
                line_end=25,
            ),
        ],
        parameters=[],
        remarks=[],
        citations=[],
    )

    assert len(extraction.assumptions) == 2
    assert extraction.assumptions[0].label == "assumption-bounded-domain"
    assert extraction.assumptions[1].label == "assumption-lipschitz-regularity"
    print("✓ Test 2 passed: Assumptions in ChapterExtraction")


def test_assumption_label_validation():
    """Test that assumption label validation works (must start with 'assumption-')."""
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = """  1: Test
  2: Content
  3: More
"""

    # Test with invalid label (should fail)
    extraction_dict_invalid = {
        "section_id": "Test Section",
        "definitions": [],
        "theorems": [],
        "proofs": [],
        "axioms": [],
        "assumptions": [
            {
                "label": "assump-invalid",  # ❌ Wrong prefix
                "line_start": 1,
                "line_end": 2,
            }
        ],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    result = validate_extraction(
        extraction_dict_invalid,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text,
    )

    assert not result.is_valid
    assert any(
        "assumption" in err.lower() and "must start with 'assumption-'" in err
        for err in result.errors
    )
    print("✓ Test 3 passed: Invalid assumption label caught by validation")


def test_assumption_label_validation_valid():
    """Test that valid assumption labels pass validation."""
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = """  1: Test
  2: Content
  3: More
"""

    extraction_dict_valid = {
        "section_id": "Test Section",
        "definitions": [],
        "theorems": [],
        "proofs": [],
        "axioms": [],
        "assumptions": [
            {
                "label": "assumption-bounded-domain",  # ✓ Valid prefix
                "line_start": 1,
                "line_end": 2,
            }
        ],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    result = validate_extraction(
        extraction_dict_valid,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text,
    )

    # Should not have assumption label errors
    assumption_errors = [
        err for err in result.errors if "assumption" in err.lower() and "must start" in err.lower()
    ]
    assert len(assumption_errors) == 0
    print("✓ Test 4 passed: Valid assumption label passes validation")


def test_assumption_conversion_to_raw():
    """Test conversion from AssumptionExtraction to RawAssumption."""
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = """  1: Test
  2: Assume the domain is bounded
  3: More content
"""

    extraction = ChapterExtraction(
        section_id="Test Section",
        definitions=[],
        theorems=[],
        proofs=[],
        axioms=[],
        assumptions=[
            AssumptionExtraction(
                label="assumption-bounded-domain",
                line_start=2,
                line_end=2,
            )
        ],
        parameters=[],
        remarks=[],
        citations=[],
    )

    raw_section, _warnings = convert_to_raw_document_section(
        extraction,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text,
    )

    assert len(raw_section.assumptions) == 1
    assert raw_section.assumptions[0].label == "assumption-bounded-domain"
    assert raw_section.assumptions[0].source.label == "assumption-bounded-domain"
    # TextLocation.lines is a list of (start, end) tuples
    assert raw_section.assumptions[0].source.line_range.lines[0] == (2, 2)
    print("✓ Test 5 passed: Conversion to RawAssumption works")


def test_assumption_in_entity_count():
    """Test that assumptions are included in entity count and summary."""
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = """  1: Test
  2: Assume the domain is bounded
  3: More content
"""

    extraction = ChapterExtraction(
        section_id="Test Section",
        definitions=[],
        theorems=[],
        proofs=[],
        axioms=[],
        assumptions=[
            AssumptionExtraction(
                label="assumption-bounded-domain",
                line_start=2,
                line_end=2,
            )
        ],
        parameters=[],
        remarks=[],
        citations=[],
    )

    raw_section, _warnings = convert_to_raw_document_section(
        extraction,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text,
    )

    # Check total_entities includes assumptions
    assert raw_section.total_entities == 1

    # Check summary includes assumptions
    summary = raw_section.get_summary()
    assert "Assumptions: 1" in summary

    print("✓ Test 6 passed: Assumptions included in entity count and summary")


def main():
    """Run all tests."""
    print("Testing Assumption entity extraction")
    print("=" * 60)

    try:
        test_assumption_extraction_basic()
        test_assumption_in_chapter_extraction()
        test_assumption_label_validation()
        test_assumption_label_validation_valid()
        test_assumption_conversion_to_raw()
        test_assumption_in_entity_count()

        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Summary:")
        print("  - AssumptionExtraction class works correctly")
        print("  - Assumptions can be added to ChapterExtraction")
        print("  - Label validation enforces 'assumption-' prefix")
        print("  - Conversion to RawAssumption preserves all fields")
        print("  - Assumptions counted in total_entities and summary")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
