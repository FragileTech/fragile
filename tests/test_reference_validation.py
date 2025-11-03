"""
Tests for reference validation in DSPy extraction workflow.
"""

import sys

sys.path.insert(0, "src")

from mathster.parsing.extract_workflow import validate_extraction


def test_strict_proves_label_validation():
    """Test that proves_label validation is strict (errors on text references)."""
    # Use existing file path
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = """
  1: # Test Chapter
  2:
  3: This is a test chapter with some content.
  4:
  5: Proof of Theorem 1.1:
  6: This is the proof.
  7:
  8: QED.
  9:
 10: More content here.
"""

    extraction_dict = {
        "section_id": "Test Section",
        "definitions": [],
        "theorems": [],
        "proofs": [
            {
                "label": "proof-test",
                "proves_label": "Theorem 1.1",  # ❌ Not a label - should fail
                "line_start": 1,
                "line_end": 10,
                "theorem_references": [],
                "citations": [],
            }
        ],
        "axioms": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    result = validate_extraction(
        extraction_dict,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text
    )

    # Should have validation error for proves_label
    assert not result.is_valid, "Expected validation to fail"
    assert any("proves_label" in err and "MUST be a theorem label" in err for err in result.errors), \
        f"Expected proves_label error, got: {result.errors}"
    print("✓ Test 1 passed: Strict validation catches invalid proves_label")


def test_proves_label_with_valid_label():
    """Test that proves_label validation passes with valid label."""
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = "  1: Test content\n  2: More content\n"

    extraction_dict = {
        "section_id": "Test Section",
        "definitions": [],
        "theorems": [],
        "proofs": [
            {
                "label": "proof-test",
                "proves_label": "thm-main-result",  # ✓ Valid label
                "line_start": 1,
                "line_end": 2,
                "theorem_references": [],
                "citations": [],
            }
        ],
        "axioms": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    result = validate_extraction(
        extraction_dict,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text
    )

    # Should not have proves_label error
    proves_label_errors = [err for err in result.errors if "proves_label" in err and "MUST be" in err]
    assert len(proves_label_errors) == 0, f"Expected no proves_label errors, got: {proves_label_errors}"
    print("✓ Test 2 passed: Valid proves_label passes validation")


def test_permissive_definition_references():
    """Test that definition_references validation is permissive (warnings only)."""
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = "  1: Test\n  2: Content\n"

    extraction_dict = {
        "section_id": "Test Section",
        "definitions": [],
        "theorems": [
            {
                "label": "thm-test",
                "statement_type": "theorem",
                "line_start": 1,
                "line_end": 2,
                "definition_references": ["Lipschitz continuous"],  # ⚠ Text reference - should warn
            }
        ],
        "proofs": [],
        "axioms": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    result = validate_extraction(
        extraction_dict,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text
    )

    # Should have warning (not error) for definition_references
    def_ref_warnings = [w for w in result.warnings if "definition_references" in w]
    assert len(def_ref_warnings) > 0, f"Expected definition_references warning, got warnings: {result.warnings}"

    # Should NOT have error (permissive)
    def_ref_errors = [err for err in result.errors if "definition_references" in err]
    assert len(def_ref_errors) == 0, f"Expected no definition_references errors (permissive), got: {def_ref_errors}"

    print("✓ Test 3 passed: Permissive validation warns on text definition_references")


def test_permissive_theorem_references():
    """Test that theorem_references validation is permissive (warnings only)."""
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = "  1: Test\n  2: Content\n"

    extraction_dict = {
        "section_id": "Test Section",
        "definitions": [],
        "theorems": [],
        "proofs": [
            {
                "label": "proof-test",
                "proves_label": "thm-main",  # Valid
                "line_start": 1,
                "line_end": 2,
                "theorem_references": ["Theorem 1.4", "Lemma 2.3"],  # ⚠ Text references - should warn
                "citations": [],
            }
        ],
        "axioms": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    result = validate_extraction(
        extraction_dict,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text
    )

    # Should have warnings (not errors) for theorem_references
    thm_ref_warnings = [w for w in result.warnings if "theorem_references" in w]
    assert len(thm_ref_warnings) > 0, f"Expected theorem_references warnings, got: {result.warnings}"

    # Should NOT have errors (permissive)
    thm_ref_errors = [err for err in result.errors if "theorem_references" in err and "should be labels" in err]
    assert len(thm_ref_errors) == 0, f"Expected no theorem_references errors (permissive), got: {thm_ref_errors}"

    print("✓ Test 4 passed: Permissive validation warns on text theorem_references")


def test_valid_references_no_warnings():
    """Test that valid label-based references produce no warnings."""
    file_path = "/home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md"
    chapter_text = """  1: Test
  2: Content
  3: More
  4: Lines
  5: Here
"""

    extraction_dict = {
        "section_id": "Test Section",
        "definitions": [],
        "theorems": [
            {
                "label": "thm-test",
                "statement_type": "theorem",
                "line_start": 1,
                "line_end": 2,
                "definition_references": ["def-lipschitz", "def-continuous"],  # ✓ Valid labels
            }
        ],
        "proofs": [
            {
                "label": "proof-test",
                "proves_label": "thm-test",  # ✓ Valid
                "line_start": 3,
                "line_end": 5,
                "theorem_references": ["thm-convergence", "lem-bound"],  # ✓ Valid labels
                "citations": [],
            }
        ],
        "axioms": [],
        "parameters": [],
        "remarks": [],
        "citations": [],
    }

    result = validate_extraction(
        extraction_dict,
        file_path=file_path,
        article_id="05_kinetic_contraction",
        chapter_text=chapter_text
    )

    # Should have no reference-related warnings
    ref_warnings = [w for w in result.warnings if "definition_references" in w or "theorem_references" in w]
    assert len(ref_warnings) == 0, f"Expected no reference warnings for valid labels, got: {ref_warnings}"

    print("✓ Test 5 passed: Valid label-based references produce no warnings")


def main():
    """Run all validation tests."""
    print("Testing reference validation")
    print("=" * 60)

    try:
        test_strict_proves_label_validation()
        test_proves_label_with_valid_label()
        test_permissive_definition_references()
        test_permissive_theorem_references()
        test_valid_references_no_warnings()

        print()
        print("=" * 60)
        print("✓ All validation tests passed!")
        print("=" * 60)
        print()
        print("Validation behavior summary:")
        print("  - proves_label: STRICT (errors on invalid format)")
        print("  - definition_references: PERMISSIVE (warnings only)")
        print("  - theorem_references: PERMISSIVE (warnings only)")
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
