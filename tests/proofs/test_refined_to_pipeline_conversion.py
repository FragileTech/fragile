"""
Test suite for refined_data to pipeline_data conversion.

Tests the conversion script that transforms enriched JSON files from
refined_data/ directory to pipeline-ready math_types format.

This test suite prevents regressions like:
- Missing source_location fallback causing empty fields
- Inconsistent handling of source vs source_location fields
- Loss of semantic data during transformation
"""

import json
from pathlib import Path

# Import conversion functions from the agent script
import sys
import tempfile
from typing import Any, Dict

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fragile.agents.convert_refined_to_pipeline import (
    add_source_fields,
    convert_axiom,
    convert_definition_to_object,
    convert_property_to_attribute,
    convert_theorem,
)


# =============================================================================
# SOURCE FIELD HANDLING TESTS (Regression Prevention)
# =============================================================================


def test_add_source_fields_with_all_data():
    """Test source field handling with complete data."""
    source = {
        "document_id": "01_fragile_gas_framework",
        "file_path": "docs/source/1_euclidean_gas/01_fragile_gas_framework.md",
        "section": "9. Rescale Transformation",
        "directive_label": "def-axiom-rescale-function",
        "line_range": [1641, 1663],
    }

    result = add_source_fields(source)

    assert result is not None
    assert result["document_id"] == "01_fragile_gas_framework"
    assert result["file_path"] == "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
    assert result["section"] == "9. Rescale Transformation"
    assert result["directive_label"] == "def-axiom-rescale-function"
    assert result["line_range"] == [1641, 1663]
    assert result["equation"] is None  # Not provided
    assert result["url_fragment"] is None  # Not provided


def test_add_source_fields_returns_none_for_empty_dict():
    """Test that empty source dict returns None."""
    source = {}
    result = add_source_fields(source)
    assert result is None


def test_add_source_fields_returns_none_for_all_null():
    """Test that source with all null values returns None."""
    source = {"document_id": None, "file_path": None, "section": None, "directive_label": None}
    result = add_source_fields(source)
    assert result is None


def test_add_source_fields_with_partial_data():
    """Test source field handling with partial data."""
    source = {
        "document_id": "04_convergence",
        "file_path": None,
        "section": "Main Results",
        "directive_label": None,
    }

    result = add_source_fields(source)

    assert result is not None  # Has some data
    assert result["document_id"] == "04_convergence"
    assert result["section"] == "Main Results"


# =============================================================================
# AXIOM CONVERSION TESTS (Bug Fix Regression)
# =============================================================================


def test_convert_axiom_with_source_location_field():
    """
    REGRESSION TEST: Ensure axiom conversion checks source_location first.

    This test prevents the bug where convert_axiom only checked "source"
    but enriched data used "source_location", causing empty fields.
    """
    refined = {
        "label": "axiom-rescale-function",
        "name": "Axiom of a Well-Behaved Rescale Function",
        "statement": "Any function g_A must satisfy C1 smoothness...",
        "mathematical_expression": "g'_A(z) >= 0",
        "chapter": "1_euclidean_gas",
        "document": "01_fragile_gas_framework",
        "source_location": {  # Note: Uses source_location not source
            "document_id": "01_fragile_gas_framework",
            "file_path": "docs/source/1_euclidean_gas/01_fragile_gas_framework.md",
            "section": "9. Rescale Transformation",
            "directive_label": "def-axiom-rescale-function",
        },
    }

    result = convert_axiom(refined)

    # Check all fields preserved
    assert result["label"] == "axiom-rescale-function"
    assert result["statement"] == "Any function g_A must satisfy C1 smoothness..."
    assert result["mathematical_expression"] == "g'_A(z) >= 0"
    assert result["chapter"] == "1_euclidean_gas"
    assert result["document"] == "01_fragile_gas_framework"

    # CRITICAL: Source must be preserved (not None, not empty)
    assert result["source"] is not None
    assert result["source"]["document_id"] == "01_fragile_gas_framework"
    assert (
        result["source"]["file_path"] == "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
    )


def test_convert_axiom_with_legacy_source_field():
    """Test axiom conversion with legacy 'source' field."""
    refined = {
        "label": "axiom-test",
        "name": "Test Axiom",
        "statement": "Test statement",
        "mathematical_expression": "x = y",
        "source": {  # Legacy field name
            "document_id": "test_doc",
            "file_path": "test.md",
        },
    }

    result = convert_axiom(refined)

    assert result["source"] is not None
    assert result["source"]["document_id"] == "test_doc"


def test_convert_axiom_with_both_source_fields():
    """Test axiom conversion prioritizes source_location over source."""
    refined = {
        "label": "axiom-test",
        "name": "Test Axiom",
        "statement": "Test",
        "mathematical_expression": "x = y",
        "source_location": {"document_id": "correct_doc", "file_path": "correct.md"},
        "source": {  # Should be ignored
            "document_id": "wrong_doc",
            "file_path": "wrong.md",
        },
    }

    result = convert_axiom(refined)

    # Should use source_location (not source)
    assert result["source"]["document_id"] == "correct_doc"
    assert result["source"]["file_path"] == "correct.md"


def test_convert_axiom_with_no_source():
    """Test axiom conversion when no source is provided."""
    refined = {
        "label": "axiom-test",
        "name": "Test Axiom",
        "statement": "Test",
        "mathematical_expression": "x = y",
    }

    result = convert_axiom(refined)

    # Should handle gracefully
    assert result["source"] is None


def test_convert_axiom_with_empty_statement():
    """Test axiom conversion with empty statement (validation error case)."""
    refined = {
        "label": "axiom-incomplete",
        "name": "Incomplete Axiom",
        "statement": "",  # Empty
        "mathematical_expression": "",  # Empty
    }

    result = convert_axiom(refined)

    # Conversion should succeed (validation happens later)
    assert result["label"] == "axiom-incomplete"
    assert result["statement"] == ""
    assert result["mathematical_expression"] == ""


# =============================================================================
# PROPERTY/ATTRIBUTE CONVERSION TESTS
# =============================================================================


def test_convert_property_to_attribute_with_source_location():
    """
    REGRESSION TEST: Ensure property conversion checks source_location first.
    """
    prop = {
        "label": "prop-lipschitz",
        "expression": "L < ∞",
        "object_label": "obj-operator",
        "established_by": "thm-1",
        "source_location": {  # Note: Uses source_location
            "document_id": "test_doc",
            "file_path": "test.md",
        },
    }

    result = convert_property_to_attribute(prop)

    assert result["label"] == "prop-lipschitz"
    assert result["expression"] == "L < ∞"
    assert result["object_label"] == "obj-operator"

    # CRITICAL: Source must be preserved
    assert result["source"] is not None
    assert result["source"]["document_id"] == "test_doc"


def test_convert_property_to_attribute_with_legacy_source():
    """Test property conversion with legacy 'source' field."""
    prop = {
        "label": "prop-test",
        "expression": "test",
        "object_label": "obj-test",
        "established_by": "thm-test",
        "source": {  # Legacy field
            "document_id": "legacy_doc",
            "file_path": "legacy.md",
        },
    }

    result = convert_property_to_attribute(prop)

    assert result["source"] is not None
    assert result["source"]["document_id"] == "legacy_doc"


def test_convert_property_to_attribute_prioritizes_source_location():
    """Test property conversion prioritizes source_location over source."""
    prop = {
        "label": "prop-test",
        "expression": "test",
        "object_label": "obj-test",
        "established_by": "thm-test",
        "source_location": {"document_id": "correct", "file_path": "correct.md"},
        "source": {"document_id": "wrong", "file_path": "wrong.md"},
    }

    result = convert_property_to_attribute(prop)

    # Should use source_location
    assert result["source"]["document_id"] == "correct"


# =============================================================================
# THEOREM CONVERSION TESTS
# =============================================================================


def test_convert_theorem_with_source_location():
    """Test theorem conversion with source_location field."""
    refined = {
        "label": "thm-convergence",
        "name": "Convergence Theorem",
        "statement_type": "theorem",
        "input_objects": ["obj-system"],
        "input_axioms": ["axiom-bounded"],
        "properties_added": [],
        "source_location": {
            "document_id": "04_convergence",
            "file_path": "docs/source/1_euclidean_gas/04_convergence.md",
        },
    }

    result = convert_theorem(refined)

    assert result["label"] == "thm-convergence"
    assert result["name"] == "Convergence Theorem"
    assert result["source"] is not None
    assert result["source"]["document_id"] == "04_convergence"


def test_convert_theorem_converts_properties_to_attributes():
    """Test theorem conversion renames properties_added to attributes_added."""
    refined = {
        "label": "thm-test",
        "name": "Test",
        "properties_added": [
            {
                "label": "prop-lipschitz",
                "expression": "L < ∞",
                "object_label": "obj-test",
                "established_by": "thm-test",
            }
        ],
    }

    result = convert_theorem(refined)

    # Check renaming happened
    assert "attributes_added" in result
    assert len(result["attributes_added"]) == 1
    assert result["attributes_added"][0]["label"] == "prop-lipschitz"


def test_convert_theorem_converts_properties_required():
    """Test theorem conversion renames properties_required to attributes_required."""
    refined = {
        "label": "thm-test",
        "name": "Test",
        "properties_required": {"obj-system": ["prop-bounded", "prop-lipschitz"]},
    }

    result = convert_theorem(refined)

    # Check structure preserved with new name
    assert "attributes_required" in result
    assert result["attributes_required"]["obj-system"] == ["prop-bounded", "prop-lipschitz"]


# =============================================================================
# DEFINITION TO OBJECT CONVERSION TESTS
# =============================================================================


def test_convert_definition_to_object_with_def_prefix():
    """Test definition conversion with def- prefix."""
    refined = {
        "label": "def-walker",
        "name": "Walker",
        "formal_statement": "w = (x, v, s)",
        "chapter": "1_euclidean_gas",
        "document": "01_fragile_gas_framework",
        "source_location": {"document_id": "01_fragile_gas_framework", "file_path": "test.md"},
    }

    result = convert_definition_to_object(refined)

    assert result is not None
    assert result["label"] == "obj-walker"  # Converted to obj-
    assert result["definition_label"] == "def-walker"  # Original preserved
    assert result["name"] == "Walker"
    assert result["source"] is not None


def test_convert_definition_to_object_handles_double_obj_prefix():
    """Test definition conversion handles double obj-obj- prefix."""
    refined = {
        "label": "obj-obj-walker",  # Double prefix
        "name": "Walker",
        "formal_statement": "w = (x, v, s)",
    }

    result = convert_definition_to_object(refined)

    # Should remove extra prefix
    assert result["label"] == "obj-walker"


def test_convert_definition_to_object_standalone_object():
    """Test conversion of standalone object (not a definition)."""
    refined = {
        "label": "obj-walker",  # Already has obj- prefix
        "name": "Walker",
        "formal_statement": "w = (x, v, s)",
    }

    result = convert_definition_to_object(refined)

    assert result["label"] == "obj-walker"
    # Should NOT have definition_label (not from definition)
    assert "definition_label" not in result


def test_convert_definition_to_object_missing_label():
    """Test definition conversion with missing label."""
    refined = {"name": "Walker", "formal_statement": "w = (x, v, s)"}

    result = convert_definition_to_object(refined)

    # Should return None and log warning
    assert result is None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_full_axiom_conversion_pipeline():
    """Test complete axiom conversion from refined_data to pipeline_data."""
    # Simulate enriched axiom file from refined_data
    refined_axiom = {
        "label": "axiom-rescale-function",
        "name": "Axiom of a Well-Behaved Rescale Function",
        "entity_type": "axiom",
        "statement_type": "axiom",
        "statement": "Any function g_A: ℝ → ℝ>0 chosen for the rescale transformation must satisfy...",
        "natural_language_statement": "Any function chosen for the rescale transformation must be...",
        "mathematical_expression": "g'_A(z) >= 0 ∀ z ∈ ℝ",
        "chapter": "1_euclidean_gas",
        "document": "01_fragile_gas_framework",
        "description": "This axiom establishes the fundamental properties...",
        "assumptions_hypotheses": [
            {
                "property": "C1 Smoothness",
                "description": "The function must be continuously differentiable...",
            }
        ],
        "source_location": {
            "document_id": "01_fragile_gas_framework",
            "file_path": "docs/source/1_euclidean_gas/01_fragile_gas_framework.md",
            "section": "9. Rescale Transformation",
            "directive_label": "def-axiom-rescale-function",
            "line_range": [1641, 1663],
        },
        "proof_status": "unproven",
    }

    # Convert to pipeline format
    pipeline_axiom = convert_axiom(refined_axiom)

    # Verify all critical fields preserved
    assert pipeline_axiom["label"] == "axiom-rescale-function"
    assert pipeline_axiom["name"] == "Axiom of a Well-Behaved Rescale Function"
    assert pipeline_axiom["statement"] != ""  # Not empty!
    assert pipeline_axiom["mathematical_expression"] != ""  # Not empty!
    assert pipeline_axiom["chapter"] == "1_euclidean_gas"
    assert pipeline_axiom["document"] == "01_fragile_gas_framework"

    # Verify source preserved
    assert pipeline_axiom["source"] is not None
    assert pipeline_axiom["source"]["document_id"] == "01_fragile_gas_framework"
    assert (
        pipeline_axiom["source"]["file_path"]
        == "docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
    )
    assert pipeline_axiom["source"]["section"] == "9. Rescale Transformation"

    # Verify pipeline format
    assert "foundational_framework" in pipeline_axiom
    assert isinstance(pipeline_axiom["source"], dict)


def test_conversion_preserves_all_source_metadata():
    """Test that conversion preserves all source metadata fields."""
    source_with_all_fields = {
        "document_id": "test_doc",
        "file_path": "test.md",
        "section": "Test Section",
        "directive_label": "test-label",
        "equation": "eq:test",
        "line_range": [10, 20],
        "url_fragment": "#test-section",
    }

    refined = {
        "label": "axiom-test",
        "name": "Test",
        "statement": "Test",
        "mathematical_expression": "x = y",
        "source_location": source_with_all_fields,
    }

    result = convert_axiom(refined)

    # All source fields preserved
    assert result["source"]["document_id"] == "test_doc"
    assert result["source"]["file_path"] == "test.md"
    assert result["source"]["section"] == "Test Section"
    assert result["source"]["directive_label"] == "test-label"
    assert result["source"]["equation"] == "eq:test"
    assert result["source"]["line_range"] == [10, 20]
    assert result["source"]["url_fragment"] == "#test-section"


# =============================================================================
# ERROR CASE TESTS
# =============================================================================


def test_axiom_conversion_with_minimal_data():
    """Test axiom conversion with only required fields."""
    refined = {"label": "axiom-minimal"}

    result = convert_axiom(refined)

    # Should not crash, fill defaults
    assert result["label"] == "axiom-minimal"
    assert result["statement"] == ""
    assert result["mathematical_expression"] == ""
    assert result["foundational_framework"] == "Fragile Gas Framework"


def test_theorem_conversion_with_empty_properties():
    """Test theorem conversion with empty properties lists."""
    refined = {
        "label": "thm-test",
        "name": "Test",
        "properties_added": [],
        "properties_required": {},
    }

    result = convert_theorem(refined)

    assert result["attributes_added"] == []
    assert result["attributes_required"] == {}


def test_definition_with_no_formal_statement():
    """Test definition conversion when formal_statement missing."""
    refined = {"label": "def-test", "name": "Test Object"}

    result = convert_definition_to_object(refined)

    assert result is not None
    # Should fall back to name
    assert result["mathematical_expression"] == "Test Object"


# =============================================================================
# FIELD CONSISTENCY TESTS
# =============================================================================


def test_all_conversions_use_consistent_source_handling():
    """
    CRITICAL TEST: Ensure all conversion functions use consistent source handling.

    This prevents future regressions where new conversion functions might
    forget to check source_location first.
    """
    source_location_data = {"document_id": "test", "file_path": "test.md"}

    # Test axiom
    axiom = convert_axiom({"label": "axiom-test", "source_location": source_location_data})
    assert axiom["source"] is not None
    assert axiom["source"]["document_id"] == "test"

    # Test property
    prop = convert_property_to_attribute({
        "label": "prop-test",
        "expression": "x",
        "object_label": "obj-test",
        "established_by": "thm",
        "source_location": source_location_data,
    })
    assert prop["source"] is not None
    assert prop["source"]["document_id"] == "test"

    # Test theorem
    thm = convert_theorem({
        "label": "thm-test",
        "name": "Test",
        "source_location": source_location_data,
    })
    assert thm["source"] is not None
    assert thm["source"]["document_id"] == "test"

    # Test definition/object
    obj = convert_definition_to_object({
        "label": "def-test",
        "name": "Test",
        "source_location": source_location_data,
    })
    assert obj is not None
    assert obj["source"] is not None
    assert obj["source"]["document_id"] == "test"
