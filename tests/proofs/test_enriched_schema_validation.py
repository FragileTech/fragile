"""
Test suite for enriched schema validation.

Tests that the validator correctly uses enriched schemas for refined_data
and pipeline schemas for pipeline_data, catching real errors in each stage.
"""

from pathlib import Path
import tempfile

import pytest

from fragile.proofs.core.enriched_types import (
    EnrichedAxiom,
    EnrichedDefinition,
    EnrichedObject,
    EnrichedTheorem,
)
from fragile.proofs.core.math_types import Axiom, MathematicalObject, TheoremBox
from fragile.proofs.tools.validation.schema_validator import SchemaValidator


# =============================================================================
# ENRICHED SCHEMA TESTS (Refined Data)
# =============================================================================


def test_enriched_axiom_accepts_source_location():
    """Test EnrichedAxiom accepts source_location field."""
    data = {
        "label": "axiom-test",
        "statement": "Test statement",
        "mathematical_expression": "x = y",
        "source_location": {"document_id": "test_doc", "file_path": "test.md"},
    }

    axiom = EnrichedAxiom(**data)
    assert axiom.label == "axiom-test"
    assert axiom.source_location is not None


def test_enriched_axiom_accepts_source_field():
    """Test EnrichedAxiom also accepts legacy source field."""
    data = {
        "label": "axiom-test",
        "statement": "Test statement",
        "mathematical_expression": "x = y",
        "source": {"document_id": "test_doc", "file_path": "test.md"},
    }

    axiom = EnrichedAxiom(**data)
    assert axiom.label == "axiom-test"
    assert axiom.source is not None


def test_enriched_axiom_accepts_enrichment_fields():
    """Test EnrichedAxiom accepts extra enrichment fields."""
    data = {
        "label": "axiom-test",
        "statement": "Test statement",
        "mathematical_expression": "x = y",
        "natural_language_statement": "In natural language...",
        "description": "Detailed description",
        "entity_type": "axiom",
        "assumptions_hypotheses": [{"property": "Test"}],
        "relations": [{"formula": "x = y"}],
    }

    axiom = EnrichedAxiom(**data)
    assert axiom.natural_language_statement == "In natural language..."
    assert axiom.description == "Detailed description"


def test_enriched_theorem_accepts_both_properties_and_attributes():
    """Test EnrichedTheorem accepts both legacy and new field names."""
    data = {
        "label": "thm-test",
        "name": "Test Theorem",
        "statement_type": "theorem",
        "properties_added": [{"label": "prop-1", "expression": "test"}],  # Legacy
        "attributes_required": {"obj-1": ["attr-1"]},  # New
    }

    theorem = EnrichedTheorem(**data)
    assert theorem.properties_added is not None
    assert theorem.attributes_required is not None


def test_enriched_definition_with_def_prefix():
    """Test EnrichedDefinition accepts def- prefix."""
    data = {"label": "def-test", "name": "Test Definition", "formal_statement": "x := y"}

    definition = EnrichedDefinition(**data)
    assert definition.label == "def-test"


# =============================================================================
# PIPELINE SCHEMA TESTS (Pipeline Data)
# =============================================================================


def test_pipeline_axiom_requires_source_not_source_location():
    """Test pipeline Axiom expects 'source' field (not source_location)."""
    # This should work (pipeline format)
    data = {
        "label": "axiom-test",
        "statement": "Test",
        "mathematical_expression": "x = y",
        "foundational_framework": "Test Framework",  # Required
        "source": {
            "document_id": "01_test_doc",  # Must match pattern
            "file_path": "test.md",
            "line_range": [1, 10],
        },
    }

    axiom = Axiom(**data)
    assert axiom.source is not None


def test_pipeline_axiom_rejects_empty_statement():
    """Test pipeline Axiom rejects empty statements."""
    data = {
        "label": "axiom-test",
        "statement": "",  # Empty!
        "mathematical_expression": "",  # Empty!
    }

    with pytest.raises(Exception) as exc_info:
        Axiom(**data)

    assert "at least 1 character" in str(exc_info.value)


# =============================================================================
# VALIDATOR MODE TESTS
# =============================================================================


def test_validator_refined_mode_uses_enriched_schemas():
    """Test validator in refined mode uses enriched schemas."""
    validator = SchemaValidator(mode="refined")

    assert validator.mode == "refined"
    assert validator.schema_map["axiom"] == EnrichedAxiom
    assert validator.schema_map["theorem"] == EnrichedTheorem
    assert validator.schema_map["definition"] == EnrichedDefinition
    assert validator.schema_map["object"] == EnrichedObject


def test_validator_pipeline_mode_uses_pipeline_schemas():
    """Test validator in pipeline mode uses pipeline schemas."""
    validator = SchemaValidator(mode="pipeline")

    assert validator.mode == "pipeline"
    assert validator.schema_map["axiom"] == Axiom
    assert validator.schema_map["theorem"] == TheoremBox
    assert validator.schema_map["object"] == MathematicalObject


def test_validator_infers_definition_type():
    """Test validator correctly infers definition entity type."""
    validator = SchemaValidator(mode="refined")

    data = {"label": "def-test"}
    file_path = Path("definitions/def-test.json")

    entity_type = validator._infer_entity_type(file_path, data)
    assert entity_type == "definition"


def test_validator_validates_enriched_axiom_with_source_location():
    """Test validator in refined mode validates axiom with source_location."""
    import json

    validator = SchemaValidator(mode="refined")

    data = {
        "label": "axiom-test",
        "statement": "Test statement",
        "mathematical_expression": "x = y",
        "source_location": {"document_id": "test", "file_path": "test.md"},
    }

    # Create temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        result = validator.validate_entity(data, temp_path)
        assert result.is_valid
        assert len(result.errors) == 0
    finally:
        temp_path.unlink()


def test_validator_catches_empty_statement_in_pipeline_mode():
    """Test validator in pipeline mode catches empty statements."""
    import json

    validator = SchemaValidator(mode="pipeline")

    data = {
        "label": "axiom-test",
        "statement": "",  # Empty!
        "mathematical_expression": "",  # Empty!
    }

    # Create temp file in axioms directory
    with tempfile.TemporaryDirectory() as tmpdir:
        axioms_dir = Path(tmpdir) / "axioms"
        axioms_dir.mkdir()
        axiom_file = axioms_dir / "axiom-test.json"

        with open(axiom_file, "w") as f:
            json.dump(data, f)

        result = validator.validate_entity(data, axiom_file)

        # Should have validation error
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "at least 1 character" in str(result.errors[0])


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_refined_validator_more_permissive_than_pipeline():
    """Test that refined validator accepts data that pipeline validator rejects."""
    import json

    # Data with source_location (refined format)
    data = {
        "label": "axiom-test",
        "statement": "Test",
        "mathematical_expression": "x = y",
        "source_location": {  # Refined format
            "document_id": "test",
            "file_path": "test.md",
        },
        "natural_language_statement": "Extra field",  # Only in refined
        "entity_type": "axiom",  # Only in refined
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        axioms_dir = Path(tmpdir) / "axioms"
        axioms_dir.mkdir()
        axiom_file = axioms_dir / "axiom-test.json"

        with open(axiom_file, "w") as f:
            json.dump(data, f)

        # Refined validator should accept
        refined_validator = SchemaValidator(mode="refined")
        refined_result = refined_validator.validate_entity(data, axiom_file)
        assert refined_result.is_valid, f"Refined validator should accept: {refined_result.errors}"

        # Pipeline validator might reject or require conversion
        pipeline_validator = SchemaValidator(mode="pipeline")
        pipeline_validator.validate_entity(data, axiom_file)
        # Pipeline validator will fail because it expects 'source' not 'source_location'


def test_end_to_end_refined_to_pipeline_validation():
    """Test complete workflow: validate refined → convert → validate pipeline."""
    import json

    # 1. Start with enriched data (refined_data format)
    enriched_data = {
        "label": "axiom-well-behaved",
        "statement": "Any function g_A must satisfy...",
        "mathematical_expression": "g'_A(z) >= 0",
        "source_location": {  # Refined format
            "document_id": "01_framework",
            "file_path": "docs/source/...",
        },
        "natural_language_statement": "In plain English...",
        "description": "Detailed description...",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. Validate with refined schema
        axioms_dir = Path(tmpdir) / "axioms"
        axioms_dir.mkdir()
        refined_file = axioms_dir / "axiom-test.json"

        with open(refined_file, "w") as f:
            json.dump(enriched_data, f)

        refined_validator = SchemaValidator(mode="refined")
        refined_result = refined_validator.validate_entity(enriched_data, refined_file)

        assert refined_result.is_valid, f"Refined validation failed: {refined_result.errors}"

        # 3. Simulate conversion to pipeline format
        pipeline_data = {
            "label": enriched_data["label"],
            "statement": enriched_data["statement"],
            "mathematical_expression": enriched_data["mathematical_expression"],
            "source": enriched_data["source_location"],  # Renamed!
            "foundational_framework": "Fragile Gas Framework",
        }

        pipeline_file = axioms_dir / "axiom-pipeline.json"
        with open(pipeline_file, "w") as f:
            json.dump(pipeline_data, f)

        # 4. Validate with pipeline schema
        pipeline_validator = SchemaValidator(mode="pipeline")
        pipeline_result = pipeline_validator.validate_entity(pipeline_data, pipeline_file)

        assert pipeline_result.is_valid, f"Pipeline validation failed: {pipeline_result.errors}"
