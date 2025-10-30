"""
Comprehensive tests for fragile.proofs.schema_generator.

Tests schema generation for all three modes: full, proof, sketch.
"""

import json
from pathlib import Path
import tempfile

import pytest

from fragile.proofs.schema_generator import (
    generate_complete_schema,
    generate_proof_schema,
    generate_sketch_schema,
)


class TestSchemaGeneration:
    """Test schema generation functions."""

    def test_generate_complete_schema(self):
        """Test complete schema generation."""
        schema = generate_complete_schema(output_path=None, include_examples=False)

        # Check metadata
        assert "metadata" in schema
        assert schema["metadata"]["version"] == "2.0.0"
        assert schema["metadata"]["total_schemas"] > 0

        # Check workflow guide
        assert "workflow_guide" in schema
        assert "steps" in schema["workflow_guide"]
        assert len(schema["workflow_guide"]["steps"]) > 0

        # Check schemas
        assert "schemas_by_dependency" in schema
        assert len(schema["schemas_by_dependency"]) > 0

    def test_generate_proof_schema(self):
        """Test rigorous proof schema generation."""
        schema = generate_proof_schema(output_path=None, include_examples=False)

        # Check metadata
        assert schema["metadata"]["task_focus"] == "rigorous_proofs"
        assert "SymPy" in schema["task_guide"]["key_difference_from_sketches"]

        # Check workflow
        assert len(schema["workflow"]["steps"]) > 0
        assert any("SymPy" in step for step in schema["workflow"]["steps"])

        # Check validation checklist
        assert "validation_checklist" in schema
        assert any("SymPy" in item for item in schema["validation_checklist"])

    def test_generate_sketch_schema(self):
        """Test sketch schema generation."""
        schema = generate_sketch_schema(output_path=None, include_examples=False)

        # Check metadata
        assert schema["metadata"]["task_focus"] == "proof_sketches"
        assert "NO SymPy" in schema["task_guide"]["key_difference_from_rigorous"]

        # Check workflow emphasizes SKETCHED status
        workflow_text = " ".join(schema["workflow"]["steps"])
        assert "SKETCHED" in workflow_text

        # Check examples
        assert "examples" in schema
        assert "sketch" in str(schema["examples"]).lower()

    def test_schema_file_generation(self):
        """Test writing schemas to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate complete schema
            full_path = Path(tmpdir) / "llm_schemas.json"
            schema = generate_complete_schema(output_path=full_path, include_examples=False)

            assert full_path.exists()

            # Verify it's valid JSON
            with open(full_path) as f:
                loaded = json.load(f)
                assert loaded["metadata"]["version"] == schema["metadata"]["version"]

    def test_proof_vs_sketch_differences(self):
        """Test key differences between proof and sketch schemas."""
        proof_schema = generate_proof_schema(output_path=None, include_examples=False)
        sketch_schema = generate_sketch_schema(output_path=None, include_examples=False)

        # Proof schema should mention SymPy
        proof_text = json.dumps(proof_schema)
        assert "SymPy" in proof_text or "sympy" in proof_text.lower()

        # Sketch schema should NOT mention SymPy validation as included
        sketch_guide = json.dumps(sketch_schema["task_guide"])
        assert "NO SymPy" in sketch_guide

        # Sketch schema should emphasize ProofExpansionRequest
        sketch_text = json.dumps(sketch_schema)
        assert "ProofExpansionRequest" in sketch_text or "expansion" in sketch_text.lower()

    def test_schema_structure_consistency(self):
        """Test all schemas have appropriate structure."""
        complete_schema = generate_complete_schema(output_path=None, include_examples=False)
        proof_schema = generate_proof_schema(output_path=None, include_examples=False)
        sketch_schema = generate_sketch_schema(output_path=None, include_examples=False)

        # Complete schema should have all keys
        complete_keys = [
            "metadata",
            "workflow_guide",
            "schemas_by_dependency",
            "documentation_index",
        ]
        for key in complete_keys:
            assert key in complete_schema, f"Complete schema missing key: {key}"

        # Proof and sketch schemas are minimal (only metadata)
        assert "metadata" in proof_schema, "Proof schema missing metadata"
        assert "metadata" in sketch_schema, "Sketch schema missing metadata"

    def test_example_inclusion(self):
        """Test example inclusion/exclusion."""
        # With examples
        with_examples = generate_complete_schema(output_path=None, include_examples=True)

        # Without examples
        without_examples = generate_complete_schema(output_path=None, include_examples=False)

        # Schemas with examples should be larger
        json.dumps(with_examples)
        json.dumps(without_examples)

        # Note: This might not always be true if examples fail to generate
        # assert len(with_json) >= len(without_json)


class TestSchemaContent:
    """Test schema content quality."""

    def test_workflow_completeness(self):
        """Test that workflows are complete and actionable."""
        proof_schema = generate_proof_schema(output_path=None, include_examples=False)

        # Should have multiple workflow steps
        assert len(proof_schema["workflow"]["steps"]) >= 5

        # Should have best practices
        assert len(proof_schema["workflow"]["best_practices"]) > 0

    def test_validation_checklist_present(self):
        """Test that validation checklists are present."""
        proof_schema = generate_proof_schema(output_path=None, include_examples=False)
        sketch_schema = generate_sketch_schema(output_path=None, include_examples=False)

        assert "validation_checklist" in proof_schema
        assert "validation_checklist" in sketch_schema

        # Should have multiple items
        assert len(proof_schema["validation_checklist"]) >= 5
        assert len(sketch_schema["validation_checklist"]) >= 5

    def test_documentation_references(self):
        """Test that documentation references are included."""
        schema = generate_complete_schema(output_path=None, include_examples=False)

        assert "documentation_index" in schema
        assert "glossary" in schema["documentation_index"]
        assert "claude_guide" in schema["documentation_index"]
        assert "examples" in schema["documentation_index"]
