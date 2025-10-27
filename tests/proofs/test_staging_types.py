"""
Tests for Staging Types (Raw Extraction Models).

Tests the new RawAxiom staging type and updates to StagingDocument.
"""

import pytest
from pydantic import ValidationError

from fragile.proofs.staging_types import (
    RawAxiom,
    RawDefinition,
    RawTheorem,
    RawProof,
    StagingDocument,
)


class TestRawAxiom:
    """Tests for RawAxiom staging type."""

    def test_raw_axiom_creation_minimal(self):
        """Test creating RawAxiom with minimal required fields."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-test",
            name="Test Axiom",
            core_assumption_text="The fundamental assumption",
            source_section="§1"
        )

        assert axiom.temp_id == "raw-axiom-001"
        assert axiom.label_text == "axiom-test"
        assert axiom.name == "Test Axiom"
        assert axiom.core_assumption_text == "The fundamental assumption"
        assert axiom.parameters_text == []  # Default
        assert axiom.condition_text == ""  # Default
        assert axiom.failure_mode_analysis_text is None  # Default
        assert axiom.source_section == "§1"

    def test_raw_axiom_creation_full(self):
        """Test creating RawAxiom with all fields."""
        axiom = RawAxiom(
            temp_id="raw-axiom-002",
            label_text="axiom-bounded-displacement",
            name="Axiom of Bounded Displacement",
            core_assumption_text="All walkers satisfy |x(t+Δt) - x(t)| ≤ ε√Δt",
            parameters_text=["ε > 0", "Δt"],
            condition_text="When Δt < ε²",
            failure_mode_analysis_text="Unphysical teleportation behavior",
            source_section="§1.1"
        )

        assert axiom.temp_id == "raw-axiom-002"
        assert len(axiom.parameters_text) == 2
        assert axiom.parameters_text[0] == "ε > 0"
        assert axiom.condition_text == "When Δt < ε²"
        assert axiom.failure_mode_analysis_text == "Unphysical teleportation behavior"

    def test_raw_axiom_temp_id_validation(self):
        """Test temp_id pattern validation."""
        # Valid patterns
        valid_ids = ["raw-axiom-1", "raw-axiom-001", "raw-axiom-999"]
        for temp_id in valid_ids:
            axiom = RawAxiom(
                temp_id=temp_id,
                label_text="test",
                name="Test",
                core_assumption_text="Test",
                source_section="§1"
            )
            assert axiom.temp_id == temp_id

        # Invalid patterns should raise ValidationError
        invalid_ids = ["raw-def-1", "axiom-1", "raw-axiom-", "raw-axiom-abc"]
        for temp_id in invalid_ids:
            with pytest.raises(ValidationError):
                RawAxiom(
                    temp_id=temp_id,
                    label_text="test",
                    name="Test",
                    core_assumption_text="Test",
                    source_section="§1"
                )

    def test_raw_axiom_immutability(self):
        """Test that RawAxiom is immutable (frozen=True)."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="test",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )

        # Should not be able to modify fields
        with pytest.raises(ValidationError):
            axiom.name = "Modified"

    def test_raw_axiom_empty_parameters(self):
        """Test axiom with empty parameters list."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="test",
            name="Test",
            core_assumption_text="Test",
            parameters_text=[],
            source_section="§1"
        )
        assert axiom.parameters_text == []

    def test_raw_axiom_model_dump(self):
        """Test serialization to dict."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-test",
            name="Test Axiom",
            core_assumption_text="Test assumption",
            parameters_text=["param1"],
            condition_text="condition",
            failure_mode_analysis_text="failure mode",
            source_section="§1"
        )

        data = axiom.model_dump()
        assert data["temp_id"] == "raw-axiom-001"
        assert data["label_text"] == "axiom-test"
        assert data["parameters_text"] == ["param1"]
        assert data["condition_text"] == "condition"
        assert data["failure_mode_analysis_text"] == "failure mode"


class TestStagingDocumentWithAxioms:
    """Tests for StagingDocument with axioms field."""

    def test_staging_document_with_axioms(self):
        """Test StagingDocument includes axioms field."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="test",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )

        doc = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[axiom],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        assert len(doc.axioms) == 1
        assert doc.axioms[0].temp_id == "raw-axiom-001"

    def test_staging_document_total_entities_includes_axioms(self):
        """Test that total_entities includes axioms in count."""
        axiom1 = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="test1",
            name="Test 1",
            core_assumption_text="Test",
            source_section="§1"
        )
        axiom2 = RawAxiom(
            temp_id="raw-axiom-002",
            label_text="test2",
            name="Test 2",
            core_assumption_text="Test",
            source_section="§1"
        )

        definition = RawDefinition(
            temp_id="raw-def-001",
            term_being_defined="Test Term",
            full_text="Test definition",
            source_section="§1"
        )

        theorem = RawTheorem(
            temp_id="raw-thm-001",
            label_text="thm-test",
            statement_type="theorem",
            full_statement_text="Test statement",
            source_section="§1"
        )

        doc = StagingDocument(
            section_id="§1",
            definitions=[definition],
            theorems=[theorem],
            proofs=[],
            axioms=[axiom1, axiom2],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        # Should count: 1 definition + 1 theorem + 2 axioms = 4
        assert doc.total_entities == 4

    def test_staging_document_get_summary_includes_axioms(self):
        """Test that get_summary() includes axioms in output."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="test",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )

        doc = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[axiom],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        summary = doc.get_summary()
        assert "Axioms: 1" in summary
        assert "§1" in summary

    def test_staging_document_empty_axioms_list(self):
        """Test StagingDocument with empty axioms list."""
        doc = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        assert doc.axioms == []
        assert doc.total_entities == 0

    def test_staging_document_model_dump_includes_axioms(self):
        """Test serialization includes axioms."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="test",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )

        doc = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[axiom],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        data = doc.model_dump()
        assert "axioms" in data
        assert len(data["axioms"]) == 1
        assert data["axioms"][0]["temp_id"] == "raw-axiom-001"

    def test_staging_document_model_validate_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="test",
            name="Test",
            core_assumption_text="Test",
            parameters_text=["param1", "param2"],
            condition_text="condition",
            failure_mode_analysis_text="failure",
            source_section="§1"
        )

        doc = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[axiom],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        # Serialize
        data = doc.model_dump()

        # Deserialize
        doc_restored = StagingDocument.model_validate(data)

        assert len(doc_restored.axioms) == 1
        assert doc_restored.axioms[0].temp_id == "raw-axiom-001"
        assert doc_restored.axioms[0].parameters_text == ["param1", "param2"]
        assert doc_restored.axioms[0].condition_text == "condition"
        assert doc_restored.axioms[0].failure_mode_analysis_text == "failure"
