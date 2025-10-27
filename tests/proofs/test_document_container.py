"""
Tests for Document Container.

Tests MathematicalDocument and EnrichedEntities containers.
"""

import pytest

from fragile.proofs.llm import (
    MathematicalDocument,
    EnrichedEntities,
)
from fragile.proofs.core import (
    Axiom,
    DefinitionBox,
    TheoremBox,
    ProofBox,
    TheoremOutputType,
    MathematicalObject,
    ObjectType,
    Parameter,
    ParameterType,
)
from fragile.proofs.staging_types import (
    RawAxiom,
    RawDefinition,
    RawTheorem,
    StagingDocument,
)
from fragile.proofs.sympy import DualStatement


class TestEnrichedEntities:
    """Tests for EnrichedEntities container."""

    def test_enriched_entities_creation_empty(self):
        """Test creating empty EnrichedEntities."""
        entities = EnrichedEntities()

        assert entities.definitions == {}
        assert entities.theorems == {}
        assert entities.axioms == {}
        assert entities.proofs == {}
        assert entities.objects == {}
        assert entities.parameters == {}
        assert entities.total_entities == 0

    def test_enriched_entities_with_content(self):
        """Test EnrichedEntities with content."""
        # Create a simple definition
        definition = DefinitionBox(
            label="def-test",
            term="Test Term",
            formal_statement=DualStatement(
                latex="Test definition",
                natural_language="Test",
                sympy_expr=None,
                paper_context=None
            )
        )

        # Create a simple theorem
        theorem = TheoremBox(
            label="thm-test",
            name="Test Theorem",
            output_type=TheoremOutputType.PROPERTY
        )

        # Create enriched entities
        entities = EnrichedEntities(
            definitions={"def-test": definition},
            theorems={"thm-test": theorem}
        )

        assert len(entities.definitions) == 1
        assert len(entities.theorems) == 1
        assert entities.total_entities == 2

    def test_enriched_entities_get_summary(self):
        """Test get_summary() method."""
        # Create some entities
        definition = DefinitionBox(
            label="def-test",
            term="Test",
            formal_statement=DualStatement(
                latex="Test",
                natural_language="Test",
                sympy_expr=None,
                paper_context=None
            )
        )

        entities = EnrichedEntities(
            definitions={"def-test": definition}
        )

        summary = entities.get_summary()
        assert "Enriched Entities" in summary
        assert "Definitions: 1" in summary

    def test_enriched_entities_immutability(self):
        """Test that EnrichedEntities is immutable."""
        entities = EnrichedEntities()

        # Should not be able to modify
        with pytest.raises(Exception):  # ValidationError or similar
            entities.definitions = {"test": None}


class TestMathematicalDocument:
    """Tests for MathematicalDocument container."""

    def test_mathematical_document_creation_minimal(self):
        """Test creating MathematicalDocument with minimal fields."""
        doc = MathematicalDocument(
            document_id="test_doc",
            chapter="1_euclidean_gas"
        )

        assert doc.document_id == "test_doc"
        assert doc.chapter == "1_euclidean_gas"
        assert doc.file_path is None
        assert len(doc.staging_documents) == 0
        assert doc.total_raw_entities == 0
        assert doc.total_enriched_entities == 0

    def test_mathematical_document_add_staging_document(self):
        """Test adding staging documents."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        # Create a staging document
        raw_axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="test",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )

        staging = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[raw_axiom],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        # Add staging document
        doc = doc.add_staging_document(staging)

        assert len(doc.staging_documents) == 1
        assert doc.total_raw_entities == 1

    def test_mathematical_document_add_enriched_entities(self):
        """Test adding enriched entities."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        # Create and add definition
        definition = DefinitionBox(
            label="def-test",
            term="Test",
            formal_statement=DualStatement(
                latex="Test",
                natural_language="Test",
                sympy_expr=None,
                paper_context=None
            )
        )
        doc = doc.add_enriched_definition(definition)

        # Create and add theorem
        theorem = TheoremBox(
            label="thm-test",
            name="Test",
            output_type=TheoremOutputType.PROPERTY
        )
        doc = doc.add_enriched_theorem(theorem)

        # Create and add axiom
        axiom = Axiom(
            label="axiom-test",
            statement="Test axiom",
            mathematical_expression="Test",
            foundational_framework="Test"
        )
        doc = doc.add_enriched_axiom(axiom)

        assert doc.total_enriched_entities == 3
        assert len(doc.enriched.definitions) == 1
        assert len(doc.enriched.theorems) == 1
        assert len(doc.enriched.axioms) == 1

    def test_mathematical_document_enrichment_rate(self):
        """Test enrichment rate calculation."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        # Add raw entities (10 total)
        raw_axioms = [
            RawAxiom(
                temp_id=f"raw-axiom-{i:03d}",
                label_text=f"axiom-{i}",
                name=f"Axiom {i}",
                core_assumption_text="Test",
                source_section="§1"
            )
            for i in range(10)
        ]

        staging = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=raw_axioms,
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )
        doc = doc.add_staging_document(staging)

        # Add enriched entities (5 total)
        for i in range(5):
            axiom = Axiom(
                label=f"axiom-{i}",
                statement="Test",
                mathematical_expression="Test",
                foundational_framework="Test"
            )
            doc = doc.add_enriched_axiom(axiom)

        # Enrichment rate should be 50%
        assert doc.total_raw_entities == 10
        assert doc.total_enriched_entities == 5
        assert doc.enrichment_rate == 50.0

    def test_mathematical_document_get_summary(self):
        """Test get_summary() method."""
        doc = MathematicalDocument(
            document_id="01_framework",
            chapter="1_euclidean_gas",
            file_path="/path/to/file.md"
        )

        # Add some data
        staging = StagingDocument(
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
        doc = doc.add_staging_document(staging)

        summary = doc.get_summary()
        assert "01_framework" in summary
        assert "1_euclidean_gas" in summary
        assert "/path/to/file.md" in summary
        assert "Raw Entities" in summary
        assert "Enriched Entities" in summary

    def test_mathematical_document_lookup_methods(self):
        """Test entity lookup methods."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        # Add entities
        definition = DefinitionBox(
            label="def-walker",
            term="Walker",
            formal_statement=DualStatement(
                latex="Test",
                natural_language="Test",
                sympy_expr=None,
                paper_context=None
            )
        )
        doc = doc.add_enriched_definition(definition)

        theorem = TheoremBox(
            label="thm-convergence",
            name="Convergence",
            output_type=TheoremOutputType.ASYMPTOTIC
        )
        doc = doc.add_enriched_theorem(theorem)

        axiom = Axiom(
            label="axiom-bounded",
            statement="Test",
            mathematical_expression="Test",
            foundational_framework="Test"
        )
        doc = doc.add_enriched_axiom(axiom)

        # Test lookups
        assert doc.get_definition("def-walker") is not None
        assert doc.get_definition("def-walker").term == "Walker"

        assert doc.get_theorem("thm-convergence") is not None
        assert doc.get_theorem("thm-convergence").name == "Convergence"

        assert doc.get_axiom("axiom-bounded") is not None

        # Non-existent entities
        assert doc.get_definition("def-nonexistent") is None
        assert doc.get_theorem("thm-nonexistent") is None

    def test_mathematical_document_add_multiple_staging(self):
        """Test adding multiple staging documents."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        # Add three staging documents
        for i in range(3):
            raw_def = RawDefinition(
                temp_id=f"raw-def-{i:03d}",
                label_text=f"def-{i}",
                term=f"Term {i}",
                statement_text="Test",
                source_section=f"§{i}"
            )
            staging = StagingDocument(
                section_id=f"§{i}",
                definitions=[raw_def],
                theorems=[],
                proofs=[],
                axioms=[],
                citations=[],
                equations=[],
                parameters=[],
                remarks=[]
            )
            doc = doc.add_staging_document(staging)

        assert len(doc.staging_documents) == 3
        assert doc.total_raw_entities == 3

    def test_mathematical_document_immutability(self):
        """Test that MathematicalDocument uses immutable pattern."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        # Adding entities returns new instance
        definition = DefinitionBox(
            label="def-test",
            term="Test",
            formal_statement=DualStatement(
                latex="Test",
                natural_language="Test",
                sympy_expr=None,
                paper_context=None
            )
        )

        doc2 = doc.add_enriched_definition(definition)

        # Original doc unchanged
        assert doc.total_enriched_entities == 0
        # New doc has entity
        assert doc2.total_enriched_entities == 1

    def test_mathematical_document_add_enriched_object(self):
        """Test adding enriched mathematical object."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        obj = MathematicalObject(
            label="obj-walker",
            name="Walker",
            mathematical_expression="w = (x, v, s)",
            object_type=ObjectType.TUPLE
        )

        doc = doc.add_enriched_object(obj)

        assert len(doc.enriched.objects) == 1
        assert doc.get_object("obj-walker") is not None

    def test_mathematical_document_add_enriched_parameter(self):
        """Test adding enriched parameter."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        param = Parameter(
            label="param-epsilon",
            name="Epsilon",
            symbol="ε",
            parameter_type=ParameterType.REGULARIZATION
        )

        doc = doc.add_enriched_parameter(param)

        assert len(doc.enriched.parameters) == 1
        assert doc.get_parameter("param-epsilon") is not None

    def test_mathematical_document_add_enriched_proof(self):
        """Test adding enriched proof."""
        doc = MathematicalDocument(
            document_id="test",
            chapter="1_euclidean_gas"
        )

        from fragile.proofs.core import (
            ProofStep,
            ProofStepType,
            ProofStepStatus,
            DirectDerivation,
        )

        step = ProofStep(
            step_id="step-1",
            step_number=1,
            description="Test step",
            justification="Test",
            step_type=ProofStepType.DIRECT,
            status=ProofStepStatus.EXPANDED,
            inputs=[],
            outputs=[],
            derivation=DirectDerivation(
                from_properties=[],
                conclusion="Test",
                reasoning="Test"
            )
        )

        proof = ProofBox(
            proof_id="proof-test",
            label="Proof of thm-test",
            proves="thm-test",
            inputs=[],
            outputs=[],
            strategy="Test",
            steps=[step]
        )

        doc = doc.add_enriched_proof(proof)

        assert len(doc.enriched.proofs) == 1
