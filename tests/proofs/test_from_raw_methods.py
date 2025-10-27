"""
Tests for from_raw() enrichment methods.

Tests all the from_raw() classmethods for converting staging types
to enriched models.
"""

import pytest

from fragile.proofs.core import (
    Axiom,
    AxiomaticParameter,
    DefinitionBox,
    TheoremBox,
    ProofBox,
    TheoremOutputType,
)
from fragile.proofs.staging_types import (
    RawAxiom,
    RawDefinition,
    RawTheorem,
    RawProof,
)


class TestAxiomFromRaw:
    """Tests for Axiom.from_raw() method."""

    def test_axiom_from_raw_minimal(self):
        """Test enriching axiom with minimal fields."""
        raw = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-test",
            name="Test Axiom",
            core_assumption_text="The fundamental assumption",
            source_section="§1"
        )

        axiom = Axiom.from_raw(raw, chapter="1_euclidean_gas")

        assert axiom.label == "axiom-test"
        assert axiom.name == "Test Axiom"
        assert axiom.foundational_framework == "Test Axiom"
        assert "fundamental assumption" in axiom.statement
        assert axiom.chapter == "1_euclidean_gas"
        assert axiom.parameters is None  # No parameters provided

    def test_axiom_from_raw_with_parameters(self):
        """Test enriching axiom with parameters."""
        raw = RawAxiom(
            temp_id="raw-axiom-002",
            label_text="axiom-bounded",
            name="Bounded Displacement",
            core_assumption_text="|x(t+Δt) - x(t)| ≤ ε√Δt",
            parameters_text=["ε > 0", "Δt: time step"],
            condition_text="When Δt < ε²",
            failure_mode_analysis_text="Teleportation",
            source_section="§1"
        )

        axiom = Axiom.from_raw(raw, chapter="1_euclidean_gas", document="01_framework")

        assert axiom.label == "axiom-bounded"
        assert axiom.name == "Bounded Displacement"
        assert len(axiom.parameters) == 2
        assert isinstance(axiom.parameters[0], AxiomaticParameter)
        assert axiom.parameters[0].symbol == "ε"
        assert axiom.parameters[0].description == "ε > 0"
        assert "Condition: When Δt < ε²" in axiom.statement
        assert "Failure mode: Teleportation" in axiom.statement
        assert axiom.failure_mode_analysis == "Teleportation"
        assert axiom.document == "01_framework"

    def test_axiom_from_raw_label_normalization(self):
        """Test label normalization for axioms."""
        # Label already has axiom- prefix
        raw1 = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-test",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )
        axiom1 = Axiom.from_raw(raw1)
        assert axiom1.label == "axiom-test"

        # Label has def-axiom- prefix (legacy)
        raw2 = RawAxiom(
            temp_id="raw-axiom-002",
            label_text="def-axiom-legacy",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )
        axiom2 = Axiom.from_raw(raw2)
        assert axiom2.label == "axiom-legacy"

    def test_axiom_from_raw_empty_optional_fields(self):
        """Test handling of empty optional fields."""
        raw = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-test",
            name="Test",
            core_assumption_text="Test assumption",
            parameters_text=[],
            condition_text="",
            failure_mode_analysis_text=None,
            source_section="§1"
        )

        axiom = Axiom.from_raw(raw)

        assert axiom.parameters is None  # Empty list → None
        assert "Condition:" not in axiom.statement  # Empty condition not added
        assert axiom.failure_mode_analysis is None


class TestDefinitionBoxFromRaw:
    """Tests for DefinitionBox.from_raw() method."""

    def test_definition_from_raw_minimal(self):
        """Test enriching definition with minimal fields."""
        raw = RawDefinition(
            temp_id="raw-def-001",
            term_being_defined="Walker",
            full_text="A walker is a tuple (x, v, s)",
            source_section="§1"
        )

        definition = DefinitionBox.from_raw(raw, chapter="1_euclidean_gas")

        assert definition.label == "def-walker"
        assert definition.term == "Walker"
        assert definition.formal_statement.latex == "A walker is a tuple (x, v, s)"
        assert definition.chapter == "1_euclidean_gas"
        assert definition.applies_to_object_type is None  # Requires semantic analysis
        assert definition.parameters == []

    def test_definition_from_raw_with_explanation(self):
        """Test enriching definition with full text."""
        raw = RawDefinition(
            temp_id="raw-def-001",
            term_being_defined="Test Term",
            full_text="Formal: Let X be a set",
            source_section="§1"
        )

        definition = DefinitionBox.from_raw(raw)

        assert definition.natural_language_description == "Formal: Let X be a set"
        assert definition.formal_statement.latex == "Formal: Let X be a set"

    def test_definition_from_raw_label_normalization(self):
        """Test label generation for definitions."""
        raw = RawDefinition(
            temp_id="raw-def-001",
            term_being_defined="Walker State",
            full_text="...",
            source_section="§1"
        )
        definition = DefinitionBox.from_raw(raw)
        assert definition.label == "def-walker-state"

    def test_definition_from_raw_preserves_raw_fallback(self):
        """Test that raw_fallback is set."""
        raw = RawDefinition(
            temp_id="raw-def-001",
            term_being_defined="Test",
            full_text="...",
            source_section="§1"
        )

        definition = DefinitionBox.from_raw(raw)

        assert definition.raw_fallback is not None
        assert definition.raw_fallback["temp_id"] == "raw-def-001"


class TestTheoremBoxFromRaw:
    """Tests for TheoremBox.from_raw() method."""

    def test_theorem_from_raw_minimal(self):
        """Test enriching theorem with minimal fields."""
        raw = RawTheorem(
            temp_id="raw-thm-001",
            label_text="Theorem 1.1 (Test Theorem)",
            statement_type="theorem",
            full_statement_text="The system converges",
            source_section="§3"
        )

        theorem = TheoremBox.from_raw(raw, chapter="1_euclidean_gas")

        assert theorem.label.startswith("thm-")
        assert theorem.name == "Theorem 1.1 (Test Theorem)"
        assert theorem.statement_type == "theorem"
        assert theorem.chapter == "1_euclidean_gas"
        assert theorem.natural_language_statement == "The system converges"

    def test_theorem_from_raw_output_type_detection(self):
        """Test automatic output type detection from name."""
        # Existence
        raw1 = RawTheorem(
            temp_id="raw-thm-001",
            label_text="Theorem 1.1 (Existence of Solution)",
            statement_type="theorem",
            full_statement_text="There exists a unique solution",
            source_section="§1"
        )
        thm1 = TheoremBox.from_raw(raw1)
        assert thm1.output_type == TheoremOutputType.EXISTENCE

        # Uniqueness
        raw2 = RawTheorem(
            temp_id="raw-thm-002",
            label_text="Theorem 1.2 (Uniqueness Theorem)",
            statement_type="theorem",
            full_statement_text="The solution is unique",
            source_section="§1"
        )
        thm2 = TheoremBox.from_raw(raw2)
        assert thm2.output_type == TheoremOutputType.UNIQUENESS

        # Inequality/Bound
        raw3 = RawTheorem(
            temp_id="raw-thm-003",
            label_text="Theorem 1.3 (Upper Bound Estimate)",
            statement_type="theorem",
            full_statement_text="We have |x| ≤ C",
            source_section="§1"
        )
        thm3 = TheoremBox.from_raw(raw3)
        assert thm3.output_type == TheoremOutputType.INEQUALITY

        # Convergence
        raw4 = RawTheorem(
            temp_id="raw-thm-004",
            label_text="Theorem 1.4 (Convergence Result)",
            statement_type="theorem",
            full_statement_text="The sequence converges",
            source_section="§1"
        )
        thm4 = TheoremBox.from_raw(raw4)
        assert thm4.output_type == TheoremOutputType.ASYMPTOTIC

        # Equivalence
        raw5 = RawTheorem(
            temp_id="raw-thm-005",
            label_text="Theorem 1.5 (Equivalence of Norms)",
            statement_type="theorem",
            full_statement_text="A iff B",
            source_section="§1"
        )
        thm5 = TheoremBox.from_raw(raw5)
        assert thm5.output_type == TheoremOutputType.EQUIVALENCE

    def test_theorem_from_raw_label_prefix_detection(self):
        """Test statement_type from RawTheorem.statement_type field."""
        # Theorem
        raw_thm = RawTheorem(
            temp_id="raw-thm-001",
            label_text="Theorem 1.1",
            statement_type="theorem",
            full_statement_text="...",
            source_section="§1"
        )
        thm = TheoremBox.from_raw(raw_thm)
        assert thm.statement_type == "theorem"

        # Lemma
        raw_lem = RawTheorem(
            temp_id="raw-thm-002",
            label_text="Lemma 2.3",
            statement_type="lemma",
            full_statement_text="...",
            source_section="§1"
        )
        lem = TheoremBox.from_raw(raw_lem)
        assert lem.statement_type == "lemma"

        # Proposition
        raw_prop = RawTheorem(
            temp_id="raw-thm-003",
            label_text="Proposition 3.4",
            statement_type="proposition",
            full_statement_text="...",
            source_section="§1"
        )
        prop = TheoremBox.from_raw(raw_prop)
        assert prop.statement_type == "proposition"

    def test_theorem_from_raw_with_informal_explanation(self):
        """Test handling of context before field."""
        raw = RawTheorem(
            temp_id="raw-thm-001",
            label_text="Theorem 1.1",
            statement_type="theorem",
            full_statement_text="Formal statement",
            context_before="This means X informally",
            source_section="§1"
        )

        theorem = TheoremBox.from_raw(raw)

        assert theorem.natural_language_statement == "Formal statement"

    def test_theorem_from_raw_proof_status(self):
        """Test proof status is unproven initially."""
        raw = RawTheorem(
            temp_id="raw-thm-001",
            label_text="Theorem 1.1",
            statement_type="theorem",
            full_statement_text="...",
            source_section="§1"
        )

        theorem = TheoremBox.from_raw(raw)

        assert theorem.proof_status == "unproven"
        assert theorem.proof is None


class TestProofBoxFromRaw:
    """Tests for ProofBox.from_raw() method."""

    def test_proof_from_raw_minimal(self):
        """Test enriching proof with minimal fields."""
        raw = RawProof(
            temp_id="raw-proof-001",
            proves_label_text="Theorem 1.1",
            full_body_text="We proceed by induction...",
            source_section="§3"
        )

        proof = ProofBox.from_raw(raw, proves="thm-test")

        assert proof.proof_id == "proof-test"
        assert proof.label == "Proof of thm-test"
        assert proof.proves == "thm-test"
        assert len(proof.steps) == 1
        assert proof.steps[0].description == "We proceed by induction..."
        assert proof.steps[0].status.value == "sketched"

    def test_proof_from_raw_single_step_structure(self):
        """Test that proof is stored as single sketched step."""
        raw = RawProof(
            temp_id="raw-proof-001",
            proves_label_text="Theorem 2.1",
            full_body_text="First, we establish...\n\nThen, we apply...\n\nFinally, we conclude...",
            source_section="§3"
        )

        proof = ProofBox.from_raw(raw, proves="thm-keystone")

        # Single step with full text
        assert len(proof.steps) == 1
        assert proof.steps[0].step_number == 1
        assert "First, we establish" in proof.steps[0].description
        assert "Finally, we conclude" in proof.steps[0].description

    def test_proof_from_raw_proof_id_generation(self):
        """Test proof_id generation from theorem label."""
        # Standard theorem
        raw1 = RawProof(
            temp_id="raw-proof-001",
            proves_label_text="Theorem 3.1",
            full_body_text="...",
            source_section="§1"
        )
        proof1 = ProofBox.from_raw(raw1, proves="thm-convergence")
        assert proof1.proof_id == "proof-convergence"

        # Lemma
        raw2 = RawProof(
            temp_id="raw-proof-002",
            proves_label_text="Lemma 4.2",
            full_body_text="...",
            source_section="§1"
        )
        proof2 = ProofBox.from_raw(raw2, proves="lem-helper")
        assert proof2.proof_id == "proof-helper"

    def test_proof_from_raw_strategy_truncation(self):
        """Test that strategy is truncated from full text."""
        long_proof = "A" * 200  # Long proof text

        raw = RawProof(
            temp_id="raw-proof-001",
            proves_label_text="Theorem 5.1",
            full_body_text=long_proof,
            source_section="§1"
        )

        proof = ProofBox.from_raw(raw, proves="thm-test")

        # Strategy should be truncated
        assert len(proof.strategy) < len(long_proof)
        assert proof.strategy.endswith("...")

    def test_proof_from_raw_empty_inputs_outputs(self):
        """Test that inputs/outputs are empty (require semantic analysis)."""
        raw = RawProof(
            temp_id="raw-proof-001",
            proves_label_text="Theorem 6.1",
            full_body_text="...",
            source_section="§1"
        )

        proof = ProofBox.from_raw(raw, proves="thm-test")

        assert proof.inputs == []
        assert proof.outputs == []
        assert proof.sub_proofs == {}


class TestAxiomaticParameter:
    """Tests for AxiomaticParameter model."""

    def test_axiomatic_parameter_creation(self):
        """Test creating AxiomaticParameter."""
        param = AxiomaticParameter(
            symbol="ε",
            description="Displacement scale",
            constraints="ε > 0"
        )

        assert param.symbol == "ε"
        assert param.description == "Displacement scale"
        assert param.constraints == "ε > 0"

    def test_axiomatic_parameter_no_constraints(self):
        """Test parameter without constraints."""
        param = AxiomaticParameter(
            symbol="x",
            description="Position vector",
            constraints=None
        )

        assert param.symbol == "x"
        assert param.constraints is None

    def test_axiomatic_parameter_immutability(self):
        """Test that AxiomaticParameter is immutable."""
        param = AxiomaticParameter(
            symbol="ε",
            description="Test",
            constraints=None
        )

        # Should not be able to modify
        with pytest.raises(Exception):  # ValidationError or AttributeError
            param.symbol = "modified"
