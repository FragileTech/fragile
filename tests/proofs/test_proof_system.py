"""
Comprehensive tests for fragile.proofs.core.proof_system.

Tests ProofBox, ProofStep, dataflow validation, and ProofEngine.
"""

import pytest
from pydantic import ValidationError

from fragile.proofs.core.proof_system import (
    AssumptionReference,
    DirectDerivation,
    LemmaApplication,
    ProofBox,
    ProofEngine,
    ProofExpansionRequest,
    ProofInput,
    ProofOutput,
    ProofStep,
    ProofStepStatus,
    ProofStepType,
    PropertyReference,
    SubProofReference,
)


class TestPropertyReference:
    """Test PropertyReference type."""

    def test_minimal_property_reference(self):
        """Test minimal property reference."""
        prop_ref = PropertyReference(
            object_id="obj-test",
            property_id="prop-test",
            property_statement="Test property",
        )
        assert prop_ref.object_id == "obj-test"
        assert prop_ref.property_id == "prop-test"

    def test_property_reference_with_expression(self):
        """Test property reference with mathematical expression."""
        prop_ref = PropertyReference(
            object_id="obj-function",
            property_id="prop-lipschitz",
            property_statement=r"Function is Lipschitz: |\nabla f(x)| \leq L",
        )
        assert r"\nabla" in prop_ref.property_statement


class TestAssumptionReference:
    """Test AssumptionReference type."""

    def test_assumption_reference(self):
        """Test assumption reference creation."""
        assump = AssumptionReference(
            object_id="obj-system",
            assumption_id="assump-independent",
            assumption_statement="Initial positions are i.i.d.",
        )
        assert assump.assumption_id == "assump-independent"


class TestProofInputOutput:
    """Test ProofInput and ProofOutput types."""

    def test_proof_input(self):
        """Test ProofInput creation."""
        prop_ref = PropertyReference(
            object_id="obj-test",
            property_id="prop-test",
            property_statement="Test",
        )
        proof_input = ProofInput(
            object_id="obj-test",
            required_properties=[prop_ref],
            required_assumptions=[],
        )
        assert len(proof_input.required_properties) == 1

    def test_proof_output(self):
        """Test ProofOutput creation."""
        prop_ref = PropertyReference(
            object_id="obj-test",
            property_id="prop-new",
            property_statement="New property",
        )
        proof_output = ProofOutput(
            object_id="obj-test",
            properties_established=[prop_ref],
        )
        assert len(proof_output.properties_established) == 1


class TestDirectDerivation:
    """Test DirectDerivation type."""

    def test_direct_derivation(self):
        """Test direct derivation creation."""
        derivation = DirectDerivation(
            mathematical_content="Apply Gronwall inequality to get...",
            techniques=["gronwall", "contraction"],
        )
        assert "Gronwall" in derivation.mathematical_content
        assert "gronwall" in derivation.techniques


class TestProofStep:
    """Test ProofStep type."""

    def test_sketched_step(self):
        """Test SKETCHED proof step."""
        prop_in = PropertyReference(object_id="obj-a", property_id="prop-1", property_statement="Property 1")
        prop_out = PropertyReference(object_id="obj-a", property_id="prop-2", property_statement="Property 2")

        step = ProofStep(
            step_id="step-1",
            description="Establish well-posedness",
            inputs=[ProofInput(object_id="obj-a", required_properties=[prop_in], required_assumptions=[])],
            outputs=[ProofOutput(object_id="obj-a", properties_established=[prop_out])],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=DirectDerivation(mathematical_content="To be filled", techniques=[]),
            status=ProofStepStatus.SKETCHED,
        )
        assert step.status == ProofStepStatus.SKETCHED

    def test_expanded_step(self):
        """Test EXPANDED proof step with full derivation."""
        prop_in = PropertyReference(object_id="obj-a", property_id="prop-1", property_statement="Property 1")
        prop_out = PropertyReference(object_id="obj-a", property_id="prop-2", property_statement="Property 2")

        derivation = DirectDerivation(
            mathematical_content="Full mathematical derivation here...",
            techniques=["analysis"],
        )

        step = ProofStep(
            step_id="step-1",
            description="Full derivation",
            inputs=[ProofInput(object_id="obj-a", required_properties=[prop_in], required_assumptions=[])],
            outputs=[ProofOutput(object_id="obj-a", properties_established=[prop_out])],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=derivation,
            status=ProofStepStatus.EXPANDED,
        )
        assert step.status == ProofStepStatus.EXPANDED
        assert len(step.derivation.mathematical_content) > 0

    def test_sub_proof_step(self):
        """Test step with SubProofReference."""
        step = ProofStep(
            step_id="step-1",
            description="Defer to technical lemma",
            inputs=[],
            outputs=[],
            step_type=ProofStepType.SUB_PROOF,
            derivation=SubProofReference(proof_id="proof-lem-technical", proof_label="Technical Lemma"),
            status=ProofStepStatus.SKETCHED,
        )
        assert step.step_type == ProofStepType.SUB_PROOF
        assert step.derivation.proof_id == "proof-lem-technical"

    def test_lemma_application_step(self):
        """Test step with LemmaApplication."""
        step = ProofStep(
            step_id="step-1",
            description="Apply Sznitman coupling lemma",
            inputs=[],
            outputs=[],
            step_type=ProofStepType.LEMMA_APPLICATION,
            derivation=LemmaApplication(lemma_id="lem-sznitman", input_mapping={"x": "obj-a"}),
            status=ProofStepStatus.EXPANDED,
        )
        assert step.step_type == ProofStepType.LEMMA_APPLICATION


class TestProofBox:
    """Test ProofBox type."""

    def test_minimal_proof(self):
        """Test minimal proof creation."""
        prop_in = PropertyReference(object_id="obj-a", property_id="prop-1", property_statement="Input property")
        prop_out = PropertyReference(object_id="obj-a", property_id="prop-2", property_statement="Output property")

        step = ProofStep(
            step_id="step-1",
            description="Single step proof",
            inputs=[ProofInput(object_id="obj-a", required_properties=[prop_in], required_assumptions=[])],
            outputs=[ProofOutput(object_id="obj-a", properties_established=[prop_out])],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=DirectDerivation(mathematical_content="Proof content", techniques=[]),
            status=ProofStepStatus.EXPANDED,
        )

        proof = ProofBox(
            proof_id="proof-thm-simple",
            label="Simple Proof",
            proves="thm-simple",
            inputs=[ProofInput(object_id="obj-a", required_properties=[prop_in], required_assumptions=[])],
            outputs=[ProofOutput(object_id="obj-a", properties_established=[prop_out])],
            strategy="Single-step direct derivation",
            steps=[step],
            sub_proofs={},
        )

        assert proof.proof_id == "proof-thm-simple"
        assert len(proof.steps) == 1

    def test_hierarchical_proof(self):
        """Test proof with sub-proofs."""
        # Main proof step referencing sub-proof
        main_step = ProofStep(
            step_id="step-1",
            description="Use technical lemma",
            inputs=[],
            outputs=[],
            step_type=ProofStepType.SUB_PROOF,
            derivation=SubProofReference(proof_id="proof-lem-technical", proof_label="proof-lem-technical"),
            status=ProofStepStatus.EXPANDED,
        )

        # Sub-proof
        sub_step = ProofStep(
            step_id="step-1",
            description="Prove technical result",
            inputs=[],
            outputs=[],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=DirectDerivation(mathematical_content="Technical proof", techniques=[]),
            status=ProofStepStatus.EXPANDED,
        )

        sub_proof = ProofBox(
            proof_id="proof-lem-technical",
            label="Technical Lemma",
            proves="lem-technical",
            inputs=[],
            outputs=[],
            strategy="Direct proof",
            steps=[sub_step],
            sub_proofs={},
        )

        main_proof = ProofBox(
            proof_id="proof-thm-main",
            label="Main Theorem",
            proves="thm-main",
            inputs=[],
            outputs=[],
            strategy="Use technical lemma",
            steps=[main_step],
            sub_proofs={"proof-lem-technical": sub_proof},
        )

        assert len(main_proof.sub_proofs) == 1
        assert "proof-lem-technical" in main_proof.sub_proofs

    def test_proof_dataflow_validation(self):
        """Test proof dataflow validation."""
        # Create proof with valid dataflow
        prop1 = PropertyReference(object_id="obj-a", property_id="prop-1", property_statement="Property 1")
        prop2 = PropertyReference(object_id="obj-a", property_id="prop-2", property_statement="Property 2")

        step = ProofStep(
            step_id="step-1",
            description="Transform prop-1 to prop-2",
            inputs=[ProofInput(object_id="obj-a", required_properties=[prop1], required_assumptions=[])],
            outputs=[ProofOutput(object_id="obj-a", properties_established=[prop2])],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=DirectDerivation(mathematical_content="Transform", techniques=[]),
            status=ProofStepStatus.EXPANDED,
        )

        proof = ProofBox(
            proof_id="proof-test",
            label="Test",
            proves="thm-test",
            inputs=[ProofInput(object_id="obj-a", required_properties=[prop1], required_assumptions=[])],
            outputs=[ProofOutput(object_id="obj-a", properties_established=[prop2])],
            strategy="Single step",
            steps=[step],
            sub_proofs={},
        )

        # Validate dataflow
        errors = proof.validate_dataflow()
        assert len(errors) == 0, f"Dataflow should be valid, got errors: {errors}"


class TestProofEngine:
    """Test ProofEngine type."""

    def test_proof_engine_registration(self):
        """Test registering proof with engine."""
        engine = ProofEngine()

        prop1 = PropertyReference(object_id="obj-a", property_id="prop-1", property_statement="Property 1")
        prop2 = PropertyReference(object_id="obj-a", property_id="prop-2", property_statement="Property 2")

        step = ProofStep(
            step_id="step-1",
            description="Test step",
            inputs=[ProofInput(object_id="obj-a", required_properties=[prop1], required_assumptions=[])],
            outputs=[ProofOutput(object_id="obj-a", properties_established=[prop2])],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=DirectDerivation(mathematical_content="Test", techniques=[]),
            status=ProofStepStatus.SKETCHED,  # Needs expansion
        )

        proof = ProofBox(
            proof_id="proof-test",
            label="Test",
            proves="thm-test",
            inputs=[ProofInput(object_id="obj-a", required_properties=[prop1], required_assumptions=[])],
            outputs=[ProofOutput(object_id="obj-a", properties_established=[prop2])],
            strategy="Test",
            steps=[step],
            sub_proofs={},
        )

        engine.register_proof(proof)
        assert "proof-test" in engine.proofs

    def test_proof_expansion_requests(self):
        """Test getting expansion requests for SKETCHED steps."""
        engine = ProofEngine()

        step = ProofStep(
            step_id="step-1",
            description="Needs expansion",
            inputs=[],
            outputs=[],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=DirectDerivation(mathematical_content="To be filled", techniques=[]),
            status=ProofStepStatus.SKETCHED,
        )

        proof = ProofBox(
            proof_id="proof-test",
            label="Test",
            proves="thm-test",
            inputs=[],
            outputs=[],
            strategy="Test",
            steps=[step],
            sub_proofs={},
        )

        engine.register_proof(proof)
        requests = engine.get_expansion_requests("proof-test")

        assert len(requests) > 0
        assert any(r.step_id == "step-1" for r in requests)


class TestProofImmutability:
    """Test that all proof types are frozen."""

    def test_proof_box_mutable(self):
        """Test ProofBox is mutable (for expansion)."""
        step = ProofStep(
            step_id="step-1",
            description="Test",
            inputs=[],
            outputs=[],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=DirectDerivation(mathematical_content="Test", techniques=[]),
            status=ProofStepStatus.EXPANDED,
        )
        proof = ProofBox(
            proof_id="proof-test",
            label="Test",
            proves="thm-test",
            inputs=[],
            outputs=[],
            strategy="Test",
            steps=[step],
            sub_proofs={},
        )
        # ProofBox is intentionally mutable for expansion during proof development
        # This should NOT raise an exception
        proof.strategy = "Modified strategy"
        assert proof.strategy == "Modified strategy"

    def test_proof_step_mutable(self):
        """Test ProofStep is NOT frozen (can be modified for expansion)."""
        step = ProofStep(
            step_id="step-1",
            description="Test",
            inputs=[],
            outputs=[],
            step_type=ProofStepType.DIRECT_DERIVATION,
            derivation=DirectDerivation(mathematical_content="Test", techniques=[]),
            status=ProofStepStatus.SKETCHED,
        )
        # ProofStep is mutable to allow expansion
        step.status = ProofStepStatus.EXPANDED
        assert step.status == ProofStepStatus.EXPANDED
