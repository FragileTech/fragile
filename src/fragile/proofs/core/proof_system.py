"""
Proof System: Compositional Proof Engine with Attribute-Level Dataflow.

This module implements a hierarchical, recursive proof system where:
1. Proofs are first-class transformers: Inputs (with properties) → Outputs (with properties)
2. Attribute-level granularity: Track exactly which properties are needed/produced
3. Recursive expansion: Complex steps become sub-proofs, simple steps are LLM-derived
4. Dataflow graph: Create explicit dependency graph for proof validation
5. Composability: Proofs reference other proofs as black-box transformations

Architecture:
- ProofInput: Object + specific properties required
- ProofOutput: Object + properties established
- ProofStep: Atomic transformation or sub-proof reference
- ProofBox: Recursive proof container with validation
- ProofEngine: Orchestrates proof expansion and validation

All types follow Lean-compatible patterns from docs/LEAN_EMULATION_GUIDE.md.

Version: 1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from fragile.proofs.core.article_system import SourceLocation
    from fragile.proofs.core.pipeline_types import TheoremBox


# =============================================================================
# PROPERTY-LEVEL SPECIFICATIONS
# =============================================================================


class AttributeReference(BaseModel):
    """
    Reference to a specific property of an object.

    This is the fundamental unit of proof dataflow: we don't say
    "I need object X", we say "I need property P of object X".

    Maps to Lean:
        structure AttributeReference where
          object_id : String
          property_id : String
          property_statement : String
    """

    model_config = ConfigDict(frozen=True)

    object_id: str = Field(..., min_length=1, description="Object ID (e.g., 'obj-discrete-system')")
    property_id: str = Field(..., min_length=1, description="Attribute ID (e.g., 'prop-lipschitz-continuity')")
    property_statement: str = Field(..., description="Mathematical statement of the property")

    def __str__(self) -> str:
        return f"{self.object_id}::{self.property_id}"


class AssumptionReference(BaseModel):
    """
    Reference to an assumption (property that may not be proven yet).

    Assumptions are properties that we require but may not have proven.
    They make the proof conditional.

    Maps to Lean:
        structure AssumptionReference where
          object_id : String
          assumption_id : String
          assumption_statement : String
          justification : Option String
    """

    model_config = ConfigDict(frozen=True)

    object_id: str = Field(..., description="Object ID")
    assumption_id: str = Field(..., description="Assumption ID")
    assumption_statement: str = Field(..., description="Mathematical statement of assumption")
    justification: Optional[str] = Field(None, description="Why this assumption is reasonable")

    def __str__(self) -> str:
        return f"{self.object_id}::{self.assumption_id} [assumed]"


class ProofInput(BaseModel):
    """
    Input specification for a proof: object + specific properties needed.

    This is like a function signature: we specify exactly what properties
    of what objects we need as input.

    Maps to Lean:
        structure ProofInput where
          object_id : String
          required_properties : List AttributeReference
          required_assumptions : List AssumptionReference
    """

    model_config = ConfigDict(frozen=True)

    object_id: str = Field(..., description="Object ID")
    required_properties: List[AttributeReference] = Field(
        default_factory=list,
        description="Proven properties required"
    )
    required_assumptions: List[AssumptionReference] = Field(
        default_factory=list,
        description="Assumptions required (make proof conditional)"
    )

    def get_all_property_ids(self) -> Set[str]:
        """Get all property IDs (proven + assumed)."""
        return {prop.property_id for prop in self.required_properties} | \
               {assump.assumption_id for assump in self.required_assumptions}


class ProofOutput(BaseModel):
    """
    Output specification for a proof: object + properties established.

    After the proof, these properties are now available for the output object.

    Maps to Lean:
        structure ProofOutput where
          object_id : String
          properties_established : List AttributeReference
    """

    model_config = ConfigDict(frozen=True)

    object_id: str = Field(..., description="Object ID")
    properties_established: List[AttributeReference] = Field(
        ...,
        min_items=1,
        description="Properties proven to hold"
    )


# =============================================================================
# PROOF STEPS
# =============================================================================


class ProofStepType(str, Enum):
    """
    Type of proof step.

    Maps to Lean:
        inductive ProofStepType where
          | direct_derivation : ProofStepType
          | sub_proof : ProofStepType
          | lemma_application : ProofStepType
          | computation : ProofStepType
    """

    DIRECT_DERIVATION = "direct_derivation"  # LLM provides mathematical derivation
    SUB_PROOF = "sub_proof"  # Complex step → nested ProofBox
    LEMMA_APPLICATION = "lemma_application"  # Apply existing lemma/theorem
    COMPUTATION = "computation"  # Computational verification


class ProofStepStatus(str, Enum):
    """Status of proof step."""

    SKETCHED = "sketched"  # Only description exists
    EXPANDED = "expanded"  # Full mathematical derivation provided
    VERIFIED = "verified"  # Verified by LLM or human


class DirectDerivation(BaseModel):
    """
    Direct mathematical derivation provided by LLM.

    This is the base case: a simple enough step that doesn't need
    recursive decomposition.

    Maps to Lean:
        structure DirectDerivation where
          mathematical_content : String
          techniques : List String
    """

    model_config = ConfigDict(frozen=True)

    mathematical_content: str = Field(
        ...,
        description="Full mathematical derivation (LaTeX/markdown)"
    )
    techniques: List[str] = Field(
        default_factory=list,
        description="Mathematical techniques used (e.g., 'cauchy-schwarz', 'integration-by-parts')"
    )
    verification_status: Optional[str] = Field(
        None,
        description="LLM verification status"
    )


class SubProofReference(BaseModel):
    """
    Reference to a nested sub-proof.

    This is the recursive case: a complex step becomes its own ProofBox.

    Maps to Lean:
        structure SubProofReference where
          proof_id : String
          proof_label : String
    """

    model_config = ConfigDict(frozen=True)

    proof_id: str = Field(..., description="Unique proof ID")
    proof_label: str = Field(..., description="Human-readable label")


class LemmaApplication(BaseModel):
    """
    Application of an existing lemma/theorem.

    Maps to Lean:
        structure LemmaApplication where
          lemma_id : String
          input_mapping : HashMap String String
    """

    model_config = ConfigDict(frozen=True)

    lemma_id: str = Field(..., description="ID of lemma/theorem being applied")
    input_mapping: Dict[str, str] = Field(
        ...,
        description="Map from lemma's input IDs to actual object IDs"
    )
    justification: Optional[str] = Field(
        None,
        description="Why this lemma applies"
    )


class ProofStep(BaseModel):
    """
    Single step in a proof with explicit input/output dataflow.

    This is the core abstraction: each step specifies:
    1. What properties it needs (inputs)
    2. What properties it produces (outputs)
    3. How to get from input to output (derivation/sub-proof/lemma)

    Maps to Lean:
        structure ProofStep where
          step_id : String
          natural_language_description : String
          inputs : List ProofInput
          outputs : List ProofOutput
          step_type : ProofStepType
          derivation : Option (DirectDerivation ⊕ SubProofReference ⊕ LemmaApplication)
          citations : List String
    """

    model_config = ConfigDict(frozen=False)  # Mutable for expansion

    step_id: str = Field(..., pattern=r"^step-[0-9]+(-[0-9]+)*$", description="Hierarchical step ID")
    natural_language_description: str = Field(
        ...,
        description="What this step accomplishes, preferably in the paper's own words. "
                   "Should be clear and self-contained."
    )

    # Dataflow specification
    inputs: List[ProofInput] = Field(..., description="Input objects + properties needed")
    outputs: List[ProofOutput] = Field(..., description="Output objects + properties produced")

    # Step implementation
    step_type: ProofStepType = Field(..., description="Type of step")
    derivation: Optional[Union[DirectDerivation, SubProofReference, LemmaApplication]] = Field(
        None,
        description="How to derive output from input"
    )

    # Metadata
    status: ProofStepStatus = Field(
        default=ProofStepStatus.SKETCHED,
        description="Current status of this step"
    )

    # Citation tracking (NEW - for paper processing)
    citations: List[str] = Field(
        default_factory=list,
        description="A list of bibliographic keys (from the article's Bibliography) cited to justify this step. "
                   "Enables tracking which external theorems or papers are used in each proof step."
    )
    estimated_complexity: Optional[str] = Field(
        None,
        description="Estimated complexity (simple/moderate/complex)"
    )

    @model_validator(mode='after')
    def validate_derivation(self) -> 'ProofStep':
        """Ensure derivation matches step_type."""
        if self.status == ProofStepStatus.EXPANDED:
            if self.derivation is None:
                raise ValueError(f"Step {self.step_id} is marked EXPANDED but has no derivation")

            # Check type consistency
            if self.step_type == ProofStepType.DIRECT_DERIVATION:
                if not isinstance(self.derivation, DirectDerivation):
                    raise ValueError(f"Step {self.step_id} has type DIRECT_DERIVATION but derivation is not DirectDerivation")
            elif self.step_type == ProofStepType.SUB_PROOF:
                if not isinstance(self.derivation, SubProofReference):
                    raise ValueError(f"Step {self.step_id} has type SUB_PROOF but derivation is not SubProofReference")
            elif self.step_type == ProofStepType.LEMMA_APPLICATION:
                if not isinstance(self.derivation, LemmaApplication):
                    raise ValueError(f"Step {self.step_id} has type LEMMA_APPLICATION but derivation is not LemmaApplication")

        return self

    def is_sketched(self) -> bool:
        """Check if step is still just a sketch."""
        return self.status == ProofStepStatus.SKETCHED

    def is_expanded(self) -> bool:
        """Check if step has been expanded."""
        return self.status in (ProofStepStatus.EXPANDED, ProofStepStatus.VERIFIED)

    def get_required_property_ids(self) -> Set[str]:
        """Get all property IDs required by this step."""
        result = set()
        for inp in self.inputs:
            result.update(inp.get_all_property_ids())
        return result

    def get_produced_property_ids(self) -> Set[str]:
        """Get all property IDs produced by this step."""
        result = set()
        for out in self.outputs:
            for prop in out.properties_established:
                result.add(prop.property_id)
        return result


# =============================================================================
# PROOF BOX
# =============================================================================


class ProofBox(BaseModel):
    """
    Recursive proof container with property-level dataflow.

    This is the main proof object. It can contain:
    1. Direct derivation steps (base case)
    2. Sub-proofs (recursive case)
    3. Lemma applications (compositional case)

    The proof is valid if:
    - Input properties satisfy all step requirements
    - Step outputs provide inputs for subsequent steps
    - Final outputs match claimed theorem conclusion

    Maps to Lean:
        structure ProofBox where
          proof_id : String
          proves : String
          inputs : List ProofInput
          outputs : List ProofOutput
          steps : List ProofStep
          sub_proofs : HashMap String ProofBox
    """

    model_config = ConfigDict(frozen=False)  # Mutable for expansion

    proof_id: str = Field(..., pattern=r"^proof-[a-z0-9-]+$", description="Unique proof ID")
    label: str = Field(..., description="Human-readable label")

    proves: str = Field(
        ...,
        pattern=r"^(thm|lem|prop)-[a-z0-9-]+$",
        description="What theorem/lemma/proposition this proves"
    )

    # Proof signature (like a function type)
    inputs: List[ProofInput] = Field(..., description="Input objects + properties required")
    outputs: List[ProofOutput] = Field(..., description="Output objects + properties established")

    # Proof body
    strategy: str = Field(..., description="High-level proof strategy")
    steps: List[ProofStep] = Field(..., min_items=1, description="Proof steps")

    # Recursive structure
    sub_proofs: Dict[str, 'ProofBox'] = Field(
        default_factory=dict,
        description="Nested sub-proofs (for complex steps)"
    )

    # Metadata
    proof_type: Optional[str] = Field(None, description="Proof type (direct, contradiction, induction, etc.)")
    complexity: Optional[str] = Field(None, description="Overall complexity (routine/standard/technical/intricate/deep)")
    source: Optional["SourceLocation"] = Field(
        None,
        description="Source location in documentation where this proof is defined",
    )

    # Theorem Integration (NEW)
    theorem: Optional['TheoremBox'] = Field(
        None,
        description="Back-reference to theorem this proves (enables bidirectional navigation)"
    )

    @model_validator(mode='after')
    def validate_against_theorem(self) -> 'ProofBox':
        """
        Validate proof against theorem if theorem is attached.

        This runs automatically when ProofBox is created/modified.

        Maps to Lean:
            def validate_against_theorem (p : ProofBox) : Except String ProofBox :=
              match p.theorem with
              | none => Except.ok p
              | some thm =>
                  if p.proves != thm.label then
                    Except.error s!"Proof.proves ({p.proves}) doesn't match theorem.label ({thm.label})"
                  else
                    Except.ok p
        """
        if self.theorem is not None:
            # Check proves field matches
            if self.proves != self.theorem.label:
                raise ValueError(
                    f"Proof.proves ({self.proves}) doesn't match "
                    f"theorem.label ({self.theorem.label})"
                )
        return self

    def get_step(self, step_id: str) -> Optional[ProofStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_sub_proof(self, proof_id: str) -> Optional['ProofBox']:
        """Get nested sub-proof by ID."""
        return self.sub_proofs.get(proof_id)

    def all_steps_expanded(self) -> bool:
        """Check if all steps are expanded."""
        return all(step.is_expanded() for step in self.steps)

    def get_sketched_steps(self) -> List[ProofStep]:
        """Get all steps that are still sketched (not expanded)."""
        return [step for step in self.steps if step.is_sketched()]

    def get_required_properties(self) -> Dict[str, Set[str]]:
        """
        Get all properties required, grouped by object.

        Returns: {object_id: {property_id1, property_id2, ...}}
        """
        result: Dict[str, Set[str]] = {}
        for inp in self.inputs:
            if inp.object_id not in result:
                result[inp.object_id] = set()
            result[inp.object_id].update(inp.get_all_property_ids())
        return result

    def get_established_properties(self) -> Dict[str, Set[str]]:
        """
        Get all properties established, grouped by object.

        Returns: {object_id: {property_id1, property_id2, ...}}
        """
        result: Dict[str, Set[str]] = {}
        for out in self.outputs:
            if out.object_id not in result:
                result[out.object_id] = set()
            for prop in out.properties_established:
                result[out.object_id].add(prop.property_id)
        return result

    def validate_dataflow(self) -> List[str]:
        """
        Validate that dataflow is consistent.

        Checks:
        1. Each step's inputs are satisfied by previous outputs or proof inputs
        2. Final outputs match claimed outputs
        3. No circular dependencies

        Returns: List of error messages (empty if valid)
        """
        errors = []

        # Track available properties after each step
        available: Dict[str, Set[str]] = {}

        # Initialize with proof inputs
        for inp in self.inputs:
            if inp.object_id not in available:
                available[inp.object_id] = set()
            available[inp.object_id].update(inp.get_all_property_ids())

        # Check each step
        for step in self.steps:
            # Check inputs are satisfied
            for inp in step.inputs:
                required = inp.get_all_property_ids()
                if inp.object_id not in available:
                    errors.append(
                        f"Step {step.step_id}: Object {inp.object_id} not available"
                    )
                else:
                    missing = required - available[inp.object_id]
                    if missing:
                        errors.append(
                            f"Step {step.step_id}: Missing properties for {inp.object_id}: {missing}"
                        )

            # Add outputs to available properties
            for out in step.outputs:
                if out.object_id not in available:
                    available[out.object_id] = set()
                for prop in out.properties_established:
                    available[out.object_id].add(prop.property_id)

        # Check final outputs are satisfied
        for out in self.outputs:
            required_props = {prop.property_id for prop in out.properties_established}
            if out.object_id not in available:
                errors.append(
                    f"Proof output: Object {out.object_id} never produced"
                )
            else:
                missing = required_props - available[out.object_id]
                if missing:
                    errors.append(
                        f"Proof output: Missing properties for {out.object_id}: {missing}"
                    )

        return errors

    def to_graph(self) -> Dict:
        """
        Convert proof to dataflow graph representation.

        Returns: Graph structure suitable for visualization
        """
        nodes = []
        edges = []

        # Add input nodes
        for inp in self.inputs:
            nodes.append({
                "id": f"input-{inp.object_id}",
                "type": "input",
                "label": inp.object_id,
                "properties": list(inp.get_all_property_ids())
            })

        # Add step nodes
        for step in self.steps:
            nodes.append({
                "id": step.step_id,
                "type": "step",
                "label": step.description,
                "status": step.status.value
            })

            # Edges from inputs to step
            for inp in step.inputs:
                edges.append({
                    "source": f"input-{inp.object_id}",
                    "target": step.step_id,
                    "properties": list(inp.get_all_property_ids())
                })

            # Edges from step to outputs
            for out in step.outputs:
                edges.append({
                    "source": step.step_id,
                    "target": f"output-{out.object_id}",
                    "properties": [p.property_id for p in out.properties_established]
                })

        # Add output nodes
        for out in self.outputs:
            nodes.append({
                "id": f"output-{out.object_id}",
                "type": "output",
                "label": out.object_id,
                "properties": [p.property_id for p in out.properties_established]
            })

        return {
            "proof_id": self.proof_id,
            "label": self.label,
            "nodes": nodes,
            "edges": edges
        }

    @classmethod
    def from_raw(
        cls,
        raw: "RawProof",  # type: ignore  # Forward reference
        proves: str,
    ) -> "ProofBox":
        """
        Create a ProofBox from a RawProof staging model.

        This is a SIMPLE enrichment that stores the proof as a single step.
        The full proof is stored in the step's description. A specialized agent
        can later break the proof into detailed steps.

        Args:
            raw: The raw proof from Stage 1 extraction
            proves: Label of the theorem this proves (e.g., "thm-keystone")

        Returns:
            ProofBox with proof stored as single sketched step

        Examples:
            >>> raw_proof = RawProof(
            ...     temp_id="raw-proof-1",
            ...     theorem_label="thm-keystone",
            ...     proof_text="We proceed by...",
            ...     source_section="§3"
            ... )
            >>> proof = ProofBox.from_raw(raw_proof, proves="thm-keystone")
            >>> proof.proof_id
            'proof-thm-keystone'
        """
        from fragile.proofs.staging_types import RawProof

        # Generate proof_id from theorem label
        proof_id = f"proof-{proves.replace('thm-', '').replace('lem-', '').replace('prop-', '')}"
        if not proof_id.startswith("proof-"):
            proof_id = f"proof-{proof_id}"

        # Get proof text: use full_body_text if available, otherwise concatenate steps
        if raw.full_body_text:
            proof_text = raw.full_body_text
        elif raw.steps:
            proof_text = "\n\n".join(raw.steps)
        else:
            proof_text = "No proof text available"

        # Create a single step with the full proof text
        # This is the "simple" approach - a specialized agent can expand later
        single_step = ProofStep(
            step_id=f"{proof_id}-step-1",
            step_number=1,
            description=proof_text,
            justification="Full proof text (to be expanded by specialized agent)",
            step_type=ProofStepType.DIRECT,
            status=ProofStepStatus.SKETCHED,  # Not yet broken into detailed steps
            inputs=[],
            outputs=[],
            derivation=DirectDerivation(
                from_properties=[],
                conclusion="See description",
                reasoning=proof_text[:200] + "..." if len(proof_text) > 200 else proof_text
            )
        )

        # Use strategy_text if available, otherwise use beginning of proof_text
        strategy = raw.strategy_text if raw.strategy_text else (proof_text[:100] + "..." if len(proof_text) > 100 else proof_text)

        return cls(
            proof_id=proof_id,
            label=f"Proof of {proves}",
            proves=proves,
            inputs=[],  # Requires semantic analysis to extract from explicit_theorem_references
            outputs=[],  # Requires semantic analysis to extract
            strategy=strategy,
            steps=[single_step],
            sub_proofs={},
            proof_type=None,  # Requires semantic analysis
            complexity=None,  # Requires semantic analysis
            source=None,  # Requires line finder
            theorem=None  # Can be attached later
        )


# =============================================================================
# PROOF ENGINE
# =============================================================================


class ProofExpansionRequest(BaseModel):
    """
    Request to expand a sketched step.

    This is what gets sent to the LLM for expansion.
    """

    model_config = ConfigDict(frozen=True)

    proof_id: str = Field(..., description="Parent proof ID")
    step_id: str = Field(..., description="Step to expand")
    step_description: str = Field(..., description="What the step should accomplish")
    inputs: List[ProofInput] = Field(..., description="Available inputs")
    outputs: List[ProofOutput] = Field(..., description="Required outputs")
    context: Optional[str] = Field(None, description="Additional context from proof strategy")


class ProofEngine:
    """
    Orchestrates proof expansion and validation.

    This is the main interface for:
    1. Creating proof boxes
    2. Expanding sketched steps
    3. Validating dataflow
    4. Managing recursive sub-proofs
    """

    def __init__(self):
        self.proofs: Dict[str, ProofBox] = {}

    def register_proof(self, proof: ProofBox) -> None:
        """Register a proof."""
        self.proofs[proof.proof_id] = proof

    def get_proof(self, proof_id: str) -> Optional[ProofBox]:
        """Get proof by ID."""
        return self.proofs.get(proof_id)

    def validate_proof(self, proof_id: str) -> List[str]:
        """
        Validate proof dataflow.

        Returns: List of errors (empty if valid)
        """
        proof = self.get_proof(proof_id)
        if proof is None:
            return [f"Proof {proof_id} not found"]

        return proof.validate_dataflow()

    def get_expansion_requests(self, proof_id: str) -> List[ProofExpansionRequest]:
        """
        Get all sketched steps that need expansion.

        Returns: List of expansion requests for LLM
        """
        proof = self.get_proof(proof_id)
        if proof is None:
            return []

        requests = []
        for step in proof.get_sketched_steps():
            requests.append(ProofExpansionRequest(
                proof_id=proof_id,
                step_id=step.step_id,
                step_description=step.description,
                inputs=step.inputs,
                outputs=step.outputs,
                context=proof.strategy
            ))

        return requests

    def expand_step(
        self,
        proof_id: str,
        step_id: str,
        derivation: Union[DirectDerivation, SubProofReference, LemmaApplication]
    ) -> bool:
        """
        Expand a sketched step with its derivation.

        Returns: True if successful
        """
        proof = self.get_proof(proof_id)
        if proof is None:
            return False

        step = proof.get_step(step_id)
        if step is None:
            return False

        # Update step (proof is mutable)
        step.derivation = derivation
        step.status = ProofStepStatus.EXPANDED

        return True

    def add_sub_proof(self, parent_proof_id: str, sub_proof: ProofBox) -> bool:
        """
        Add a nested sub-proof.

        Returns: True if successful
        """
        proof = self.get_proof(parent_proof_id)
        if proof is None:
            return False

        proof.sub_proofs[sub_proof.proof_id] = sub_proof
        return True

    def compute_proof_complexity(self, proof_id: str) -> Dict:
        """
        Compute complexity metrics for a proof.

        Returns: Dictionary with metrics
        """
        proof = self.get_proof(proof_id)
        if proof is None:
            return {}

        total_steps = len(proof.steps)
        expanded_steps = len([s for s in proof.steps if s.is_expanded()])
        sub_proof_count = len(proof.sub_proofs)

        return {
            "total_steps": total_steps,
            "expanded_steps": expanded_steps,
            "sketched_steps": total_steps - expanded_steps,
            "sub_proofs": sub_proof_count,
            "completion": expanded_steps / total_steps if total_steps > 0 else 0,
            "depth": self._compute_depth(proof),
        }

    def _compute_depth(self, proof: ProofBox) -> int:
        """Compute maximum depth of proof tree."""
        if not proof.sub_proofs:
            return 1
        return 1 + max(self._compute_depth(sp) for sp in proof.sub_proofs.values())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_proof_from_theorem(
    theorem_id: str,
    theorem_label: str,
    inputs: List[ProofInput],
    outputs: List[ProofOutput],
    strategy: str,
    steps: List[ProofStep]
) -> ProofBox:
    """
    Helper: Create ProofBox from theorem specification.

    Pure function (no side effects).
    """
    proof_id = f"proof-{theorem_id}"

    return ProofBox(
        proof_id=proof_id,
        label=theorem_label,
        proves=theorem_id,
        inputs=inputs,
        outputs=outputs,
        strategy=strategy,
        steps=steps
    )


def create_simple_step(
    step_id: str,
    description: str,
    inputs: List[ProofInput],
    outputs: List[ProofOutput],
    complexity: str = "simple"
) -> ProofStep:
    """
    Helper: Create simple proof step (to be expanded by LLM).

    Pure function (no side effects).
    """
    return ProofStep(
        step_id=step_id,
        description=description,
        inputs=inputs,
        outputs=outputs,
        step_type=ProofStepType.DIRECT_DERIVATION,
        estimated_complexity=complexity
    )
