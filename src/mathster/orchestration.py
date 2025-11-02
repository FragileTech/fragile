"""
Orchestration Types for Extract-then-Enrich Pipeline.

This module provides the core orchestration infrastructure for the two-stage
mathematical paper processing pipeline:

- ResolutionContext: Cross-referencing and entity resolution during enrichment
- EnrichmentError: Exception handling for enrichment failures
- Supporting types for error tracking and validation

The orchestration layer coordinates between:
1. Raw extraction (Stage 1: raw_data.py)
2. Semantic enrichment (Stage 2: prompts/enrichment.py)
3. Final structured models (math_types.py, proof_system.py)

Maps to Lean:
    structure ResolutionContext where
      definitions : HashMap String RawDefinition
      theorems : HashMap String RawTheorem
      mathster : HashMap String RawProof
      ...

    inductive EnrichmentError where
      | ParseFailure : String → String → EnrichmentError
      | ReferenceUnresolved : String → EnrichmentError
      | ValidationFailure : String → EnrichmentError
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mathster.core.raw_data import (
    RawCitation,
    RawDefinition,
    RawEquation,
    RawParameter,
    RawProof,
    RawRemark,
    RawTheorem,
    RawDocument,
)


# =============================================================================
# RESOLUTION CONTEXT
# =============================================================================


class ResolutionContext(BaseModel):
    """
    Context for cross-referencing entities during enrichment.

    The ResolutionContext serves as a knowledge base during Stage 2 enrichment,
    storing all extracted raw entities and providing methods to resolve
    references between them (e.g., "Theorem 2.1" → "thm-keystone").

    This enables:
    - Linking mathster to theorems
    - Resolving definition references in theorems
    - Connecting citations to bibliographic entries
    - Tracking ambiguous references for manual resolution

    Examples:
        >>> ctx = ResolutionContext()
        >>> # Add raw entities
        >>> ctx.add_definition(raw_def)
        >>> ctx.add_theorem(raw_thm)
        >>>
        >>> # Resolve references
        >>> thm_id = ctx.resolve_theorem_reference("Theorem 2.1")
        >>> # → "raw-thm-001"
        >>>
        >>> # Get entity by ID
        >>> theorem = ctx.get_theorem("raw-thm-001")

    Maps to Lean:
        structure ResolutionContext where
          definitions : HashMap String RawDefinition
          theorems : HashMap String RawTheorem
          mathster : HashMap String RawProof
          equations : HashMap String RawEquation
          parameters : HashMap String RawParameter
          remarks : HashMap String RawRemark
          citations : HashMap String RawCitation

          def resolve_reference (ctx : ResolutionContext) (ref : String) : Option String
    """

    # Entity storage (temp_id → entity)
    definitions: dict[str, RawDefinition] = Field(default_factory=dict)
    theorems: dict[str, RawTheorem] = Field(default_factory=dict)
    proofs: dict[str, RawProof] = Field(default_factory=dict)
    equations: dict[str, RawEquation] = Field(default_factory=dict)
    parameters: dict[str, RawParameter] = Field(default_factory=dict)
    remarks: dict[str, RawRemark] = Field(default_factory=dict)
    citations: dict[str, RawCitation] = Field(default_factory=dict)

    # Reverse lookups (for fast resolution)
    label_text_to_theorem: dict[str, str] = Field(default_factory=dict)
    # Maps "Theorem 2.1" → "raw-thm-001"

    term_to_definition: dict[str, str] = Field(default_factory=dict)
    # Maps "Walker State" → "raw-def-001"

    equation_label_to_id: dict[str, str] = Field(default_factory=dict)
    # Maps "(2.3)" → "raw-eq-001"

    symbol_to_parameter: dict[str, str] = Field(default_factory=dict)
    # Maps "γ" → "raw-param-001"

    # Ambiguous references (to be resolved manually or by LLM)
    ambiguous_references: list[AmbiguousReference] = Field(default_factory=list)

    # Metadata
    document_id: str | None = None
    source_file: str | None = None

    # =============================================================================
    # ADDING ENTITIES
    # =============================================================================

    def add_definition(self, definition: RawDefinition) -> None:
        """Add a definition and create reverse lookup."""
        self.definitions[definition.temp_id] = definition
        self.term_to_definition[definition.term_being_defined.lower()] = definition.temp_id

    def add_theorem(self, theorem: RawTheorem) -> None:
        """Add a theorem and create reverse lookup."""
        self.theorems[theorem.temp_id] = theorem
        self.label_text_to_theorem[theorem.label_text] = theorem.temp_id

    def add_proof(self, proof: RawProof) -> None:
        """Add a proof."""
        self.proofs[proof.temp_id] = proof

    def add_equation(self, equation: RawEquation) -> None:
        """Add an equation and create reverse lookup."""
        self.equations[equation.temp_id] = equation
        if equation.equation_label:
            self.equation_label_to_id[equation.equation_label] = equation.temp_id

    def add_parameter(self, parameter: RawParameter) -> None:
        """Add a parameter and create reverse lookup."""
        self.parameters[parameter.temp_id] = parameter
        self.symbol_to_parameter[parameter.symbol] = parameter.temp_id

    def add_remark(self, remark: RawRemark) -> None:
        """Add a remark."""
        self.remarks[remark.temp_id] = remark

    def add_citation(self, citation: RawCitation) -> None:
        """Add a citation."""
        self.citations[citation.key_in_text] = citation

    def add_staging_document(self, doc: RawDocument) -> None:
        """
        Batch-add all entities from a StagingDocument.

        Args:
            doc: StagingDocument from Stage 1 extraction
        """
        for definition in doc.definitions:
            self.add_definition(definition)
        for theorem in doc.theorems:
            self.add_theorem(theorem)
        for proof in doc.proofs:
            self.add_proof(proof)
        for equation in doc.equations:
            self.add_equation(equation)
        for parameter in doc.parameters:
            self.add_parameter(parameter)
        for remark in doc.remarks:
            self.add_remark(remark)
        for citation in doc.citations:
            self.add_citation(citation)

    # =============================================================================
    # RESOLUTION METHODS
    # =============================================================================

    def resolve_theorem_reference(self, label_text: str) -> str | None:
        """
        Resolve theorem reference (e.g., "Theorem 2.1") to temp_id.

        Args:
            label_text: Reference like "Theorem 2.1", "Lemma 3.5"

        Returns:
            Temp ID (e.g., "raw-thm-001") or None if not found
        """
        return self.label_text_to_theorem.get(label_text)

    def resolve_definition_reference(self, term: str) -> str | None:
        """
        Resolve definition reference by term name.

        Args:
            term: Term like "Walker State", "potential"

        Returns:
            Temp ID (e.g., "raw-def-001") or None if not found
        """
        return self.term_to_definition.get(term.lower())

    def resolve_equation_reference(self, equation_label: str) -> str | None:
        """
        Resolve equation reference (e.g., "(2.3)") to temp_id.

        Args:
            equation_label: Label like "(2.3)", "(17)"

        Returns:
            Temp ID (e.g., "raw-eq-001") or None if not found
        """
        return self.equation_label_to_id.get(equation_label)

    def resolve_parameter_reference(self, symbol: str) -> str | None:
        """
        Resolve parameter reference by symbol.

        Args:
            symbol: Symbol like "γ", "N", "h"

        Returns:
            Temp ID (e.g., "raw-param-001") or None if not found
        """
        return self.symbol_to_parameter.get(symbol)

    def find_proof_for_theorem(self, theorem_label_text: str) -> RawProof | None:
        """
        Find the proof for a given theorem.

        Args:
            theorem_label_text: Label like "Theorem 2.1"

        Returns:
            RawProof if found, None otherwise
        """
        for proof in self.proofs.values():
            if proof.proves_label_text == theorem_label_text:
                return proof
        return None

    # =============================================================================
    # ENTITY RETRIEVAL
    # =============================================================================

    def get_definition(self, temp_id: str) -> RawDefinition | None:
        """Get definition by temp_id."""
        return self.definitions.get(temp_id)

    def get_theorem(self, temp_id: str) -> RawTheorem | None:
        """Get theorem by temp_id."""
        return self.theorems.get(temp_id)

    def get_proof(self, temp_id: str) -> RawProof | None:
        """Get proof by temp_id."""
        return self.proofs.get(temp_id)

    def get_equation(self, temp_id: str) -> RawEquation | None:
        """Get equation by temp_id."""
        return self.equations.get(temp_id)

    def get_parameter(self, temp_id: str) -> RawParameter | None:
        """Get parameter by temp_id."""
        return self.parameters.get(temp_id)

    def get_remark(self, temp_id: str) -> RawRemark | None:
        """Get remark by temp_id."""
        return self.remarks.get(temp_id)

    def get_citation(self, key: str) -> RawCitation | None:
        """Get citation by key."""
        return self.citations.get(key)

    # =============================================================================
    # AMBIGUOUS REFERENCE TRACKING
    # =============================================================================

    def flag_ambiguous_reference(
        self,
        reference_text: str,
        context: str,
        candidates: list[str],
        entity_type: str,
    ) -> None:
        """
        Flag a reference that couldn't be resolved automatically.

        Args:
            reference_text: The ambiguous reference (e.g., "the main theorem")
            context: Surrounding text for disambiguation
            candidates: Possible temp_ids it could refer to
            entity_type: Type of entity ("theorem", "definition", etc.)
        """
        self.ambiguous_references.append(
            AmbiguousReference(
                reference_text=reference_text,
                context=context,
                candidates=candidates,
                entity_type=entity_type,
            )
        )

    def get_ambiguous_references(self) -> list[AmbiguousReference]:
        """Get all ambiguous references for manual resolution."""
        return self.ambiguous_references

    # =============================================================================
    # STATISTICS
    # =============================================================================

    def get_stats(self) -> dict[str, int]:
        """Get statistics on extracted entities."""
        return {
            "definitions": len(self.definitions),
            "theorems": len(self.theorems),
            "mathster": len(self.proofs),
            "equations": len(self.equations),
            "parameters": len(self.parameters),
            "remarks": len(self.remarks),
            "citations": len(self.citations),
            "ambiguous_references": len(self.ambiguous_references),
        }


# =============================================================================
# AMBIGUOUS REFERENCE
# =============================================================================


class AmbiguousReference(BaseModel):
    """
    Tracks an ambiguous reference that needs manual resolution.

    Examples:
        >>> ref = AmbiguousReference(
        ...     reference_text="the main theorem",
        ...     context="By the main theorem, we have...",
        ...     candidates=["raw-thm-001", "raw-thm-005"],
        ...     entity_type="theorem",
        ... )
    """

    reference_text: str = Field(..., description="The ambiguous reference text")
    context: str = Field(..., description="Surrounding context for disambiguation")
    candidates: list[str] = Field(
        default_factory=list, description="Possible temp_ids it could refer to"
    )
    entity_type: str = Field(..., description="Type of entity (theorem, definition, etc.)")
    resolved_id: str | None = Field(None, description="Resolved temp_id (once determined)")


# =============================================================================
# ENRICHMENT ERROR
# =============================================================================


class ErrorType(str, Enum):
    """Types of enrichment errors."""

    PARSE_FAILURE = "parse_failure"
    REFERENCE_UNRESOLVED = "reference_unresolved"
    VALIDATION_FAILURE = "validation_failure"
    LLM_CALL_FAILURE = "llm_call_failure"
    MISSING_DEPENDENCY = "missing_dependency"
    CIRCULAR_DEPENDENCY = "circular_dependency"


class EnrichmentError(Exception):
    """
    Exception for enrichment failures during Stage 2.

    Preserves raw data and error context for debugging and retries.

    Attributes:
        error_type: Classification of the error
        message: Human-readable error description
        entity_id: Temp ID of the entity being enriched
        raw_data: Original raw entity data (preserved for retry)
        context: Additional context (e.g., which reference failed)

    Examples:
        >>> raise EnrichmentError(
        ...     error_type=ErrorType.REFERENCE_UNRESOLVED,
        ...     message="Cannot resolve 'Theorem 2.1' reference",
        ...     entity_id="raw-proof-001",
        ...     raw_data={"proves_label_text": "Theorem 2.1", ...},
        ...     context={"reference": "Theorem 2.1"}
        ... )

    Maps to Lean:
        inductive EnrichmentError where
          | ParseFailure : String → String → EnrichmentError
          | ReferenceUnresolved : String → EnrichmentError
          | ValidationFailure : String → EnrichmentError
          | LLMCallFailure : String → EnrichmentError
    """

    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        entity_id: str | None = None,
        raw_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ):
        """
        Initialize EnrichmentError.

        Args:
            error_type: Classification of error
            message: Human-readable description
            entity_id: Temp ID of entity being enriched
            raw_data: Original raw entity (for debugging/retry)
            context: Additional context dictionary
        """
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.entity_id = entity_id
        self.raw_data = raw_data or {}
        self.context = context or {}

    def __repr__(self) -> str:
        return (
            f"EnrichmentError(type={self.error_type.value}, "
            f"message='{self.message}', entity_id={self.entity_id})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "entity_id": self.entity_id,
            "raw_data": self.raw_data,
            "context": self.context,
        }

    def to_info(self) -> EnrichmentErrorInfo:
        """
        Convert exception to serializable EnrichmentErrorInfo model.

        Returns:
            EnrichmentErrorInfo: Pydantic-compatible error information

        Note:
            Import is done inside method to avoid circular dependency
            since EnrichmentErrorInfo is defined after this class.
        """
        # Import here to avoid circular dependency
        # EnrichmentErrorInfo is defined later in this file
        return EnrichmentErrorInfo(
            error_type=self.error_type,
            message=self.message,
            entity_id=self.entity_id,
            raw_data=self.raw_data or None,
            context=self.context or None,
        )


# =============================================================================
# VALIDATION RESULT
# =============================================================================


class ValidationResult(BaseModel):
    """
    Result of validating an enriched entity.

    Tracks validation success/failure and any errors encountered.

    Examples:
        >>> result = ValidationResult(
        ...     is_valid=False, errors=["Missing source location", "Invalid label format"]
        ... )
    """

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(default_factory=list, description="Validation error messages")
    warnings: list[str] = Field(default_factory=list, description="Non-critical warnings")
    entity_id: str | None = Field(None, description="Entity being validated")


# =============================================================================
# ENRICHMENT STATUS
# =============================================================================


class EnrichmentStatus(str, Enum):
    """Status of entity enrichment."""

    PENDING = "pending"  # Not yet enriched
    IN_PROGRESS = "in_progress"  # Currently being enriched
    COMPLETED = "completed"  # Successfully enriched
    FAILED = "failed"  # Enrichment failed
    SKIPPED = "skipped"  # Skipped due to missing dependencies


class EnrichmentErrorInfo(BaseModel):
    """
    Serializable error information for enrichment failures.

    This is the Pydantic-compatible version of EnrichmentError exception.
    Use this in Pydantic models to store error information.
    """

    error_type: ErrorType = Field(..., description="Classification of the error")
    message: str = Field(..., description="Human-readable error description")
    entity_id: str | None = Field(None, description="Temp ID of the entity being enriched")
    raw_data: dict[str, Any] | None = Field(None, description="Original raw entity data")
    context: dict[str, Any] | None = Field(None, description="Additional error context")

    model_config = ConfigDict(frozen=True)


class EntityEnrichmentStatus(BaseModel):
    """
    Tracks enrichment status for a single entity.

    Examples:
        >>> status = EntityEnrichmentStatus(
        ...     temp_id="raw-thm-001", status=EnrichmentStatus.COMPLETED
        ... )
    """

    temp_id: str = Field(..., description="Entity temp ID")
    status: EnrichmentStatus = Field(default=EnrichmentStatus.PENDING)
    error: EnrichmentErrorInfo | None = Field(None, description="Error information if failed")
    attempts: int = Field(default=0, description="Number of enrichment attempts")
    enriched_label: str | None = Field(None, description="Final label if successfully enriched")
