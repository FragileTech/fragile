"""
Enriched Mathematical Entity Types for LLM Pipeline.

This module provides additional mathematical entity types created during
the Extract-then-Enrich pipeline:

- EquationBox: Mathematical equations (numbered and unnumbered)
- ParameterBox: Parameters, constants, and variables
- RemarkBox: Remarks, notes, observations, examples

These complement the core types in math_types.py and enable full semantic
capture of mathematical research papers.

Maps to Lean:
    structure EquationBox where
      label : String
      latex_content : String
      introduces_symbols : List String
      ...

    structure ParameterBox where
      label : String
      symbol : String
      domain : ParameterType
      ...

    structure RemarkBox where
      label : String
      remark_type : RemarkType
      content : String
      ...
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mathster.proof_pipeline.math_types import ParameterType


if TYPE_CHECKING:
    from mathster.core.article_system import SourceLocation
    from mathster.core.raw_data import RawEquation, RawParameter, RawRemark
    from mathster.sympy_integration.dual_representation import DualStatement


# =============================================================================
# ENUMS
# =============================================================================


class RemarkType(str, Enum):
    """Types of mathematical remarks."""

    NOTE = "note"
    REMARK = "remark"
    OBSERVATION = "observation"
    COMMENT = "comment"
    EXAMPLE = "example"
    INTUITION = "intuition"
    WARNING = "warning"


class ParameterScope(str, Enum):
    """Scope of parameter definition."""

    GLOBAL = "global"  # Document-wide parameter
    LOCAL = "local"  # Section or theorem-specific
    UNIVERSAL = "universal"  # Cross-document constant (e.g., π, e)


# =============================================================================
# EQUATION BOX
# =============================================================================


class EquationBox(BaseModel):
    """
    Mathematical equation extracted from document.

    Represents both numbered and unnumbered display equations with full context,
    symbol tracking, and dual LaTeX/SymPy representation.

    Examples:
        >>> eq = EquationBox(
        ...     label="eq-langevin",
        ...     equation_number="(2.1)",
        ...     latex_content="dx_t = v_t dt, \\quad dv_t = -\\gamma v_t dt + \\sqrt{2\\gamma} dW_t",
        ...     dual_statement=DualStatement(...),
        ...     introduces_symbols=["x_t", "v_t", "gamma"],
        ...     context_before="The kinetic operator is defined by Langevin dynamics:",
        ...     source=SourceLocation(...),
        ... )

    Maps to Lean:
        structure EquationBox where
          label : String
          equation_number : Option String
          latex_content : String
          dual_statement : Option DualStatement
          introduces_symbols : List String
          uses_symbols : List String
          context_before : Option String
          context_after : Option String
          referenced_by : List String
          source : Option SourceLocation
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    label: str = Field(
        ...,
        pattern=r"^eq-[a-z0-9-]+$",
        description="Unique equation label (e.g., 'eq-langevin', 'eq-main-bound')",
    )

    equation_number: str | None = Field(
        None, description="Equation number if numbered (e.g., '(2.1)', '(17)')"
    )

    # Mathematical Content
    latex_content: str = Field(..., min_length=1, description="LaTeX content (verbatim)")

    dual_statement: DualStatement | None = Field(
        None, description="Dual LaTeX/SymPy representation if parseable"
    )

    # Symbol Tracking
    introduces_symbols: list[str] = Field(
        default_factory=list,
        description="Symbols first defined in this equation (e.g., ['x_t', 'v_t'])",
    )

    uses_symbols: list[str] = Field(
        default_factory=list, description="Symbols used from earlier definitions"
    )

    # Context
    context_before: str | None = Field(
        None, description="Text immediately before equation (for understanding)"
    )

    context_after: str | None = Field(
        None, description="Text immediately after equation (for understanding)"
    )

    # Cross-References
    referenced_by: list[str] = Field(
        default_factory=list,
        description="Entities referencing this equation (e.g., ['thm-main', 'proof-001'])",
    )

    appears_in_theorems: list[str] = Field(
        default_factory=list,
        description="Theorem labels where this equation appears in statement",
    )

    # Source
    source: SourceLocation = Field(description="Source location in documentation")

    # Error Tracking (for enrichment failures)
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Validation errors encountered during enrichment (if any)",
    )

    raw_fallback: dict[str, Any] | None = Field(
        None, description="Original raw data preserved on partial enrichment failure"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validator → Lean proof obligation: label well-formed."""
        if not v.startswith("eq-"):
            raise ValueError(f"Equation labels must start with 'eq-': {v}")
        return v

    @classmethod
    def from_raw(cls, raw: RawEquation, label: str, source: SourceLocation) -> EquationBox:
        """
        Create EquationBox from RawEquation (Stage 1 → Stage 2).

        Args:
            raw: RawEquation from extraction
            label: Final label to assign (e.g., "eq-langevin")
            source: Source location with line range

        Returns:
            EquationBox with enriched data

        Example:
            >>> raw_eq = RawEquation(
            ...     temp_id="raw-eq-001",
            ...     equation_label="(2.1)",
            ...     latex_content="f(x) = x^2",
            ...     ...
            ... )
            >>> eq = EquationBox.from_raw(raw_eq, "eq-quadratic", source_loc)
        """
        return cls(
            label=label,
            equation_number=raw.equation_label,
            latex_content=raw.latex_content,
            context_before=raw.context_before,
            context_after=raw.context_after,
            source=source,
            raw_fallback=raw.model_dump(),  # Preserve raw data
        )


# =============================================================================
# PARAMETER BOX
# =============================================================================


class ParameterBox(BaseModel):
    """
    Mathematical parameter, constant, or variable.

    Represents parameters that appear throughout a mathematical document,
    tracking their domain, scope, and meaning.

    Examples:
        >>> param = ParameterBox(
        ...     label="param-gamma",
        ...     symbol="γ",
        ...     latex="\\gamma",
        ...     domain=ParameterType.REAL,
        ...     meaning="friction coefficient controlling damping",
        ...     scope=ParameterScope.GLOBAL,
        ...     constraints=["γ > 0"],
        ...     source=SourceLocation(...),
        ... )

    Maps to Lean:
        structure ParameterBox where
          label : String
          symbol : String
          latex : String
          domain : ParameterType
          meaning : String
          scope : ParameterScope
          constraints : List String
          default_value : Option String
          source : Option SourceLocation
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    label: str = Field(
        ...,
        pattern=r"^param-[a-z0-9-]+$",
        description="Unique parameter label (e.g., 'param-gamma', 'param-N')",
    )

    symbol: str = Field(..., min_length=1, description="Symbol (e.g., 'γ', 'N', 'h')")

    latex: str = Field(
        ..., min_length=1, description="LaTeX representation (e.g., '\\gamma', 'N')"
    )

    # Mathematical Properties
    domain: ParameterType = Field(..., description="Mathematical domain")

    meaning: str = Field(..., min_length=1, description="Brief meaning/description")

    scope: ParameterScope = Field(default=ParameterScope.LOCAL, description="Parameter scope")

    # Constraints
    constraints: list[str] = Field(
        default_factory=list,
        description="Mathematical constraints (e.g., ['γ > 0', 'N >= 2'])",
    )

    default_value: str | None = Field(None, description="Default or typical value (if applicable)")

    # Context
    full_definition_text: str | None = Field(
        None, description="Complete defining text from source"
    )

    appears_in: list[str] = Field(
        default_factory=list,
        description="Entities using this parameter (e.g., ['thm-main', 'eq-langevin'])",
    )

    # Source
    source: SourceLocation = Field(..., description="Source location in documentation")

    # Error Tracking
    validation_errors: list[str] = Field(default_factory=list)
    raw_fallback: dict[str, Any] | None = Field(None)

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validator → Lean proof obligation: label well-formed."""
        if not v.startswith("param-"):
            raise ValueError(f"Parameter labels must start with 'param-': {v}")
        return v

    @classmethod
    def from_raw(
        cls,
        raw: RawParameter,
        label: str,
        domain: ParameterType,
        source: SourceLocation,
    ) -> ParameterBox:
        """
        Create ParameterBox from RawParameter (Stage 1 → Stage 2).

        Args:
            raw: RawParameter from extraction
            label: Final label to assign (e.g., "param-gamma")
            domain: Inferred mathematical domain
            source: Source location with line range

        Returns:
            ParameterBox with enriched data

        Example:
            >>> raw_param = RawParameter(
            ...     temp_id="raw-param-001",
            ...     symbol="γ",
            ...     meaning="friction coefficient",
            ...     scope="global",
            ...     ...
            ... )
            >>> param = ParameterBox.from_raw(
            ...     raw_param, "param-gamma", ParameterType.REAL, source_loc
            ... )
        """
        scope = ParameterScope.GLOBAL if raw.scope == "global" else ParameterScope.LOCAL

        return cls(
            label=label,
            symbol=raw.symbol,
            latex=raw.symbol,  # Can be enhanced with LaTeX parsing
            domain=domain,
            meaning=raw.meaning,
            scope=scope,
            full_definition_text=raw.full_text,
            source=source,
            raw_fallback=raw.model_dump(),
        )


# =============================================================================
# REMARK BOX
# =============================================================================


class RemarkBox(BaseModel):
    """
    Mathematical remark, note, observation, or example.

    Represents informal mathematical commentary that provides intuition,
    examples, or clarifications without being formal results.

    Examples:
        >>> remark = RemarkBox(
        ...     label="remark-kinetic-necessity",
        ...     remark_type=RemarkType.REMARK,
        ...     content="The condition v > 0 is essential. Without kinetic energy...",
        ...     relates_to=["thm-convergence", "def-walker"],
        ...     source=SourceLocation(...),
        ... )

    Maps to Lean:
        structure RemarkBox where
          label : String
          remark_type : RemarkType
          content : String
          relates_to : List String
          provides_intuition_for : List String
          source : Option SourceLocation
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    label: str = Field(
        ...,
        pattern=r"^remark-[a-z0-9-]+$",
        description="Unique remark label (e.g., 'remark-kinetic-necessity')",
    )

    remark_type: RemarkType = Field(..., description="Type of remark")

    # Content
    content: str = Field(..., min_length=1, description="Full text of the remark")

    # Cross-References
    relates_to: list[str] = Field(
        default_factory=list,
        description="Entities this remark relates to (e.g., ['thm-main', 'def-walker'])",
    )

    provides_intuition_for: list[str] = Field(
        default_factory=list,
        description="Entities this remark provides intuition for",
    )

    # Source
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation",
    )

    # Error Tracking
    validation_errors: list[str] = Field(default_factory=list)
    raw_fallback: dict[str, Any] | None = Field(None)

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validator → Lean proof obligation: label well-formed."""
        if not v.startswith("remark-"):
            raise ValueError(f"Remark labels must start with 'remark-': {v}")
        return v

    @classmethod
    def from_raw(
        cls,
        raw: RawRemark,
        label: str,
        source: SourceLocation,
    ) -> RemarkBox:
        """
        Create RemarkBox from RawRemark (Stage 1 → Stage 2).

        Args:
            raw: RawRemark from extraction
            label: Final label to assign (e.g., "remark-kinetic-necessity")
            source: Source location with line range

        Returns:
            RemarkBox with enriched data

        Example:
            >>> raw_remark = RawRemark(
            ...     temp_id="raw-remark-001",
            ...     remark_type="remark",
            ...     full_text="The condition v > 0 is essential...",
            ...     ...
            ... )
            >>> remark = RemarkBox.from_raw(raw_remark, "remark-kinetic-necessity", source_loc)
        """
        # Map string type to enum
        remark_type_map = {
            "note": RemarkType.NOTE,
            "remark": RemarkType.REMARK,
            "observation": RemarkType.OBSERVATION,
            "comment": RemarkType.COMMENT,
            "example": RemarkType.EXAMPLE,
        }
        remark_type = remark_type_map.get(raw.remark_type, RemarkType.REMARK)

        return cls(
            label=label,
            remark_type=remark_type,
            content=raw.full_text,
            source=source,
            raw_fallback=raw.model_dump(),
        )


# =============================================================================
# ENRICHED AXIOM BOX (Refined Data Stage)
# =============================================================================


class EnrichedAxiom(BaseModel):
    """
    Axiom in refined_data format (transitional stage between raw and pipeline).

    This schema is MORE PERMISSIVE than the pipeline Axiom schema because
    refined_data is a work-in-progress format that may have:
    - Mixed field names (source vs source_location)
    - Extra enrichment fields not in pipeline
    - Optional semantic metadata

    Use this for validating files in refined_data/ BEFORE conversion to pipeline_data/.

    Maps to Lean:
        structure EnrichedAxiom where
          label : String
          statement : String
          mathematical_expression : String
          name : Option String
          ...
    """

    model_config = ConfigDict(frozen=True, extra="allow")  # Allow extra fields

    # Required Core Fields
    label: str = Field(..., pattern=r"^(axiom-|ax-)[a-z0-9-]+$")
    statement: str = Field(..., min_length=1)
    mathematical_expression: str = Field(..., min_length=1)

    # Optional Enrichment Fields
    name: str | None = None
    natural_language_statement: str | None = None
    description: str | None = None
    entity_type: str | None = None
    statement_type: str | None = None

    # Framework Context
    foundational_framework: str | None = Field(default="Fragile Gas Framework")
    chapter: str | None = None
    document: str | None = None

    # Axiom-Specific Enrichment
    assumptions_hypotheses: list[dict[str, Any]] | None = None
    relations: list[dict[str, Any]] | None = None
    core_assumption: Any | None = None  # DualStatement or dict
    parameters: Any | None = None  # List[AxiomaticParameter] or list[dict]
    condition: Any | None = None  # DualStatement or dict
    failure_mode_analysis: str | None = None

    # Dependencies
    input_objects: list[str] = Field(default_factory=list)
    input_axioms: list[str] = Field(default_factory=list)
    input_parameters: list[str] = Field(default_factory=list)

    # Internal Structure
    lemma_dag_edges: list[list[str]] = Field(default_factory=list)
    internal_lemmas: list[str] = Field(default_factory=list)

    # Source
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation where this axiom is defined",
    )

    # Status
    proof_status: str | None = None
    conclusion_main_result: Any | None = None

    # Fallback
    raw_fallback: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# ENRICHED THEOREM BOX (Refined Data Stage)
# =============================================================================


class EnrichedTheorem(BaseModel):
    """
    Theorem/Lemma/Proposition in refined_data format.

    This schema handles the transitional format with mixed field names
    and optional enrichment metadata.

    Use this for validating theorems in refined_data/ BEFORE conversion.

    Maps to Lean:
        structure EnrichedTheorem where
          label : String
          name : String
          statement_type : String
          ...
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    # Required Core Fields
    label: str = Field(..., pattern=r"^(thm-|lem-|prop-|cor-)[a-z0-9-]+$")
    name: str = Field(..., min_length=1)
    statement_type: str  # "theorem", "lemma", "proposition", "corollary"

    # Optional Statement Fields
    natural_language_statement: str | None = None
    assumptions: list[Any] = Field(default_factory=list)  # DualStatement or dict or str
    conclusion: Any | None = None  # DualStatement or dict or str

    # Framework Context
    chapter: str | None = None
    document: str | None = None

    # Dependencies
    input_objects: list[str] = Field(default_factory=list)
    input_axioms: list[str] = Field(default_factory=list)
    input_parameters: list[str] = Field(default_factory=list)

    # Output Classification
    output_type: str | None = None  # TheoremOutputType (may be non-standard in refined)

    # Attributes/Properties (handles both names)
    attributes_added: list[dict[str, Any]] = Field(default_factory=list)
    attributes_required: dict[str, list[str]] = Field(default_factory=dict)
    properties_added: list[dict[str, Any]] | None = None  # Legacy name
    properties_required: dict[str, list[str]] | None = None  # Legacy name

    # Relationships
    relations_established: list[Any] = Field(default_factory=list)

    # Internal Structure
    internal_lemmas: list[str] = Field(default_factory=list)
    internal_propositions: list[str] = Field(default_factory=list)
    lemma_dag_edges: list[list[str]] = Field(default_factory=list)
    uses_definitions: list[str] = Field(default_factory=list)

    # Source
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation where this theorem is defined",
    )

    # Proof
    proof: Any | None = None  # ProofBox or dict
    proof_status: str | None = None

    # Additional Metadata
    equation_label: str | None = None
    validation_errors: list[str] = Field(default_factory=list)

    # Fallback
    raw_fallback: dict[str, Any] | None = None


# =============================================================================
# ENRICHED DEFINITION BOX (Refined Data Stage)
# =============================================================================


class EnrichedDefinition(BaseModel):
    """
    Definition in refined_data format (becomes MathematicalObject in pipeline).

    Definitions in refined_data have def-* labels and rich semantic metadata.
    During conversion, they become obj-* labeled MathematicalObject instances.

    Use this for validating definitions in refined_data/ BEFORE conversion.

    Maps to Lean:
        structure EnrichedDefinition where
          label : String
          name : String
          entity_type : String
          ...
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    # Required Core Fields
    label: str = Field(..., pattern=r"^(def-|obj-)[a-z0-9-]+$")
    name: str = Field(..., min_length=1)

    # Type Classification
    entity_type: str = Field(default="definition")
    statement_type: str | None = None

    # Content
    natural_language_statement: str | None = None
    description: str | None = None
    formal_statement: str | None = None  # Mathematical expression

    # Framework Context
    chapter: str | None = None
    document: str | None = None

    # Dependencies
    input_objects: list[str] = Field(default_factory=list)
    input_axioms: list[str] = Field(default_factory=list)
    input_parameters: list[str] = Field(default_factory=list)

    # Relationships
    relations: list[dict[str, Any]] = Field(default_factory=list)

    # Source
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation where this definition is defined",
    )

    # Status
    proof_status: str | None = None

    # Fallback
    raw_fallback: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# ENRICHED OBJECT BOX (Refined Data Stage - Standalone Objects)
# =============================================================================


class EnrichedObject(BaseModel):
    """
    Standalone mathematical object in refined_data format (not from definition).

    Similar to EnrichedDefinition but for obj-* labeled entities that weren't
    originally definitions.

    Use this for validating standalone objects in refined_data/.

    Maps to Lean:
        structure EnrichedObject where
          label : String
          name : String
          ...
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    # Required Core Fields
    label: str = Field(..., pattern=r"^obj-[a-z0-9-]+$")
    name: str = Field(..., min_length=1)

    # Type Classification
    entity_type: str | None = Field(default="object")
    object_type: str | None = None  # ObjectType enum value

    # Content
    mathematical_expression: str | None = None
    description: str | None = None

    # Framework Context
    chapter: str | None = None
    document: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Attributes (properties assigned by theorems)
    current_attributes: list[dict[str, Any]] = Field(default_factory=list)
    attribute_history: list[dict[str, Any]] = Field(default_factory=list)

    # Source
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation where this object is defined",
    )

    # Fallback
    raw_fallback: dict[str, Any] = Field(default_factory=dict)
