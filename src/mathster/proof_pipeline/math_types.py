"""
Mathematical Type System for Document Parsing and Representation.

This module contains all mathematical entities that appear in documents and are
created by the document-parser agent when extracting structured data from
markdown mathematical specifications.

Mathematical entities include:
- Objects (MathematicalObject): Sets, functions, measures, operators, etc.
- Properties (Property): Characteristics assigned to objects by theorems
- Relationships (Relationship): Connections between objects (equivalence, embedding, etc.)
- Axioms (Axiom): Foundational truths that are never proved
- Parameters (Parameter): Configuration values and constraints
- Theorems (TheoremBox): Mathematical statements (theorems, lemmas, propositions)

All types follow Lean-compatible patterns from docs/LEAN_EMULATION_GUIDE.md:
- frozen=True (immutability)
- Pure functions (no side effects)
- Total functions (Optional[T] instead of exceptions)
- Explicit types (no Any)
- Validators that map to Lean proof obligations

Version: 2.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


if TYPE_CHECKING:
    from mathster.core.article_system import SourceLocation
    from mathster.proof_pipeline.proof_system import ProofBox
    from mathster.sympy_integration.dual_representation import DualStatement


# =============================================================================
# ENUMS (Map to Lean Inductives)
# =============================================================================


class TheoremOutputType(str, Enum):
    """
    The 16 fundamental theorem output types providing complete coverage
    of mathematical reasoning.

    Maps to Lean:
        inductive TheoremOutputType where
          | property : TheoremOutputType
          | relation : TheoremOutputType
          ...
    """

    PROPERTY = "Property"
    RELATION = "Relation"
    EXISTENCE = "Existence"
    CONSTRUCTION = "Construction"
    CLASSIFICATION = "Classification"
    UNIQUENESS = "Uniqueness"
    IMPOSSIBILITY = "Impossibility"
    EMBEDDING = "Embedding"
    APPROXIMATION = "Approximation"
    EQUIVALENCE = "Equivalence"
    DECOMPOSITION = "Decomposition"
    EXTENSION = "Extension"
    REDUCTION = "Reduction"
    BOUND = "Bound"
    CONVERGENCE = "Convergence"
    CONTRACTION = "Contraction"


class ObjectType(str, Enum):
    """Mathematical object categories."""

    SET = "set"
    FUNCTION = "function"
    MEASURE = "measure"
    SPACE = "space"
    OPERATOR = "operator"
    DISTRIBUTION = "distribution"
    FIELD = "field"
    STRUCTURE = "structure"


class ParameterType(str, Enum):
    """Parameter value types."""

    REAL = "real"
    INTEGER = "integer"
    NATURAL = "natural"
    RATIONAL = "rational"
    COMPLEX = "complex"
    BOOLEAN = "boolean"
    SYMBOLIC = "symbolic"


class RefinementType(str, Enum):
    """Attribute refinement types."""

    STRENGTHENING = "strengthening"  # e.g., continuous → C^∞
    GENERALIZATION = "generalization"  # e.g., bounded → L^p
    QUANTIFICATION = "quantification"  # e.g., exists → forall


class AttributeEventType(str, Enum):
    """Attribute timeline event types."""

    ADDED = "added"
    REFINED = "refined"
    CONDITIONAL_UPGRADE = "conditional_upgrade"


class RelationType(str, Enum):
    """
    Types of relationships between mathematical objects.

    Maps to Lean:
        inductive RelationType where
          | equivalence : RelationType
          | embedding : RelationType
          ...
    """

    EQUIVALENCE = "equivalence"  # A ≡ B (bidirectional, transitive)
    EMBEDDING = "embedding"  # A ↪ B (directed, structure-preserving)
    APPROXIMATION = "approximation"  # A ≈ B (directed, with error bounds)
    REDUCTION = "reduction"  # A → B (directed, complexity reduction)
    EXTENSION = "extension"  # A → B (directed, extension of scope)
    GENERALIZATION = "generalization"  # A → B (directed, broader applicability)
    SPECIALIZATION = "specialization"  # A → B (directed, narrower scope)
    OTHER = "other"  # Custom relationship type


# =============================================================================
# ATTRIBUTE TYPES
# =============================================================================


class AttributeRefinement(BaseModel):
    """
    Tracks when a attribute is strengthened or generalized.

    Example: attr-continuous → attr-smooth (strengthening)

    Maps to Lean:
        structure AttributeRefinement where
          original_attribute : String
          refined_attribute : String
          ...
    """

    model_config = ConfigDict(frozen=True)

    original_attribute: str = Field(..., pattern=r"^attr-[a-z0-9-]+$")
    refined_attribute: str = Field(..., pattern=r"^attr-[a-z0-9-]+$")
    refinement_theorem: str = Field(..., pattern=r"^(thm|lem|prop)-[a-z0-9-]+$")
    refinement_type: RefinementType

    @field_validator("original_attribute", "refined_attribute")
    @classmethod
    def validate_attribute_label(cls, v: str) -> str:
        """Validator → Lean proof obligation: labels well-formed."""
        if not v.startswith("attr-"):
            raise ValueError(f"Attribute labels must start with 'attr-': {v}")
        return v


class Attribute(BaseModel):
    """
    Attribute assigned to a mathematical object by a theorem.

    Conditionality is COMPUTED dynamically (not stored here).
    Check theorem.attributes_required vs object.current_attributes.

    Maps to Lean:
        structure Attribute where
          label : String
          expression : String
          object_label : String
          established_by : String
          ...

          def is_unconditional (a : Attribute) (conds : List String) : Bool :=
            conds.isEmpty
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(..., pattern=r"^attr-[a-z0-9-]+$")
    expression: str = Field(..., min_length=1, description="LaTeX mathematical expression")
    object_label: str = Field(..., pattern=r"^obj-[a-z0-9-]+$")
    established_by: str = Field(..., pattern=r"^(thm|lem|prop)-[a-z0-9-]+$")
    timestamp: int | None = Field(None, ge=0, description="Pipeline execution step")
    can_be_refined: bool = Field(True, description="Whether property can be strengthened")
    refinements: list[AttributeRefinement] = Field(default_factory=list)
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation where this property is defined",
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validator → Lean proof obligation."""
        if not v.startswith("attr-"):
            raise ValueError(f"Attribute labels must start with 'attr-': {v}")
        return v


class AttributeEvent(BaseModel):
    """Event in attribute history timeline."""

    model_config = ConfigDict(frozen=True)

    timestamp: int = Field(..., ge=0)
    attribute_label: str = Field(..., pattern=r"^attr-[a-z0-9-]+$")
    added_by_theorem: str = Field(..., pattern=r"^(thm|lem|prop)-[a-z0-9-]+$")
    event_type: AttributeEventType


# =============================================================================
# RELATIONSHIP TYPES
# =============================================================================


class RelationshipAttribute(BaseModel):
    """
    Attribute of a relationship (e.g., error bounds, convergence rates).

    Example: For an approximation relationship, might have error = O(N^{-1/d})

    Maps to Lean:
        structure RelationshipAttribute where
          label : String
          expression : String
          description : String
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(
        ...,
        pattern=r"^[a-z][a-zA-Z0-9-]*$",
        description="Attribute label (e.g., 'approx-error-N')",
    )
    expression: str = Field(..., min_length=1, description="Mathematical expression")
    description: str | None = None

    @field_validator("label")
    @classmethod
    def validate_label_format(cls, v: str) -> str:
        """Validator → Lean proof obligation: label well-formed."""
        if not v[0].islower():
            raise ValueError(f"Relationship property labels must start with lowercase: {v}")
        return v


class Relationship(BaseModel):
    """
    Relationship between two mathematical objects established by a theorem.

    First-class object with label, type, directionality, and properties.

    Examples:
        - Equivalence: discrete ≡ continuous (bidirectional)
        - Embedding: particles ↪ fluid (directed)
        - Approximation: discrete ≈ continuous with O(N^{-1/d}) (directed)
        - Reduction: PDE → ODE (directed)

    Maps to Lean:
        structure Relationship where
          label : String
          relationship_type : RelationType
          source_object : String
          target_object : String
          bidirectional : Bool
          established_by : String
          attributes : List RelationshipAttribute
          expression : String

          def is_symmetric (r : Relationship) : Bool :=
            r.bidirectional

          def is_directed (r : Relationship) : Bool :=
            !r.bidirectional
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    label: str = Field(
        ...,
        pattern=r"^rel-[a-z0-9]+(-[a-z0-9]+)*-(equivalence|embedding|approximation|reduction|extension|generalization|specialization|other)$",
        description="Relationship label (format: rel-{source}-{target}-{type})",
    )

    # Type and directionality
    relationship_type: RelationType
    bidirectional: bool = Field(
        ...,
        description="Whether relationship is symmetric (auto-computed from type if not provided)",
    )

    # Objects involved
    source_object: str = Field(..., pattern=r"^obj-[a-z0-9-]+$", description="Source object label")
    target_object: str = Field(..., pattern=r"^obj-[a-z0-9-]+$", description="Target object label")

    # Metadata
    established_by: str = Field(
        ...,
        pattern=r"^(thm|lem|prop)-[a-z0-9-]+$",
        description="Theorem/lemma/proposition that established this relationship",
    )
    expression: str = Field(
        ..., min_length=1, description="Mathematical expression of relationship"
    )

    # Properties
    attributes: list[RelationshipAttribute] = Field(
        default_factory=list,
        description="Relationship-specific attributes (e.g., error bounds, convergence rates)",
    )

    # Tags for categorization
    tags: list[str] = Field(
        default_factory=list,
        description="Category tags (e.g., 'mean-field', 'discrete-continuous')",
    )

    # Timestamp
    timestamp: int | None = Field(
        None, ge=0, description="Pipeline execution step when established"
    )

    # Source
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation where this relationship is defined",
    )

    @field_validator("label")
    @classmethod
    def validate_label_format(cls, v: str) -> str:
        """Validator → Lean proof obligation: label well-formed."""
        if not v.startswith("rel-"):
            raise ValueError(f"Relationship labels must start with 'rel-': {v}")
        # Check ends with valid relationship type
        valid_types = [t.value for t in RelationType]
        if not any(v.endswith(f"-{t}") for t in valid_types):
            raise ValueError(f"Relationship label must end with relationship type: {v}")
        return v

    @model_validator(mode="after")
    def compute_bidirectionality(self) -> Relationship:
        """
        Auto-compute bidirectionality based on relationship type if needed.
        Equivalence is always bidirectional, others are typically directed.

        Maps to Lean:
            def compute_bidirectionality (r : Relationship) : Relationship :=
              match r.relationship_type with
              | RelationType.equivalence => { r with bidirectional := true }
              | _ => r
        """
        # Equivalence relationships are always bidirectional
        if self.relationship_type == RelationType.EQUIVALENCE:
            if not self.bidirectional:
                # Create updated copy with bidirectional=True
                return self.model_copy(update={"bidirectional": True})
        return self

    # Pure function: Check if symmetric
    def is_symmetric(self) -> bool:
        """
        Pure function: Check if relationship is symmetric (bidirectional).

        Maps to Lean:
            def is_symmetric (r : Relationship) : Bool :=
              r.bidirectional
        """
        return self.bidirectional

    # Pure function: Check if directed
    def is_directed(self) -> bool:
        """
        Pure function: Check if relationship is directed (not bidirectional).

        Maps to Lean:
            def is_directed (r : Relationship) : Bool :=
              !r.bidirectional
        """
        return not self.bidirectional

    # Pure function: Get reverse relationship label
    def get_reverse_label(self) -> str | None:
        """
        Pure function: Get label for reverse relationship (if bidirectional).

        For symmetric relationships (equivalence), returns same label.
        For directed relationships, would need inverse type.

        Maps to Lean:
            def get_reverse_label (r : Relationship) : Option String :=
              if r.bidirectional then
                some r.label
              else
                none  -- Directed relationships don't have automatic reverses
        """
        if self.is_symmetric():
            return self.label
        return None  # Directed relationships don't have automatic reverses

    # Pure function: Get related objects
    def get_objects(self) -> tuple[str, str]:
        """
        Pure function: Get (source, target) object labels.

        Maps to Lean:
            def get_objects (r : Relationship) : (String × String) :=
              (r.source_object, r.target_object)
        """
        return (self.source_object, self.target_object)


# =============================================================================
# DEFINITION BOX
# =============================================================================


class DefinitionBox(BaseModel):
    """
    Represents a formal mathematical definition from a source document.

    A DefinitionBox captures the formal definition of a concept (e.g., "v-porous on lines"),
    distinguishing it from a MathematicalObject which is an instance. Definitions introduce
    new terminology and specify the precise conditions under which that terminology applies.

    Examples:
        - Definition of "v-porous on lines": Set E ⊂ ℝ^n is v-porous on lines if...
        - Definition of "Lipschitz continuous": Function f is Lipschitz if...
        - Definition of "uniformly convex": Space X is uniformly convex if...

    Maps to Lean:
        structure DefinitionBox where
          label : String
          term : String
          formal_statement : DualStatement
          applies_to_object_type : Option ObjectType
          parameters : List String
          source : Option SourceLocation
          natural_language_description : Option String
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    label: str = Field(
        ...,
        pattern=r"^def-[a-z0-9-]+$",
        description="Unique label for the definition (format: def-{concept-name})",
    )

    term: str = Field(
        ...,
        min_length=1,
        description="The term being defined (e.g., 'v-porous on lines', 'Lipschitz continuous')",
    )

    # The formal statement defining the term
    formal_statement: DualStatement | None = Field(
        None,
        description="The 'if and only if' or 'we say that...' statement expressing the definition formally. "
        "None if using simple from_raw() enrichment; populated by LLM enrichment pipeline.",
    )

    # The object type this definition applies to, if specific
    applies_to_object_type: ObjectType | None = Field(
        None,
        description="The category of object this definition applies to (e.g., SET, FUNCTION, SPACE)",
    )

    # Parameters used within the definition
    parameters: list[str] = Field(
        default_factory=list,
        description="Parameter labels used in the definition (e.g., ['param-v', 'param-h', 'param-beta'])",
    )

    # Link back to the source document
    source: SourceLocation | None = Field(
        None, description="Source location in documentation where this definition appears"
    )

    # The original prose from the paper
    natural_language_description: str | None = Field(
        None,
        description="The original prose from the paper describing the definition, for reference",
    )

    # Tags for categorization
    tags: list[str] = Field(
        default_factory=list,
        description="Category tags for filtering/searching (e.g., 'porosity', 'regularity', 'geometric')",
    )

    # Chapter and document for organization
    chapter: str | None = Field(
        None,
        description="High-level chapter/folder name (e.g., '1_euclidean_gas', '2_geometric_gas')",
    )

    document: str | None = Field(
        None, description="Document/subdirectory name (e.g., '03_cloning', '06_convergence')"
    )

    # Error Tracking (for LLM enrichment failures)
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
        if not v.startswith("def-"):
            raise ValueError(f"Definition labels must start with 'def-': {v}")
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v: list[str]) -> list[str]:
        """Validator → Lean proof obligation: all parameter labels well-formed."""
        for label in v:
            if not label.startswith("param-"):
                raise ValueError(f"Parameter labels must start with 'param-': {label}")
        return v

    @classmethod
    def from_raw(
        cls,
        raw: RawDefinition,  # type: ignore  # Forward reference
        source: SourceLocation,
        chapter: str | None = None,
        document: str | None = None,
    ) -> DefinitionBox:
        """
        Create a DefinitionBox from a RawDefinition staging model.

        This is a simple enrichment that stores the raw text as-is.
        For full enrichment with DualStatement parsing, use the LLM-based
        enrichment pipeline.

        Args:
            raw: The raw definition from Stage 1 extraction
            source: Source location in documentation (REQUIRED)
            chapter: Chapter identifier (e.g., "1_euclidean_gas")
            document: Document identifier (e.g., "01_fragile_gas_framework")

        Returns:
            Enriched DefinitionBox instance

        Raises:
            ValueError: If source is None

        Examples:
            >>> raw_def = RawDefinition(
            ...     temp_id="raw-def-1",
            ...     label_text="def-walker-state",
            ...     term="Walker State",
            ...     statement_text="A walker state is...",
            ...     source_section="§1",
            ... )
            >>> definition = DefinitionBox.from_raw(raw_def)
            >>> definition.label
            'def-walker-state'
        """

        # Generate label from term (normalize to def- prefix)
        # RawDefinition doesn't have label_text, we create it from temp_id or term
        term_slug = raw.term_being_defined.lower().replace(" ", "-").replace("_", "-")
        label = f"def-{term_slug}"

        # Simple enrichment: store raw text, don't create DualStatement
        # Full DualStatement parsing requires LLM enrichment pipeline
        return cls(
            label=label,
            term=raw.term_being_defined,
            formal_statement=None,  # Requires LLM enrichment pipeline
            applies_to_object_type=None,  # Requires semantic analysis
            parameters=[],  # Requires semantic analysis to extract from parameters_mentioned
            source=source,
            natural_language_description=raw.full_text,
            tags=[],  # Requires semantic analysis
            chapter=chapter,
            document=document,
            validation_errors=[],
            raw_fallback=raw.model_dump() if raw else None,
        )


# =============================================================================
# MATHEMATICAL OBJECT
# =============================================================================


class MathematicalObject(BaseModel):
    """
    Mathematical object created by Definition directives.

    Objects accumulate attributes as they flow through theorems.
    Only definitions create objects; theorems only add attributes.

    Note: current_attributes is mutable (tracked externally in pipeline state).
    This model is immutable (frozen), but we track mutations via model_copy.

    Maps to Lean:
        structure MathematicalObject where
          label : String
          name : String
          mathematical_expression : String
          object_type : ObjectType
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(..., pattern=r"^obj-[a-z0-9-]+$")
    name: str = Field(..., min_length=1)
    mathematical_expression: str = Field(..., min_length=1)
    object_type: ObjectType
    current_attributes: list[Attribute] = Field(default_factory=list)
    attribute_history: list[AttributeEvent] = Field(default_factory=list)
    tags: list[str] = Field(
        default_factory=list,
        description="Category tags for filtering/searching (e.g., 'discrete', 'euclidean-gas', 'particle')",
    )
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation where this object is defined",
    )
    chapter: str | None = Field(
        None,
        description="High-level chapter/folder name (e.g., '1_euclidean_gas', '2_geometric_gas')",
    )
    document: str | None = Field(
        None, description="Document/subdirectory name (e.g., '03_cloning', '06_convergence')"
    )

    # Link to formal definition (NEW)
    definition_label: str | None = Field(
        None,
        pattern=r"^def-[a-z0-9-]+$",
        description="The label of the DefinitionBox that formally defines this type of object. "
        "Links objects back to their conceptual definitions.",
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validator → Lean proof obligation."""
        if not v.startswith("obj-"):
            raise ValueError(f"Object labels must start with 'obj-': {v}")
        return v

    # Pure function: Check if object has property
    def has_attribute(self, attribute_label: str) -> bool:
        """
        Pure function: Check if object has a specific attribute.

        Maps to Lean:
            def has_attribute (obj : MathematicalObject) (attr : String) : Bool :=
              obj.current_attributes.any (fun a => a.label == attr)
        """
        return any(p.label == attribute_label for p in self.current_attributes)

    # Pure function: Get property labels
    def get_attribute_labels(self) -> set[str]:
        """
        Pure function: Get set of attribute labels this object has.

        Maps to Lean:
            def get_attribute_labels (obj : MathematicalObject) :=
              obj.current_attributes.map (fun a => a.label)
        """
        return {p.label for p in self.current_attributes}

    # Total function: Get property by label
    def get_attribute(self, attribute_label: str) -> Attribute | None:
        """
        Total function: Get attribute by label (returns None if not found).

        Maps to Lean:
            def get_attribute (obj : MathematicalObject) (label : String) : Option Attribute :=
              obj.current_attributes.find? (fun a => a.label == label)
        """
        for prop in self.current_attributes:
            if prop.label == attribute_label:
                return prop
        return None

    # Pure function: Add property (returns new object)
    def add_attribute(self, attr: Attribute, timestamp: int) -> MathematicalObject:
        """
        Pure function: Add attribute to object (immutable update).

        Returns new MathematicalObject with attribute added.

        Maps to Lean:
            def add_attribute (obj : MathematicalObject) (a : Attribute) : MathematicalObject :=
              { obj with
                current_attributes := obj.current_attributes ++ [a]
                attribute_history := obj.attribute_history ++ [event] }
        """
        event = AttributeEvent(
            timestamp=timestamp,
            attribute_label=attr.label,
            added_by_theorem=attr.established_by,
            event_type=AttributeEventType.ADDED,
        )

        return self.model_copy(
            update={
                "current_attributes": [*self.current_attributes, attr],
                "attribute_history": [*self.attribute_history, event],
            }
        )


# =============================================================================
# AXIOM
# =============================================================================


class Axiom(BaseModel):
    """
    Immutable foundational truth never modified by theorems.

    Axioms can be used by any theorem but are never proved.

    The Fragile framework uses multi-part axioms with optional structured fields:
    - name: Human-readable title
    - core_assumption: The fundamental claim (enriched to DualStatement)
    - parameters: Mathematical objects used in the axiom
    - condition: Formal applicability condition (enriched to DualStatement)
    - failure_mode_analysis: What happens if the axiom is violated
    - source: Where the axiom appears in the document

    Maps to Lean:
        structure Axiom where
          label : String
          statement : String
          mathematical_expression : String
          foundational_framework : String
          name : Option String
          core_assumption : Option DualStatement
          parameters : Option (List AxiomaticParameter)
          condition : Option DualStatement
          failure_mode_analysis : Option String
          source : Option SourceLocation
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(..., pattern=r"^axiom-[a-z0-9-]+$")
    statement: str = Field(..., min_length=1)
    mathematical_expression: str = Field(..., min_length=1)
    foundational_framework: str = Field(..., min_length=1)
    chapter: str | None = Field(
        None,
        description="High-level chapter/folder name (e.g., '1_euclidean_gas', '2_geometric_gas')",
    )
    document: str | None = Field(
        None, description="Document/subdirectory name (e.g., '03_cloning', '06_convergence')"
    )

    # Optional structured fields (from Extract-then-Enrich pipeline)
    name: str | None = Field(None, description="Human-readable title/name of the axiom")

    core_assumption: DualStatement | None = Field(
        None, description="The fundamental assumption or claim, enriched to dual representation"
    )

    parameters: list[AxiomaticParameter] | None = Field(
        None, description="Mathematical objects and symbols used in the axiom definition"
    )

    condition: DualStatement | None = Field(
        None,
        description="Formal statement of when the axiom applies, enriched to dual representation",
    )

    failure_mode_analysis: str | None = Field(
        None, description="Analysis of what happens if the axiom is violated"
    )

    source: SourceLocation = Field(
        ..., description="Location in the source document where this axiom appears"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validator → Lean proof obligation."""
        if not v.startswith("axiom-"):
            raise ValueError(f"Axiom labels must start with 'axiom-': {v}")
        return v

    @classmethod
    def from_raw(
        cls,
        raw: RawAxiom,  # type: ignore  # Forward reference
        source: SourceLocation,
        chapter: str | None = None,
        document: str | None = None,
    ) -> Axiom:
        """
        Create an Axiom from a RawAxiom staging model.

        This is a simple enrichment that stores raw text fields directly.
        For full enrichment with DualStatement parsing, use the LLM-based
        enrichment pipeline.

        Args:
            raw: The raw axiom from Stage 1 extraction
            source: Source location in documentation (REQUIRED)
            chapter: Chapter identifier (e.g., "1_euclidean_gas")
            document: Document identifier (e.g., "01_fragile_gas_framework")

        Returns:
            Enriched Axiom instance

        Examples:
            >>> raw_axiom = RawAxiom(
            ...     temp_id="raw-axiom-1",
            ...     label_text="axiom-bounded-displacement",
            ...     name="Axiom of Bounded Displacement",
            ...     core_assumption_text="All walkers move within bounded distances",
            ...     parameters_text=["ε > 0"],
            ...     condition_text="When dt < ε",
            ...     source_section="§1",
            ... )
            >>> axiom = Axiom.from_raw(raw_axiom, chapter="1_euclidean_gas")
            >>> axiom.label
            'axiom-bounded-displacement'
        """

        # Extract basic fields
        label = raw.label_text
        if not label.startswith("axiom-"):
            # Normalize label if needed
            label = f"axiom-{label.replace('def-axiom-', '').replace('axiom-', '')}"

        # For simple enrichment, combine all text into the statement field
        statement_parts = [raw.core_assumption_text]
        if raw.condition_text:
            statement_parts.append(f"Condition: {raw.condition_text}")
        if raw.failure_mode_analysis_text:
            statement_parts.append(f"Failure mode: {raw.failure_mode_analysis_text}")
        statement = "\n\n".join(statement_parts)

        # Create enriched parameters (if present)
        parameters = None
        if raw.parameters_text:
            from mathster.core.math_types import AxiomaticParameter

            def extract_symbol(param_text: str) -> str:
                """Extract symbol from parameter text like 'ε > 0' or 'Δt: time step'."""
                # If there's a colon, everything before it is the symbol
                if ":" in param_text:
                    return param_text.split(":")[0].strip()

                # Otherwise, extract first word/symbol before operators
                import re

                # Match symbol before operators like >, <, =, ∈, ≤, ≥, etc.
                match = re.match(
                    r"^([a-zA-Zα-ωΑ-Ω\u0370-\u03FF\u1F00-\u1FFF]+|[^\s><=∈≤≥]+)\s*[><=∈≤≥]",
                    param_text,
                )
                if match:
                    return match.group(1).strip()

                # If no operator, take first word
                return param_text.split()[0] if param_text.split() else param_text[:10]

            parameters = [
                AxiomaticParameter(
                    symbol=extract_symbol(param_text), description=param_text, constraints=None
                )
                for param_text in raw.parameters_text
            ]

        return cls(
            label=label,
            statement=statement,
            mathematical_expression=raw.core_assumption_text,
            foundational_framework=raw.name,
            chapter=chapter,
            document=document,
            name=raw.name,
            # Store structured fields (not yet enriched to DualStatement)
            core_assumption=None,  # Requires LLM for DualStatement parsing
            parameters=parameters,
            condition=None,  # Requires LLM for DualStatement parsing
            failure_mode_analysis=raw.failure_mode_analysis_text,
            source=source,
        )


# =============================================================================
# PARAMETER
# =============================================================================


class Parameter(BaseModel):
    """
    Configuration value or constraint that controls theorem applicability.

    Not a mathematical object (doesn't accumulate properties).

    Maps to Lean:
        structure Parameter where
          label : String
          name : String
          symbol : String
          parameter_type : ParameterType
          constraints : Option String
          default_value : Option String
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(..., pattern=r"^param-[a-z0-9-]+$")
    name: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1)
    parameter_type: ParameterType
    constraints: str | None = None
    default_value: str | None = None
    chapter: str | None = Field(
        None,
        description="High-level chapter/folder name (e.g., '1_euclidean_gas', '2_geometric_gas')",
    )
    document: str | None = Field(
        None, description="Document/subdirectory name (e.g., '03_cloning', '06_convergence')"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validator → Lean proof obligation."""
        if not v.startswith("param-"):
            raise ValueError(f"Parameter labels must start with 'param-': {v}")
        return v


class AxiomaticParameter(BaseModel):
    """
    A mathematical object or symbol used in an axiom definition.

    Unlike the global Parameter class, AxiomaticParameter is local to a specific
    axiom and describes the entities that appear in the axiom's formal statement.

    Examples:
        - "v > 0" (viscosity parameter with constraint)
        - "U: 𝒳 → ℝ is Lipschitz" (potential function with property)
        - "γ ∈ (0, 1]" (friction coefficient with bounds)

    Maps to Lean:
        structure AxiomaticParameter where
          symbol : String
          description : String
          constraints : Option String
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(
        ..., min_length=1, description="Symbol or name (e.g., 'v', 'U', 'γ', 'walker state w')"
    )

    description: str = Field(
        ..., min_length=1, description="Natural language description of what this represents"
    )

    constraints: str | None = Field(
        None, description="Formal constraints or properties (e.g., 'v > 0', 'U is Lipschitz')"
    )


# =============================================================================
# THEOREM BOX
# =============================================================================


class TheoremBox(BaseModel):
    """
    Mathematical statement as processing box (theorem, lemma, or proposition).

    All three types work identically - they are distinguished only by label prefix:
    - thm-* : Major results (theorems)
    - lem-* : Supporting results (lemmas)
    - prop-* : Intermediate results (propositions)

    The statement_type field is auto-detected from the label prefix.

    Inputs: objects, axioms, parameters
    Attributes Required: API signature (attributes objects must have)
    Internal Processing: DAG of lemmas
    Outputs: properties, relations, existence statements, etc.

    Conditionality is COMPUTED by checking attributes_required
    against each object's current_attributes.

    Maps to Lean:
        structure TheoremBox where
          label : String
          name : String
          statement_type : StatementType
          input_objects : List String
          input_axioms : List String
          input_parameters : List String
          attributes_required : HashMap String (List String)
          ...

          def compute_conditionality
            (thm : TheoremBox)
            (objects : HashMap String MathematicalObject)
            : List String := ...

          def is_conditional
            (thm : TheoremBox)
            (objects : HashMap String MathematicalObject)
            : Bool :=
            !(thm.compute_conditionality objects).isEmpty
    """

    model_config = ConfigDict(frozen=True)

    # Metadata
    label: str = Field(..., pattern=r"^(thm|lem|prop)-[a-z0-9-]+$")
    name: str = Field(..., min_length=1)
    statement_type: Literal["theorem", "lemma", "proposition"] = Field(
        default="theorem",
        description="Type of mathematical statement (auto-detected from label prefix)",
    )
    source: SourceLocation = Field(
        ...,
        description="Source location in documentation where this theorem/lemma/proposition is defined",
    )
    chapter: str | None = Field(
        None,
        description="High-level chapter/folder name (e.g., '1_euclidean_gas', '2_geometric_gas')",
    )
    document: str | None = Field(
        None, description="Document/subdirectory name (e.g., '03_cloning', '06_convergence')"
    )

    # Proof Integration (NEW)
    proof: ProofBox | None = Field(
        None,
        description="Complete proof of this theorem (optional - enables sketch-first workflow)",
    )
    proof_status: Literal["unproven", "sketched", "expanded", "verified"] = Field(
        default="unproven", description="Current proof development status"
    )

    # Inputs (3 categories)
    input_objects: list[str] = Field(default_factory=list)
    input_axioms: list[str] = Field(default_factory=list)
    input_parameters: list[str] = Field(default_factory=list)

    # Properties Required (API Signature)
    attributes_required: dict[str, list[str]] = Field(
        default_factory=dict,
        description="API signature: {object_label: [required_attributes]}",
    )

    # Internal DAG
    internal_lemmas: list[str] = Field(default_factory=list)
    internal_propositions: list[str] = Field(default_factory=list)
    lemma_dag_edges: list[tuple[str, str]] = Field(
        default_factory=list, description="DAG edges: (from, to)"
    )

    # Outputs
    output_type: TheoremOutputType
    attributes_added: list[Attribute] = Field(default_factory=list)
    relations_established: list[Relationship] = Field(
        default_factory=list,
        description="Relationships between objects established by this theorem",
    )

    # Enhanced Semantic Capture (NEW - for PDF/paper processing)
    natural_language_statement: str | None = Field(
        None,
        description="The original, full prose of the theorem statement as it appears in the source text. "
        "Preserves the author's wording and presentation.",
    )

    assumptions: list[DualStatement] = Field(
        default_factory=list,
        description="A structured list of the theorem's explicit assumptions and preconditions "
        "(e.g., 'Let v > 0', 'Assume E is compact'). Each assumption is a DualStatement "
        "combining LaTeX and SymPy representations.",
    )

    conclusion: DualStatement | None = Field(
        None,
        description="The core conclusion of the theorem, as a machine-readable dual LaTeX/SymPy statement. "
        "For example, the main inequality or identity that the theorem establishes.",
    )

    equation_label: str | None = Field(
        None,
        description="The equation number or label associated with the conclusion in the source text "
        "(e.g., '(1.1)', '(3.5a)', 'eq:main-result').",
    )

    uses_definitions: list[str] = Field(
        default_factory=list,
        description="Labels of DefinitionBox entities required to understand the theorem statement "
        "(e.g., ['def-v-porous', 'def-lipschitz']). Links theorem to conceptual prerequisites.",
    )

    # Error Tracking (for LLM enrichment failures)
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
        if not v.startswith(("thm-", "lem-", "prop-", "attr-")):
            raise ValueError(f"Labels must start with 'thm-', 'lem-', 'prop-', or 'attr-': {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def auto_detect_statement_type(cls, data: dict) -> dict:
        """
        Auto-detect statement_type from label prefix.

        Maps to Lean:
            def auto_detect_statement_type (box : TheoremBox) : TheoremBox :=
              let type_val :=
                if box.label.startsWith "thm-" then "theorem"
                else if box.label.startsWith "lem-" then "lemma"
                else if box.label.startsWith "attr-" then "proposition"
                else "theorem"
              if box.statement_type != type_val then
                { box with statement_type := type_val }
              else box
        """
        # Get label from data
        label = data.get("label", "")

        # Determine statement_type from label prefix
        if label.startswith("thm-"):
            type_val = "theorem"
        elif label.startswith("lem-"):
            type_val = "lemma"
        elif label.startswith("attr-"):
            type_val = "proposition"
        else:
            type_val = "theorem"  # fallback

        # Set statement_type if not explicitly provided
        if "statement_type" not in data or data["statement_type"] == "theorem":
            data["statement_type"] = type_val

        return data

    @field_validator("input_objects")
    @classmethod
    def validate_input_objects(cls, v: list[str]) -> list[str]:
        """Validator → Lean proof obligation: all object labels well-formed."""
        # Accept obj- (preferred) or def- (legacy) prefixes
        for label in v:
            if not (label.startswith(("obj-", "def-"))):
                raise ValueError(f"Object labels must start with 'obj-' or 'def-': {label}")
        return v

    @field_validator("input_axioms")
    @classmethod
    def validate_input_axioms(cls, v: list[str]) -> list[str]:
        """Validator → Lean proof obligation: all axiom labels well-formed."""
        # Accept axiom- (preferred), ax- (short), or def-axiom- (legacy) prefixes
        for label in v:
            if not (label.startswith(("axiom-", "ax-", "def-axiom-"))):
                raise ValueError(
                    f"Axiom labels must start with 'axiom-', 'ax-', or 'def-axiom-': {label}"
                )
        return v

    # Pure function: Compute conditionality
    def compute_conditionality(self, objects: dict[str, MathematicalObject]) -> list[str]:
        """
        Pure function: Compute missing properties (implicit assumptions).

        Returns list of missing properties in format "obj-label:attr-label".
        Empty list = unconditional.

        Maps to Lean:
            def compute_conditionality
              (thm : TheoremBox)
              (objects : HashMap String MathematicalObject)
              : List String :=
              thm.attributes_required.foldl
                (fun acc (obj_label, required_attrs) =>
                  match objects.find? obj_label with
                  | none => acc ++ [s!"{obj_label}:OBJECT_NOT_FOUND"]
                  | some obj =>
                      let obj_attrs := obj.get_attribute_labels
                      required_attrs.foldl
                        (fun acc2 prop =>
                          if prop ∉ obj_props then
                            acc2 ++ [s!"{obj_label}:{prop}"]
                          else acc2)
                        acc)
                []
        """
        missing: list[str] = []

        for obj_label, required_props in self.attributes_required.items():
            if obj_label not in objects:
                missing.append(f"{obj_label}:OBJECT_NOT_FOUND")
                continue

            obj = objects[obj_label]
            obj_props = obj.get_attribute_labels()

            for prop in required_props:
                if prop not in obj_props:
                    missing.append(f"{obj_label}:{prop}")

        return missing

    # Pure function: Check if conditional
    def is_conditional(self, objects: dict[str, MathematicalObject]) -> bool:
        """
        Pure function: Check if theorem has unmet property requirements.

        Maps to Lean:
            def is_conditional
              (thm : TheoremBox)
              (objects : HashMap String MathematicalObject)
              : Bool :=
              !(thm.compute_conditionality objects).isEmpty
        """
        return len(self.compute_conditionality(objects)) > 0

    # Pure function: Check if can execute
    def can_execute(
        self, available_objects: set[str], object_states: dict[str, MathematicalObject]
    ) -> bool:
        """
        Pure function: Check if theorem can execute.

        Requirements:
        1. All input objects must be available
        2. All property requirements must be satisfied

        Maps to Lean:
            def can_execute
              (thm : TheoremBox)
              (available : List String)
              (states : HashMap String MathematicalObject)
              : Bool :=
              thm.input_objects.all (fun obj => obj ∈ available) &&
              !thm.is_conditional states
        """
        # All input objects available?
        objects_available = all(obj in available_objects for obj in self.input_objects)

        # All property requirements satisfied?
        properties_satisfied = not self.is_conditional(object_states)

        return objects_available and properties_satisfied

    # Pure function: Check if theorem has proof
    def has_proof(self) -> bool:
        """
        Pure function: Check if theorem has an attached proof.

        Maps to Lean:
            def has_proof (thm : TheoremBox) : Bool :=
              thm.proof.isSome
        """
        return self.proof is not None

    # Pure function: Check if theorem is fully proven
    def is_proven(self) -> bool:
        """
        Pure function: Check if theorem is fully proven (all steps expanded).

        Maps to Lean:
            def is_proven (thm : TheoremBox) : Bool :=
              match thm.proof with
              | none => false
              | some p => p.all_steps_expanded
        """
        return self.proof is not None and self.proof.all_steps_expanded()

    # Pure function: Attach proof to theorem
    def attach_proof(self, proof: ProofBox) -> TheoremBox:
        """
        Pure function: Attach proof to theorem (immutable update).

        Validates that proof.proves matches self.label.

        Args:
            proof: ProofBox to attach

        Returns:
            New TheoremBox with proof attached

        Raises:
            ValueError: If proof.proves doesn't match self.label

        Maps to Lean:
            def attach_proof (thm : TheoremBox) (p : ProofBox) : Except String TheoremBox :=
              if p.proves != thm.label then
                Except.error s!"Proof claims to prove {p.proves}, not {thm.label}"
              else
                let status := if p.all_steps_expanded then "expanded" else "sketched"
                Except.ok { thm with proof := some p, proof_status := status }
        """
        # Validate proof matches theorem
        if proof.proves != self.label:
            raise ValueError(f"Proof claims to prove {proof.proves}, not {self.label}")

        # Determine proof status
        if proof.all_steps_expanded():
            status = "expanded"
        elif any(step.status.value != "sketched" for step in proof.steps):
            status = "sketched"
        else:
            status = "sketched"

        return self.model_copy(update={"proof": proof, "proof_status": status})

    # Total function: Validate proof against theorem
    def validate_proof(self, objects: dict[str, MathematicalObject]) -> ProofValidationResult:
        """
        Total function: Validate attached proof against theorem specification.

        Args:
            objects: Available mathematical objects (for property validation)

        Returns:
            ProofValidationResult with validation status and any errors

        Maps to Lean:
            def validate_proof
              (thm : TheoremBox)
              (objects : HashMap String MathematicalObject)
              : ProofValidationResult :=
              match thm.proof with
              | none => {
                  is_valid := false,
                  mismatches := [],
                  warnings := ["No proof attached to theorem"]
                }
              | some p => validate_proof_for_theorem p thm objects
        """
        # Import here to avoid circular dependency
        from mathster.core.proof_integration import (
            ProofValidationResult,
            validate_proof_for_theorem,
        )

        if self.proof is None:
            return ProofValidationResult(
                is_valid=False, mismatches=[], warnings=["No proof attached to theorem"]
            )

        return validate_proof_for_theorem(self.proof, self, objects)

    @classmethod
    def from_raw(
        cls,
        raw: RawTheorem,  # type: ignore  # Forward reference
        source: SourceLocation,
        chapter: str | None = None,
        document: str | None = None,
    ) -> TheoremBox:
        """
        Create a TheoremBox from a RawTheorem staging model.

        This is a simple enrichment that stores the raw text as-is.
        For full enrichment with semantic analysis, use the LLM-based
        enrichment pipeline.

        Args:
            raw: The raw theorem from Stage 1 extraction
            source: Source location in documentation (REQUIRED)
            chapter: Chapter identifier (e.g., "1_euclidean_gas")
            document: Document identifier (e.g., "01_fragile_gas_framework")

        Returns:
            Enriched TheoremBox instance

        Raises:
            ValueError: If source is None

        Examples:
            >>> raw_thm = RawTheorem(
            ...     temp_id="raw-thm-1",
            ...     label_text="thm-keystone",
            ...     name="Keystone Principle",
            ...     statement_text="Under axioms...",
            ...     source_section="§3",
            ... )
            >>> theorem = TheoremBox.from_raw(raw_thm)
            >>> theorem.label
            'thm-keystone'
        """

        # Extract label from label_text (e.g., "Theorem 1.1" → "thm-1.1")
        # Normalize to standard prefix format
        label = raw.label_text
        # Extract statement type from label_text or use raw.statement_type
        statement_type = (
            raw.statement_type
        )  # Already normalized: "theorem", "lemma", "proposition", "corollary"

        # Determine prefix from statement_type
        prefix_map = {
            "theorem": "thm-",
            "lemma": "lem-",
            "proposition": "prop-",
            "corollary": "cor-",
        }
        prefix = prefix_map.get(statement_type, "thm-")

        # Create label if not already prefixed
        if not label.startswith(("thm-", "lem-", "prop-", "cor-")):
            import re

            # Extract name/number from label_text (e.g., "Theorem 1.1 (Main Result)" → "main-result")
            # Remove statement type word
            label_slug = label.lower().replace(statement_type, "")
            # Extract content from parentheses if present, otherwise use the whole text
            paren_match = re.search(r"\(([^)]+)\)", label_slug)
            if paren_match:
                label_slug = paren_match.group(1)
            # Clean up: remove dots, parentheses, replace spaces/underscores with hyphens
            label_slug = re.sub(r"[.\(\)]", "", label_slug)  # Remove dots and parentheses
            label_slug = re.sub(
                r"[\s_]+", "-", label_slug
            )  # Replace whitespace/underscores with hyphens
            label_slug = re.sub(r"-+", "-", label_slug)  # Collapse multiple hyphens
            label_slug = label_slug.strip("-")  # Remove leading/trailing hyphens
            # If no meaningful slug extracted, use a generic one from temp_id or auto-generate
            if not label_slug or label_slug == statement_type:
                label_slug = (
                    raw.temp_id.replace("raw-thm-", "")
                    .replace("raw-lem-", "")
                    .replace("raw-prop-", "")
                )
            label = f"{prefix}{label_slug}"

        # Determine output type from label_text and full_statement_text
        # Use both label and statement for heuristics
        combined_text = f"{raw.label_text} {raw.full_statement_text}".lower()
        if any(
            word in combined_text for word in ["existence", "exists", "there is", "there exists"]
        ):
            output_type = TheoremOutputType.EXISTENCE
        elif any(word in combined_text for word in ["uniqueness", "unique"]):
            output_type = TheoremOutputType.UNIQUENESS
        elif any(word in combined_text for word in ["bound", "inequality", "estimate"]):
            output_type = TheoremOutputType.BOUND
        elif any(word in combined_text for word in ["equivalence", "iff", "if and only if"]):
            output_type = TheoremOutputType.EQUIVALENCE
        elif any(word in combined_text for word in ["convergence", "converges"]):
            output_type = TheoremOutputType.CONVERGENCE
        elif any(word in combined_text for word in ["contraction", "contractive"]):
            output_type = TheoremOutputType.CONTRACTION
        else:
            output_type = TheoremOutputType.PROPERTY  # Default

        # Simple enrichment: store raw text, don't create DualStatement
        # Full DualStatement parsing requires LLM enrichment pipeline
        return cls(
            label=label,
            name=raw.label_text,  # Use label_text as name (e.g., "Theorem 1.1 (Main Result)")
            statement_type=statement_type,
            source=source,
            chapter=chapter,
            document=document,
            proof=None,  # Can be attached later
            proof_status="unproven",
            input_objects=[],  # Requires semantic analysis
            input_axioms=[],  # Requires semantic analysis
            input_parameters=[],  # Requires semantic analysis
            attributes_required={},  # Requires semantic analysis
            internal_lemmas=[],
            internal_propositions=[],
            lemma_dag_edges=[],
            output_type=output_type,
            attributes_added=[],  # Requires semantic analysis
            relations_established=[],  # Requires semantic analysis
            natural_language_statement=raw.full_statement_text,
            assumptions=[],  # Requires LLM parsing
            conclusion=None,  # Requires LLM enrichment pipeline
            equation_label=raw.equation_label,
            uses_definitions=raw.explicit_definition_references,  # Use extracted references
            validation_errors=[],
            raw_fallback=raw.model_dump() if raw else None,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_simple_object(
    label: str, name: str, expr: str, obj_type: ObjectType, tags: list[str] | None = None
) -> MathematicalObject:
    """
    Helper: Create simple object with no properties.

    Pure function (no side effects).
    """
    return MathematicalObject(
        label=label,
        name=name,
        mathematical_expression=expr,
        object_type=obj_type,
        current_attributes=[],
        attribute_history=[],
        tags=tags or [],
    )


def create_simple_theorem(
    label: str, name: str, output_type: TheoremOutputType, input_objects: list[str] | None = None
) -> TheoremBox:
    """
    Helper: Create simple theorem with minimal configuration.

    Pure function (no side effects).
    """
    return TheoremBox(
        label=label,
        name=name,
        input_objects=input_objects or [],
        output_type=output_type,
    )


# =============================================================================
# MODEL REBUILD (Fix forward references)
# =============================================================================

# Import forward-referenced types at runtime to resolve forward references
try:
    from mathster.core.article_system import SourceLocation
    from mathster.core.proof_system import ProofBox
    from mathster.sympy_integration.dual_representation import DualStatement

    # Rebuild models that use forward references
    DefinitionBox.model_rebuild()
    Axiom.model_rebuild()
    Parameter.model_rebuild()
    MathematicalObject.model_rebuild()
    TheoremBox.model_rebuild()
except ImportError:
    # Dependencies not available yet (e.g., during initial import)
    pass
