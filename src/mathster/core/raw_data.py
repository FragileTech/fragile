"""
Staging Models for LLM-Based Mathematical Paper Extraction.

This module defines the "raw extraction" models that serve as the direct JSON
output target for Stage 1 of the Extract-then-Enrich pipeline. These models
are intentionally simple and string-heavy, designed to make the LLM's transcription
task as straightforward as possible.

Stage 1 (Raw Extraction):
    LLM performs shallow parse, extracting semantic blocks into these staging models.
    Goal: Completeness of capture over semantic understanding.

Stage 2 (Enrichment & Assembly):
    Python orchestrator processes raw data, triggers focused secondary LLM calls,
    and assembles validated final Pydantic models (TheoremBox, ProofBox, etc.).
    Goal: Structural and logical integrity.

Design Principles:
- Simple field types (strings, lists of strings)
- Direct transcription, minimal interpretation
- Preserve full context (text before/after)
- Track source locations for debuggability
- Enable robust error recovery
"""

import re
import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


from mathster.core.article_system import TextLocation, SourceLocation



# =============================================================================
# LABEL NORMALIZATION UTILITIES
# =============================================================================


def normalize_term_to_label(term: str, prefix: str = "def") -> str:
    """
    Generate label from term by normalizing to pattern: {prefix}-{normalized-term}.

    This function provides conservative term-to-label conversion:
    - Converts to lowercase
    - Replaces spaces and underscores with hyphens
    - Removes special characters (keeps only alphanumeric and hyphens)
    - Collapses multiple consecutive hyphens to single hyphen
    - Strips leading/trailing hyphens
    - Prepends prefix

    Args:
        term: The term to normalize (e.g., "Lipschitz continuous", "v-porous on balls")
        prefix: Label prefix to prepend (default: "def")

    Returns:
        Normalized label string matching pattern ^{prefix}-[a-z0-9-]+$

    Raises:
        ValueError: If normalization produces empty or invalid label
    """
    if not term or not term.strip():
        raise ValueError(f"Cannot normalize empty term to label")

    # Step 1: Convert to lowercase
    normalized = term.lower()

    # Step 2: Replace spaces and underscores with hyphens
    normalized = normalized.replace(" ", "-").replace("_", "-")

    # Step 3: Remove special characters (keep only alphanumeric and hyphens)
    # Keep: a-z, 0-9, hyphens
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)

    # Step 4: Collapse multiple consecutive hyphens to single hyphen
    normalized = re.sub(r"-+", "-", normalized)

    # Step 5: Strip leading/trailing hyphens
    normalized = normalized.strip("-")

    # Validate result
    if not normalized:
        raise ValueError(
            f"Term normalization produced empty label. Original term: '{term}'"
        )

    # Construct final label
    label = f"{prefix}-{normalized}"

    # Validate against expected pattern
    pattern = r"^[a-z]+-[a-z0-9-]+$"
    if not re.match(pattern, label):
        raise ValueError(
            f"Normalized label '{label}' does not match required pattern '{pattern}'. "
            f"Original term: '{term}'"
        )

    return label


# =============================================================================
# BASE STAGING MODEL
# =============================================================================


class RawDataModel(BaseModel):
    """
    Common base class for raw extraction staging models.

    Provides shared configuration (frozen models) and source location metadata
    so downstream enrichment stages can uniformly access provenance details.
    """

    model_config = ConfigDict(frozen=True)

    source: SourceLocation = Field(
        ..., description="Source location metadata for this extracted entity."
    )
    full_text: str = Field(..., description="Full verbatim text content of this entity.")

    @classmethod
    def from_source(
        cls, source: SourceLocation
    ) -> "RawDataModel":
        """Create instance from source location and full text."""
        full_text = source.extract_full_text()
        return cls(source=source, full_text=full_text)

    @model_validator(mode="after")
    def validate_label(self):
        """Ensure label field matches expected pattern if present."""
        if hasattr(self, "label"):
            label = getattr(self, "label")
            # Skip validation if label is None (will be auto-populated by subclass)
            if label is not None:
                pattern = r"^[a-z]+-[a-z0-9-]+$"
                if not isinstance(label, str) or not re.match(pattern, label):
                    raise ValueError(
                        f"Invalid label format: '{label}'. Must match pattern '{pattern}'."
                    )
        return self



# =============================================================================
# CORE STAGING MODELS
# =============================================================================


class RawDefinition(RawDataModel):
    """
    Direct transcription of a mathematical definition block.

    The LLM's job is simply to identify definition blocks and copy their content
    verbatim. No semantic parsing required at this stage.
    """

    label: str | None = Field(
        None,
        pattern=r"^def-[a-z0-9-]+$",
        description="Unique definition label (e.g., 'def-lipschitz-continuous'). "
        "Auto-generated from term if not provided.",
    )

    term: str = Field(
        ...,
        min_length=1,
        description="The exact term being defined (e.g., 'v-porous on balls', 'Lipschitz continuous'). "
        "Should be the canonical name as it appears in the text.",
    )

    full_text: TextLocation = Field(
        ...,
        description="Line ranges in the source document where the complete definition text is located. "
        "The LLM should identify the start and end lines of the definition paragraph(s). "
        "Text extraction will be performed automatically by Python using these line ranges. "
        "Example: TextLocation(lines=[(142, 158)]) for a definition spanning lines 142-158.",
    )

    parameters_mentioned: list[str] = Field(
        default_factory=list,
        description="Symbols that act as parameters for this definition (e.g., ['v', 'α₀', 'α₁', 'h']). "
        "LLM should identify free parameters mentioned in the definition text.",
    )

    @model_validator(mode="before")
    @classmethod
    def auto_populate_and_validate_label(cls, data: dict) -> dict:
        """
        Always use label from source location and validate user-provided label.

        This validator runs before field validation and:
        1. Gets label from source.label
        2. Validates it matches the expected pattern for definitions (^def-[a-z0-9-]+$)
        3. If user provided label that doesn't match source.label, emits warning
        4. Keeps user-provided label (permissive policy) but alerts to inconsistencies
        5. Raises error if source.label doesn't match expected pattern (strict validation)

        Args:
            data: Raw input data dict before validation

        Returns:
            Modified data dict with label populated

        Raises:
            ValueError: If source.label doesn't match expected definition label pattern
        """
        # Step 1: Extract source label (source might be dict or SourceLocation object)
        source = data.get("source")
        if source is None:
            # Let Pydantic handle the missing required field error
            return data

        # Handle both dict and SourceLocation object
        if isinstance(source, dict):
            source_label = source.get("label")
            source_file_path = source.get("file_path", "unknown")
        else:
            # Already a SourceLocation object
            source_label = source.label
            source_file_path = source.file_path

        if source_label is None:
            # Let Pydantic handle missing label in source
            return data

        # Step 2: Validate source label matches expected pattern for definitions
        if not source_label.startswith("def-"):
            raise ValueError(
                f"Source label '{source_label}' does not match expected pattern for definitions "
                f"(must start with 'def-'). Source: {source_file_path}"
            )

        # Step 3: Compare user-provided value with source label and warn on mismatch
        user_label = data.get("label")
        if user_label is not None and user_label != source_label:
            warnings.warn(
                f"RawDefinition label mismatch: User provided label='{user_label}' "
                f"but source.label='{source_label}'. "
                f"Using user value. Source: {source_file_path}",
                UserWarning,
                stacklevel=2,
            )
        else:
            # No user value or values match - use source label
            data["label"] = source_label

        return data


class RawTheorem(RawDataModel):
    """
    Direct transcription of a Theorem, Lemma, Proposition, or Corollary.

    Captures the theorem statement and surrounding context. The LLM should identify:
    1. What type of statement it is (theorem/lemma/prop/corollary)
    2. The exact label from the text
    3. The full statement (assumptions + conclusion)
    4. Any explicitly mentioned definitions or prior results
    """

    label: str = Field(
        ...,
        pattern=r"^(thm|lem|prop|cor)-[a-z0-9-]+$",
        description="Unique statement label (e.g., 'thm-main-result', 'lem-gradient-bound'). "
        "If the concept has no assigned label then we should create one for it.",
    )
    statement_type: Literal["theorem", "lemma", "proposition", "corollary"] = Field(
        ..., description="The type of mathematical statement. Inferred from the label prefix."
    )

    conclusion_formula_latex: str | None = Field(
        None,
        description="The primary mathematical formula of the conclusion, isolated in raw LaTeX. "
        "Example: '||f1_x||_2 \\leq C h^\\beta ||f||_2'. "
        "Can be None if conclusion is stated in prose rather than formula.",
    )

    explicit_definition_references: list[str] = Field(
        default_factory=list,
        description="Terms mentioned that are clearly defined elsewhere, as they appear in text. "
        "Examples: ['v-porous on lines', 'Lipschitz continuous', 'uniformly convex']. "
        "LLM should identify formal terminology that likely has a Definition.",
    )


class RawProof(RawDataModel):
    """
    Direct transcription of a proof block.

    Captures the proof structure, steps, and citations. The LLM should identify:
    1. Which theorem this proof is for (by matching label)
    2. The overall proof strategy (if stated)
    3. Explicit steps (if numbered/enumerated)
    4. Citations to other theorems and bibliographic references
    """

    label: str = Field(
        ...,
        pattern=r"^proof-[a-z0-9-]+$",
        description="Unique proof label (e.g., 'proof-thm-main-result'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    proves_label: str = Field(
        ...,
        min_length=1,
        description="The label of the theorem this proof is for (e.g., 'Theorem 1.1', 'Lemma 3.4'). "
        "Should match the label from the corresponding RawTheorem. "
        "If proof says just 'Proof.' without explicit label, infer from context.",
    )

    strategy_text: TextLocation | None = Field(
        None,
        description="Line ranges for the initial paragraph(s) describing the overall proof strategy. "
        "The LLM should identify start and end lines of strategy description. "
        "Example: Strategy like 'We proceed in three steps. First, we establish...'. "
        "None if proof dives directly into details without stating strategy.",
    )

    steps: list[TextLocation] | None = Field(
        None,
        description="An ordered list of line ranges for each explicitly numbered/enumerated proof step. "
        "The LLM should identify start and end lines for each step separately. "
        "Example: [TextLocation(lines=[(50, 55)]), TextLocation(lines=[(56, 60)])]. "
        "None if proof is not structured into explicit steps.",
    )

    full_body_text: TextLocation | None = Field(
        None,
        description="Line ranges for the entire body of the proof if it is NOT broken into explicit steps. "
        "Use this field for prose-style mathster. The LLM should identify start and end lines. "
        "If steps is populated, this should be None (avoid duplication).",
    )

    explicit_theorem_references: list[str] = Field(
        default_factory=list,
        description="Explicit references to other results in the paper. "
        "Examples: ['Theorem 1.4', 'Lemma 2.3', 'Proposition 3.9', 'Corollary 1.2']. "
        "Preserve exact text as cited.",
    )

    citations_in_text: list[str] = Field(
        default_factory=list,
        description="All bibliographic citations found in the proof. "
        "Examples: ['[16]', '[13, 21, 22]', 'Han (2016)', '[Slepčev, 2018]']. "
        "Preserve exact citation format from text.",
    )


class RawCitation(RawDataModel):
    """
    Direct transcription of a single bibliographic entry.

    The LLM should copy bibliography entries verbatim, preserving all formatting.
    """

    key_in_text: str = Field(
        ...,
        min_length=1,
        description="The key used in the text for in-text citations. "
        "Examples: '[5]', '[16]', 'Han2016', 'Donoho09'. "
        "Preserve exact format including brackets if present.",
    )

    full_entry_text: TextLocation = Field(
        ...,
        description="Line ranges for the complete reference entry text from the bibliography. "
        "The LLM should identify start and end lines of the bibliographic entry. "
        "Text extraction will include authors, title, journal, year, pages, DOI, etc.",
    )


# =============================================================================
# EXTENDED STAGING MODELS
# =============================================================================


class RawEquation(RawDataModel):
    """
    Standalone numbered or important equation.

    Tracks significant equations that appear outside of theorem statements.
    These are often numbered equations that define operators, relationships, or
    intermediate results that are referenced later.

    Note: Not all equations are numbered in markdown, but we track them anyway
    and assign unique labels that follow the Stage 2 conventions.
    """

    label: str = Field(
        ...,
        pattern=r"^eq-[a-z0-9-]+$",
        description="Unique equation label (e.g., 'eq-langevin-drift'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    equation_label: str | None = Field(
        None,
        description="The equation number if present in the text (e.g., '(2.3)', '(4.1a)', 'Eq. (17)'). "
        "None if equation is unnumbered (display equation without label).",
    )

    latex_content: TextLocation = Field(
        ...,
        description="Line ranges for the equation's LaTeX content in the source document. "
        "The LLM should identify start and end lines of the equation block. "
        "Text extraction will preserve exact LaTeX syntax without $$...$$or \\[...\\] delimiters.",
    )

    context_before: TextLocation | None = Field(
        None,
        description="Line ranges for the sentence(s) immediately before the equation, providing context. "
        "The LLM should identify start and end lines of contextual text. "
        "Example: Text like 'We define the operator T_h as follows:'. "
        "Helps understand what the equation represents.",
    )

    context_after: TextLocation | None = Field(
        None,
        description="Line ranges for the sentence(s) immediately after the equation, often explaining notation. "
        "The LLM should identify start and end lines of explanatory text. "
        "Example: Text like 'where K is the kernel function and w_i are weights'. "
        "Helps understand equation components.",
    )


class RawParameter(RawDataModel):
    """
    Parameter or notation definition.

    Captures statements like "Throughout this paper, h denotes a small parameter"
    or parameter tables that define notation used in the document.

    These are crucial for understanding the mathematical context and for building
    the global PaperContext.
    """

    label: str = Field(
        ...,
        pattern=r"^param-[a-z0-9-]+$",
        description="Unique parameter label (e.g., 'param-gamma', 'param-step-size'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    symbol: str = Field(
        ...,
        min_length=1,
        description="The mathematical symbol being defined. "
        "Examples: 'h', 'v', 'Ω', 'N', 'β', 'C'. "
        "Preserve LaTeX if present (e.g., '\\beta', '\\Omega').",
    )

    meaning: str = Field(
        ...,
        min_length=1,
        description="The meaning or definition of the symbol. "
        "Example: 'a small parameter', 'the number of walkers', 'porosity constant'. "
        "Extract the key semantic content.",
    )
    scope: Literal["global", "local"] = Field(
        ...,
        description="Whether this parameter is defined globally (for the whole document) "
        "or locally (within a specific theorem/section). "
        "Global: 'Throughout this paper...', 'We always assume...'. "
        "Local: 'In this section...', 'For the remainder of this proof...'.",
    )


class RawRemark(RawDataModel):
    """
    Informal observation, remark, or note.

    Captures non-formal statements that provide important mathematical intuition,
    context, or connections to other work. These are valuable for understanding
    the paper's narrative and for explaining results to readers.
    """

    label: str = Field(
        ...,
        pattern=r"^remark-[a-z0-9-]+$",
        description="Unique remark label (e.g., 'remark-kinetic-necessity'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    remark_type: Literal["note", "remark", "observation", "comment", "example"] = Field(
        ...,
        description="The type of informal statement. "
        "Typically inferred from the label prefix (e.g., 'Remark', 'Note', 'Observation').",
    )


class RawAxiom(RawDataModel):
    """
    Direct transcription of an axiom block.

    Axioms in the Fragile framework have a multi-part structure:
    - Core assumption (the fundamental claim)
    - Parameters (mathematical objects used in the axiom)
    - Condition (formal statement of when the axiom applies)
    - Failure mode analysis (optional - what happens if violated)

    The LLM should extract all parts verbatim, leaving fields empty if not present.
    """

    label: str = Field(
        ...,
        pattern=r"^(axiom-|ax-|def-axiom-)[a-z0-9-]+$",
        description="Unique axiom label (e.g., 'axiom-bounded-diameter', 'def-axiom-reward-regularity'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    name: str = Field(
        ...,
        min_length=1,
        description="The title/name of the axiom (e.g., 'Axiom of Guaranteed Revival', 'Bounded Displacement Axiom').",
    )

    core_assumption_text: TextLocation = Field(
        ...,
        description="Line ranges for the fundamental assumption or claim of the axiom. "
        "The LLM should identify start and end lines of the core statement. "
        "Text extraction will provide the verbatim transcription.",
    )

    parameters_text: list[str] = Field(
        default_factory=list,
        description="List of parameter definitions mentioned in the axiom. "
        "Each string is a verbatim block describing one parameter. "
        "Example: ['v > 0 is the velocity magnitude', 'N is the number of walkers']. "
        "May be empty if no explicit parameters.",
    )

    condition_text: TextLocation | None = Field(
        None,
        description="Line ranges for the formal condition statement (when the axiom applies). "
        "The LLM should identify start and end lines of the condition. "
        "Example: Text like 'v > 0 and N >= 2'. None if not explicitly stated.",
    )

    failure_mode_analysis_text: TextLocation | None = Field(
        None,
        description="Line ranges for the optional analysis of what happens when the axiom is violated. "
        "The LLM should identify start and end lines of the failure mode text. "
        "None if not present in the document.",
    )


# =============================================================================
# CONTAINER MODEL
# =============================================================================


class RawDocumentSection(RawDataModel):
    """
    Aggregates all raw extractions from a document section.

    This is the top-level output from Stage 1 (raw extraction). It contains
    lists of all identified mathematical entities in their raw, unprocessed form.

    The orchestrator will process sections in parallel, producing one RawDocumentSection
    per section, then merge them into a single document-level registry.
    """

    section_id: str = Field(
        ...,
        description="Identifier for the section this extraction covers. "
        "Examples: '## 2 The Cloning Operator', '## 3 Measuring Pipeline'. "
        "Used for organizing parallel processing.",
    )

    definitions: list[RawDefinition] = Field(
        default_factory=list, description="All definitions extracted from this section."
    )

    theorems: list[RawTheorem] = Field(
        default_factory=list,
        description="All theorems, lemmas, propositions, and corollaries extracted from this section.",
    )

    proofs: list[RawProof] = Field(
        default_factory=list, description="All proofs extracted from this section."
    )

    citations: list[RawCitation] = Field(
        default_factory=list,
        description="All bibliographic citations (typically from a references section). "
        "May be empty for non-bibliography sections.",
    )

    # equations: list[RawEquation] = Field(
    #     default_factory=list,
    #     description="All standalone equations (numbered or significant unnumbered) extracted from this section.",
    # )

    parameters: list[RawParameter] = Field(
        default_factory=list,
        description="All parameter/notation definitions extracted from this section.",
    )

    remarks: list[RawRemark] = Field(
        default_factory=list,
        description="All remarks, notes, and observations extracted from this section.",
    )

    axioms: list[RawAxiom] = Field(
        default_factory=list, description="All axioms extracted from this section."
    )

    # Statistics for validation
    @property
    def total_entities(self) -> int:
        """Total number of entities extracted."""
        return (
            len(self.definitions)
            + len(self.theorems)
            + len(self.proofs)
            + len(self.citations)
            # + len(self.equations)
            + len(self.parameters)
            + len(self.remarks)
            + len(self.axioms)
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of extraction results."""
        return (
            f"Section '{self.section_id}': {self.total_entities} entities extracted\n"
            f"  - Definitions: {len(self.definitions)}\n"
            f"  - Theorems/Lemmas/Props: {len(self.theorems)}\n"
            f"  - Proofs: {len(self.proofs)}\n"
            f"  - Axioms: {len(self.axioms)}\n"
            f"  - Citations: {len(self.citations)}\n"
            # f"  - Equations: {len(self.equations)}\n"
            f"  - Parameters: {len(self.parameters)}\n"
            f"  - Remarks: {len(self.remarks)}"
        )
