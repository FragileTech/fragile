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

Maps to Lean:
    namespace StagingTypes
      structure RawDefinition where ...
      structure RawTheorem where ...
      ...
    end StagingTypes
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from fragile.proofs.core.article_system import SourceLocation


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


# =============================================================================
# CORE STAGING MODELS
# =============================================================================


class RawDefinition(RawDataModel):
    """
    Direct transcription of a mathematical definition block.

    The LLM's job is simply to identify definition blocks and copy their content
    verbatim. No semantic parsing required at this stage.

    Examples:
        - "Definition 2.1 (v-porous on lines). A set E ⊂ ℝ^n is..."
        - "We say that a function f is Lipschitz continuous if..."

    Maps to Lean:
        structure RawDefinition where
          label : String
          term : String
          full_text : String
          parameters_mentioned : List String
    """

    label: str = Field(
        ...,
        pattern=r"^def-[a-z0-9-]+$",
        description="Unique definition label (e.g., 'def-lipschitz-continuous'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    term: str = Field(
        ...,
        min_length=1,
        description="The exact term being defined (e.g., 'v-porous on balls', 'Lipschitz continuous'). "
        "Should be the canonical name as it appears in the text.",
    )

    full_text: str = Field(
        ...,
        min_length=1,
        description="The complete, verbatim text of the definition paragraph(s). "
        "Includes the definition statement and any explanatory text. "
        "Preserves LaTeX formatting exactly as written.",
    )

    parameters_mentioned: list[str] = Field(
        default_factory=list,
        description="Symbols that act as parameters for this definition (e.g., ['v', 'α₀', 'α₁', 'h']). "
        "LLM should identify free parameters mentioned in the definition text.",
    )


class RawTheorem(RawDataModel):
    """
    Direct transcription of a Theorem, Lemma, Proposition, or Corollary.

    Captures the theorem statement and surrounding context. The LLM should identify:
    1. What type of statement it is (theorem/lemma/prop/corollary)
    2. The exact label from the text
    3. The full statement (assumptions + conclusion)
    4. Any explicitly mentioned definitions or prior results

    Examples:
        - "Theorem 1.1 (Main Result). Let v > 0 and assume E is v-porous..."
        - "Lemma 3.4. Under the assumptions of Theorem 3.1, we have..."

    Maps to Lean:
        structure RawTheorem where
          label : String
          label_text : String
          statement_type : StatementType
          context_before : Option String
          full_statement_text : String
          conclusion_formula_latex : Option String
          equation_label : Option String
          explicit_definition_references : List String
    """

    label: str = Field(
        ...,
        pattern=r"^(thm|lem|prop|cor)-[a-z0-9-]+$",
        description="Unique statement label (e.g., 'thm-main-result', 'lem-gradient-bound'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    label_text: str = Field(
        ...,
        min_length=1,
        description="The exact label from the text (e.g., 'Theorem 1.1', 'Lemma 3.4', 'Proposition 2.9'). "
        "Preserve the exact formatting including numbers and optional name in parentheses.",
    )

    statement_type: Literal["theorem", "lemma", "proposition", "corollary"] = Field(
        ..., description="The type of mathematical statement. Inferred from the label prefix."
    )

    context_before: str | None = Field(
        None,
        description="The paragraph(s) immediately preceding the theorem statement. "
        "Provides context for understanding assumptions and setup. "
        "Can be None if theorem starts a new section.",
    )

    full_statement_text: str = Field(
        ...,
        min_length=1,
        description="The verbatim text of the entire theorem statement, from the opening "
        "(e.g., 'Let...', 'Assume...', 'For all...') to the final formula or conclusion. "
        "Preserves all LaTeX, numbered equations, and formatting.",
    )

    conclusion_formula_latex: str | None = Field(
        None,
        description="The primary mathematical formula of the conclusion, isolated in raw LaTeX. "
        "Example: '||f1_x||_2 \\leq C h^\\beta ||f||_2'. "
        "Can be None if conclusion is stated in prose rather than formula.",
    )

    equation_label: str | None = Field(
        None,
        description="The equation number if the conclusion has one (e.g., '(1.1)', '(3.5a)', 'Eq. (12)'). "
        "None if conclusion is unnumbered.",
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

    Examples:
        - "Proof of Theorem 1.1. We proceed in three steps..."
        - "Proof. The result follows immediately from Lemma 2.3 and [16]..."

    Maps to Lean:
        structure RawProof where
          label : String
          proves_label_text : String
          strategy_text : Option String
          steps : Option (List String)
          full_body_text : Option String
          explicit_theorem_references : List String
          citations_in_text : List String
    """

    label: str = Field(
        ...,
        pattern=r"^proof-[a-z0-9-]+$",
        description="Unique proof label (e.g., 'proof-thm-main-result'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    proves_label_text: str = Field(
        ...,
        min_length=1,
        description="The label of the theorem this proof is for (e.g., 'Theorem 1.1', 'Lemma 3.4'). "
        "Should match the label_text from the corresponding RawTheorem. "
        "If proof says just 'Proof.' without explicit label, infer from context.",
    )

    strategy_text: str | None = Field(
        None,
        description="The initial paragraph(s) describing the overall proof strategy. "
        "Example: 'We proceed in three steps. First, we establish...'. "
        "None if proof dives directly into details without stating strategy.",
    )

    steps: list[str] | None = Field(
        None,
        description="An ordered list of verbatim text for each explicitly numbered/enumerated step. "
        "Example: ['Step 1: Establish the bound...', 'Step 2: Apply Lemma 2.3...']. "
        "None if proof is not structured into explicit steps.",
    )

    full_body_text: str | None = Field(
        None,
        description="The entire body of the proof if it is NOT broken into explicit steps. "
        "Use this field for prose-style proofs. "
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

    Examples:
        - "[16] R. Han and D. Slepčev. Stochastic dynamics on hypergraphs..."
        - "[5] D.L. Donoho. Compressed sensing. IEEE Trans. Inform. Theory..."

    Maps to Lean:
        structure RawCitation where
          key_in_text : String
          full_entry_text : String
    """

    key_in_text: str = Field(
        ...,
        min_length=1,
        description="The key used in the text for in-text citations. "
        "Examples: '[5]', '[16]', 'Han2016', 'Donoho09'. "
        "Preserve exact format including brackets if present.",
    )

    full_entry_text: str = Field(
        ...,
        min_length=1,
        description="The complete, verbatim text of the reference entry from the bibliography. "
        "Includes authors, title, journal, year, pages, DOI, etc. "
        "Preserve all formatting exactly as written.",
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

    Examples:
        - Equation (2.3) defining a specific operator
        - Display equations with important formulas
        - Numbered equations referenced in proofs

    Note: Not all equations are numbered in markdown, but we track them anyway
    and assign unique labels that follow the Stage 2 conventions.

    Maps to Lean:
        structure RawEquation where
          label : String
          equation_label : Option String
          latex_content : String
          context_before : Option String
          context_after : Option String
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

    latex_content: str = Field(
        ...,
        min_length=1,
        description="The raw LaTeX content of the equation. "
        "Example: 'T_h f(x) = \\sum_{i=1}^N w_i K(x - x_i) f(x_i)'. "
        "Preserve exact LaTeX syntax without $$...$$or \\[...\\] delimiters.",
    )

    context_before: str | None = Field(
        None,
        description="The sentence(s) immediately before the equation, providing context. "
        "Example: 'We define the operator T_h as follows:'. "
        "Helps understand what the equation represents.",
    )

    context_after: str | None = Field(
        None,
        description="The sentence(s) immediately after the equation, often explaining the notation. "
        "Example: 'where K is the kernel function and w_i are weights'. "
        "Helps understand equation components.",
    )


class RawParameter(RawDataModel):
    """
    Parameter or notation definition.

    Captures statements like "Throughout this paper, h denotes a small parameter"
    or parameter tables that define notation used in the document.

    These are crucial for understanding the mathematical context and for building
    the global PaperContext.

    Examples:
        - "Throughout this paper, Ω denotes a bounded domain in ℝ^n"
        - "Let h > 0 be a small parameter"
        - Table entries: "v: porosity parameter"

    Maps to Lean:
        structure RawParameter where
          label : String
          symbol : String
          meaning : String
          full_text : String
          scope : ScopeType
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

    full_text: str = Field(
        ...,
        min_length=1,
        description="The complete verbatim text of the parameter definition. "
        "Example: 'Throughout this paper, h > 0 denotes a small parameter'. "
        "Preserves full context.",
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

    Examples:
        - "Remark 2.1. The condition v > 0 is essential..."
        - "Note that this generalizes the result of [16]..."
        - "Observation. The proof technique extends to higher dimensions..."

    Maps to Lean:
        structure RawRemark where
          label : String
          remark_type : RemarkType
          full_text : String
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

    full_text: str = Field(
        ...,
        min_length=1,
        description="The complete verbatim text of the remark, including the label. "
        "Example: 'Remark 2.1. The condition v > 0 is essential because...'. "
        "Preserves all mathematical content and references.",
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

    Example:
        "Axiom of Guaranteed Revival. Core Assumption: Every walker that dies
        is guaranteed to revive. Parameters: v (velocity magnitude), N (number
        of walkers). Condition: v > 0. Failure Mode: If v = 0, walkers cannot
        escape local minima."

    Maps to Lean:
        structure RawAxiom where
          label : String
          label_text : String
          axiom_name : String
          statement : String
    """

    label: str = Field(
        ...,
        pattern=r"^(axiom-|ax-|def-axiom-)[a-z0-9-]+$",
        description="Unique axiom label (e.g., 'axiom-bounded-diameter', 'def-axiom-reward-regularity'). "
        "If the concept has no assigned label then we should create one for it.",
    )

    label_text: str = Field(
        ...,
        description="The exact label from the text (e.g., 'def-axiom-guaranteed-revival', 'Axiom 1').",
    )

    name: str = Field(
        ...,
        min_length=1,
        description="The title/name of the axiom (e.g., 'Axiom of Guaranteed Revival', 'Bounded Displacement Axiom').",
    )

    core_assumption_text: str = Field(
        ...,
        min_length=1,
        description="The fundamental assumption or claim of the axiom. "
        "Verbatim transcription of the core statement.",
    )

    parameters_text: list[str] = Field(
        default_factory=list,
        description="List of parameter definitions mentioned in the axiom. "
        "Each string is a verbatim block describing one parameter. "
        "Example: ['v > 0 is the velocity magnitude', 'N is the number of walkers']. "
        "May be empty if no explicit parameters.",
    )

    condition_text: str = Field(
        default="",
        description="The formal condition statement (when the axiom applies). "
        "Verbatim LaTeX or mathematical statement. "
        "Example: 'v > 0 and N >= 2'. Empty string if not stated.",
    )

    failure_mode_analysis_text: str | None = Field(
        None,
        description="Optional analysis of what happens when the axiom is violated. "
        "Verbatim text if present, None otherwise.",
    )


# =============================================================================
# CONTAINER MODEL
# =============================================================================


class StagingDocument(RawDataModel):
    """
    Aggregates all raw extractions from a document section.

    This is the top-level output from Stage 1 (raw extraction). It contains
    lists of all identified mathematical entities in their raw, unprocessed form.

    The orchestrator will process sections in parallel, producing one StagingDocument
    per section, then merge them into a single document-level registry.

    Maps to Lean:
        structure StagingDocument where
          section_id : String
          definitions : List RawDefinition
          theorems : List RawTheorem
          proofs : List RawProof
          axioms : List RawAxiom
          citations : List RawCitation
          equations : List RawEquation
          parameters : List RawParameter
          remarks : List RawRemark
    """

    section_id: str = Field(
        ...,
        description="Identifier for the section this extraction covers. "
        "Examples: '§1-intro', '§2.1-preliminaries', '§4-main-results'. "
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

    equations: list[RawEquation] = Field(
        default_factory=list,
        description="All standalone equations (numbered or significant unnumbered) extracted from this section.",
    )

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
            + len(self.equations)
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
            f"  - Equations: {len(self.equations)}\n"
            f"  - Parameters: {len(self.parameters)}\n"
            f"  - Remarks: {len(self.remarks)}"
        )
