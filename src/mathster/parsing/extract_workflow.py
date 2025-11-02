"""
Extraction workflow for mathematical concept extraction using DSPy ReAct agents.

This module implements the FRESH EXTRACTION workflow that extracts mathematical
entities from markdown documents for the first time.

Workflow:
    1. Receive numbered chapter text
    2. Extract entities using ReAct agent with self-validation
    3. Convert to RawDocumentSection
    4. Return results with any errors/warnings

Key Components:
    - MathematicalConceptExtractor: ReAct agent for extraction
    - ExtractWithValidation: DSPy signature for agent
    - validate_extraction_tool: Self-validation tool for agent
    - convert_to_raw_document_section: Convert extraction to structured format

Usage:
    from mathster.parsing.extract_workflow import extract_chapter

    raw_section, errors = extract_chapter(
        chapter_text=chapter_with_line_numbers,
        chapter_number=0,
        file_path="docs/source/...",
        article_id="01_fragile_gas_framework"
    )
"""

import json
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from mathster.core.article_system import SourceLocation, TextLocation
from mathster.core.raw_data import (
    RawAxiom,
    RawCitation,
    RawDefinition,
    RawDocumentSection,
    RawParameter,
    RawProof,
    RawRemark,
    RawTheorem,
)


# =============================================================================
# EXTRACTION SCHEMA - Lightweight models for DSPy output
# =============================================================================


class ExtractedEntity(BaseModel):
    """Base model for extracted entities with line range information."""

    label: str = Field(
        ..., description="Unique label for this entity (e.g., 'def-lipschitz', 'thm-main-result')"
    )
    line_start: int = Field(..., description="Starting line number (1-indexed, inclusive)")
    line_end: int = Field(..., description="Ending line number (1-indexed, inclusive)")


class DefinitionExtraction(ExtractedEntity):
    """Extracted definition with minimal structure."""

    term: str = Field(..., description="The exact term being defined")
    parameters_mentioned: list[str] = Field(
        default_factory=list,
        description="Mathematical symbols/parameters mentioned (e.g., ['v', 'α₀', 'h'])",
    )


class TheoremExtraction(ExtractedEntity):
    """Extracted theorem, lemma, proposition, or corollary."""

    statement_type: Literal["theorem", "lemma", "proposition", "corollary"] = Field(
        ..., description="Type of mathematical statement"
    )
    conclusion_formula_latex: str | None = Field(
        None,
        description="Primary conclusion formula in LaTeX (without delimiters like $$ or \\[ \\])",
    )
    definition_references: list[str] = Field(
        default_factory=list,
        description="Terms that are clearly defined elsewhere (e.g., ['v-porous on lines', 'Lipschitz continuous'])",
    )


class ProofExtraction(ExtractedEntity):
    """Extracted proof with optional structure."""

    proves_label: str = Field(
        ..., description="Label or reference to theorem being proved (e.g., 'Theorem 1.1', 'thm-main-result')"
    )
    strategy_line_start: int | None = Field(
        None, description="Starting line of proof strategy description if present"
    )
    strategy_line_end: int | None = Field(
        None, description="Ending line of proof strategy description if present"
    )
    steps: list[tuple[int, int]] = Field(
        default_factory=list,
        description="List of (start_line, end_line) tuples for each numbered proof step",
    )
    full_body_line_start: int | None = Field(
        None, description="Starting line of full proof body if not broken into steps"
    )
    full_body_line_end: int | None = Field(
        None, description="Ending line of full proof body if not broken into steps"
    )
    theorem_references: list[str] = Field(
        default_factory=list,
        description="Explicit theorem references (e.g., ['Theorem 1.4', 'Lemma 2.3'])",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Bibliographic citations (e.g., ['[16]', 'Han (2016)'])",
    )


class AxiomExtraction(ExtractedEntity):
    """Extracted axiom with structured components."""

    name: str = Field(..., description="Title/name of the axiom")
    core_assumption_line_start: int = Field(..., description="Starting line of core assumption text")
    core_assumption_line_end: int = Field(..., description="Ending line of core assumption text")
    parameters: list[str] = Field(
        default_factory=list,
        description="List of parameter definitions (verbatim text blocks)",
    )
    condition_line_start: int | None = Field(
        None, description="Starting line of condition text if present"
    )
    condition_line_end: int | None = Field(
        None, description="Ending line of condition text if present"
    )
    failure_mode_line_start: int | None = Field(
        None, description="Starting line of failure mode analysis if present"
    )
    failure_mode_line_end: int | None = Field(
        None, description="Ending line of failure mode analysis if present"
    )


class ParameterExtraction(ExtractedEntity):
    """Extracted parameter/notation definition."""

    symbol: str = Field(
        ..., description="Mathematical symbol (e.g., 'h', '\\beta', 'N')"
    )
    meaning: str = Field(
        ..., description="Definition/meaning of the symbol"
    )
    scope: Literal["global", "local"] = Field(
        ..., description="Whether parameter is global (whole document) or local (section/proof)"
    )


class RemarkExtraction(ExtractedEntity):
    """Extracted remark, note, or observation."""

    remark_type: Literal["note", "remark", "observation", "comment", "example"] = Field(
        ..., description="Type of informal statement"
    )


class CitationExtraction(ExtractedEntity):
    """Extracted bibliographic citation."""

    key_in_text: str = Field(
        ..., description="Citation key as it appears in text (e.g., '[5]', 'Han2016')"
    )


class ChapterExtraction(BaseModel):
    """Aggregate extraction result for a single chapter."""

    section_id: str = Field(
        ..., description="Section identifier (e.g., '## 1. Introduction', 'Chapter 0 - Preamble')"
    )
    definitions: list[DefinitionExtraction] = Field(default_factory=list)
    theorems: list[TheoremExtraction] = Field(default_factory=list)
    proofs: list[ProofExtraction] = Field(default_factory=list)
    axioms: list[AxiomExtraction] = Field(default_factory=list)
    parameters: list[ParameterExtraction] = Field(default_factory=list)
    remarks: list[RemarkExtraction] = Field(default_factory=list)
    citations: list[CitationExtraction] = Field(default_factory=list)


# =============================================================================
# VALIDATION LOGIC
# =============================================================================


class ValidationResult(BaseModel):
    """Result of validating an extraction attempt."""

    is_valid: bool = Field(..., description="Whether the extraction is valid")
    errors: list[str] = Field(default_factory=list, description="List of validation errors")
    warnings: list[str] = Field(default_factory=list, description="List of warnings")
    entities_validated: dict[str, int] = Field(
        default_factory=dict,
        description="Count of successfully validated entities by type"
    )

    def get_feedback(self) -> str:
        """Get human-readable feedback for the agent."""
        if self.is_valid:
            return (
                f"✓ Validation successful! Entities validated: {self.entities_validated}. "
                "All entities have valid labels, line numbers, and required fields."
            )

        feedback = "✗ Validation failed. Please fix the following errors:\n"
        for i, error in enumerate(self.errors, 1):
            feedback += f"{i}. {error}\n"

        if self.warnings:
            feedback += "\nWarnings:\n"
            for i, warning in enumerate(self.warnings, 1):
                feedback += f"{i}. {warning}\n"

        return feedback


def validate_extraction(
    extraction_dict: dict,
    file_path: str,
    article_id: str,
    chapter_text: str
) -> ValidationResult:
    """
    Validation tool that attempts to build RawDocumentSection from extraction.

    This tool is called by the ReAct agent to validate its extraction attempts.
    It provides detailed feedback on what went wrong so the agent can retry.

    Args:
        extraction_dict: Dictionary representing ChapterExtraction
        file_path: Path to source markdown file
        article_id: Article identifier
        chapter_text: Original chapter text

    Returns:
        ValidationResult with success status and error details
    """
    errors = []
    warnings = []
    entities_validated = {
        "definitions": 0,
        "theorems": 0,
        "proofs": 0,
        "axioms": 0,
        "parameters": 0,
        "remarks": 0,
        "citations": 0
    }

    try:
        # Parse as ChapterExtraction
        extraction = ChapterExtraction(**extraction_dict)

        # Attempt to convert to RawDocumentSection
        try:
            raw_section, conversion_warnings = convert_to_raw_document_section(
                extraction,
                file_path=file_path,
                article_id=article_id,
                chapter_text=chapter_text
            )

            # Add conversion warnings to warnings list
            if conversion_warnings:
                warnings.extend(conversion_warnings)

            # Count successfully validated entities
            entities_validated["definitions"] = len(raw_section.definitions)
            entities_validated["theorems"] = len(raw_section.theorems)
            entities_validated["proofs"] = len(raw_section.proofs)
            entities_validated["axioms"] = len(raw_section.axioms)
            entities_validated["parameters"] = len(raw_section.parameters)
            entities_validated["remarks"] = len(raw_section.remarks)
            entities_validated["citations"] = len(raw_section.citations)

            # Check if we got any entities
            total_entities = sum(entities_validated.values())
            if total_entities == 0:
                warnings.append("No entities were extracted from this chapter. Is the chapter empty?")

            # Validate label patterns
            for defn in extraction.definitions:
                if not defn.label.startswith("def-"):
                    errors.append(f"Definition label '{defn.label}' must start with 'def-'")

            for thm in extraction.theorems:
                if not any(thm.label.startswith(p) for p in ["thm-", "lem-", "prop-", "cor-"]):
                    errors.append(f"Theorem label '{thm.label}' must start with thm-/lem-/prop-/cor-")

            for proof in extraction.proofs:
                if not proof.label.startswith("proof-"):
                    errors.append(f"Proof label '{proof.label}' must start with 'proof-'")

            for axiom in extraction.axioms:
                if not any(axiom.label.startswith(p) for p in ["axiom-", "ax-", "def-axiom-"]):
                    errors.append(f"Axiom label '{axiom.label}' must start with axiom-/ax-/def-axiom-")

            for param in extraction.parameters:
                if not param.label.startswith("param-"):
                    errors.append(f"Parameter label '{param.label}' must start with 'param-'")

            for remark in extraction.remarks:
                if not remark.label.startswith("remark-"):
                    errors.append(f"Remark label '{remark.label}' must start with 'remark-'")

            # Validate line numbers are reasonable
            for defn in extraction.definitions:
                if defn.line_start > defn.line_end:
                    errors.append(
                        f"Definition '{defn.label}': line_start ({defn.line_start}) "
                        f"must be <= line_end ({defn.line_end})"
                    )

            for thm in extraction.theorems:
                if thm.line_start > thm.line_end:
                    errors.append(
                        f"Theorem '{thm.label}': line_start ({thm.line_start}) "
                        f"must be <= line_end ({thm.line_end})"
                    )

        except Exception as e:
            errors.append(f"Failed to convert to RawDocumentSection: {str(e)}")

    except Exception as e:
        errors.append(f"Failed to parse ChapterExtraction: {str(e)}")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        entities_validated=entities_validated
    )


def validate_extraction_tool(extraction_json: str, context: str) -> str:
    """
    Tool for ReAct agent to validate its extraction.

    Args:
        extraction_json: JSON string of ChapterExtraction
        context: Context string containing file_path, article_id, chapter_text

    Returns:
        Validation feedback string
    """
    try:
        # Parse context (format: "file_path|||article_id|||chapter_text")
        parts = context.split("|||")
        if len(parts) != 3:
            return "Error: Invalid context format"

        file_path, article_id, chapter_text = parts

        # Parse extraction JSON
        extraction_dict = json.loads(extraction_json)

        # Validate
        validation = validate_extraction(
            extraction_dict,
            file_path=file_path,
            article_id=article_id,
            chapter_text=chapter_text
        )

        return validation.get_feedback()

    except Exception as e:
        return f"Validation error: {str(e)}"


# =============================================================================
# DSPY SIGNATURE AND MODULE
# =============================================================================


class ExtractMathematicalConcepts(dspy.Signature):
    """
    Extract all mathematical entities from a numbered markdown chapter.

    You are an expert mathematical document parser. Your task is to identify and extract
    ALL mathematical entities from the provided chapter text. The text has line numbers
    in the format "NNN: content" which you must use to identify precise boundaries.

    ENTITY TYPES TO EXTRACT:

    1. DEFINITIONS (label: def-*):
       - Terms being formally defined
       - Include the exact term and any parameters mentioned
       - Example: "Lipschitz continuous", "v-porous on balls"

    2. THEOREMS/LEMMAS/PROPOSITIONS/COROLLARIES (label: thm-*, lem-*, prop-*, cor-*):
       - Mathematical statements with claims
       - Identify type and extract conclusion formula if present
       - Note any definition references

    3. PROOFS (label: proof-*):
       - Link to theorem being proved
       - Extract strategy if stated
       - Identify numbered steps if structured
       - Note theorem references and citations

    4. AXIOMS (label: axiom-*, ax-*, def-axiom-*):
       - Foundational assumptions
       - Extract core assumption, parameters, conditions, failure modes

    5. PARAMETERS (label: param-*):
       - Mathematical notation definitions
       - Symbol, meaning, and scope (global/local)

    6. REMARKS (label: remark-*):
       - Notes, observations, comments, examples
       - Informal mathematical intuition

    7. CITATIONS:
       - Bibliographic references
       - Key in text and location

    LABEL GENERATION:
    - If entity has an explicit label (e.g., :label: thm-main-result), use it
    - Otherwise, generate a descriptive label following the pattern:
      * def-{normalized-term} for definitions
      * thm-{descriptive-name} for theorems
      * lem-{descriptive-name} for lemmas
      * proof-{theorem-label} for proofs
      * etc.
    - Labels must be lowercase with hyphens, matching pattern: ^[a-z]+-[a-z0-9-]+$

    LINE NUMBER TRACKING (CRITICAL):
    - Use line numbers from the text to identify exact boundaries
    - Line numbers are CONTINUOUS across the entire document
    - Format: "NNN: content" where NNN is the line number
    - Extract both start and end lines for each entity
    - ONLY OUTPUT LINE NUMBERS - do NOT copy/extract the actual text content
    - The text will be extracted later from the source file using these line numbers

    OUTPUT FORMAT:
    - Return ONLY the metadata: labels, line numbers, entity types, references
    - DO NOT include the actual mathematical text content in your output
    - This minimizes output size and reduces token usage

    IMPORTANT:
    - Be thorough - extract ALL mathematical content
    - Preserve exact terminology from source
    - Don't skip content that lacks explicit labels
    - Use line numbers for precise location tracking
    """

    chapter_with_lines: str = dspy.InputField(
        desc="Chapter text with line numbers in format 'NNN: content'"
    )
    chapter_number: int = dspy.InputField(
        desc="Chapter number (0 for preamble, 1+ for sections)"
    )

    extraction: ChapterExtraction = dspy.OutputField(
        desc="All mathematical entities found with precise line ranges"
    )


class ExtractWithValidation(dspy.Signature):
    """
    Extract mathematical entities from a chapter and validate the results.

    You have access to a validation tool that checks if your extraction is correct.
    Use the tool to validate your work and fix any errors found.

    Workflow:
    1. Extract entities from the chapter text
    2. Call validate_extraction_tool with your extraction as JSON
    3. Read validation feedback
    4. If errors exist, revise and try again
    5. Repeat until validation passes or you've made max attempts
    """

    chapter_with_lines: str = dspy.InputField(
        desc="Chapter text with line numbers in format 'NNN: content'"
    )
    chapter_number: int = dspy.InputField(
        desc="Chapter number (0 for preamble, 1+ for sections)"
    )
    validation_context: str = dspy.InputField(
        desc="Context for validation: file_path|||article_id|||chapter_text"
    )

    extraction: ChapterExtraction = dspy.OutputField(
        desc="Validated ChapterExtraction with all entities"
    )


class MathematicalConceptExtractor(dspy.Module):
    """
    ReAct-based DSPy module for extracting mathematical concepts from chapters.

    Uses DSPy's ReAct agent with validation tool to iteratively improve extraction.
    The agent can call the validation tool and self-correct based on feedback.
    """

    def __init__(self, max_iters: int = 3):
        """
        Initialize extractor.

        Args:
            max_iters: Maximum number of ReAct iterations (default: 3)
        """
        super().__init__()
        # Create ReAct agent with validation tool
        self.react_agent = dspy.ReAct(
            ExtractWithValidation,
            tools=[validate_extraction_tool],
            max_iters=max_iters
        )

    def forward(
        self,
        chapter_with_lines: str,
        chapter_number: int,
        file_path: str = "",
        article_id: str = ""
    ) -> ChapterExtraction:
        """
        Extract concepts from a numbered chapter using ReAct agent.

        Args:
            chapter_with_lines: Chapter text with line numbers
            chapter_number: Chapter index
            file_path: Path to source file (for validation)
            article_id: Article identifier (for validation)

        Returns:
            Validated ChapterExtraction
        """
        # Prepare validation context
        validation_context = f"{file_path}|||{article_id}|||{chapter_with_lines}"

        try:
            # Run ReAct agent
            result = self.react_agent(
                chapter_with_lines=chapter_with_lines,
                chapter_number=chapter_number,
                validation_context=validation_context
            )

            extraction = result.extraction
            return extraction

        except Exception as e:
            print(f"  ✗ ReAct agent failed: {e}")
            # Return empty extraction as fallback
            return ChapterExtraction(
                section_id=f"Chapter {chapter_number}",
                definitions=[],
                theorems=[],
                proofs=[],
                axioms=[],
                parameters=[],
                remarks=[],
                citations=[]
            )


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================


def sanitize_label(raw_label: str) -> str:
    """
    Sanitize a raw label into a valid SourceLocation label format.

    Converts any string into a valid label matching pattern: ^[a-z][a-z0-9-]*$

    Args:
        raw_label: Raw label string (may contain spaces, periods, etc.)

    Returns:
        Sanitized label with only lowercase letters, digits, and hyphens

    Examples:
        >>> sanitize_label("## 1. Introduction")
        'section-1-introduction'
        >>> sanitize_label("Chapter 0 - Preamble")
        'section-chapter-0-preamble'
        >>> sanitize_label("section-Test Value")
        'section-test-value'
    """
    import re

    # Convert to lowercase
    label = raw_label.lower()

    # Remove leading markdown headers (##, ###, etc.)
    label = re.sub(r'^#+\s*', '', label)

    # Replace any sequence of non-alphanumeric characters with a single hyphen
    label = re.sub(r'[^a-z0-9]+', '-', label)

    # Remove leading/trailing hyphens
    label = label.strip('-')

    # Ensure it starts with a letter (not a digit)
    if label and label[0].isdigit():
        label = f"section-{label}"
    elif not label or not label[0].isalpha():
        label = f"section-{label}" if label else "section-unknown"

    # Collapse multiple consecutive hyphens
    label = re.sub(r'-+', '-', label)

    return label


def create_source_location(
    label: str,
    line_start: int,
    line_end: int,
    file_path: str,
    article_id: str
) -> SourceLocation:
    """
    Create a SourceLocation object from line range information.

    Args:
        label: Entity label
        line_start: Starting line number (1-indexed, inclusive)
        line_end: Ending line number (1-indexed, inclusive)
        file_path: Path to source markdown file
        article_id: Article identifier

    Returns:
        SourceLocation object with populated metadata
    """
    return SourceLocation(
        file_path=file_path,
        line_range=TextLocation.from_single_range(line_start, line_end),
        label=label,
        article_id=article_id
    )


def convert_to_raw_document_section(
    extraction: ChapterExtraction,
    file_path: str,
    article_id: str,
    chapter_text: str,
) -> tuple[RawDocumentSection, list[str]]:
    """
    Convert DSPy extraction output to RawDocumentSection.

    This function creates RawDocumentSection with SourceLocation metadata but WITHOUT
    extracting full text content. The full_text fields are left empty as they can be
    extracted later from the source file using the SourceLocation line ranges.

    This minimizes output size and simplifies the extraction process.

    Args:
        extraction: DSPy extraction result
        file_path: Path to source markdown file
        article_id: Article identifier (e.g., "01_fragile_gas_framework")
        chapter_text: Original chapter text (used only for parsing section_id)

    Returns:
        Tuple of (RawDocumentSection, list of conversion warnings)
    """

    conversion_warnings = []

    # Helper to create source location
    def make_source(label: str, line_start: int, line_end: int) -> SourceLocation:
        return create_source_location(label, line_start, line_end, file_path, article_id)

    # Convert definitions
    raw_definitions = []
    for d in extraction.definitions:
        try:
            raw_def = RawDefinition(
                label=d.label,
                term=d.term,
                full_text=TextLocation.from_single_range(d.line_start, d.line_end),
                parameters_mentioned=d.parameters_mentioned,
                source=make_source(d.label, d.line_start, d.line_end),
            )
            raw_definitions.append(raw_def)
        except Exception as e:
            warning = f"Failed to convert definition {d.label}: {e}"
            conversion_warnings.append(warning)
            print(f"  ⚠ {warning}")

    # Convert theorems
    raw_theorems = []
    for t in extraction.theorems:
        try:
            raw_thm = RawTheorem(
                label=t.label,
                statement_type=t.statement_type,
                conclusion_formula_latex=t.conclusion_formula_latex,
                explicit_definition_references=t.definition_references,
                full_text="",  # Text can be extracted later from source location
                source=make_source(t.label, t.line_start, t.line_end),
            )
            raw_theorems.append(raw_thm)
        except Exception as e:
            warning = f"Failed to convert theorem {t.label}: {e}"
            conversion_warnings.append(warning)
            print(f"  ⚠ {warning}")

    # Convert proofs
    raw_proofs = []
    for p in extraction.proofs:
        try:
            # Handle strategy text
            strategy_text = None
            if p.strategy_line_start is not None and p.strategy_line_end is not None:
                strategy_text = TextLocation.from_single_range(
                    p.strategy_line_start, p.strategy_line_end
                )

            # Handle steps
            steps = None
            if p.steps:
                steps = [
                    TextLocation.from_single_range(start, end)
                    for start, end in p.steps
                ]

            # Handle full body
            full_body_text = None
            if p.full_body_line_start is not None and p.full_body_line_end is not None:
                full_body_text = TextLocation.from_single_range(
                    p.full_body_line_start, p.full_body_line_end
                )

            raw_proof = RawProof(
                label=p.label,
                proves_label=p.proves_label,
                strategy_text=strategy_text,
                steps=steps,
                full_body_text=full_body_text,
                explicit_theorem_references=p.theorem_references,
                citations_in_text=p.citations,
                full_text="",  # Text can be extracted later from source location
                source=make_source(p.label, p.line_start, p.line_end),
            )
            raw_proofs.append(raw_proof)
        except Exception as e:
            warning = f"Failed to convert proof {p.label}: {e}"
            conversion_warnings.append(warning)
            print(f"  ⚠ {warning}")

    # Convert axioms
    raw_axioms = []
    for a in extraction.axioms:
        try:
            # Handle condition text
            condition_text = None
            if a.condition_line_start is not None and a.condition_line_end is not None:
                condition_text = TextLocation.from_single_range(
                    a.condition_line_start, a.condition_line_end
                )

            # Handle failure mode text
            failure_mode_text = None
            if a.failure_mode_line_start is not None and a.failure_mode_line_end is not None:
                failure_mode_text = TextLocation.from_single_range(
                    a.failure_mode_line_start, a.failure_mode_line_end
                )

            raw_axiom = RawAxiom(
                label=a.label,
                name=a.name,
                core_assumption_text=TextLocation.from_single_range(
                    a.core_assumption_line_start, a.core_assumption_line_end
                ),
                parameters_text=a.parameters,
                condition_text=condition_text,
                failure_mode_analysis_text=failure_mode_text,
                full_text="",  # Text can be extracted later from source location
                source=make_source(a.label, a.line_start, a.line_end),
            )
            raw_axioms.append(raw_axiom)
        except Exception as e:
            warning = f"Failed to convert axiom {a.label}: {e}"
            conversion_warnings.append(warning)
            print(f"  ⚠ {warning}")

    # Convert parameters
    raw_parameters = []
    for param in extraction.parameters:
        try:
            raw_param = RawParameter(
                label=param.label,
                symbol=param.symbol,
                meaning=param.meaning,
                scope=param.scope,
                full_text="",  # Text can be extracted later from source location
                source=make_source(param.label, param.line_start, param.line_end),
            )
            raw_parameters.append(raw_param)
        except Exception as e:
            warning = f"Failed to convert parameter {param.label}: {e}"
            conversion_warnings.append(warning)
            print(f"  ⚠ {warning}")

    # Convert remarks
    raw_remarks = []
    for r in extraction.remarks:
        try:
            raw_remark = RawRemark(
                label=r.label,
                remark_type=r.remark_type,
                full_text="",  # Text can be extracted later from source location
                source=make_source(r.label, r.line_start, r.line_end),
            )
            raw_remarks.append(raw_remark)
        except Exception as e:
            warning = f"Failed to convert remark {r.label}: {e}"
            conversion_warnings.append(warning)
            print(f"  ⚠ {warning}")

    # Convert citations
    raw_citations = []
    for c in extraction.citations:
        try:
            raw_citation = RawCitation(
                key_in_text=c.key_in_text,
                full_entry_text=TextLocation.from_single_range(c.line_start, c.line_end),
                full_text="",  # Text can be extracted later from source location
                source=make_source(c.key_in_text, c.line_start, c.line_end),
            )
            raw_citations.append(raw_citation)
        except Exception as e:
            warning = f"Failed to convert citation {c.key_in_text}: {e}"
            conversion_warnings.append(warning)
            print(f"  ⚠ {warning}")

    # Create section source (use first line of chapter)
    import re
    first_line = 1
    last_line = 1

    # Parse line numbers from chapter text
    lines = chapter_text.split('\n')
    if lines:
        first_match = re.match(r"\s*(\d+):\s", lines[0])
        if first_match:
            first_line = int(first_match.group(1))

        last_match = re.match(r"\s*(\d+):\s", lines[-1])
        if last_match:
            last_line = int(last_match.group(1))

    # Sanitize section_id to create a valid label
    section_label = sanitize_label(extraction.section_id)
    if not section_label.startswith("section-"):
        section_label = f"section-{section_label}"

    section_source = make_source(
        section_label,
        first_line,
        last_line
    )

    # Create RawDocumentSection (full_text can be extracted later from source location)
    raw_section = RawDocumentSection(
        section_id=extraction.section_id,
        definitions=raw_definitions,
        theorems=raw_theorems,
        proofs=raw_proofs,
        axioms=raw_axioms,
        parameters=raw_parameters,
        remarks=raw_remarks,
        citations=raw_citations,
        source=section_source,
        full_text="",  # Text can be extracted later from source location
    )

    return raw_section, conversion_warnings


# =============================================================================
# MAIN EXTRACTION WORKFLOW FUNCTION
# =============================================================================


def extract_chapter(
    chapter_text: str,
    chapter_number: int,
    file_path: str,
    article_id: str,
    max_iters: int = 3,
    verbose: bool = True
) -> tuple[RawDocumentSection | None, list[str]]:
    """
    Extract mathematical concepts from a chapter using ReAct agent.

    This is the main entry point for the EXTRACTION workflow.

    Args:
        chapter_text: Chapter text with line numbers (format: "NNN: content")
        chapter_number: Chapter index (0 for preamble, 1+ for sections)
        file_path: Path to source markdown file
        article_id: Article identifier (e.g., "01_fragile_gas_framework")
        max_iters: Maximum ReAct iterations (default: 3)
        verbose: Print progress information

    Returns:
        Tuple of (RawDocumentSection or None, list of errors)
        - RawDocumentSection: Extracted and validated data (None if extraction failed)
        - list[str]: Any errors or warnings encountered during extraction
    """
    errors_encountered = []

    # Create extractor
    extractor = MathematicalConceptExtractor(max_iters=max_iters)

    # Extract entities
    try:
        extraction = extractor(
            chapter_with_lines=chapter_text,
            chapter_number=chapter_number,
            file_path=file_path,
            article_id=article_id
        )

        if verbose:
            print(f"  ✓ Extraction completed")

    except Exception as e:
        error_msg = f"Extraction failed: {str(e)}"
        errors_encountered.append(error_msg)
        if verbose:
            print(f"  ✗ {error_msg}")

        # Create empty extraction as fallback
        import re
        section_id = f"Chapter {chapter_number}"
        # Try to extract section ID from chapter text
        for line in chapter_text.split('\n')[:20]:
            content = re.sub(r"^\s*\d+:\s*", "", line)
            if content.startswith("## "):
                section_id = content.strip()
                break

        extraction = ChapterExtraction(
            section_id=section_id,
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            parameters=[],
            remarks=[],
            citations=[]
        )

    # Convert to RawDocumentSection
    try:
        raw_section, conversion_warnings = convert_to_raw_document_section(
            extraction,
            file_path=file_path,
            article_id=article_id,
            chapter_text=chapter_text
        )

        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        if verbose and raw_section:
            print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")

        return raw_section, errors_encountered

    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        errors_encountered.append(error_msg)
        if verbose:
            print(f"  ✗ {error_msg}")

        return None, errors_encountered
