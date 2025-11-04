"""
Extraction workflow for mathematical concept extraction using DSPy ReAct agents.

**⚠️ DEPRECATED: This file is legacy code. Use the modular API instead:**

```python
# OLD (deprecated):
from mathster.parsing.extract_workflow import extract_chapter

# NEW (recommended):
from mathster.parsing import workflows

workflows.extract_chapter(...)

# Or directly:
from mathster.parsing.workflows import extract_chapter
```

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

⚠️ For new code, use: `from mathster.parsing.workflows import extract_chapter`
"""

from itertools import starmap
import json
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from mathster.core.article_system import SourceLocation, TextLocation
from mathster.core.raw_data import (
    RawAssumption,
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
# ERROR HANDLING HELPER
# =============================================================================


def make_error_dict(error_msg: str, value: any | None = None) -> dict:
    """
    Create structured error dictionary for tracking extraction/improvement failures.

    This helper ensures consistent error format across the codebase:
    - 'error': The error message string (human-readable description)
    - 'value': The incorrectly generated value that caused the error (for debugging)

    Args:
        error_msg: Error message string describing what went wrong
        value: The malformed data, entity, or context that caused the error.
               Can be dict, list, exception details, or any JSON-serializable value.
               Use None if no meaningful value is available.

    Returns:
        Dictionary with 'error' and 'value' keys

    Examples:
        >>> make_error_dict("Failed to parse entity", {"label": "def-x", "term": "..."})
        {'error': 'Failed to parse entity', 'value': {'label': 'def-x', 'term': '...'}}

        >>> make_error_dict("LLM timeout", {"attempt": 1, "exception": "TimeoutError"})
        {'error': 'LLM timeout', 'value': {'attempt': 1, 'exception': 'TimeoutError'}}
    """
    return {"error": error_msg, "value": value}


# =============================================================================
# LABEL LOOKUP HELPER - Resolve text references to labels
# =============================================================================


def lookup_label_from_context(
    reference_text: str,
    context: str,
    reference_type: Literal["theorem", "definition", "proof"],
) -> str:
    """
    Look up the actual label for a text reference in the document context.

    Strategy:
    1. Try to find :label: directive near the reference text in context
    2. If not found, generate standardized label from text

    Args:
        reference_text: Text reference like "Theorem 1.4" or "Lipschitz continuous"
        context: Chapter text with line numbers
        reference_type: Type of reference to help with pattern matching

    Returns:
        Label string (e.g., "thm-convergence" or "def-lipschitz-continuous")

    Examples:
        >>> lookup_label_from_context("Theorem 1.4", chapter_text, "theorem")
        "thm-convergence"  # Found :label: thm-convergence near "Theorem 1.4"

        >>> lookup_label_from_context("Lemma 2.3", chapter_text, "theorem")
        "lem-2-3"  # No :label: found, generated from text

        >>> lookup_label_from_context("Lipschitz continuous", chapter_text, "definition")
        "def-lipschitz-continuous"  # Generated from term
    """
    import re

    # Prefix mapping
    prefix_map = {
        "theorem": ["thm", "lem", "prop", "cor"],
        "definition": ["def"],
        "proof": ["proof"],
    }
    prefixes = prefix_map.get(reference_type, [])

    # Strategy 1: Search for :label: directive near reference text
    # Pattern: Look for Jupyter Book directive with :label: nearby

    # Normalize reference text for searching
    search_text = reference_text.strip()

    # Try to find the reference in context
    lines = context.split("\n")
    for i, line in enumerate(lines):
        # Remove line numbers from the line (format: "  123: content")
        line_content = re.sub(r"^\s*\d+:\s*", "", line)

        # Check if line contains the reference text
        if search_text.lower() in line_content.lower():
            # Search nearby lines (±10 lines) for :label: directive
            start = max(0, i - 10)
            end = min(len(lines), i + 10)
            context_window = "\n".join(lines[start:end])

            # Look for :label: directives matching expected prefixes
            for prefix in prefixes:
                label_pattern = rf":label:\s+({prefix}-[a-z0-9-_]+)"
                match = re.search(label_pattern, context_window)
                if match:
                    return match.group(1)

    # Strategy 2: Generate standardized label from text
    # Use sanitize_label to ensure correct format (defined below in this file)

    if reference_type == "theorem":
        # Extract theorem number/type from text
        # "Theorem 1.4" → "thm-1-4"
        # "Lemma 2.3" → "lem-2-3"
        thm_match = re.match(
            r"(Theorem|Lemma|Proposition|Corollary)\s+([\d.]+)", reference_text, re.IGNORECASE
        )
        if thm_match:
            thm_type = thm_match.group(1).lower()
            thm_num = thm_match.group(2).replace(".", "-")
            prefix = {
                "theorem": "thm",
                "lemma": "lem",
                "proposition": "prop",
                "corollary": "cor",
            }[thm_type]
            return f"{prefix}-{thm_num}"

    elif reference_type == "definition":
        # "Lipschitz continuous" → "def-lipschitz-continuous"
        # Use sanitize_label (defined below) to normalize
        term = sanitize_label(reference_text)
        if not term.startswith("def-"):
            return f"def-{term}"
        return term

    # Fallback: generic sanitization
    label = sanitize_label(reference_text)
    if not any(label.startswith(p + "-") for p in prefixes):
        label = f"{prefixes[0]}-{label}"
    return label


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
        description=(
            "Labels of definitions referenced in this theorem (e.g., ['def-lipschitz-continuous', 'def-v-porous-on-balls']). "
            "Extract labels from :label: directives if present, or generate using pattern def-{normalized-term}. "
            "ALWAYS use labels (def-*), NEVER use text like 'Lipschitz continuous'."
        ),
    )


class ProofExtraction(ExtractedEntity):
    """Extracted proof with optional structure."""

    proves_label: str = Field(
        ...,
        description=(
            "Label of the theorem this proof proves (e.g., 'thm-main-result', 'lem-gradient-bound'). "
            "Extract from :label: directive near the theorem statement, or generate from theorem numbering (e.g., 'Theorem 1.1' → 'thm-1-1'). "
            "MUST match pattern: thm-*|lem-*|prop-*|cor-*. This is REQUIRED - extraction will fail if not a valid label."
        ),
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
        description=(
            "Labels of theorems/lemmas referenced in this proof (e.g., ['thm-convergence', 'lem-gradient-bound']). "
            "Extract from :label: directives or generate from numbering (e.g., 'Theorem 1.4' → 'thm-1-4'). "
            "Use patterns: thm-*, lem-*, prop-*, cor-*. PREFER labels over text when possible."
        ),
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Bibliographic citations (e.g., ['[16]', 'Han (2016)'])",
    )


class AxiomExtraction(ExtractedEntity):
    """Extracted axiom with structured components."""

    name: str = Field(..., description="Title/name of the axiom")
    core_assumption_line_start: int = Field(
        ..., description="Starting line of core assumption text"
    )
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

    symbol: str = Field(..., description="Mathematical symbol (e.g., 'h', '\\beta', 'N')")
    meaning: str = Field(..., description="Definition/meaning of the symbol")
    scope: Literal["global", "local"] = Field(
        ..., description="Whether parameter is global (whole document) or local (section/proof)"
    )


class RemarkExtraction(ExtractedEntity):
    """Extracted remark, note, or observation."""

    remark_type: Literal["note", "remark", "observation", "comment", "example"] = Field(
        ..., description="Type of informal statement"
    )


class AssumptionExtraction(ExtractedEntity):
    """Extracted local assumption or hypothesis.

    Assumptions are local conditions that apply to specific theorems or proofs,
    such as 'Assume the domain is bounded' or 'Suppose f is Lipschitz continuous'.
    Unlike axioms (global framework principles), assumptions are statement-specific.
    """


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
    assumptions: list[AssumptionExtraction] = Field(default_factory=list)
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
        default_factory=dict, description="Count of successfully validated entities by type"
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
    extraction_dict: dict, file_path: str, article_id: str, chapter_text: str
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
        "assumptions": 0,
        "parameters": 0,
        "remarks": 0,
        "citations": 0,
    }

    try:
        # Parse as ChapterExtraction
        extraction = ChapterExtraction(**extraction_dict)

        # Attempt to convert to RawDocumentSection
        try:
            raw_section, conversion_warnings = convert_to_raw_document_section(
                extraction, file_path=file_path, article_id=article_id, chapter_text=chapter_text
            )

            # Add conversion warnings to warnings list
            if conversion_warnings:
                warnings.extend(conversion_warnings)

            # Count successfully validated entities
            entities_validated["definitions"] = len(raw_section.definitions)
            entities_validated["theorems"] = len(raw_section.theorems)
            entities_validated["proofs"] = len(raw_section.proofs)
            entities_validated["axioms"] = len(raw_section.axioms)
            entities_validated["assumptions"] = len(raw_section.assumptions)
            entities_validated["parameters"] = len(raw_section.parameters)
            entities_validated["remarks"] = len(raw_section.remarks)
            entities_validated["citations"] = len(raw_section.citations)

            # Check if we got any entities
            total_entities = sum(entities_validated.values())
            if total_entities == 0:
                warnings.append(
                    "No entities were extracted from this chapter. Is the chapter empty?"
                )

            # Validate label patterns
            for defn in extraction.definitions:
                if not defn.label.startswith("def-"):
                    errors.append(f"Definition label '{defn.label}' must start with 'def-'")

            for thm in extraction.theorems:
                if not any(thm.label.startswith(p) for p in ["thm-", "lem-", "prop-", "cor-"]):
                    errors.append(
                        f"Theorem label '{thm.label}' must start with thm-/lem-/prop-/cor-"
                    )

            for proof in extraction.proofs:
                if not proof.label.startswith("proof-"):
                    errors.append(f"Proof label '{proof.label}' must start with 'proof-'")

            for axiom in extraction.axioms:
                if not any(axiom.label.startswith(p) for p in ["axiom-", "ax-", "def-axiom-"]):
                    errors.append(
                        f"Axiom label '{axiom.label}' must start with axiom-/ax-/def-axiom-"
                    )

            for param in extraction.parameters:
                if not param.label.startswith("param-"):
                    errors.append(f"Parameter label '{param.label}' must start with 'param-'")

            for remark in extraction.remarks:
                if not remark.label.startswith("remark-"):
                    errors.append(f"Remark label '{remark.label}' must start with 'remark-'")

            for assumption in extraction.assumptions:
                if not assumption.label.startswith("assumption-"):
                    errors.append(
                        f"Assumption label '{assumption.label}' must start with 'assumption-'"
                    )

            # NEW: Validate reference formats
            # PERMISSIVE: Validate theorem definition_references (warnings only)
            for thm in extraction.theorems:
                for def_ref in thm.definition_references:
                    if not def_ref.startswith("def-"):
                        warnings.append(
                            f"Theorem '{thm.label}': definition_references should be labels "
                            f"starting with 'def-', got '{def_ref}'. Consider extracting proper label."
                        )

            # Validate proofs
            for proof in extraction.proofs:
                # STRICT: Validate proves_label (MUST be label format)
                if not any(
                    proof.proves_label.startswith(p) for p in ["thm-", "lem-", "prop-", "cor-"]
                ):
                    errors.append(
                        f"Proof '{proof.label}': proves_label MUST be a theorem label "
                        f"(thm-*|lem-*|prop-*|cor-*), got '{proof.proves_label}'. "
                        f"This is a CRITICAL error - extraction cannot proceed."
                    )

                # PERMISSIVE: Validate theorem_references (warnings only)
                for thm_ref in proof.theorem_references:
                    if not any(thm_ref.startswith(p) for p in ["thm-", "lem-", "prop-", "cor-"]):
                        warnings.append(
                            f"Proof '{proof.label}': theorem_references should be labels "
                            f"(thm-*|lem-*|prop-*|cor-*), got '{thm_ref}'. Consider extracting proper label."
                        )

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
            errors.append(f"Failed to convert to RawDocumentSection: {e!s}")

    except Exception as e:
        errors.append(f"Failed to parse ChapterExtraction: {e!s}")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid, errors=errors, warnings=warnings, entities_validated=entities_validated
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
            extraction_dict, file_path=file_path, article_id=article_id, chapter_text=chapter_text
        )

        return validation.get_feedback()

    except Exception as e:
        return f"Validation error: {e!s}"


def compare_labels_tool(extraction_json: str, context: str) -> str:
    """
    Tool for ReAct agent to compare extracted labels against source document.

    This tool validates that all labels in the extraction actually exist in the
    source document, and identifies any labels that were missed. It helps the
    agent detect and fix:
    - Hallucinated labels (invented by LLM, not in source)
    - Missed labels (in source but not extracted)

    Use this tool BEFORE validate_extraction_tool to catch label issues early.

    Args:
        extraction_json: JSON string of ChapterExtraction
        context: Context string containing file_path, article_id, chapter_text

    Returns:
        Comparison report string showing found/hallucinated/missed labels
    """
    try:
        from mathster.parsing.tools import compare_extraction_with_source

        # Parse context (format: "file_path|||article_id|||chapter_text")
        parts = context.split("|||")
        if len(parts) != 3:
            return "Error: Invalid context format"

        _file_path, _article_id, chapter_text = parts

        # Parse extraction JSON
        extraction_dict = json.loads(extraction_json)

        # Compare extraction with source
        _comparison, report = compare_extraction_with_source(extraction_dict, chapter_text)

        # Return report with actionable feedback
        return report

    except Exception as e:
        return f"Label comparison error: {e!s}"


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

    8. ASSUMPTIONS (label: assumption-*):
       - Local hypotheses and conditions for theorems/proofs
       - Statement-specific assumptions (unlike global axioms)
       - Examples: "Assume the domain is bounded", "Suppose f is Lipschitz continuous"

    LABEL GENERATION:
    - If entity has an explicit label (e.g., :label: thm-main-result), use it
    - Otherwise, generate a descriptive label following the pattern:
      * def-{normalized-term} for definitions
      * thm-{descriptive-name} for theorems
      * lem-{descriptive-name} for lemmas
      * proof-{theorem-label} for proofs
      * etc.
    - Labels must be lowercase with hyphens, matching pattern: ^[a-z]+-[a-z0-9-]+$

    REFERENCE EXTRACTION GUIDELINES (CRITICAL):
    ==========================================

    1. **Extract Labels, Not Text**:
       - ❌ WRONG: "Theorem 1.4", "Lipschitz continuous"
       - ✅ CORRECT: "thm-convergence", "def-lipschitz-continuous"

    2. **Label Lookup Strategy**:
       - First: Search for :label: directive in Jupyter Book markup near the reference
       - Fallback: Generate standardized label from text using patterns below

    3. **Label Patterns**:
       - Definitions: def-{normalized-term} (e.g., "def-lipschitz-continuous")
       - Theorems: thm-{number-or-name} (e.g., "thm-1-4" or "thm-main-result")
       - Lemmas: lem-{number-or-name} (e.g., "lem-2-3" or "lem-gradient-bound")
       - Propositions: prop-{number-or-name}
       - Corollaries: cor-{number-or-name}

    4. **Examples**:

       Source text: "By Theorem 1.4, we have..."
       Search context for: :::{prf:theorem} ... :label: thm-convergence
       Extract: theorem_references = ["thm-convergence"]

       Source text: "where f is Lipschitz continuous"
       Search context for: :::{prf:definition} ... :label: def-lipschitz
       Extract: definition_references = ["def-lipschitz"]

       Source text: "Proof of Theorem 1.1"
       Search context for: :::{prf:theorem} Theorem 1.1 ... :label: thm-main-result
       Extract: proves_label = "thm-main-result"

    5. **Critical Field - proves_label**:
       - This field is MANDATORY and MUST be a valid label
       - Pattern: thm-*|lem-*|prop-*|cor-*
       - Invalid examples will cause validation failure

    6. **When :label: Not Found**:
       - Generate standardized label from text numbering:
         * "Theorem 1.4" → "thm-1-4"
         * "Lemma 2.3" → "lem-2-3"
         * "Lipschitz continuous" → "def-lipschitz-continuous"

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
    chapter_number: int = dspy.InputField(desc="Chapter number (0 for preamble, 1+ for sections)")

    extraction: ChapterExtraction = dspy.OutputField(
        desc="All mathematical entities found with precise line ranges"
    )


class ExtractWithValidation(dspy.Signature):
    """
    Extract mathematical entities from a chapter and validate the results.

    You have access to TWO validation tools to ensure extraction quality:

    1. **compare_labels_tool**: Compares your extracted labels against the source
       document to detect:
       - Hallucinated labels (you invented labels not in the source) ← FIX THESE
       - Missed labels (labels in source you didn't extract) ← ADD THESE
       Use this FIRST to validate label accuracy before structural validation.

    2. **validate_extraction_tool**: Validates overall extraction structure and
       converts to RawDocumentSection. Checks:
       - All required fields are present
       - Label patterns are correct (def-*, thm-*, etc.)
       - Line numbers are valid
       Use this AFTER label validation passes.

    Recommended Workflow:
    1. Extract entities from the chapter text
    2. Call compare_labels_tool with your extraction as JSON
       - If hallucinated labels found: REMOVE them and retry
       - If missed labels found: ADD them and retry
    3. Once label comparison passes, call validate_extraction_tool
    4. If structural errors exist, fix and retry
    5. Repeat until both validations pass
    """

    chapter_with_lines: str = dspy.InputField(
        desc="Chapter text with line numbers in format 'NNN: content'"
    )
    chapter_number: int = dspy.InputField(desc="Chapter number (0 for preamble, 1+ for sections)")
    validation_context: str = dspy.InputField(
        desc="Context for validation: file_path|||article_id|||chapter_text"
    )
    previous_error_report: str = dspy.InputField(
        default="",
        desc=(
            "Error report from previous failed extraction attempt. "
            "If provided, READ THIS CAREFULLY and fix all issues mentioned. "
            "Empty string means this is the first attempt."
        ),
    )

    extraction: ChapterExtraction = dspy.OutputField(
        desc="Validated ChapterExtraction with all entities"
    )


class MathematicalConceptExtractor(dspy.Module):
    """
    ReAct-based DSPy module for extracting mathematical concepts from chapters.

    Uses DSPy's ReAct agent with TWO validation tools to iteratively improve extraction:
    1. compare_labels_tool: Validates label accuracy vs source (catches hallucinations)
    2. validate_extraction_tool: Validates structure and converts to RawDocumentSection

    The agent can call both tools and self-correct based on feedback, ensuring high
    extraction quality by catching both label errors and structural issues.
    """

    def __init__(self, max_iters: int = 5):
        """
        Initialize extractor.

        Args:
            max_iters: Maximum number of ReAct iterations (default: 5)
        """
        super().__init__()
        # Create ReAct agent with validation tools
        # - compare_labels_tool: Check label accuracy vs source (use first)
        # - validate_extraction_tool: Validate structure and convert to RawDocumentSection (use after)
        self.react_agent = dspy.ReAct(
            ExtractWithValidation,
            tools=[compare_labels_tool, validate_extraction_tool],
            max_iters=max_iters,
        )

    def forward(
        self,
        chapter_with_lines: str,
        chapter_number: int,
        file_path: str = "",
        article_id: str = "",
        previous_error_report: str = "",
    ) -> ChapterExtraction:
        """
        Extract concepts from a numbered chapter using ReAct agent.

        Args:
            chapter_with_lines: Chapter text with line numbers
            chapter_number: Chapter index
            file_path: Path to source file (for validation)
            article_id: Article identifier (for validation)
            previous_error_report: Error report from previous attempt (for retry logic)

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
                validation_context=validation_context,
                previous_error_report=previous_error_report,
            )

            return result.extraction

        except Exception as e:
            print(f"  ✗ ReAct agent failed: {e}")
            # Re-raise for retry logic to handle
            raise


# =============================================================================
# ERROR REPORTING FOR RETRY LOGIC
# =============================================================================


def generate_detailed_error_report(
    error: Exception, attempt_number: int, max_retries: int, extraction_context: dict | None = None
) -> str:
    """
    Generate a detailed, LLM-friendly error report for failed extractions.

    This function transforms technical errors (Pydantic ValidationError, exceptions)
    into actionable feedback that helps the ReAct agent self-correct on retry.

    Args:
        error: The exception that occurred
        attempt_number: Current attempt number (1-indexed)
        max_retries: Maximum number of retry attempts
        extraction_context: Optional context about what was being extracted

    Returns:
        Formatted error report string optimized for LLM consumption

    Example Output:
        ```
        EXTRACTION ERROR REPORT
        =======================

        Attempt: 2/3
        Previous Error: Pydantic validation failed

        VALIDATION ERRORS:
        1. Field: definitions[0].term
           Problem: Missing required field
           Fix: Every definition must have a 'term' field with the exact text being defined
           Example: "term": "Lipschitz continuous"
        ```
    """
    import traceback

    from pydantic import ValidationError

    # Build header
    report_lines = [
        "=" * 70,
        "EXTRACTION ERROR REPORT",
        "=" * 70,
        "",
        f"Attempt: {attempt_number}/{max_retries}",
        f"Error Type: {type(error).__name__}",
        "",
    ]

    # Add context if provided
    if extraction_context:
        report_lines.append("EXTRACTION CONTEXT:")
        for key, value in extraction_context.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:97] + "..."
            report_lines.append(f"  - {key}: {str_value}")
        report_lines.append("")

    # Parse error based on type
    if isinstance(error, ValidationError):
        # Pydantic validation error - parse into detailed field-level feedback
        report_lines.extend(("VALIDATION ERRORS:", ""))

        for i, err in enumerate(error.errors(), 1):
            # Extract error details
            field_path = ".".join(str(loc) for loc in err["loc"])
            error_type = err["type"]
            error_msg = err["msg"]

            report_lines.extend((f"{i}. Field: {field_path}", f"   Problem: {error_msg}"))

            # Provide specific fix guidance based on error type
            if error_type == "missing":
                report_lines.extend((
                    "   Fix: This is a REQUIRED field. You must provide a value.",
                    f"   → Ensure the field '{field_path}' is present in your output",
                ))

            elif error_type in {"string_type", "int_type", "float_type", "bool_type"}:
                expected_type = error_type.replace("_type", "")
                report_lines.extend((
                    f"   Fix: Field must be of type '{expected_type}'",
                    f"   → Check that '{field_path}' has the correct data type",
                ))

            elif "literal_error" in error_type:
                # Extract allowed values from error message
                report_lines.append("   Fix: Field must match one of the allowed literal values")
                if "Input should be" in error_msg:
                    report_lines.append(f"   → {error_msg}")

            elif "list_type" in error_type:
                report_lines.extend((
                    "   Fix: Field must be a list/array",
                    f"   → Wrap value in square brackets: [{field_path}]",
                ))

            else:
                # Generic guidance
                report_lines.append(f"   Fix: {error_msg}")

            # Add field-specific examples
            if "term" in field_path:
                report_lines.append('   Example: "term": "Lipschitz continuous"')
            elif "label" in field_path:
                report_lines.append(
                    '   Example: "label": "def-lipschitz" (must match :label: directive)'
                )
            elif "statement_type" in field_path:
                report_lines.extend((
                    '   Example: "statement_type": "theorem"',
                    '   Allowed values: "theorem", "lemma", "proposition", "corollary"',
                ))
            elif "line_start" in field_path or "line_end" in field_path:
                report_lines.extend((
                    '   Example: "line_start": 42, "line_end": 58',
                    "   → Must be integers within document line range",
                ))

            report_lines.append("")

    elif isinstance(error, json.JSONDecodeError):
        # JSON parsing error
        report_lines.extend((
            "JSON PARSING ERROR:",
            "",
            f"Problem: Invalid JSON syntax at line {error.lineno}, column {error.colno}",
            f"Message: {error.msg}",
            "",
            "Common JSON Errors:",
            "  - Missing closing bracket/brace: }, ], )",
            "  - Trailing comma in last array/object element",
            "  - Unquoted string values",
            "  - Single quotes instead of double quotes",
            "",
            "Fix: Ensure valid JSON formatting:",
            '  - All strings use double quotes: "string"',
            "  - All brackets/braces are properly closed",
            "  - No trailing commas",
            "",
        ))

    elif "timeout" in str(error).lower():
        # Timeout error
        report_lines.extend((
            "TIMEOUT ERROR:",
            "",
            "Problem: Extraction took too long and timed out",
            "",
            "Possible Causes:",
            "  - Chapter is very large (too many entities)",
            "  - Agent got stuck in reasoning loop",
            "  - Network/API latency issues",
            "",
            "Fix: Try to:",
            "  - Focus on extracting only the most important entities",
            "  - Use more concise reasoning steps",
            "  - Validate early to catch errors quickly",
            "",
        ))

    else:
        # Generic exception
        report_lines.extend(("GENERAL ERROR:", "", f"Problem: {error!s}", ""))

        # Include truncated traceback
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        report_lines.append("Traceback (last 10 lines):")
        for line in tb_lines[-10:]:
            report_lines.append(f"  {line.rstrip()}")
        report_lines.append("")

    # Add general retry guidance
    report_lines.extend((
        "=" * 70,
        "RETRY GUIDANCE:",
        "=" * 70,
        "",
        "Read the errors above CAREFULLY and:",
        "1. Fix each field issue mentioned",
        "2. Verify data types match requirements",
        "3. Ensure all required fields are present",
        "4. Validate labels match :label: directives in source",
        "5. Call validate_extraction_tool to check before submitting",
        "",
        f"Remaining attempts: {max_retries - attempt_number}",
        "=" * 70,
    ))

    return "\n".join(report_lines)


# =============================================================================
# RETRY WRAPPERS
# =============================================================================


def extract_chapter_with_retry(
    chapter_text: str,
    chapter_number: int,
    file_path: str,
    article_id: str,
    max_iters: int = 10,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True,
) -> tuple[ChapterExtraction, list[str]]:
    """
    Extract chapter with automatic retry on failure.

    This wrapper adds retry logic around MathematicalConceptExtractor,
    generating detailed error reports and passing them to subsequent attempts.
    After first failure, switches to fallback model for remaining retries.

    Args:
        chapter_text: Chapter text with line numbers
        chapter_number: Chapter index
        file_path: Path to source markdown file
        article_id: Article identifier
        max_iters: Maximum ReAct iterations per attempt (default: 10)
        max_retries: Maximum number of retry attempts (default: 3)
        fallback_model: Model to use after first failure (default: claude-haiku-4-5)
        verbose: Print progress information

    Returns:
        Tuple of (ChapterExtraction, list of error messages)
        - Raises exception if all retries fail
    """
    errors_encountered = []
    extractor = MathematicalConceptExtractor(max_iters=max_iters)
    switched_to_fallback = False

    for attempt in range(1, max_retries + 1):
        try:
            # Build error report from previous attempt
            previous_error_report = ""
            if attempt > 1 and errors_encountered:
                # Get last error
                last_error = errors_encountered[-1]
                # Note: last_error is a string, we need to wrap it back into exception
                # For now, just pass the error string as context
                previous_error_report = f"Previous attempt failed with: {last_error}"

            if verbose and attempt > 1:
                print(f"  → Retry attempt {attempt}/{max_retries}")

            # Attempt extraction
            extraction = extractor(
                chapter_with_lines=chapter_text,
                chapter_number=chapter_number,
                file_path=file_path,
                article_id=article_id,
                previous_error_report=previous_error_report,
            )

            if verbose and attempt > 1:
                print(f"  ✓ Retry successful on attempt {attempt}")

            return extraction, errors_encountered

        except Exception as e:
            # Generate detailed error report
            extraction_context = {
                "chapter_number": chapter_number,
                "file_path": file_path,
                "article_id": article_id,
            }

            error_report = generate_detailed_error_report(
                error=e,
                attempt_number=attempt,
                max_retries=max_retries,
                extraction_context=extraction_context,
            )

            error_msg = f"Attempt {attempt} failed: {type(e).__name__}: {e!s}"
            errors_encountered.append(
                make_error_dict(
                    error_msg,
                    value={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "chapter_info": {
                            "chapter_number": chapter_number,
                            "file_path": file_path,
                            "article_id": article_id,
                        },
                    },
                )
            )

            if verbose:
                print(f"  ✗ Attempt {attempt}/{max_retries} failed: {type(e).__name__}")
                if attempt < max_retries:
                    print(f"\n{error_report}\n")

            # Switch to fallback model after first failure
            if attempt == 1 and max_retries > 1 and not switched_to_fallback:
                if verbose:
                    print(f"  → Switching to fallback model: {fallback_model}")

                # Import here to avoid circular dependency
                from mathster.parsing.dspy_pipeline import configure_dspy

                try:
                    configure_dspy(model=fallback_model)
                    switched_to_fallback = True
                    if verbose:
                        print(f"  ✓ Successfully switched to {fallback_model}")
                except Exception as switch_error:
                    if verbose:
                        print(f"  ⚠ Failed to switch model: {switch_error}")
                        print("  → Continuing with current model")

            # If this was the last attempt, raise with full context
            if attempt == max_retries:
                if verbose:
                    print(f"  ✗ All {max_retries} attempts failed")
                    print(f"\n{error_report}\n")
                raise Exception(
                    f"Extraction failed after {max_retries} attempts. "
                    f"Last error: {type(e).__name__}: {e!s}"
                ) from e

    # Should never reach here, but for safety
    raise Exception(f"Extraction failed after {max_retries} attempts")


def extract_label_with_retry(
    chapter_text: str,
    target_label: str,
    entity_type: str,
    file_path: str,
    article_id: str,
    max_iters_per_label: int = 3,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = False,
) -> tuple[dict, list[str]]:
    """
    Extract single label with automatic retry on failure.

    This wrapper adds retry logic around SingleLabelExtractor,
    generating detailed error reports for each failed attempt.
    After first failure, switches to fallback model for remaining retries.

    Args:
        chapter_text: Chapter text with line numbers
        target_label: The specific label to extract
        entity_type: Type of entity (definitions, theorems, etc.)
        file_path: Path to source markdown file
        article_id: Article identifier
        max_iters_per_label: Maximum ReAct iterations per attempt (default: 3)
        max_retries: Maximum number of retry attempts (default: 3)
        fallback_model: Model to use after first failure (default: claude-haiku-4-5)
        verbose: Print detailed progress (default: False for quieter output)

    Returns:
        Tuple of (entity dict, list of error messages)
        - Raises exception if all retries fail
    """
    errors_encountered = []
    extractor = SingleLabelExtractor(max_iters=max_iters_per_label)
    switched_to_fallback = False

    for attempt in range(1, max_retries + 1):
        try:
            # Build error report from previous attempt
            previous_error_report = ""
            if attempt > 1 and errors_encountered:
                last_error = errors_encountered[-1]
                previous_error_report = f"Previous attempt failed with: {last_error}"

            if verbose and attempt > 1:
                print(f"      → Retry attempt {attempt}/{max_retries} for {target_label}")

            # Attempt extraction
            entity_dict = extractor(
                chapter_with_lines=chapter_text,
                target_label=target_label,
                file_path=file_path,
                article_id=article_id,
                previous_error_report=previous_error_report,
            )

            # Check if extraction returned an error marker
            if "extraction_error" in entity_dict:
                raise Exception(entity_dict["extraction_error"])

            if verbose and attempt > 1:
                print(f"      ✓ Retry successful on attempt {attempt}")

            return entity_dict, errors_encountered

        except Exception as e:
            # Generate detailed error report
            extraction_context = {
                "target_label": target_label,
                "entity_type": entity_type,
                "file_path": file_path,
                "article_id": article_id,
            }

            error_report = generate_detailed_error_report(
                error=e,
                attempt_number=attempt,
                max_retries=max_retries,
                extraction_context=extraction_context,
            )

            error_msg = f"Attempt {attempt} failed: {type(e).__name__}: {e!s}"
            errors_encountered.append(
                make_error_dict(
                    error_msg,
                    value={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "target_label": target_label,
                        "entity_type": entity_type,
                    },
                )
            )

            if verbose:
                print(f"      ✗ Attempt {attempt}/{max_retries} failed: {type(e).__name__}")
                if attempt < max_retries:
                    print(f"\n{error_report}\n")

            # Switch to fallback model after first failure
            if attempt == 1 and max_retries > 1 and not switched_to_fallback:
                if verbose:
                    print(f"      → Switching to fallback model: {fallback_model}")

                # Import here to avoid circular dependency
                from mathster.parsing.dspy_pipeline import configure_dspy

                try:
                    configure_dspy(model=fallback_model)
                    switched_to_fallback = True
                    if verbose:
                        print(f"      ✓ Successfully switched to {fallback_model}")
                except Exception as switch_error:
                    if verbose:
                        print(f"      ⚠ Failed to switch model: {switch_error}")
                        print("      → Continuing with current model")

            # If this was the last attempt, raise with full context
            if attempt == max_retries:
                if verbose:
                    print(f"      ✗ All {max_retries} attempts failed for {target_label}")
                raise Exception(
                    f"Failed to extract {target_label} after {max_retries} attempts. "
                    f"Last error: {type(e).__name__}: {e!s}"
                ) from e

    # Should never reach here
    raise Exception(f"Failed to extract {target_label} after {max_retries} attempts")


# =============================================================================
# SINGLE-LABEL EXTRACTION (for label-by-label mode)
# =============================================================================


def validate_single_entity_tool(entity_json: str, context: str) -> str:
    """
    Tool for validating a single extracted entity.

    Args:
        entity_json: JSON string of extracted entity
        context: Validation context (file_path|||article_id|||target_label)

    Returns:
        Validation feedback string
    """
    try:
        import json

        # Parse context
        parts = context.split("|||")
        if len(parts) != 3:
            return "Error: Invalid context format. Expected: file_path|||article_id|||target_label"

        _file_path, _article_id, target_label = parts

        # Parse entity
        entity = json.loads(entity_json)

        # Validation checks
        errors = []
        warnings = []

        # Check label matches target
        entity_label = entity.get("label")
        if not entity_label:
            errors.append("Missing 'label' field")
        elif entity_label != target_label:
            errors.append(f"Label mismatch: expected '{target_label}', got '{entity_label}'")

        # Check line numbers present
        if "line_start" not in entity:
            errors.append("Missing 'line_start' field")
        if "line_end" not in entity:
            errors.append("Missing 'line_end' field")

        # Check line numbers valid
        if "line_start" in entity and "line_end" in entity:
            if entity["line_start"] > entity["line_end"]:
                errors.append(
                    f"Invalid line range: line_start ({entity['line_start']}) > line_end ({entity['line_end']})"
                )

        # Check label pattern (basic check)
        if entity_label:
            if not entity_label[0].isalpha():
                errors.append(f"Label must start with a letter: '{entity_label}'")
            if not all(c.isalnum() or c in "-_" for c in entity_label):
                errors.append(f"Label contains invalid characters: '{entity_label}'")

        # Build report
        if errors:
            report = "✗ VALIDATION FAILED\n"
            report += "\nErrors:\n"
            for err in errors:
                report += f"  - {err}\n"
        else:
            report = f"✓ VALIDATION PASSED for {target_label}\n"

        if warnings:
            report += "\nWarnings:\n"
            for warn in warnings:
                report += f"  - {warn}\n"

        return report

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON: {e}"
    except Exception as e:
        return f"Error: Validation failed: {e}"


class ExtractSingleLabel(dspy.Signature):
    """
    Extract ONE specific mathematical entity by its label.

    CRITICAL INSTRUCTIONS:
    - Extract ONLY the entity with target_label
    - DO NOT extract any other entities
    - Search for the :label: directive matching target_label
    - Extract all metadata for that specific entity

    WORKFLOW:
    1. Search chapter_with_lines for ":label: {target_label}"
    2. Identify the entity type from the directive (prf:definition, prf:theorem, etc.)
    3. Extract entity metadata:
       - label (must match target_label exactly)
       - line_start, line_end (line range of the entity)
       - Entity-specific fields (term for definitions, statement_type for theorems, etc.)
    4. Return entity as dict with all required fields
    5. Call validate_single_entity_tool to verify correctness

    IMPORTANT:
    - Do NOT extract entities with different labels
    - Do NOT modify or invent the target_label
    - line_start and line_end are REQUIRED
    - Ensure entity dict matches the expected format for its type

    EXAMPLE:
    target_label = "def-lipschitz"
    → Search for ":label: def-lipschitz"
    → Found in {prf:definition} directive at lines 45-52
    → Extract: {"label": "def-lipschitz", "line_start": 45, "line_end": 52, "term": "Lipschitz Continuous", ...}
    """

    chapter_with_lines: str = dspy.InputField(
        desc="Chapter text with line numbers in format 'NNN: content'"
    )
    target_label: str = dspy.InputField(
        desc="The specific label to extract (e.g., 'def-lipschitz', 'thm-main'). "
        "Extract ONLY this label, no others."
    )
    validation_context: str = dspy.InputField(
        desc="Context for validation: file_path|||article_id|||target_label"
    )
    previous_error_report: str = dspy.InputField(
        default="",
        desc=(
            "Error report from previous failed extraction attempt for this label. "
            "If provided, READ THIS CAREFULLY and fix all issues mentioned. "
            "Empty string means this is the first attempt."
        ),
    )

    entity: dict = dspy.OutputField(
        desc="Single extracted entity as dict with label, line_start, line_end, and entity-specific fields"
    )


class SingleLabelExtractor(dspy.Module):
    """
    ReAct-based DSPy module for extracting a single specific label.

    This extractor focuses on extracting just ONE entity at a time, identified
    by its label. Used for label-by-label extraction mode for higher accuracy.
    """

    def __init__(self, max_iters: int = 3):
        """
        Initialize single-label extractor.

        Args:
            max_iters: Maximum number of ReAct iterations (default: 3)
        """
        super().__init__()
        self.react_agent = dspy.ReAct(
            ExtractSingleLabel, tools=[validate_single_entity_tool], max_iters=max_iters
        )

    def forward(
        self,
        chapter_with_lines: str,
        target_label: str,
        file_path: str = "",
        article_id: str = "",
        previous_error_report: str = "",
    ) -> dict:
        """
        Extract a single entity by its label.

        Args:
            chapter_with_lines: Chapter text with line numbers
            target_label: The specific label to extract
            file_path: Path to source file (for validation)
            article_id: Article identifier (for validation)
            previous_error_report: Error report from previous attempt (for retry logic)

        Returns:
            Entity dict with label, line_start, line_end, and entity-specific fields
        """
        # Prepare validation context
        validation_context = f"{file_path}|||{article_id}|||{target_label}"

        try:
            # Run ReAct agent
            result = self.react_agent(
                chapter_with_lines=chapter_with_lines,
                target_label=target_label,
                validation_context=validation_context,
                previous_error_report=previous_error_report,
            )

            return result.entity

        except Exception as e:
            print(f"  ✗ Failed to extract {target_label}: {e}")
            # Re-raise for retry logic to handle
            raise


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================


def sanitize_label(raw_label: str) -> str:
    """
    Sanitize a raw label into a valid SourceLocation label format.

    Converts any string into a valid label matching pattern: ^[a-z][a-z0-9_-]*$
    - Converts to lowercase
    - Uses hyphens to separate tag sections (type-name)
    - Preserves underscores ONLY within names (e.g., my_param)
    - Replaces other special characters with hyphens
    - Ensures it starts with a letter

    Tag structure rules:
    - Hyphens separate tag sections: param-my_param ✓
    - Underscores only within names: param_my_param ✗ (converted to param-my-param)
    - Common prefixes: param, def, thm, lem, cor, ax, section, prop, rem, cite

    Args:
        raw_label: Raw label string (may contain uppercase, spaces, periods, etc.)

    Returns:
        Sanitized label with only lowercase letters, digits, underscores, and hyphens

    Examples:
        >>> sanitize_label("## 1. Introduction")
        'section-1-introduction'
        >>> sanitize_label("param-Theta")
        'param-theta'
        >>> sanitize_label("param_my_param")
        'param-my-param'
        >>> sanitize_label("param-my_param")
        'param-my_param'
        >>> sanitize_label("def_Energy")
        'def-energy'
    """
    import re

    # Convert to lowercase (CRITICAL: prevents uppercase validation errors)
    label = raw_label.lower()

    # Remove leading markdown headers (##, ###, etc.)
    label = re.sub(r"^#+\s*", "", label)

    # Replace any sequence of non-alphanumeric characters (except underscores/hyphens) with a single hyphen
    # This preserves underscores and hyphens while converting other special chars
    label = re.sub(r"[^a-z0-9_-]+", "-", label)

    # Remove leading/trailing hyphens and underscores
    label = label.strip("-_")

    # Known tag prefixes that should be separated from names with hyphens
    # Format: {prefix}-{name}, where name can contain underscores
    prefixes = [
        "param",
        "def",
        "thm",
        "lem",
        "cor",
        "ax",
        "axiom",
        "section",
        "prop",
        "rem",
        "remark",
        "cite",
        "eq",
        "obj",
        "const",
        "notation",
    ]

    # If label starts with a known prefix followed by underscore,
    # convert that first underscore to hyphen (it's a section separator, not part of the name)
    for prefix in prefixes:
        # Match: prefix + underscore + rest
        # Example: "param_my_param" → "param" + "_" + "my_param"
        pattern = f"^({prefix})_(.+)$"
        match = re.match(pattern, label)
        if match:
            # Convert prefix_name to prefix-name
            label = f"{match.group(1)}-{match.group(2)}"
            break

    # Ensure it starts with a letter (not a digit or underscore)
    if label and (label[0].isdigit() or label[0] == "_"):
        label = f"section-{label}"
    elif not label or not label[0].isalpha():
        label = f"section-{label}" if label else "section-unknown"

    # Collapse multiple consecutive hyphens (but preserve underscores)
    return re.sub(r"-+", "-", label)


def create_source_location(
    label: str, line_start: int, line_end: int, file_path: str, article_id: str
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
        article_id=article_id,
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

    # Helper to create source location with automatic label sanitization
    def make_source(label: str, line_start: int, line_end: int) -> SourceLocation:
        # Sanitize label to ensure it's lowercase and valid
        sanitized_label = sanitize_label(label)
        return create_source_location(sanitized_label, line_start, line_end, file_path, article_id)

    # Convert definitions
    raw_definitions = []
    for d in extraction.definitions:
        try:
            # Sanitize label to ensure lowercase and valid format
            sanitized_label = sanitize_label(d.label)
            raw_def = RawDefinition(
                label=sanitized_label,
                term=d.term,
                full_text=TextLocation.from_single_range(d.line_start, d.line_end),
                parameters_mentioned=d.parameters_mentioned,
                source=make_source(d.label, d.line_start, d.line_end),
            )
            raw_definitions.append(raw_def)
        except Exception as e:
            warning = f"Failed to convert definition {d.label}: {e}"
            conversion_warnings.append(make_error_dict(warning, value=d.model_dump()))
            print(f"  ⚠ {warning}")

    # Convert theorems
    raw_theorems = []
    for t in extraction.theorems:
        try:
            # Sanitize label to ensure lowercase and valid format
            sanitized_label = sanitize_label(t.label)
            raw_thm = RawTheorem(
                label=sanitized_label,
                statement_type=t.statement_type,
                conclusion_formula_latex=t.conclusion_formula_latex,
                explicit_definition_references=t.definition_references,
                full_text="",  # Text can be extracted later from source location
                source=make_source(t.label, t.line_start, t.line_end),
            )
            raw_theorems.append(raw_thm)
        except Exception as e:
            warning = f"Failed to convert theorem {t.label}: {e}"
            conversion_warnings.append(make_error_dict(warning, value=t.model_dump()))
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
                steps = list(starmap(TextLocation.from_single_range, p.steps))

            # Handle full body
            full_body_text = None
            if p.full_body_line_start is not None and p.full_body_line_end is not None:
                full_body_text = TextLocation.from_single_range(
                    p.full_body_line_start, p.full_body_line_end
                )

            # Sanitize labels to ensure lowercase and valid format
            sanitized_label = sanitize_label(p.label)
            sanitized_proves_label = sanitize_label(p.proves_label)

            raw_proof = RawProof(
                label=sanitized_label,
                proves_label=sanitized_proves_label,
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
            conversion_warnings.append(make_error_dict(warning, value=p.model_dump()))
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

            # Sanitize label to ensure lowercase and valid format
            sanitized_label = sanitize_label(a.label)

            raw_axiom = RawAxiom(
                label=sanitized_label,
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
            conversion_warnings.append(make_error_dict(warning, value=a.model_dump()))
            print(f"  ⚠ {warning}")

    # Convert parameters
    raw_parameters = []
    for param in extraction.parameters:
        try:
            # Sanitize label to ensure lowercase and valid format
            sanitized_label = sanitize_label(param.label)
            raw_param = RawParameter(
                label=sanitized_label,
                symbol=param.symbol,
                meaning=param.meaning,
                scope=param.scope,
                full_text="",  # Text can be extracted later from source location
                source=make_source(param.label, param.line_start, param.line_end),
            )
            raw_parameters.append(raw_param)
        except Exception as e:
            warning = f"Failed to convert parameter {param.label}: {e}"
            conversion_warnings.append(make_error_dict(warning, value=param.model_dump()))
            print(f"  ⚠ {warning}")

    # Convert remarks
    raw_remarks = []
    for r in extraction.remarks:
        try:
            # Sanitize label to ensure lowercase and valid format
            sanitized_label = sanitize_label(r.label)
            raw_remark = RawRemark(
                label=sanitized_label,
                remark_type=r.remark_type,
                full_text="",  # Text can be extracted later from source location
                source=make_source(r.label, r.line_start, r.line_end),
            )
            raw_remarks.append(raw_remark)
        except Exception as e:
            warning = f"Failed to convert remark {r.label}: {e}"
            conversion_warnings.append(make_error_dict(warning, value=r.model_dump()))
            print(f"  ⚠ {warning}")

    # Convert citations
    raw_citations = []
    for c in extraction.citations:
        try:
            # Sanitize key_in_text (used as label) to ensure lowercase and valid format
            sanitized_label = sanitize_label(c.key_in_text)
            raw_citation = RawCitation(
                key_in_text=c.key_in_text,  # Keep original format for citation key
                full_entry_text=TextLocation.from_single_range(c.line_start, c.line_end),
                full_text="",  # Text can be extracted later from source location
                source=make_source(c.key_in_text, c.line_start, c.line_end),
            )
            raw_citations.append(raw_citation)
        except Exception as e:
            warning = f"Failed to convert citation {c.key_in_text}: {e}"
            conversion_warnings.append(make_error_dict(warning, value=c.model_dump()))
            print(f"  ⚠ {warning}")

    # Convert assumptions
    raw_assumptions = []
    for assumption in extraction.assumptions:
        try:
            # Sanitize label to ensure lowercase and valid format
            sanitized_label = sanitize_label(assumption.label)
            raw_assumption = RawAssumption(
                label=sanitized_label,
                full_text="",  # Text can be extracted later from source location
                source=make_source(assumption.label, assumption.line_start, assumption.line_end),
            )
            raw_assumptions.append(raw_assumption)
        except Exception as e:
            warning = f"Failed to convert assumption {assumption.label}: {e}"
            conversion_warnings.append(make_error_dict(warning, value=assumption.model_dump()))
            print(f"  ⚠ {warning}")

    # Create section source (use first line of chapter)
    import re

    first_line = 1
    last_line = 1

    # Parse line numbers from chapter text
    lines = chapter_text.split("\n")
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

    section_source = make_source(section_label, first_line, last_line)

    # Create RawDocumentSection (full_text can be extracted later from source location)
    raw_section = RawDocumentSection(
        section_id=extraction.section_id,
        definitions=raw_definitions,
        theorems=raw_theorems,
        proofs=raw_proofs,
        axioms=raw_axioms,
        assumptions=raw_assumptions,
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
    max_iters: int = 10,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True,
) -> tuple[RawDocumentSection | None, list[str]]:
    """
    Extract mathematical concepts from a chapter using ReAct agent with retry logic.

    This is the main entry point for the EXTRACTION workflow.
    After first failure, switches to fallback model for remaining retries.

    Args:
        chapter_text: Chapter text with line numbers (format: "NNN: content")
        chapter_number: Chapter index (0 for preamble, 1+ for sections)
        file_path: Path to source markdown file
        article_id: Article identifier (e.g., "01_fragile_gas_framework")
        max_iters: Maximum ReAct iterations per attempt (default: 10)
        max_retries: Maximum number of retry attempts on failure (default: 3)
        fallback_model: Model to use after first failure (default: claude-haiku-4-5)
        verbose: Print progress information

    Returns:
        Tuple of (RawDocumentSection or None, list of errors)
        - RawDocumentSection: Extracted and validated data (None if extraction failed)
        - list[str]: Any errors or warnings encountered during extraction
    """
    errors_encountered = []

    # Extract entities with retry logic
    try:
        extraction, retry_errors = extract_chapter_with_retry(
            chapter_text=chapter_text,
            chapter_number=chapter_number,
            file_path=file_path,
            article_id=article_id,
            max_iters=max_iters,
            max_retries=max_retries,
            fallback_model=fallback_model,
            verbose=verbose,
        )

        # Add retry errors to our error list
        errors_encountered.extend(retry_errors)

        if verbose:
            if retry_errors:
                print(f"  ✓ Extraction completed after {len(retry_errors) + 1} attempts")
            else:
                print("  ✓ Extraction completed")

    except Exception as e:
        error_msg = f"Extraction failed after {max_retries} attempts: {e!s}"
        errors_encountered.append(
            make_error_dict(
                error_msg,
                value={
                    "chapter_number": chapter_number,
                    "file_path": file_path,
                    "article_id": article_id,
                    "exception": str(e),
                },
            )
        )
        if verbose:
            print(f"  ✗ {error_msg}")

        # Create empty extraction as fallback
        import re

        section_id = f"Chapter {chapter_number}"
        # Try to extract section ID from chapter text
        for line in chapter_text.split("\n")[:20]:
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
            assumptions=[],
            parameters=[],
            remarks=[],
            citations=[],
        )

    # Convert to RawDocumentSection
    try:
        raw_section, conversion_warnings = convert_to_raw_document_section(
            extraction, file_path=file_path, article_id=article_id, chapter_text=chapter_text
        )

        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        if verbose and raw_section:
            print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")

            # Display extraction report showing which labels were parsed
            try:
                from mathster.parsing.tools import compare_extraction_with_source

                _, report = compare_extraction_with_source(raw_section, chapter_text)
                print("\n" + "=" * 70)
                print("EXTRACTION REPORT")
                print("=" * 70)
                print(report)
                print("=" * 70 + "\n")
            except Exception as e:
                print(f"  ⚠ Could not generate extraction report: {e}")

        return raw_section, errors_encountered

    except Exception as e:
        error_msg = f"Conversion failed: {e!s}"
        errors_encountered.append(make_error_dict(error_msg, value=extraction.model_dump()))
        if verbose:
            print(f"  ✗ {error_msg}")

        return None, errors_encountered


def extract_chapter_by_labels(
    chapter_text: str,
    chapter_number: int,
    file_path: str,
    article_id: str,
    max_iters_per_label: int = 3,
    max_retries: int = 3,
    fallback_model: str = "anthropic/claude-haiku-4-5",
    verbose: bool = True,
) -> tuple[RawDocumentSection | None, list[str]]:
    """
    Extract chapter by iterating over individual labels (single-label mode) with retry logic.

    This function implements the NESTED LOOP structure requested:
    - Outer loop: sections (currently just one section = full chapter)
    - Inner loop: labels within each section
    After first failure per label, switches to fallback model for remaining retries.

    Strategy:
    1. Discover all labels using analyze_labels_in_chapter()
    2. For each label: extract using SingleLabelExtractor with retry
    3. Accumulate entities into ChapterExtraction
    4. Convert to RawDocumentSection

    Args:
        chapter_text: Chapter text with line numbers (format: "NNN: content")
        chapter_number: Chapter index (0 for preamble, 1+ for sections)
        file_path: Path to source markdown file
        article_id: Article identifier (e.g., "01_fragile_gas_framework")
        max_iters_per_label: Maximum ReAct iterations per label per attempt (default: 3)
        max_retries: Maximum number of retry attempts per label (default: 3)
        fallback_model: Model to use after first failure per label (default: claude-haiku-4-5)
        verbose: Print progress information

    Returns:
        Tuple of (RawDocumentSection or None, list of errors)
    """
    errors_encountered = []

    # STEP 1: Discover all labels in chapter
    from mathster.parsing.tools import analyze_labels_in_chapter

    labels_by_type, report = analyze_labels_in_chapter(chapter_text)

    total_labels = sum(len(labels) for labels in labels_by_type.values())

    if verbose:
        print(f"  → Found {total_labels} labels to extract")
        for entity_type, labels in labels_by_type.items():
            if labels:
                print(f"    • {entity_type}: {len(labels)}")

    if total_labels == 0:
        if verbose:
            print("  ⚠ No labels found in chapter")
        # Return empty extraction
        import re

        section_id = f"Chapter {chapter_number}"
        for line in chapter_text.split("\n")[:20]:
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
            assumptions=[],
            parameters=[],
            remarks=[],
            citations=[],
        )
        raw_section, _ = convert_to_raw_document_section(
            extraction, file_path, article_id, chapter_text
        )
        return raw_section, []

    # STEP 2: Initialize accumulator
    import re

    section_id = f"Chapter {chapter_number}"
    for line in chapter_text.split("\n")[:20]:
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
        assumptions=[],
        parameters=[],
        remarks=[],
        citations=[],
    )

    # STEP 3: **NESTED LOOP** - Iterate over each label with retry
    label_counter = 0
    successful_extractions = 0

    for entity_type, labels in labels_by_type.items():
        for label in labels:
            label_counter += 1
            if verbose:
                print(f"  [{label_counter}/{total_labels}] {label}...", end=" ")

            try:
                # Extract single label with retry
                entity_dict, retry_errors = extract_label_with_retry(
                    chapter_text=chapter_text,
                    target_label=label,
                    entity_type=entity_type,
                    file_path=file_path,
                    article_id=article_id,
                    max_iters_per_label=max_iters_per_label,
                    max_retries=max_retries,
                    fallback_model=fallback_model,
                    verbose=True,  # Suppress verbose retry output for cleaner display
                )

                # Add retry errors to our error list
                if retry_errors:
                    errors_encountered.extend(retry_errors)

                # Convert dict to appropriate Extraction class based on entity type
                entity = convert_dict_to_extraction_entity(entity_dict, entity_type)

                # Add to appropriate list in accumulator
                entity_list = getattr(extraction, entity_type)
                entity_list.append(entity)

                successful_extractions += 1

                if verbose:
                    if retry_errors:
                        print(f"✓ (after {len(retry_errors) + 1} attempts)")
                    else:
                        print("✓")

            except Exception as e:
                error_msg = f"Failed to extract {label} after {max_retries} attempts: {e}"
                errors_encountered.append(
                    make_error_dict(
                        error_msg,
                        value={
                            "target_label": label,
                            "entity_type": entity_type_key,
                            "exception": str(e),
                        },
                    )
                )
                if verbose:
                    print("✗")
                    print(f"      Error: {error_msg}")

    if verbose:
        print(f"  ✓ Extracted {successful_extractions}/{total_labels} labels")

    # STEP 5: Convert accumulated extraction to RawDocumentSection
    try:
        raw_section, conversion_warnings = convert_to_raw_document_section(
            extraction, file_path=file_path, article_id=article_id, chapter_text=chapter_text
        )

        if conversion_warnings:
            errors_encountered.extend(conversion_warnings)

        if verbose and raw_section:
            print(f"  ✓ Conversion completed: {raw_section.total_entities} entities")

            # Display extraction report
            try:
                from mathster.parsing.tools import compare_extraction_with_source

                _, report = compare_extraction_with_source(raw_section, chapter_text)
                print("\n" + "=" * 70)
                print("EXTRACTION REPORT")
                print("=" * 70)
                print(report)
                print("=" * 70 + "\n")
            except Exception as e:
                print(f"  ⚠ Could not generate extraction report: {e}")

        return raw_section, errors_encountered

    except Exception as e:
        error_msg = f"Conversion failed: {e!s}"
        errors_encountered.append(make_error_dict(error_msg, value=extraction.model_dump()))
        if verbose:
            print(f"  ✗ {error_msg}")

        return None, errors_encountered


def convert_dict_to_extraction_entity(entity_dict: dict, entity_type: str):
    """
    Convert entity dict to appropriate Extraction class.

    Args:
        entity_dict: Entity data as dict
        entity_type: Entity type (e.g., "definitions", "theorems")

    Returns:
        Appropriate Extraction object (DefinitionExtraction, TheoremExtraction, etc.)
    """
    # Map entity type to extraction class
    if entity_type == "definitions":
        return DefinitionExtraction(**entity_dict)
    if entity_type in {"theorems", "lemmas", "propositions", "corollaries"}:
        # All theorem-like entities use TheoremExtraction
        return TheoremExtraction(**entity_dict)
    if entity_type == "proofs":
        return ProofExtraction(**entity_dict)
    if entity_type == "axioms":
        return AxiomExtraction(**entity_dict)
    if entity_type == "assumptions":
        return AssumptionExtraction(**entity_dict)
    if entity_type == "parameters":
        return ParameterExtraction(**entity_dict)
    if entity_type == "remarks":
        return RemarkExtraction(**entity_dict)
    if entity_type == "citations":
        return CitationExtraction(**entity_dict)
    raise ValueError(f"Unknown entity type: {entity_type}")
