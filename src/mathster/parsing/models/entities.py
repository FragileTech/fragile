"""
Mathematical entity extraction models.

Defines Pydantic models for all mathematical entities that can be extracted
from markdown documents: definitions, theorems, proofs, axioms, parameters,
remarks, assumptions, and citations.
"""

from typing import Literal

from pydantic import BaseModel, Field


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
