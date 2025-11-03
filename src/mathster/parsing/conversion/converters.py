"""
Conversion functions for transforming extraction results into raw data structures.

Provides converters to transform DSPy extraction outputs (ChapterExtraction) into
RawDocumentSection structures with proper SourceLocation metadata.
"""

import re

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
from mathster.parsing.conversion.labels import sanitize_label
from mathster.parsing.conversion.sources import create_source_location
from mathster.parsing.models.entities import (
    AssumptionExtraction,
    AxiomExtraction,
    ChapterExtraction,
    CitationExtraction,
    DefinitionExtraction,
    ParameterExtraction,
    ProofExtraction,
    RemarkExtraction,
    TheoremExtraction,
)
from mathster.parsing.validation.errors import make_error_dict


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
            conversion_warnings.append(make_error_dict(
                warning,
                value=d.model_dump()
            ))
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
            conversion_warnings.append(make_error_dict(
                warning,
                value=t.model_dump()
            ))
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
            conversion_warnings.append(make_error_dict(
                warning,
                value=p.model_dump()
            ))
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
            conversion_warnings.append(make_error_dict(
                warning,
                value=a.model_dump()
            ))
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
            conversion_warnings.append(make_error_dict(
                warning,
                value=param.model_dump()
            ))
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
            conversion_warnings.append(make_error_dict(
                warning,
                value=r.model_dump()
            ))
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
            conversion_warnings.append(make_error_dict(
                warning,
                value=c.model_dump()
            ))
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
            conversion_warnings.append(make_error_dict(
                warning,
                value=assumption.model_dump()
            ))
            print(f"  ⚠ {warning}")

    # Create section source (use first line of chapter)
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
        assumptions=raw_assumptions,
        parameters=raw_parameters,
        remarks=raw_remarks,
        citations=raw_citations,
        source=section_source,
        full_text="",  # Text can be extracted later from source location
    )

    return raw_section, conversion_warnings


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
    elif entity_type in ("theorems", "lemmas", "propositions", "corollaries"):
        # All theorem-like entities use TheoremExtraction
        return TheoremExtraction(**entity_dict)
    elif entity_type == "proofs":
        return ProofExtraction(**entity_dict)
    elif entity_type == "axioms":
        return AxiomExtraction(**entity_dict)
    elif entity_type == "assumptions":
        return AssumptionExtraction(**entity_dict)
    elif entity_type == "parameters":
        return ParameterExtraction(**entity_dict)
    elif entity_type == "remarks":
        return RemarkExtraction(**entity_dict)
    elif entity_type == "citations":
        return CitationExtraction(**entity_dict)
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")
