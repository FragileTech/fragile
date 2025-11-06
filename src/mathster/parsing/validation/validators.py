"""
Core validation logic for mathematical entity extraction.

Provides validation functions that check extraction results for correctness,
completeness, and adherence to labeling conventions.
"""

from mathster.parsing.models.entities import ChapterExtraction
from mathster.parsing.models.results import ValidationResult


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
    # Import here to avoid circular dependency
    from mathster.parsing.conversion.converters import convert_to_raw_document_section

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

            # Validate reference formats
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


def validate_parameter(
    parameter_dict: dict,
    file_path: str,
    article_id: str,
    chapter_text: str,
) -> ValidationResult:
    """
    Validate a single parameter extraction.

    This validates that a parameter has been properly extracted with all required
    fields and correct formatting. Used by DSPy agents for self-validation.

    Args:
        parameter_dict: Dictionary representing ParameterExtraction
        file_path: Path to source markdown file
        article_id: Article identifier
        chapter_text: Original chapter text

    Returns:
        ValidationResult with success status and error details
    """
    from mathster.parsing.models.entities import ParameterExtraction

    errors = []
    warnings = []
    entities_validated = {"parameters": 0}

    try:
        # Parse as ParameterExtraction
        param = ParameterExtraction(**parameter_dict)

        # Validate label pattern
        if not param.label.startswith("param-"):
            errors.append(f"Parameter label '{param.label}' must start with 'param-'")

        # Validate symbol is not empty
        if not param.symbol or param.symbol.strip() == "":
            errors.append(f"Parameter '{param.label}': symbol cannot be empty")

        # Validate meaning is descriptive (not just the symbol repeated)
        if not param.meaning or param.meaning.strip() == "":
            errors.append(f"Parameter '{param.label}': meaning cannot be empty")
        elif param.meaning.strip().lower() == param.symbol.strip().lower():
            warnings.append(
                f"Parameter '{param.label}': meaning should be descriptive, "
                f"not just repeat the symbol"
            )

        # Validate line range
        if param.line_start > param.line_end:
            errors.append(
                f"Parameter '{param.label}': line_start ({param.line_start}) "
                f"must be <= line_end ({param.line_end})"
            )

        # Validate line numbers are within chapter bounds
        lines = chapter_text.split("\n")
        max_line = len(lines)
        if param.line_start < 1 or param.line_end > max_line:
            errors.append(
                f"Parameter '{param.label}': line range [{param.line_start}, {param.line_end}] "
                f"is out of bounds (chapter has {max_line} lines)"
            )

        # Validate scope
        if param.scope not in ["global", "local"]:
            errors.append(f"Parameter '{param.label}': scope must be 'global' or 'local'")

        # Try to convert to RawParameter
        try:
            from mathster.parsing.conversion.converters import convert_parameter

            raw_param, conversion_warnings = convert_parameter(
                param, file_path=file_path, article_id=article_id, chapter_text=chapter_text
            )

            if conversion_warnings:
                warnings.extend(conversion_warnings)

            entities_validated["parameters"] = 1

        except Exception as e:
            errors.append(f"Failed to convert to RawParameter: {e!s}")

    except Exception as e:
        errors.append(f"Failed to parse ParameterExtraction: {e!s}")

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid, errors=errors, warnings=warnings, entities_validated=entities_validated
    )
