"""
Parameter validation logic.

Provides validation function for parameter extraction results.
"""

from mathster.parsing.models.entities import ParameterExtraction
from mathster.parsing.models.results import ValidationResult


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
        if param.scope not in {"global", "local"}:
            errors.append(f"Parameter '{param.label}': scope must be 'global' or 'local'")

        # Try to convert to RawParameter
        try:
            from mathster.parameter_extraction.conversion.converters import convert_parameter

            _raw_param, conversion_warnings = convert_parameter(
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
