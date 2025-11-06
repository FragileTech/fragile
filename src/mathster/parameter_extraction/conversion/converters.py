"""
Conversion functions for parameter extraction results.

Provides converters to transform ParameterExtraction objects into RawParameter
structures with proper SourceLocation metadata.
"""

from mathster.core.raw_data import RawParameter
from mathster.dspy_integration.text_utils import sanitize_label
from mathster.parsing.conversion.sources import create_source_location
from mathster.parsing.models.entities import ParameterExtraction


def convert_parameter(
    param: ParameterExtraction,
    file_path: str,
    article_id: str,
    chapter_text: str,
) -> tuple[RawParameter, list[str]]:
    """
    Convert ParameterExtraction to RawParameter.

    Args:
        param: Parameter extraction result
        file_path: Path to source markdown file
        article_id: Article identifier
        chapter_text: Original chapter text (for line validation)

    Returns:
        Tuple of (RawParameter, list of conversion warnings)
    """
    warnings = []

    # Sanitize label
    sanitized_label = sanitize_label(param.label)

    # Create source location
    source = create_source_location(
        sanitized_label, param.line_start, param.line_end, file_path, article_id
    )

    # Validate symbol is not empty
    if not param.symbol or param.symbol.strip() == "":
        warnings.append(f"Parameter '{param.label}': symbol is empty")

    # Validate meaning is not empty
    if not param.meaning or param.meaning.strip() == "":
        warnings.append(f"Parameter '{param.label}': meaning is empty")

    # Create RawParameter
    try:
        raw_param = RawParameter(
            label=sanitized_label,
            symbol=param.symbol,
            meaning=param.meaning,
            scope=param.scope,
            full_text="",  # Text can be extracted later from source location
            source=source,
        )
        return raw_param, warnings

    except Exception as e:
        warnings.append(f"Failed to create RawParameter for '{param.label}': {e}")
        raise
