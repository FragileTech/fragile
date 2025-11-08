"""
Mathematical Parameter Extraction Module.

This module (Stage 2 of the pipeline) extracts structured Parameter objects from
parameters_mentioned in parsed mathematical documents.

Unlike other mathematical entities (definitions, theorems), parameters don't have
their own Jupyter Book directives. This module finds parameter definitions by:
1. Searching for declaration patterns ("Let X be", "X := ", etc.)
2. Finding first mentions in formulas
3. Using DSPy agents for edge cases

Architecture:
- validation/: Parameter validation logic
- conversion/: ParameterExtraction → RawParameter conversion
- dspy_components/: DSPy agents for parameter extraction
- text_processing/: Parameter declaration finding
- workflows/: Extraction and refinement workflows

Usage:
    from mathster.parameter_extraction import extract_parameters, refine_parameters

    # Extract parameters from chapter
    raw_parameters, errors = extract_parameters(
        chapter_text=numbered_text,
        existing_extraction=chapter_data,
        ...
    )

    # Refine parameters at line 1 using DSPy
    updated, failed, errors = refine_parameters(
        chapter_file=path_to_json,
        full_document_text=full_doc,
        ...
    )
"""

from mathster.parameter_extraction import (
    conversion,
    dspy_components,
    text_processing,
    validation,
    workflows,
)
from mathster.parameter_extraction.workflows.extract import (
    extract_parameters_from_chapter,
    improve_parameters_from_chapter,
)
from mathster.parameter_extraction.workflows.refine import refine_parameter_line_numbers


# Convenience aliases
extract_parameters = extract_parameters_from_chapter
refine_parameters = refine_parameter_line_numbers

__all__ = [
    "conversion",
    "dspy_components",
    "extract_parameters",
    "extract_parameters_from_chapter",
    "improve_parameters_from_chapter",
    "refine_parameter_line_numbers",
    "refine_parameters",
    "text_processing",
    "validation",
    "workflows",
]
