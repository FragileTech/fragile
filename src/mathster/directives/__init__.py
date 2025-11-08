"""
Directives module for parsing pipeline.

Provides automated extraction of directive structure before DSPy semantic extraction.
"""

from mathster.directives.directive_extractor import (
    extract_directive_hints,
    preview_hints,
    validate_hints,
)
from mathster.directives.directive_parser import (
    DirectiveHint,
    DocumentSection,
    extract_jupyter_directives,
    format_directive_hints_for_llm,
    generate_section_id,
    get_directive_summary,
    split_into_sections,
)

__all__ = [
    # Extractor functions
    "extract_directive_hints",
    "preview_hints",
    "validate_hints",
    # Parser functions and classes
    "DirectiveHint",
    "DocumentSection",
    "extract_jupyter_directives",
    "format_directive_hints_for_llm",
    "generate_section_id",
    "get_directive_summary",
    "split_into_sections",
]
