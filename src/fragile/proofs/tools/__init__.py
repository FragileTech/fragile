"""
Utility Tools for Mathematical Paper Extraction.

This module provides utility functions for processing markdown documents
during the Extract-then-Enrich pipeline:

- Line Finding: Locate exact line ranges for mathematical entities
- Source Helpers: Create SourceLocation objects from various inputs

These tools are designed to be called by Claude Code or subagents during
the enrichment phase to ground extracted entities to their source locations.

Usage:
    from fragile.proofs.tools import find_text_in_markdown, find_directive_lines

    # Find line range for text
    line_range = find_text_in_markdown(
        markdown_content,
        "Theorem 2.1"
    )

    # Find Jupyter Book directive
    line_range = find_directive_lines(
        markdown_content,
        "thm-keystone",
        directive_type="theorem"
    )

Maps to Lean:
    namespace Tools
      namespace LineFinder
        def find_text_in_markdown : String � String � Nat � Option (Nat � Nat)
        def find_directive_lines : String � String � Option (Nat � Nat)
      end LineFinder
      namespace SourceHelpers
        def from_jupyter_directive : ... � SourceLocation
        def from_markdown_location : ... � SourceLocation
      end SourceHelpers
    end Tools
"""

# Line finding utilities
# Directive parsing utilities
from fragile.proofs.tools.directive_parser import (
    DirectiveHint,
    DocumentSection,
    extract_jupyter_directives,
    format_directive_hints_for_llm,
    generate_section_id,
    get_directive_summary,
    split_into_sections,
)
from fragile.proofs.tools.line_finder import (
    extract_lines,
    find_all_occurrences,
    find_directive_lines,
    find_equation_lines,
    find_section_lines,
    find_text_in_markdown,
    get_file_line_count,
    validate_line_range,
)


__all__ = [
    # Directive parsing (hybrid approach)
    "DirectiveHint",
    "DocumentSection",
    "extract_jupyter_directives",
    "extract_lines",
    "find_all_occurrences",
    "find_directive_lines",
    "find_equation_lines",
    "find_section_lines",
    # Line finding functions
    "find_text_in_markdown",
    "format_directive_hints_for_llm",
    "generate_section_id",
    "get_directive_summary",
    "get_file_line_count",
    "split_into_sections",
    # Utility functions
    "validate_line_range",
]
