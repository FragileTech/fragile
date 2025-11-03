"""
Text processing utilities for markdown documents.

This module provides utilities for processing markdown text: adding line numbers,
splitting by chapters, analyzing labels, and comparing extractions with source.

Submodules:
    - numbering: Line numbering utilities
    - splitting: Chapter splitting utilities
    - analysis: Label analysis and extraction comparison
"""

from mathster.parsing.text_processing.numbering import add_line_numbers
from mathster.parsing.text_processing.splitting import (
    split_markdown_by_chapters,
    split_markdown_by_chapters_with_line_numbers,
)
from mathster.parsing.text_processing.analysis import (
    classify_label,
    analyze_labels_in_chapter,
    compare_extraction_with_source,
)

__all__ = [
    # Numbering
    "add_line_numbers",
    # Splitting
    "split_markdown_by_chapters",
    "split_markdown_by_chapters_with_line_numbers",
    # Analysis
    "classify_label",
    "analyze_labels_in_chapter",
    "compare_extraction_with_source",
]
