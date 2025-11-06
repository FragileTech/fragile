"""
High-level workflows for mathematical entity extraction and improvement.

This module provides the main workflow functions that orchestrate the
extraction and improvement processes using DSPy components.

Submodules:
    - extract: Fresh extraction workflows
    - improve: Improvement workflows for existing extractions
"""

# Import from modular workflow files
from mathster.parsing.workflows.extract import (
    extract_chapter,
    extract_chapter_by_labels,
    extract_chapter_with_retry,
    extract_label_with_retry,
)
from mathster.parsing.workflows.improve import (
    compute_changes,
    improve_chapter,
    improve_chapter_by_labels,
    improve_chapter_with_retry,
    improve_label_with_retry,
)


__all__ = [
    "compute_changes",
    # Extract workflows
    "extract_chapter",
    "extract_chapter_by_labels",
    "extract_chapter_with_retry",
    "extract_label_with_retry",
    # Improve workflows
    "improve_chapter",
    "improve_chapter_by_labels",
    "improve_chapter_with_retry",
    "improve_label_with_retry",
]
