"""
High-level workflows for mathematical entity extraction and improvement.

This module provides the main workflow functions that orchestrate the
extraction and improvement processes using DSPy components.

Submodules:
    - extract: Fresh extraction workflows
    - improve: Improvement workflows for existing extractions
    - retry: Retry logic with fallback model support

For now, these re-export from the original files to maintain backward compatibility.
The full extraction will be completed in a future refactoring phase.
"""

# Re-export from legacy files (internal implementation detail)
# Users should import from mathster.parsing.workflows or mathster.parsing
from mathster.parsing.extract_workflow import (
    extract_chapter,
    extract_chapter_by_labels,
    extract_chapter_with_retry,
    extract_label_with_retry,
)
from mathster.parsing.improve_workflow import (
    improve_chapter,
    improve_chapter_by_labels,
    improve_chapter_with_retry,
    improve_label_with_retry,
    compute_changes,
)

__all__ = [
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
    "compute_changes",
]
