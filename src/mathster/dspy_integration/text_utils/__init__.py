"""
Generic text processing utilities for DSPy-based extraction systems.

Provides utilities for markdown processing, line numbering, and label sanitization
that are reusable across different DSPy extraction projects.
"""

from mathster.dspy_integration.text_utils.labels import lookup_label_from_context, sanitize_label
from mathster.dspy_integration.text_utils.numbering import add_line_numbers
from mathster.dspy_integration.text_utils.splitting import (
    split_markdown_by_chapters,
    split_markdown_by_chapters_with_line_numbers,
)


__all__ = [
    "add_line_numbers",
    "lookup_label_from_context",
    "sanitize_label",
    "split_markdown_by_chapters",
    "split_markdown_by_chapters_with_line_numbers",
]
