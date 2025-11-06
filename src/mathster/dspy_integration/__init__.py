"""
Generic DSPy Integration Utilities.

This module provides reusable patterns and utilities for DSPy-based extraction systems.
All code here is domain-agnostic and can be used in any DSPy project.

Key components:
- config: DSPy model configuration
- errors: Structured error handling
- text_utils: Markdown processing, line numbering, label sanitization
- validation_tools: Generic validation wrappers for ReAct agents (TODO)

Usage:
    from mathster.dspy_integration import configure_dspy, make_error_dict
    from mathster.dspy_integration.text_utils import split_markdown_by_chapters, sanitize_label
"""

from mathster.dspy_integration.config import configure_dspy
from mathster.dspy_integration.errors import generate_detailed_error_report, make_error_dict
from mathster.dspy_integration import text_utils

__all__ = [
    "configure_dspy",
    "generate_detailed_error_report",
    "make_error_dict",
    "text_utils",
]
