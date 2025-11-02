"""
Mathster parsing module.

This module provides tools for parsing and extracting mathematical content
from markdown documents using DSPy-based ReAct agents with self-validation.

Architecture:
    dspy_pipeline: Main orchestration (EXTRACT + IMPROVE workflows)
      ├─> extract_workflow: Fresh extraction of mathematical entities
      └─> improve_workflow: Enhancement of existing extractions

Submodules:
    - tools: Markdown splitting and preprocessing utilities
    - dspy_pipeline: Main orchestration with CLI interface
    - extract_workflow: Fresh extraction workflow using ReAct agents
    - improve_workflow: Improvement workflow for existing extractions

For DSPy + MCP integration, see mathster.mcps module.

Usage:
    # CLI usage
    python -m mathster.parsing.dspy_pipeline <markdown_file> [options]

    # Programmatic usage
    from mathster.parsing import extract_chapter, improve_chapter

    # Fresh extraction
    raw_section, errors = extract_chapter(
        chapter_text=chapter_with_lines,
        chapter_number=0,
        file_path="docs/source/...",
        article_id="01_fragile_gas_framework"
    )

    # Improvement
    raw_section, changes, errors = improve_chapter(
        chapter_text=chapter_with_lines,
        existing_extraction=loaded_json,
        file_path="docs/source/...",
        article_id="01_fragile_gas_framework"
    )
"""

from mathster.parsing.extract_workflow import (
    ChapterExtraction,
    MathematicalConceptExtractor,
    extract_chapter,
    sanitize_label,
)
from mathster.parsing.improve_workflow import (
    ImprovementResult,
    MathematicalConceptImprover,
    improve_chapter,
)
from mathster.parsing.tools import (
    split_markdown_by_chapters_with_line_numbers,
)

__all__ = [
    # Tools
    "split_markdown_by_chapters_with_line_numbers",
    "sanitize_label",
    # Extract workflow
    "extract_chapter",
    "ChapterExtraction",
    "MathematicalConceptExtractor",
    # Improve workflow
    "improve_chapter",
    "ImprovementResult",
    "MathematicalConceptImprover",
]
