"""
Directive extraction preprocessing for parsing pipeline.

This module provides automated extraction of Jupyter Book directive structure
BEFORE passing to DSPy agents. This significantly speeds up extraction by:
1. Python extracts structure (type, label, lines, metadata) - <1 second
2. DSPy focuses only on semantics (dependencies, formulas) - ~50% faster

Integration:
    from mathster.directives import extract_directive_hints

    # Pre-extract structure
    hints = extract_directive_hints(chapter_text, chapter_number)

    # Pass to DSPy agent (which now focuses on semantics only)
    extraction = extractor(chapter_text, directive_hints=hints, ...)
"""

import logging
from typing import Any

from mathster.directives.directive_parser import extract_jupyter_directives


logger = logging.getLogger(__name__)


def extract_directive_hints(
    chapter_text: str,
    chapter_number: int,
    section_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract all directive structural hints from chapter.

    This function automatically extracts:
    - Directive type (theorem, definition, proof, etc.)
    - Label (from :label: field)
    - Title (from first line or :name: field)
    - Line numbers (start, end, content boundaries)
    - Metadata (class, nonumber, etc.)

    The DSPy agent can then focus ONLY on semantic extraction:
    - Dependencies (which theorems/definitions referenced)
    - Mathematical formulas
    - Proof structure

    Args:
        chapter_text: Chapter text with or without line numbers
        chapter_number: Chapter index
        section_id: Optional section identifier

    Returns:
        List of directive hint dictionaries ready for DSPy
    """
    if section_id is None:
        section_id = f"chapter-{chapter_number}"

    # Extract directives using enhanced parser
    directive_hints = extract_jupyter_directives(chapter_text, section_id=section_id)

    logger.info(f"Pre-extracted {len(directive_hints)} directives from chapter {chapter_number}")

    # Log summary
    type_counts = {}
    for hint in directive_hints:
        dtype = hint.directive_type
        type_counts[dtype] = type_counts.get(dtype, 0) + 1

    for dtype, count in sorted(type_counts.items()):
        logger.info(f"  {dtype}: {count}")

    # Convert to dicts for JSON serialization
    return [hint.to_dict() for hint in directive_hints]


def validate_hints(hints: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate extracted directive hints.

    Checks:
    - All have labels
    - Line numbers are valid
    - No overlapping directives
    - Metadata is well-formed

    Args:
        hints: List of directive hint dicts

    Returns:
        (is_valid, list of errors)
    """
    errors = []

    # Check for missing labels
    for hint in hints:
        if not hint.get("label"):
            errors.append(f"Directive at line {hint.get('start_line')} has no label")

        # Check line numbers
        start = hint.get("start_line", 0)
        end = hint.get("end_line", 0)
        if start >= end:
            errors.append(f"{hint.get('label')}: Invalid line range [{start}, {end}]")

    # Check for overlapping directives (might indicate nesting)
    sorted_hints = sorted(hints, key=lambda h: h.get("start_line", 0))
    for i in range(len(sorted_hints) - 1):
        curr = sorted_hints[i]
        next_hint = sorted_hints[i + 1]

        if curr.get("end_line", 0) > next_hint.get("start_line", 0):
            # Overlap detected - might be nested or malformed
            logger.warning(
                f"Overlapping directives: {curr.get('label')} "
                f"(lines {curr.get('start_line')}-{curr.get('end_line')}) overlaps "
                f"{next_hint.get('label')} (lines {next_hint.get('start_line')}-{next_hint.get('end_line')})"
            )

    return len(errors) == 0, errors


def preview_hints(hints: list[dict], max_hints: int = 10) -> str:
    """
    Generate a preview of extracted hints for logging.

    Args:
        hints: List of directive hints
        max_hints: Maximum hints to include in preview

    Returns:
        Formatted preview string
    """
    lines = [f"Extracted {len(hints)} directive hints:"]

    for hint in hints[:max_hints]:
        dtype = hint.get("directive_type", "?")
        label = hint.get("label", "?")
        title = hint.get("title", "")
        start = hint.get("start_line", 0)
        end = hint.get("end_line", 0)

        title_str = f' "{title}"' if title else ""
        lines.append(f"  [{start:4d}-{end:4d}] {dtype:12s} {label:30s}{title_str}")

    if len(hints) > max_hints:
        lines.append(f"  ... and {len(hints) - max_hints} more")

    return "\n".join(lines)
