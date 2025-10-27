"""
Line Finding Utility for Markdown Source Grounding.

This module provides simple Python utilities for finding exact line ranges of text
in markdown documents. Designed to be called by Claude Code or subagents during
Stage 2 enrichment to ground extracted mathematical entities to their source.

Key Functions:
- find_text_in_markdown: Find exact line range for a text snippet
- find_directive_lines: Find line range for Jupyter Book directive by label
- find_equation_lines: Find line range for a LaTeX equation
- find_section_lines: Find line range for a section heading

All functions return 1-indexed line ranges (start_line, end_line) as tuples,
matching the convention used in SourceLocation.line_range.

Maps to Lean:
    namespace LineFinder
      def find_text_in_markdown : String → String → Nat → Option (Nat × Nat)
      def find_directive_lines : String → String → Option (Nat × Nat)
      def find_equation_lines : String → String → Option (Nat × Nat)
      def find_section_lines : String → String → Option (Nat × Nat)
    end LineFinder
"""

import re
from typing import List, Optional, Tuple


def find_text_in_markdown(
    markdown_content: str,
    search_string: str,
    context_lines: int = 0,
    case_sensitive: bool = False
) -> Optional[Tuple[int, int]]:
    """
    Find the exact line range for a text snippet in markdown.

    This is the most general-purpose function. Use it when you have a snippet
    of text (e.g., from RawTheorem.full_statement_text) and need to find where
    it appears in the source markdown.

    Args:
        markdown_content: The full markdown file content as a string
        search_string: The text to search for (can be multi-line)
        context_lines: Number of lines to include before/after the match (default: 0)
        case_sensitive: Whether to perform case-sensitive search (default: False)

    Returns:
        Tuple of (start_line, end_line) if found (1-indexed, inclusive), None otherwise

    Examples:
        >>> content = "Line 1\\nTheorem 1.1\\nLet v > 0\\nLine 4"
        >>> find_text_in_markdown(content, "Theorem 1.1")
        (2, 2)
        >>> find_text_in_markdown(content, "Theorem 1.1\\nLet v > 0")
        (2, 3)
        >>> find_text_in_markdown(content, "not present")
        None

    Maps to Lean:
        def find_text_in_markdown
          (content : String)
          (search : String)
          (context_lines : Nat)
          : Option (Nat × Nat)
    """
    lines = markdown_content.splitlines()

    # Prepare search string for matching
    search_lines = search_string.splitlines()
    if not search_lines:
        return None

    # Handle case sensitivity
    if not case_sensitive:
        search_lines = [line.lower() for line in search_lines]
        compare_lines = [line.lower() for line in lines]
    else:
        compare_lines = lines

    # Search for the first line of the search string
    for i, line in enumerate(compare_lines):
        if search_lines[0] in line:
            # Check if subsequent lines match
            match_found = True
            for j, search_line in enumerate(search_lines):
                if i + j >= len(compare_lines) or search_line not in compare_lines[i + j]:
                    match_found = False
                    break

            if match_found:
                # Calculate line range with context
                start_line = max(1, i + 1 - context_lines)
                end_line = min(len(lines), i + len(search_lines) + context_lines)
                return (start_line, end_line)

    return None


def find_directive_lines(
    markdown_content: str,
    directive_label: str,
    directive_type: Optional[str] = None
) -> Optional[Tuple[int, int]]:
    """
    Find the line range for a Jupyter Book directive by its label.

    Jupyter Book directives have the format:
    :::{{prf:theorem}} label-name
    content
    :::

    This function finds the directive block and returns its full line range.

    Args:
        markdown_content: The full markdown file content as a string
        directive_label: The label to search for (e.g., "thm-keystone")
        directive_type: Optional directive type filter (e.g., "theorem", "definition")
                       If None, searches for any directive with the label

    Returns:
        Tuple of (start_line, end_line) if found (1-indexed, inclusive), None otherwise

    Examples:
        >>> content = '''
        ... :::{{prf:theorem}} thm-main
        ... This is the theorem
        ... :::
        ... '''
        >>> find_directive_lines(content, "thm-main")
        (2, 4)

    Maps to Lean:
        def find_directive_lines
          (content : String)
          (label : String)
          (directive_type : Option String)
          : Option (Nat × Nat)
    """
    lines = markdown_content.splitlines()

    # Build regex pattern for directive opening
    if directive_type:
        # Specific directive type (e.g., prf:theorem)
        pattern = rf"^:::\s*\{{prf:{directive_type}\}}\s+{re.escape(directive_label)}"
    else:
        # Any directive type
        pattern = rf"^:::\s*\{{prf:\w+\}}\s+{re.escape(directive_label)}"

    # Find the opening directive line
    start_line = None
    for i, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            start_line = i + 1  # 1-indexed
            break

    if start_line is None:
        return None

    # Find the closing ::: line
    for i in range(start_line, len(lines)):  # start_line is already 1-indexed
        if lines[i].strip() == ":::":
            end_line = i + 1  # 1-indexed
            return (start_line, end_line)

    # If no closing found, return from start to end of file
    return (start_line, len(lines))


def find_equation_lines(
    markdown_content: str,
    equation_latex: str,
    equation_label: Optional[str] = None
) -> Optional[Tuple[int, int]]:
    """
    Find the line range for a LaTeX equation in markdown.

    Searches for display equations delimited by $$ or \\[ \\] and matches
    the LaTeX content. Optionally can match by equation label.

    Args:
        markdown_content: The full markdown file content as a string
        equation_latex: The LaTeX content to search for (without delimiters)
        equation_label: Optional equation label (e.g., "(2.3)", "eq:main")

    Returns:
        Tuple of (start_line, end_line) if found (1-indexed, inclusive), None otherwise

    Examples:
        >>> content = '''
        ... Text before
        ... $$
        ... f(x) = x^2
        ... $$
        ... Text after
        ... '''
        >>> find_equation_lines(content, "f(x) = x^2")
        (2, 4)

    Maps to Lean:
        def find_equation_lines
          (content : String)
          (latex : String)
          (label : Option String)
          : Option (Nat × Nat)
    """
    lines = markdown_content.splitlines()

    # Normalize equation latex for comparison (remove extra whitespace)
    normalized_search = " ".join(equation_latex.split())

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for display equation start
        if line == "$$" or line.startswith("\\["):
            start_line = i + 1  # 1-indexed
            equation_content_lines = []

            # Collect equation content
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line == "$$" or line.startswith("\\]"):
                    end_line = i + 1  # 1-indexed

                    # Check if content matches
                    equation_content = " ".join(equation_content_lines)
                    normalized_content = " ".join(equation_content.split())

                    if normalized_search in normalized_content:
                        return (start_line, end_line)
                    break

                equation_content_lines.append(line)
                i += 1

        i += 1

    return None


def find_section_lines(
    markdown_content: str,
    section_heading: str,
    exact_match: bool = False
) -> Optional[Tuple[int, int]]:
    """
    Find the line range for a section (from heading to next heading or end of file).

    Args:
        markdown_content: The full markdown file content as a string
        section_heading: The section heading to search for (e.g., "Introduction", "§2.1")
        exact_match: If True, require exact match; if False, allow partial match

    Returns:
        Tuple of (start_line, end_line) if found (1-indexed, inclusive), None otherwise
        The range includes the heading line and all content up to (but not including)
        the next heading of the same or higher level.

    Examples:
        >>> content = '''
        ... # Introduction
        ... Some text
        ... ## Subsection
        ... More text
        ... # Next Section
        ... '''
        >>> find_section_lines(content, "Introduction")
        (1, 4)

    Maps to Lean:
        def find_section_lines
          (content : String)
          (heading : String)
          (exact_match : Bool)
          : Option (Nat × Nat)
    """
    lines = markdown_content.splitlines()

    # Find the heading line
    start_line = None
    heading_level = None

    for i, line in enumerate(lines):
        # Check for markdown heading (# syntax)
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Check if this is our target heading
            if exact_match:
                if title == section_heading:
                    start_line = i + 1  # 1-indexed
                    heading_level = level
                    break
            else:
                if section_heading.lower() in title.lower():
                    start_line = i + 1  # 1-indexed
                    heading_level = level
                    break

    if start_line is None or heading_level is None:
        return None

    # Find the end of the section (next heading of same or higher level)
    for i in range(start_line, len(lines)):  # start_line is 1-indexed
        heading_match = re.match(r"^(#{1,6})\s+", lines[i])
        if heading_match:
            level = len(heading_match.group(1))
            if level <= heading_level:
                # Found next section at same or higher level
                end_line = i  # Don't include the next heading (i is 0-indexed)
                return (start_line, end_line)

    # If no next heading found, return to end of file
    return (start_line, len(lines))


def find_all_occurrences(
    markdown_content: str,
    search_string: str,
    case_sensitive: bool = False
) -> List[Tuple[int, int]]:
    """
    Find all occurrences of a text snippet in markdown.

    Useful when a term appears multiple times and you need to disambiguate.

    Args:
        markdown_content: The full markdown file content as a string
        search_string: The text to search for
        case_sensitive: Whether to perform case-sensitive search

    Returns:
        List of (start_line, end_line) tuples for all matches (1-indexed, inclusive)

    Examples:
        >>> content = "Line 1\\nTest\\nLine 3\\nTest again\\nLine 5"
        >>> find_all_occurrences(content, "Test")
        [(2, 2), (4, 4)]

    Maps to Lean:
        def find_all_occurrences
          (content : String)
          (search : String)
          (case_sensitive : Bool)
          : List (Nat × Nat)
    """
    lines = markdown_content.splitlines()
    results = []

    # Prepare search string
    if not case_sensitive:
        search_lower = search_string.lower()
        compare_lines = [line.lower() for line in lines]
    else:
        search_lower = search_string
        compare_lines = lines

    for i, line in enumerate(compare_lines):
        if search_lower in line:
            results.append((i + 1, i + 1))  # 1-indexed

    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def validate_line_range(line_range: Tuple[int, int], max_lines: int) -> bool:
    """
    Validate that a line range is well-formed.

    Args:
        line_range: Tuple of (start_line, end_line)
        max_lines: Maximum number of lines in the file

    Returns:
        True if line range is valid, False otherwise

    Checks:
    - Both values are positive
    - start_line <= end_line
    - end_line <= max_lines
    """
    start, end = line_range
    return (
        start >= 1 and
        end >= 1 and
        start <= end and
        end <= max_lines
    )


def extract_lines(
    markdown_content: str,
    line_range: Tuple[int, int]
) -> str:
    """
    Extract a specific line range from markdown content.

    Args:
        markdown_content: The full markdown file content
        line_range: Tuple of (start_line, end_line) (1-indexed, inclusive)

    Returns:
        The extracted text from the specified line range

    Example:
        >>> content = "Line 1\\nLine 2\\nLine 3\\nLine 4"
        >>> extract_lines(content, (2, 3))
        'Line 2\\nLine 3'
    """
    lines = markdown_content.splitlines()
    start, end = line_range

    # Convert to 0-indexed for slicing
    start_idx = start - 1
    end_idx = end  # inclusive end in our convention

    if start_idx < 0 or end_idx > len(lines):
        raise ValueError(f"Line range {line_range} out of bounds (file has {len(lines)} lines)")

    return "\n".join(lines[start_idx:end_idx])


def get_file_line_count(markdown_content: str) -> int:
    """Get the total number of lines in markdown content."""
    return len(markdown_content.splitlines())
