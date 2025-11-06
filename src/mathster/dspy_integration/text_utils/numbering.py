"""
Line numbering utilities for markdown text.

Provides functions to add line numbers to text documents for precise
entity location tracking during extraction.
"""


def add_line_numbers(document: str, padding: bool = True, offset: int = 0) -> str:
    """
    Add line numbers to each line of a document.

    Args:
        document: The text document to add line numbers to
        padding: Whether to pad line numbers with spaces for alignment
        offset: Starting line number offset (default=0, so first line is 1)

    Returns:
        The document with line numbers prepended to each line

    Example:
        >>> text = "Line 1\\nLine 2\\nLine 3"
        >>> print(add_line_numbers(text))
          1: Line 1
          2: Line 2
          3: Line 3
    """
    lines = document.split("\n")
    max_line_num = len(lines) + offset
    max_digits = len(str(max_line_num))

    if padding:
        numbered_lines = [
            f"{str(i + 1 + offset).rjust(max_digits)}: {line}" for i, line in enumerate(lines)
        ]
    else:
        numbered_lines = [f"{i + 1 + offset}: {line}" for i, line in enumerate(lines)]

    return "\n".join(numbered_lines)
