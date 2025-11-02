from pathlib import Path

def add_line_numbers(document: str, padding: bool = True, offset: int = 0) -> str:
    """Add line numbers to each line of a document.

    Args:
        document: The text document to add line numbers to
        padding: Whether to pad line numbers with spaces for alignment
        offset: Starting line number offset (default=0, so first line is 1)

    Returns:
        The document with line numbers prepended to each line
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


def split_markdown_by_chapters(file_path: str | Path, header: str = "##") -> list[str]:
    """
    Split a markdown file by chapters based on header level.

    Args:
        file_path: Path to the markdown file to split
        header: The header marker to use for splitting (e.g., "##" for level 2 headers)

    Returns:
        A list of strings where:
        - The first item (chapter 0) contains everything before the first header
        - Subsequent items contain each chapter starting with its header

    Example:
        >>> chapters = split_markdown_by_chapters("03_cloning.md")
        >>> # chapters[0] contains content before first "##"
        >>> # chapters[1] contains "## 0. TLDR" and its content
        >>> # chapters[2] contains "## 1. Introduction" and its content
    """
    file_path = Path(file_path)

    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    chapters = []
    current_chapter = []

    for line in lines:
        # Check if the line starts with the specified header
        if line.startswith(header + " "):
            # Save the current chapter if it has content
            if current_chapter or not chapters:
                chapters.append('\n'.join(current_chapter))
                current_chapter = []
            # Start a new chapter with this header line
            current_chapter.append(line)
        else:
            current_chapter.append(line)

    # Add the last chapter
    if current_chapter:
        chapters.append('\n'.join(current_chapter))

    return chapters


def split_markdown_by_chapters_with_line_numbers(
    file_path: str | Path,
    header: str = "##",
    padding: bool = True
) -> list[str]:
    """
    Split a markdown file by chapters and add continuous line numbers across all chapters.

    Args:
        file_path: Path to the markdown file to split
        header: The header marker to use for splitting (e.g., "##" for level 2 headers)
        padding: Whether to pad line numbers with spaces for alignment

    Returns:
        A list of strings where each chapter has line numbers that continue from the previous chapter.
        The first item (chapter 0) contains everything before the first header with line numbers.
        Subsequent items contain each chapter starting with its header, with continuous line numbering.

    Example:
        >>> chapters = split_markdown_by_chapters_with_line_numbers("03_cloning.md")
        >>> # chapters[0] contains content before first "##" with lines 1, 2, 3, ...
        >>> # chapters[1] contains "## 0. TLDR" continuing with lines N, N+1, N+2, ...
        >>> # chapters[2] contains "## 1. Introduction" continuing with lines M, M+1, M+2, ...
    """
    # First, split the markdown into chapters
    chapters = split_markdown_by_chapters(file_path, header)

    # Now add line numbers with continuous counting
    numbered_chapters = []
    current_offset = 0

    for chapter in chapters:
        # Add line numbers starting from the current offset
        numbered_chapter = add_line_numbers(chapter, padding=padding, offset=current_offset)
        numbered_chapters.append(numbered_chapter)

        # Update offset for next chapter (count lines in current chapter)
        current_offset += len(chapter.split('\n'))

    return numbered_chapters



