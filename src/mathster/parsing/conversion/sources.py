"""
SourceLocation creation utilities.

Provides functions for creating SourceLocation objects from line range information.
"""

from mathster.core.article_system import SourceLocation, TextLocation


def create_source_location(
    label: str, line_start: int, line_end: int, file_path: str, article_id: str
) -> SourceLocation:
    """
    Create a SourceLocation object from line range information.

    Args:
        label: Entity label
        line_start: Starting line number (1-indexed, inclusive)
        line_end: Ending line number (1-indexed, inclusive)
        file_path: Path to source markdown file
        article_id: Article identifier

    Returns:
        SourceLocation object with populated metadata
    """
    return SourceLocation(
        file_path=file_path,
        line_range=TextLocation.from_single_range(line_start, line_end),
        label=label,
        article_id=article_id,
    )
