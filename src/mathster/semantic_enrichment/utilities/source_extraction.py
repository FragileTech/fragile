"""
Source metadata extraction utilities.

Extracts chapter and document information from source locations and file paths.
"""

import re
from pathlib import Path


def extract_chapter_from_source(source_dict: dict) -> str | None:
    """
    Extract chapter name from source location.

    Args:
        source_dict: Source location as dict

    Returns:
        Chapter name (e.g., "1_euclidean_gas") or None

    Examples:
        >>> source = {"file_path": "docs/source/1_euclidean_gas/07_mean_field.md"}
        >>> extract_chapter_from_source(source)
        "1_euclidean_gas"
    """
    file_path = source_dict.get("file_path", "")

    # Pattern: docs/source/{chapter}/{document}.md
    match = re.search(r"docs/source/([^/]+)/", file_path)
    if match:
        return match.group(1)

    return None


def extract_document_from_source(source_dict: dict) -> str | None:
    """
    Extract document name from source location.

    Args:
        source_dict: Source location as dict

    Returns:
        Document name (e.g., "07_mean_field") or None

    Examples:
        >>> source = {"file_path": "docs/source/1_euclidean_gas/07_mean_field.md"}
        >>> extract_document_from_source(source)
        "07_mean_field"
    """
    file_path = source_dict.get("file_path", "")

    # Get filename without extension
    path = Path(file_path)
    if path.suffix == ".md":
        return path.stem

    return None


def extract_volume_from_source(source_dict: dict) -> str | None:
    """
    Extract volume name from source location.

    Args:
        source_dict: Source location as dict

    Returns:
        Volume name (same as chapter for now) or None
    """
    return extract_chapter_from_source(source_dict)
