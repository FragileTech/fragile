"""
Data transformation and conversion utilities.

This module provides functions for converting between different representations
of mathematical entities, sanitizing labels, and creating source locations.

Submodules:
    - converters: Main conversion logic (ChapterExtraction → RawDocumentSection)
    - labels: Label sanitization and lookup utilities
    - sources: SourceLocation creation helpers
"""

from mathster.parsing.conversion.converters import (
    convert_dict_to_extraction_entity,
    convert_to_raw_document_section,
)
from mathster.parsing.conversion.labels import lookup_label_from_context, sanitize_label
from mathster.parsing.conversion.sources import create_source_location

__all__ = [
    # Label utilities
    "sanitize_label",
    "lookup_label_from_context",
    # Source utilities
    "create_source_location",
    # Converters
    "convert_to_raw_document_section",
    "convert_dict_to_extraction_entity",
]
