"""
Semantic enrichment utilities.

Provides automated utilities for enriching raw entities with semantic metadata.
"""

from mathster.semantic_enrichment.utilities.constraint_extraction import (
    extract_constraints_from_text,
    infer_domain_from_constraints,
)
from mathster.semantic_enrichment.utilities.cross_reference import build_parameter_usage_index
from mathster.semantic_enrichment.utilities.enum_mapping import (
    map_parameter_scope,
    map_remark_type,
)
from mathster.semantic_enrichment.utilities.source_extraction import (
    extract_chapter_from_source,
    extract_document_from_source,
    extract_volume_from_source,
)

__all__ = [
    "build_parameter_usage_index",
    "extract_chapter_from_source",
    "extract_constraints_from_text",
    "extract_document_from_source",
    "extract_volume_from_source",
    "infer_domain_from_constraints",
    "map_parameter_scope",
    "map_remark_type",
]
