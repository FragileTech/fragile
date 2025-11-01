"""
Registry and Storage System.

This module contains the registry for managing collections of mathematical objects,
theorems, relationships, and reviews, plus storage/persistence utilities:
- MathematicalRegistry for managing collections
- ReviewRegistry for managing review history
- Reference system for querying (TagQuery, etc.)
- Storage for saving/loading registries
"""

from mathster.registry.article_registry import ArticleRegistry, get_article_registry
from mathster.registry.reference_system import (
    CombinedTagQuery,
    create_reference_id,
    create_reference_resolved,
    extract_id_from_label,
    extract_tags_from_object,
    QueryResult,
    Reference,
    ResolutionContext,
    ResolvedReference,
    TagQuery,
    UnresolvedReference,
)
from mathster.registry.registry import MathematicalRegistry
from mathster.registry.review_registry import get_review_registry, ReviewRegistry
from mathster.registry.storage import (
    load_registry_from_directory,
    save_registry_to_directory,
)


__all__ = [
    # Article registry
    "ArticleRegistry",
    "CombinedTagQuery",
    # Registry
    "MathematicalRegistry",
    "QueryResult",
    "Reference",
    "ResolutionContext",
    "ResolvedReference",
    # Review registry
    "ReviewRegistry",
    # Reference system types
    "TagQuery",
    "UnresolvedReference",
    # Reference system functions
    "create_reference_id",
    "create_reference_resolved",
    "extract_id_from_label",
    "extract_tags_from_object",
    "get_article_registry",
    "get_review_registry",
    "load_registry_from_directory",
    # Storage
    "save_registry_to_directory",
]
