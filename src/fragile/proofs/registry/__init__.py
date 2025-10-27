"""
Registry and Storage System.

This module contains the registry for managing collections of mathematical objects,
theorems, relationships, and reviews, plus storage/persistence utilities:
- MathematicalRegistry for managing collections
- ReviewRegistry for managing review history
- Reference system for querying (TagQuery, etc.)
- Storage for saving/loading registries
"""

from fragile.proofs.registry.reference_system import (
    CombinedTagQuery,
    QueryResult,
    Reference,
    ResolutionContext,
    ResolvedReference,
    TagQuery,
    UnresolvedReference,
    create_reference_id,
    create_reference_resolved,
    extract_id_from_label,
    extract_tags_from_object,
)
from fragile.proofs.registry.registry import MathematicalRegistry
from fragile.proofs.registry.review_registry import ReviewRegistry, get_review_registry
from fragile.proofs.registry.article_registry import ArticleRegistry, get_article_registry
from fragile.proofs.registry.storage import (
    load_registry_from_directory,
    save_registry_to_directory,
)

__all__ = [
    # Reference system types
    "TagQuery",
    "CombinedTagQuery",
    "QueryResult",
    "Reference",
    "UnresolvedReference",
    "ResolvedReference",
    "ResolutionContext",
    # Reference system functions
    "create_reference_id",
    "create_reference_resolved",
    "extract_id_from_label",
    "extract_tags_from_object",
    # Registry
    "MathematicalRegistry",
    # Review registry
    "ReviewRegistry",
    "get_review_registry",
    # Article registry
    "ArticleRegistry",
    "get_article_registry",
    # Storage
    "save_registry_to_directory",
    "load_registry_from_directory",
]
