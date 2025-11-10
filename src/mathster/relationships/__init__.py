"""
Relationship System.

This module contains types and utilities for relationships between mathematical objects:
- Relationship types and properties (from core.pipeline_types)
- Relationship graph analysis
- Equivalence classes, lineage, framework flow
"""

# Import graph analysis
from mathster.relationships.graphs import (
    build_relationship_graph_from_registry,
    EquivalenceClassifier,
    FrameworkFlow,
    ObjectLineage,
    RelationshipGraph,
)


__all__ = [
    "EquivalenceClassifier",
    "FrameworkFlow",
    "ObjectLineage",
    # Types (from core)
    # Graphs
    "RelationshipGraph",
    "build_relationship_graph_from_registry",
]
