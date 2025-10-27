"""
Relationship System.

This module contains types and utilities for relationships between mathematical objects:
- Relationship types and properties (from core.pipeline_types)
- Relationship graph analysis
- Equivalence classes, lineage, framework flow
"""

# Import Relationship types from core (they're in pipeline_types)
from fragile.proofs.core.pipeline_types import Relationship, RelationshipAttribute

# Import graph analysis
from fragile.proofs.relationships.graphs import (
    EquivalenceClassifier,
    FrameworkFlow,
    ObjectLineage,
    RelationshipGraph,
    build_relationship_graph_from_registry,
)

__all__ = [
    # Types (from core)
    "Relationship",
    "RelationshipAttribute",
    # Graphs
    "RelationshipGraph",
    "ObjectLineage",
    "EquivalenceClassifier",
    "FrameworkFlow",
    "build_relationship_graph_from_registry",
]
