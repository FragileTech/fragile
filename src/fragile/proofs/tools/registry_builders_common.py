#!/usr/bin/env python3
"""
Common Utilities for Registry Builders.

This module contains shared functions used by all registry builders:
- Path and source location helpers
- Label normalization
- Preprocessing functions for attributes, relationships, and edges

Extracted from combine_all_chapters.py for reusability.

Version: 1.0.0
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.math_types import RelationType
from fragile.proofs.utils.source_helpers import SourceLocationBuilder


# ============================================================================
# LAYER 1: Path & Source Location Helpers
# ============================================================================


def extract_location_from_path(json_path: Path, docs_root: Path) -> tuple[str, str, str]:
    """Extract chapter, document, and markdown file path from JSON file path.

    Args:
        json_path: Path to JSON file
            e.g., docs/source/1_euclidean_gas/03_cloning/objects/obj-walker.json
        docs_root: Path to docs/source directory
            e.g., docs/source

    Returns:
        Tuple of (chapter, document, file_path):
            chapter: e.g., '1_euclidean_gas'
            document: e.g., '03_cloning'
            file_path: e.g., 'docs/source/1_euclidean_gas/03_cloning.md'

    Raises:
        ValueError: If path structure doesn't match expected format
    """
    try:
        # Get path relative to docs_root
        rel_path = json_path.relative_to(docs_root)
        parts = rel_path.parts

        # Expected structure: chapter/document/type/file.json
        # e.g., 1_euclidean_gas/03_cloning/objects/obj-walker.json
        if len(parts) < 2:
            raise ValueError(f"Path too short: {rel_path}")

        chapter = parts[0]  # e.g., '1_euclidean_gas'

        # Document could be parts[1] or 'root' if directly under chapter
        if len(parts) == 2:
            # Directly under chapter (e.g., 1_euclidean_gas/objects/obj-foo.json)
            document = "root"
        else:
            document = parts[1]  # e.g., '03_cloning'

        # Construct markdown file path
        if document == "root":
            # No specific document, use chapter-level file if it exists
            file_path = str(docs_root / chapter / f"{chapter}.md")
        else:
            file_path = str(docs_root / chapter / f"{document}.md")

        return chapter, document, file_path

    except (ValueError, IndexError) as e:
        raise ValueError(f"Cannot extract location from path {json_path}: {e}")


def create_source_location(
    chapter: str, document: str, file_path: str, label: str | None = None
) -> SourceLocation | None:
    """Create SourceLocation using SourceLocationBuilder.

    Args:
        chapter: Chapter name (e.g., '1_euclidean_gas')
        document: Document name (e.g., '03_cloning')
        file_path: Path to markdown file
        label: Optional directive label for more specific location

    Returns:
        SourceLocation or None if cannot be created (with warning)
    """
    # Validate document_id format (required by SourceLocation pattern)
    # Expected: NN_lowercase_with_underscores
    if document == "root":
        # Use chapter as document_id if at root level
        document_id = chapter
    else:
        document_id = document

    # Check if document_id matches pattern: ^[0-9]{2}_[a-z_]+$
    if not re.match(r"^[0-9]{2}_[a-z_]+$", document_id):
        warnings.warn(
            f"Document ID '{document_id}' doesn't match expected pattern (NN_lowercase). "
            f"SourceLocation may be invalid."
        )
        # Still try to create it, Pydantic will validate

    try:
        if label:
            # Use from_jupyter_directive for more specific location
            return SourceLocationBuilder.from_jupyter_directive(
                document_id=document_id, file_path=file_path, directive_label=label
            )
        # Use minimal for just document reference
        return SourceLocationBuilder.minimal(document_id=document_id, file_path=file_path)
    except Exception as e:
        warnings.warn(f"Cannot create SourceLocation for {document_id}: {e}")
        return None


def ensure_object_label_prefix(label: str) -> str:
    """Ensure object label starts with obj- and only contains lowercase alphanumeric and hyphens.

    Args:
        label: Raw label (may have underscores, uppercase, wrong prefix)

    Returns:
        Normalized label with obj- prefix
    """
    # Clean the label: lowercase, replace underscores with hyphens
    clean_label = label.lower().replace("_", "-")

    if clean_label.startswith("obj-"):
        return clean_label
    if clean_label.startswith("def-"):
        # Convert def- to obj-
        return "obj-" + clean_label[4:]
    return "obj-" + clean_label


def infer_relationship_type(expression: str) -> RelationType:
    """Infer RelationType from expression keywords.

    Args:
        expression: Relationship expression string

    Returns:
        Inferred RelationType (defaults to OTHER if no match)
    """
    expr_lower = expression.lower()

    if any(keyword in expression for keyword in ["≤", "≥", "<", ">", "bound"]):
        return RelationType.APPROXIMATION
    if any(keyword in expression for keyword in ["=", "≡", "equivalent", "equals"]):
        return RelationType.EQUIVALENCE
    if any(keyword in expression for keyword in ["⊂", "⊆", "extends", "extension"]):
        return RelationType.EXTENSION
    if "embeds" in expr_lower or "embedding" in expr_lower:
        return RelationType.EMBEDDING
    if "reduces" in expr_lower or "reduction" in expr_lower:
        return RelationType.REDUCTION
    if "generaliz" in expr_lower:
        return RelationType.GENERALIZATION
    if "specializ" in expr_lower:
        return RelationType.SPECIALIZATION
    return RelationType.OTHER


def normalize_lemma_edge(edge: list | tuple | dict | str, theorem_label: str) -> tuple[str, str]:
    """Normalize lemma DAG edge to (source, target) tuple.

    Args:
        edge: Edge in various formats (list, dict, string)
        theorem_label: Label of current theorem (used as target for string edges)

    Returns:
        Tuple of (source_label, target_label)
    """
    if isinstance(edge, list | tuple) and len(edge) == 2:
        # Already a tuple/list, convert to tuple
        return (edge[0], edge[1])
    if isinstance(edge, dict):
        # Dict format: {"depends_on": "lem-x"} or {"from": "lem-x", "to": "lem-y"}
        if "from" in edge and "to" in edge:
            return (edge["from"], edge["to"])
        if "depends_on" in edge:
            # depends_on means edge[depends_on] → current theorem
            return (edge["depends_on"], theorem_label)
        raise ValueError(f"Dict edge missing 'from'/'to' or 'depends_on': {edge}")
    if isinstance(edge, str):
        # String format: single dependency, target is the current theorem
        return (edge, theorem_label)
    raise ValueError(f"Invalid edge format: {edge}")


# ============================================================================
# LAYER 2: Preprocessing Functions (Prepare dicts for Pydantic)
# ============================================================================


def preprocess_attributes_added(props_raw: list) -> list[dict]:
    """Preprocess attributes_added from theorem JSON.

    Args:
        props_raw: Raw properties from JSON (may be dicts or strings)

    Returns:
        List of property dicts ready for Attribute.model_validate()
    """
    properties = []
    if not props_raw or not isinstance(props_raw, list):
        return properties

    for prop_data in props_raw:
        if isinstance(prop_data, dict):
            # Full property dict
            properties.append(prop_data)
        elif isinstance(prop_data, str):
            # Just a label, skip (property should exist separately)
            continue

    return properties


def preprocess_relations_established(
    rels_raw: list, theorem_label: str, input_objects: list[str], source: SourceLocation | None
) -> list[dict]:
    """Preprocess relations_established from theorem JSON.

    Handles three formats:
    1. String expressions → infer relationship from keywords
    2. Simple dicts → normalize to full Relationship format
    3. Full Relationship objects → pass through

    Args:
        rels_raw: Raw relations from JSON
        theorem_label: Label of theorem establishing these relations
        input_objects: Input objects from theorem
        source: SourceLocation for the theorem

    Returns:
        List of relationship dicts ready for Relationship.model_validate()
    """
    relations = []
    if not rels_raw or not isinstance(rels_raw, list):
        return relations

    for idx, rel_item in enumerate(rels_raw):
        try:
            if isinstance(rel_item, str):
                # String format: create synthetic relationship from expression
                rel_type = infer_relationship_type(rel_item)

                # Generate label
                theorem_clean = theorem_label.lower().replace("_", "-")
                label = f"rel-{theorem_clean}-{idx}-{rel_type.value}"

                # Infer source/target from input objects
                source_obj = (
                    ensure_object_label_prefix(input_objects[0])
                    if len(input_objects) > 0
                    else "obj-unknown"
                )
                target_obj = (
                    ensure_object_label_prefix(input_objects[1])
                    if len(input_objects) > 1
                    else source_obj
                )

                rel_dict = {
                    "label": label,
                    "relationship_type": rel_type.value,
                    "source_object": source_obj,
                    "target_object": target_obj,
                    "bidirectional": False,
                    "established_by": theorem_label,
                    "expression": rel_item,
                    "properties": [],
                    "tags": ["inferred-from-string"],
                }
                if source:
                    rel_dict["source"] = source

                relations.append(rel_dict)

            elif isinstance(rel_item, dict) and "relationship_type" in rel_item:
                # Full Relationship object format - pass through with source
                if source and "source" not in rel_item:
                    rel_item = {**rel_item, "source": source}
                relations.append(rel_item)

            elif isinstance(rel_item, dict):
                # Simple dict format: parse and create relationship
                type_str = rel_item.get("type", "other")
                source_raw = rel_item.get(
                    "from",
                    rel_item.get("source", input_objects[0] if input_objects else "unknown"),
                )
                target_raw = rel_item.get(
                    "to",
                    rel_item.get(
                        "target", input_objects[1] if len(input_objects) > 1 else "unknown"
                    ),
                )

                source_obj = ensure_object_label_prefix(source_raw)
                target_obj = ensure_object_label_prefix(target_raw)

                # Map type string to RelationType value
                try:
                    rel_type = RelationType(type_str)
                except ValueError:
                    type_lower = type_str.lower()
                    if type_lower in {"contraction", "approximation"}:
                        rel_type = RelationType.APPROXIMATION
                    elif type_lower in {"equivalence", "equal"}:
                        rel_type = RelationType.EQUIVALENCE
                    elif type_lower in {"embedding", "embeds"}:
                        rel_type = RelationType.EMBEDDING
                    elif type_lower in {"reduction", "reduces"}:
                        rel_type = RelationType.REDUCTION
                    elif type_lower in {"extension", "extends"}:
                        rel_type = RelationType.EXTENSION
                    elif type_lower == "generalization":
                        rel_type = RelationType.GENERALIZATION
                    elif type_lower == "specialization":
                        rel_type = RelationType.SPECIALIZATION
                    else:
                        rel_type = RelationType.OTHER

                # Generate label
                source_clean = source_obj.replace("obj-", "").replace("_", "-")
                target_clean = target_obj.replace("obj-", "").replace("_", "-")
                label = rel_item.get(
                    "label", f"rel-{source_clean}-{target_clean}-{rel_type.value}"
                )

                # Construct expression from dict fields
                expression_parts = []
                if "metric" in rel_item:
                    expression_parts.append(f"metric: {rel_item['metric']}")
                if "rate" in rel_item:
                    expression_parts.append(f"rate: {rel_item['rate']}")
                if "bound" in rel_item:
                    expression_parts.append(f"bound: {rel_item['bound']}")
                expression = ", ".join(expression_parts) if expression_parts else str(rel_item)

                rel_dict = {
                    "label": label,
                    "relationship_type": rel_type.value,
                    "source_object": source_obj,
                    "target_object": target_obj,
                    "bidirectional": rel_item.get("bidirectional", False),
                    "established_by": theorem_label,
                    "expression": expression,
                    "properties": [],
                    "tags": ["inferred-from-dict"],
                }
                if source:
                    rel_dict["source"] = source

                relations.append(rel_dict)

        except (KeyError, TypeError, ValueError) as e:
            warnings.warn(f"Error parsing relationship in {theorem_label}: {e}")
            continue

    return relations


def preprocess_lemma_edges(edges_raw: list, theorem_label: str) -> list[tuple[str, str]]:
    """Preprocess lemma_dag_edges from theorem JSON.

    Args:
        edges_raw: Raw edges from JSON (may be tuples, dicts, or strings)
        theorem_label: Label of current theorem

    Returns:
        List of (source, target) tuples
    """
    edges = []
    if not edges_raw or not isinstance(edges_raw, list):
        return edges

    for edge in edges_raw:
        try:
            normalized = normalize_lemma_edge(edge, theorem_label)
            edges.append(normalized)
        except ValueError as e:
            warnings.warn(f"Error normalizing edge in {theorem_label}: {e}")
            continue

    return edges


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "create_source_location",
    "ensure_object_label_prefix",
    # Path & location helpers
    "extract_location_from_path",
    "infer_relationship_type",
    "normalize_lemma_edge",
    # Preprocessing functions
    "preprocess_attributes_added",
    "preprocess_lemma_edges",
    "preprocess_relations_established",
]
