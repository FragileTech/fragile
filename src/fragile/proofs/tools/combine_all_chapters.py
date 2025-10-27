#!/usr/bin/env python3
"""
Combine All Chapter JSON Files into Unified Registry.

This script:
1. Scans all chapter directories in docs/source/
2. Loads all JSON files (objects, axioms, parameters, theorems)
3. Parses them into Pydantic models
4. Combines into a single MathematicalRegistry
5. Saves to a directory for dashboard visualization

Usage:
    python -m fragile.proofs.combine_all_chapters
    python -m fragile.proofs.combine_all_chapters --output combined_registry
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings

# Use relative imports since script is inside the package
from ..core.article_system import SourceLocation
from ..core.math_types import (
    Axiom,
    MathematicalObject,
    ObjectType,
    Parameter,
    ParameterType,
    Attribute,
    Relationship,
    RelationshipAttribute,
    RelationType,
    TheoremBox,
    TheoremOutputType,
)
from ..registry.registry import MathematicalRegistry
from ..registry.storage import save_registry_to_directory
from ..utils.source_helpers import SourceLocationBuilder


# ============================================================================
# LAYER 1: Path & Source Location Helpers
# ============================================================================

def extract_location_from_path(json_path: Path, docs_root: Path) -> Tuple[str, str, str]:
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
    chapter: str,
    document: str,
    file_path: str,
    label: Optional[str] = None
) -> Optional[SourceLocation]:
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
    import re
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
                document_id=document_id,
                file_path=file_path,
                directive_label=label
            )
        else:
            # Use minimal for just document reference
            return SourceLocationBuilder.minimal(
                document_id=document_id,
                file_path=file_path
            )
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
    elif clean_label.startswith("def-"):
        # Convert def- to obj-
        return "obj-" + clean_label[4:]
    else:
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
    elif any(keyword in expression for keyword in ["=", "≡", "equivalent", "equals"]):
        return RelationType.EQUIVALENCE
    elif any(keyword in expression for keyword in ["⊂", "⊆", "extends", "extension"]):
        return RelationType.EXTENSION
    elif "embeds" in expr_lower or "embedding" in expr_lower:
        return RelationType.EMBEDDING
    elif "reduces" in expr_lower or "reduction" in expr_lower:
        return RelationType.REDUCTION
    elif "generaliz" in expr_lower:
        return RelationType.GENERALIZATION
    elif "specializ" in expr_lower:
        return RelationType.SPECIALIZATION
    else:
        return RelationType.OTHER


def normalize_lemma_edge(
    edge: Union[list, tuple, dict, str],
    theorem_label: str
) -> Tuple[str, str]:
    """Normalize lemma DAG edge to (source, target) tuple.

    Args:
        edge: Edge in various formats (list, dict, string)
        theorem_label: Label of current theorem (used as target for string edges)

    Returns:
        Tuple of (source_label, target_label)
    """
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        # Already a tuple/list, convert to tuple
        return (edge[0], edge[1])
    elif isinstance(edge, dict):
        # Dict format: {"depends_on": "lem-x"} or {"from": "lem-x", "to": "lem-y"}
        if "from" in edge and "to" in edge:
            return (edge["from"], edge["to"])
        elif "depends_on" in edge:
            # depends_on means edge[depends_on] → current theorem
            return (edge["depends_on"], theorem_label)
        else:
            raise ValueError(f"Dict edge missing 'from'/'to' or 'depends_on': {edge}")
    elif isinstance(edge, str):
        # String format: single dependency, target is the current theorem
        return (edge, theorem_label)
    else:
        raise ValueError(f"Invalid edge format: {edge}")


# ============================================================================
# LAYER 2: Preprocessing Functions (Prepare dicts for Pydantic)
# ============================================================================

def preprocess_attributes_added(props_raw: List) -> List[dict]:
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
    rels_raw: List,
    theorem_label: str,
    input_objects: List[str],
    source: Optional[SourceLocation]
) -> List[dict]:
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
                source_obj = ensure_object_label_prefix(input_objects[0]) if len(input_objects) > 0 else "obj-unknown"
                target_obj = ensure_object_label_prefix(input_objects[1]) if len(input_objects) > 1 else source_obj

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
                source_raw = rel_item.get("from", rel_item.get("source", input_objects[0] if input_objects else "unknown"))
                target_raw = rel_item.get("to", rel_item.get("target", input_objects[1] if len(input_objects) > 1 else "unknown"))

                source_obj = ensure_object_label_prefix(source_raw)
                target_obj = ensure_object_label_prefix(target_raw)

                # Map type string to RelationType value
                try:
                    rel_type = RelationType(type_str)
                except ValueError:
                    type_lower = type_str.lower()
                    if type_lower in ("contraction", "approximation"):
                        rel_type = RelationType.APPROXIMATION
                    elif type_lower in ("equivalence", "equal"):
                        rel_type = RelationType.EQUIVALENCE
                    elif type_lower in ("embedding", "embeds"):
                        rel_type = RelationType.EMBEDDING
                    elif type_lower in ("reduction", "reduces"):
                        rel_type = RelationType.REDUCTION
                    elif type_lower in ("extension", "extends"):
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
                label = rel_item.get("label", f"rel-{source_clean}-{target_clean}-{rel_type.value}")

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


def preprocess_lemma_edges(edges_raw: List, theorem_label: str) -> List[Tuple[str, str]]:
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
# LAYER 3: JSON Loaders (Pydantic validation)
# ============================================================================

def load_object_from_json(json_path: Path, docs_root: Path) -> MathematicalObject:
    """Load MathematicalObject from JSON file using Pydantic validation.

    Args:
        json_path: Path to JSON file
        docs_root: Path to docs/source directory

    Returns:
        Validated MathematicalObject instance
    """
    with open(json_path) as f:
        data = json.load(f)

    # Extract location from path
    try:
        chapter, document, file_path = extract_location_from_path(json_path, docs_root)
        label = data.get("label", json_path.stem)
        source = create_source_location(chapter, document, file_path, label)
    except ValueError as e:
        warnings.warn(f"Cannot extract location for {json_path.name}: {e}")
        source = None

    # Add source to data if created
    if source:
        data["source"] = source

    # Pydantic will handle object_type enum conversion automatically
    return MathematicalObject.model_validate(data)


def load_axiom_from_json(json_path: Path, docs_root: Path) -> Axiom:
    """Load Axiom from JSON file using Pydantic validation.

    Note: Axiom doesn't have source field yet, uses chapter/document directly.

    Args:
        json_path: Path to JSON file
        docs_root: Path to docs/source directory

    Returns:
        Validated Axiom instance
    """
    with open(json_path) as f:
        data = json.load(f)

    # Extract location from path
    try:
        chapter, document, _ = extract_location_from_path(json_path, docs_root)
        data["chapter"] = chapter
        data["document"] = document
    except ValueError as e:
        warnings.warn(f"Cannot extract location for {json_path.name}: {e}")

    # Pydantic handles validation
    return Axiom.model_validate(data)


def load_parameter_from_json(json_path: Path, docs_root: Path) -> Parameter:
    """Load Parameter from JSON file using Pydantic validation.

    Note: Parameter doesn't have source field yet, uses chapter/document directly.

    Args:
        json_path: Path to JSON file
        docs_root: Path to docs/source directory

    Returns:
        Validated Parameter instance
    """
    with open(json_path) as f:
        data = json.load(f)

    # Extract location from path
    try:
        chapter, document, _ = extract_location_from_path(json_path, docs_root)
        data["chapter"] = chapter
        data["document"] = document
    except ValueError as e:
        warnings.warn(f"Cannot extract location for {json_path.name}: {e}")

    # Pydantic handles parameter_type enum conversion
    return Parameter.model_validate(data)


def load_theorem_from_json(json_path: Path, docs_root: Path) -> TheoremBox:
    """Load TheoremBox from JSON file using Pydantic validation.

    Args:
        json_path: Path to JSON file
        docs_root: Path to docs/source directory

    Returns:
        Validated TheoremBox instance
    """
    with open(json_path) as f:
        data = json.load(f)

    # Extract location from path
    try:
        chapter, document, file_path = extract_location_from_path(json_path, docs_root)
        label = data.get("label", json_path.stem)
        source = create_source_location(chapter, document, file_path, label)
    except ValueError as e:
        warnings.warn(f"Cannot extract location for {json_path.name}: {e}")
        source = None

    # Add source to data if created
    if source:
        data["source"] = source

    # Preprocess attributes_added
    props_raw = data.get("attributes_added", [])
    props_processed = preprocess_attributes_added(props_raw)
    if props_processed:
        # Parse Attribute objects
        attributes_added = []
        for prop_data in props_processed:
            try:
                # Add source if not present
                if source and "source" not in prop_data:
                    prop_data["source"] = source
                prop = Attribute.model_validate(prop_data)
                attributes_added.append(prop)
            except Exception as e:
                warnings.warn(f"Error parsing property in {data.get('label', 'unknown')}: {e}")
        data["attributes_added"] = attributes_added
    else:
        data["attributes_added"] = []

    # Preprocess relations_established
    rels_raw = data.get("relations_established", [])
    input_objects = data.get("input_objects", [])
    rels_processed = preprocess_relations_established(rels_raw, data["label"], input_objects, source)
    if rels_processed:
        # Parse Relationship objects
        relations_established = []
        for rel_data in rels_processed:
            try:
                # Convert source dict to SourceLocation if needed
                if "source" in rel_data and isinstance(rel_data["source"], dict):
                    rel_data["source"] = SourceLocation.model_validate(rel_data["source"])
                rel = Relationship.model_validate(rel_data)
                relations_established.append(rel)
            except Exception as e:
                warnings.warn(f"Error parsing relationship in {data.get('label', 'unknown')}: {e}")
        data["relations_established"] = relations_established
    else:
        data["relations_established"] = []

    # Preprocess lemma_dag_edges
    edges_raw = data.get("lemma_dag_edges", [])
    data["lemma_dag_edges"] = preprocess_lemma_edges(edges_raw, data["label"])

    # Pydantic handles output_type enum conversion
    return TheoremBox.model_validate(data)


def load_relationship_from_json(json_path: Path, docs_root: Path) -> Relationship:
    """Load Relationship from JSON file using Pydantic validation.

    Args:
        json_path: Path to JSON file
        docs_root: Path to docs/source directory

    Returns:
        Validated Relationship instance
    """
    with open(json_path) as f:
        data = json.load(f)

    # Extract location from path
    try:
        chapter, document, file_path = extract_location_from_path(json_path, docs_root)
        label = data.get("label", json_path.stem)
        source = create_source_location(chapter, document, file_path, label)
    except ValueError as e:
        warnings.warn(f"Cannot extract location for {json_path.name}: {e}")
        source = None

    # Add source to data if created
    if source:
        data["source"] = source

    # Pydantic handles relationship_type enum conversion
    return Relationship.model_validate(data)


class ChapterCombiner:
    """Combines JSON files from all chapters into unified registry."""

    def __init__(self, docs_root: Path):
        """Initialize combiner.

        Args:
            docs_root: Path to docs/source directory
        """
        self.docs_root = docs_root
        self.registry = MathematicalRegistry()
        self.stats = {
            "objects": 0,
            "axioms": 0,
            "parameters": 0,
            "theorems": 0,
            "relationships": 0,
            "relationships_from_strings": 0,
            "relationships_from_dicts": 0,
            "relationships_from_full_objects": 0,
            "relationships_invalid": 0,
            "errors": [],
            "chapters_processed": set(),
        }

    def find_all_chapters(self) -> List[Path]:
        """Find all chapter directories."""
        chapters = []

        # Look for numbered chapter directories
        for path in self.docs_root.iterdir():
            if path.is_dir() and (
                path.name.startswith("1_euclidean_gas")
                or path.name.startswith("2_geometric_gas")
            ):
                chapters.append(path)

        return sorted(chapters)

    def find_chapter_subdirectories(self, chapter_path: Path) -> List[Path]:
        """Find all subdirectories in a chapter that contain JSON files."""
        subdirs = []

        for path in chapter_path.rglob("*"):
            if path.is_dir() and any(
                subdir_name in path.name
                for subdir_name in ["objects", "axioms", "theorems", "parameters", "relationships"]
            ):
                subdirs.append(path)

        return subdirs

    def validate_relationship(self, rel: Relationship) -> bool:
        """Validate that relationship source and target objects exist.

        Args:
            rel: Relationship to validate

        Returns:
            True if valid (both objects exist), False otherwise
        """
        # Check if objects exist in registry
        all_objects = {obj.label for obj in self.registry.get_all_objects()}
        has_source = rel.source_object in all_objects
        has_target = rel.target_object in all_objects

        if not has_source and rel.source_object != "obj-unknown":
            print(f"    ⚠️  Relationship {rel.label}: source object '{rel.source_object}' not found")
        if not has_target and rel.target_object != "obj-unknown":
            print(f"    ⚠️  Relationship {rel.label}: target object '{rel.target_object}' not found")

        return has_source and has_target

    def process_directory(self, dir_path: Path, dir_type: str):
        """Process all JSON files in a directory using Layer 3 loaders.

        SIMPLIFIED: Uses load_*_from_json() functions instead of manual parsing.

        Args:
            dir_path: Directory containing JSON files
            dir_type: Type of files ("objects", "axioms", "theorems", etc.)
        """
        print(f"  Processing {dir_type} from: {dir_path.relative_to(self.docs_root)}")

        json_files = list(dir_path.glob("*.json"))
        if not json_files:
            return

        for json_file in json_files:
            try:
                if dir_type == "objects":
                    obj = load_object_from_json(json_file, self.docs_root)
                    self.registry.add(obj)
                    self.stats["objects"] += 1

                elif dir_type == "axioms":
                    axiom = load_axiom_from_json(json_file, self.docs_root)
                    self.registry.add(axiom)
                    self.stats["axioms"] += 1

                elif dir_type == "parameters":
                    param = load_parameter_from_json(json_file, self.docs_root)
                    self.registry.add(param)
                    self.stats["parameters"] += 1

                elif dir_type == "theorems":
                    thm = load_theorem_from_json(json_file, self.docs_root)
                    self.registry.add(thm)
                    self.stats["theorems"] += 1

                    # Also add embedded relationships and track their format
                    for rel in thm.relations_established:
                        try:
                            # Validate relationship
                            is_valid = self.validate_relationship(rel)
                            if not is_valid:
                                self.stats["relationships_invalid"] += 1

                            # Add to registry
                            self.registry.add(rel)
                            self.stats["relationships"] += 1

                            # Track format
                            if "inferred-from-string" in rel.tags:
                                self.stats["relationships_from_strings"] += 1
                            elif "inferred-from-dict" in rel.tags:
                                self.stats["relationships_from_dicts"] += 1
                            else:
                                self.stats["relationships_from_full_objects"] += 1
                        except ValueError as e:
                            # Relationship might already exist
                            if "Duplicate ID" not in str(e):
                                raise

                elif dir_type == "relationships":
                    rel = load_relationship_from_json(json_file, self.docs_root)
                    self.registry.add(rel)
                    self.stats["relationships"] += 1

            except Exception as e:
                error_msg = f"Error loading {json_file.name}: {e}"
                print(f"    ⚠️  {error_msg}")
                self.stats["errors"].append(error_msg)

    def process_chapter(self, chapter_path: Path):
        """Process all JSON files in a chapter.

        SIMPLIFIED: No longer needs to extract document/chapter names -
        Layer 1 helpers handle this from file paths.
        """
        chapter_name = chapter_path.name
        print(f"\n{'='*70}")
        print(f"Processing Chapter: {chapter_name}")
        print(f"{'='*70}")

        self.stats["chapters_processed"].add(chapter_name)

        # Find all relevant subdirectories
        subdirs = self.find_chapter_subdirectories(chapter_path)

        for subdir in subdirs:
            if "objects" in subdir.name:
                self.process_directory(subdir, "objects")
            elif "axioms" in subdir.name:
                self.process_directory(subdir, "axioms")
            elif "parameters" in subdir.name:
                self.process_directory(subdir, "parameters")
            elif "theorems" in subdir.name:
                self.process_directory(subdir, "theorems")
            elif "relationships" in subdir.name:
                self.process_directory(subdir, "relationships")

    def combine_all(self):
        """Main method to combine all chapters."""
        print("=" * 70)
        print("COMBINING ALL CHAPTERS INTO UNIFIED REGISTRY")
        print("=" * 70)
        print(f"\nDocs root: {self.docs_root}")

        # Find all chapters
        chapters = self.find_all_chapters()
        print(f"\nFound {len(chapters)} chapters:")
        for ch in chapters:
            print(f"  - {ch.name}")

        # Process each chapter
        for chapter_path in chapters:
            self.process_chapter(chapter_path)

        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        """Print final statistics."""
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"\n✓ Chapters Processed: {len(self.stats['chapters_processed'])}")
        for ch in sorted(self.stats["chapters_processed"]):
            print(f"  - {ch}")

        total_items = (
            self.stats['objects'] +
            self.stats['axioms'] +
            self.stats['parameters'] +
            self.stats['theorems'] +
            self.stats['relationships']
        )

        print(f"\n✓ Total Items Loaded:")
        print(f"  - Objects:        {self.stats['objects']}")
        print(f"  - Axioms:         {self.stats['axioms']}")
        print(f"  - Parameters:     {self.stats['parameters']}")
        print(f"  - Theorems:       {self.stats['theorems']}")
        print(f"  - Relationships:  {self.stats['relationships']}")
        print(f"  - TOTAL:          {total_items}")

        # Detailed relationship statistics
        if self.stats['relationships'] > 0:
            print(f"\n✓ Relationship Details:")
            print(f"  - From strings:         {self.stats['relationships_from_strings']} ({100 * self.stats['relationships_from_strings'] / self.stats['relationships']:.1f}%)")
            print(f"  - From simple dicts:    {self.stats['relationships_from_dicts']} ({100 * self.stats['relationships_from_dicts'] / self.stats['relationships']:.1f}%)")
            print(f"  - From full objects:    {self.stats['relationships_from_full_objects']} ({100 * self.stats['relationships_from_full_objects'] / self.stats['relationships']:.1f}%)")
            if self.stats['relationships_invalid'] > 0:
                print(f"  - Invalid (missing src/tgt): {self.stats['relationships_invalid']}")

            # Count by relationship type
            rel_type_counts = {}
            for rel in self.registry.get_all_relationships():
                rt = rel.relationship_type.value
                rel_type_counts[rt] = rel_type_counts.get(rt, 0) + 1

            print(f"\n✓ Relationships by Type:")
            for rt, count in sorted(rel_type_counts.items(), key=lambda x: -x[1]):
                pct = 100 * count / self.stats['relationships']
                print(f"  - {rt:20s} {count:4d} ({pct:.1f}%)")

        if self.stats["errors"]:
            print(f"\n⚠️  Errors Encountered: {len(self.stats['errors'])}")
            print("\nFirst 10 errors:")
            for error in self.stats["errors"][:10]:
                print(f"  - {error}")
            if len(self.stats["errors"]) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more errors")

    def save_registry(self, output_dir: Path):
        """Save combined registry to directory."""
        print("\n" + "=" * 70)
        print(f"SAVING REGISTRY TO: {output_dir}")
        print("=" * 70)

        save_registry_to_directory(self.registry, output_dir)

        print(f"\n✓ Registry saved successfully!")
        print(f"\nTo visualize with dashboard:")
        print(f"  panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show")
        print(f"\nTo regenerate:")
        print(f"  python -m fragile.proofs.combine_all_chapters --output combined_registry")
        print(f"\nOr load programmatically:")
        print(f"  from fragile.proofs import load_registry_from_directory, MathematicalRegistry")
        print(f"  registry = load_registry_from_directory(MathematicalRegistry, '{output_dir}')")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Combine all chapter JSON files into unified registry")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="combined_registry",
        help="Output directory name (default: combined_registry)",
    )
    parser.add_argument(
        "--docs-root",
        type=str,
        default="docs/source",
        help="Path to docs/source directory (default: docs/source)",
    )

    args = parser.parse_args()

    # Setup paths
    # Script is in src/fragile/proofs/tools/, go up 4 levels to project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    docs_root = project_root / args.docs_root
    output_dir = project_root / args.output

    if not docs_root.exists():
        print(f"Error: Docs root not found: {docs_root}")
        sys.exit(1)

    # Create combiner and process
    combiner = ChapterCombiner(docs_root)
    combiner.combine_all()

    # Save registry
    combiner.save_registry(output_dir)

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
