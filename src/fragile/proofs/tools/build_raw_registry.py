#!/usr/bin/env python3
"""
Build Registry from Raw Data.

This registry builder loads raw JSON files from raw_data/ directories
(output of document-parser agent) with minimal preprocessing.

Use this when:
- Loading directly from raw_data/ (skipping refinement)
- You want fastest processing with minimal transformation
- Testing extraction pipeline output

Usage:
    python -m fragile.proofs.tools.build_raw_registry \\
        --docs-root docs/source \\
        --output raw_registry

Version: 1.0.0
"""

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Set, Tuple
import warnings

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.math_types import (
    Attribute,
    Axiom,
    MathematicalObject,
    Parameter,
    Relationship,
    TheoremBox,
)
from fragile.proofs.registry.registry import MathematicalRegistry
from fragile.proofs.registry.storage import save_registry_to_directory
from fragile.proofs.tools.registry_builders_common import (
    create_source_location,
    ensure_object_label_prefix,
    extract_location_from_path,
    preprocess_attributes_added,
    preprocess_lemma_edges,
    preprocess_relations_established,
)


# ============================================================================
# JSON LOADERS (with minimal preprocessing)
# ============================================================================


def load_object_from_json(json_path: Path, docs_root: Path) -> MathematicalObject:
    """Load MathematicalObject from raw JSON file.

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
    """Load Axiom from raw JSON file.

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
    """Load Parameter from raw JSON file.

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
    """Load TheoremBox from raw JSON file.

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
    rels_processed = preprocess_relations_established(
        rels_raw, data["label"], input_objects, source
    )
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
    """Load Relationship from raw JSON file.

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


# ============================================================================
# RAW DATA COMBINER
# ============================================================================


class RawDataCombiner:
    """Combines JSON files from raw_data/ into unified registry."""

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

    def find_all_chapters(self) -> list[Path]:
        """Find all chapter directories.

        Discovers any directory matching pattern: {digit(s)}_{name}
        Examples: 1_euclidean_gas, 2_geometric_gas, 10_advanced_topics, etc.
        """
        chapters = []

        # Look for numbered chapter directories (pattern: {digit}_{name})
        chapter_pattern = re.compile(r"^\d+_\w+$")
        for path in self.docs_root.iterdir():
            if path.is_dir() and chapter_pattern.match(path.name):
                chapters.append(path)

        return sorted(chapters)

    def find_raw_data_directories(self, chapter_path: Path) -> dict[str, Path]:
        """Find raw_data subdirectories in a chapter.

        Only looks for raw_data as direct children of section directories,
        avoiding outdated data scattered elsewhere in the tree.

        Args:
            chapter_path: Path to chapter directory

        Returns:
            Dict mapping entity type to directory path
        """
        raw_dirs = {}

        # Look for section directories (direct children only)
        for section_dir in chapter_path.iterdir():
            if not section_dir.is_dir():
                continue

            # Check for raw_data as immediate child of section directory
            raw_data_dir = section_dir / "raw_data"
            if not raw_data_dir.exists():
                continue

            # Check for entity type subdirectories within raw_data
            for entity_type in ["objects", "axioms", "parameters", "theorems", "relationships"]:
                entity_dir = raw_data_dir / entity_type
                if entity_dir.exists():
                    # Store with full path for uniqueness
                    key = f"{section_dir.name}_{entity_type}"
                    raw_dirs[key] = entity_dir

        return raw_dirs

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
            print(
                f"    ⚠️  Relationship {rel.label}: source object '{rel.source_object}' not found"
            )
        if not has_target and rel.target_object != "obj-unknown":
            print(
                f"    ⚠️  Relationship {rel.label}: target object '{rel.target_object}' not found"
            )

        return has_source and has_target

    def process_raw_directory(self, dir_path: Path, dir_type: str):
        """Process all JSON files in a raw_data directory.

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
        """Process all raw_data in a chapter.

        Args:
            chapter_path: Path to chapter directory
        """
        chapter_name = chapter_path.name
        print(f"\n{'=' * 70}")
        print(f"Processing Chapter: {chapter_name}")
        print(f"{'=' * 70}")

        self.stats["chapters_processed"].add(chapter_name)

        # Find all raw_data subdirectories
        raw_dirs = self.find_raw_data_directories(chapter_path)

        if not raw_dirs:
            print("  ⚠️  No raw_data directories found")
            return

        # Process in order: objects, axioms, parameters, theorems, relationships
        # (theorems depend on objects/axioms/parameters)
        # Group directories by entity type to process all sections' objects first, then all axioms, etc.
        for entity_type in ["objects", "axioms", "parameters", "theorems", "relationships"]:
            # Find all directories for this entity type across all sections
            matching_dirs = {k: v for k, v in raw_dirs.items() if k.endswith(f"_{entity_type}")}
            for dir_path in matching_dirs.values():
                self.process_raw_directory(dir_path, entity_type)

    def combine_all(self):
        """Main method to combine all chapters."""
        print("=" * 70)
        print("COMBINING RAW DATA INTO UNIFIED REGISTRY")
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
            self.stats["objects"]
            + self.stats["axioms"]
            + self.stats["parameters"]
            + self.stats["theorems"]
            + self.stats["relationships"]
        )

        print("\n✓ Total Items Loaded:")
        print(f"  - Objects:        {self.stats['objects']}")
        print(f"  - Axioms:         {self.stats['axioms']}")
        print(f"  - Parameters:     {self.stats['parameters']}")
        print(f"  - Theorems:       {self.stats['theorems']}")
        print(f"  - Relationships:  {self.stats['relationships']}")
        print(f"  - TOTAL:          {total_items}")

        # Detailed relationship statistics
        if self.stats["relationships"] > 0:
            print("\n✓ Relationship Details:")
            print(
                f"  - From strings:         {self.stats['relationships_from_strings']} ({100 * self.stats['relationships_from_strings'] / self.stats['relationships']:.1f}%)"
            )
            print(
                f"  - From simple dicts:    {self.stats['relationships_from_dicts']} ({100 * self.stats['relationships_from_dicts'] / self.stats['relationships']:.1f}%)"
            )
            print(
                f"  - From full objects:    {self.stats['relationships_from_full_objects']} ({100 * self.stats['relationships_from_full_objects'] / self.stats['relationships']:.1f}%)"
            )
            if self.stats["relationships_invalid"] > 0:
                print(f"  - Invalid (missing src/tgt): {self.stats['relationships_invalid']}")

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

        print("\n✓ Registry saved successfully!")
        print("\nTo visualize with dashboard:")
        print("  panel serve src/fragile/proofs/proof_pipeline_dashboard.py --show")
        print("\nTo regenerate:")
        print("  python -m fragile.proofs.tools.build_raw_registry \\")
        print(f"    --docs-root {self.docs_root} \\")
        print(f"    --output {output_dir}")
        print("\nOr load programmatically:")
        print("  from fragile.proofs import load_registry_from_directory, MathematicalRegistry")
        print(f"  registry = load_registry_from_directory(MathematicalRegistry, '{output_dir}')")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build registry from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build registry from all raw_data directories
  python -m fragile.proofs.tools.build_raw_registry \\
    --docs-root docs/source \\
    --output raw_registry

  # Build with custom docs root
  python -m fragile.proofs.tools.build_raw_registry \\
    --docs-root /path/to/docs/source \\
    --output my_raw_registry
        """,
    )
    parser.add_argument(
        "--docs-root",
        type=str,
        default="docs/source",
        help="Path to docs/source directory (default: docs/source)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="raw_registry",
        help="Output directory name (default: raw_registry)",
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
    combiner = RawDataCombiner(docs_root)
    combiner.combine_all()

    # Save registry
    combiner.save_registry(output_dir)

    # Exit with error code if there were errors
    if combiner.stats["errors"]:
        print(f"\n❌ Processing completed with {len(combiner.stats['errors'])} errors")
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("✅ COMPLETE!")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
