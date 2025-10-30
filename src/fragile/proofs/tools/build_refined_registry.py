#!/usr/bin/env python3
"""
Build Registry from Refined (Enriched) Data.

This registry builder loads enriched JSON files from refined_data/ directories
(output of document-refiner agent) and applies transformations to create
a MathematicalRegistry.

Use this when:
- Loading directly from refined_data/ (skipping transformation step)
- You want automatic transformation during registry building
- Processing enriched data with full preprocessing

Usage:
    python -m fragile.proofs.tools.build_refined_registry \\
        --docs-root docs/source \\
        --output refined_registry

Version: 1.0.0
"""

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Set
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
from fragile.proofs.tools.enriched_to_math_types import EnrichedToMathTypesTransformer


# ============================================================================
# REFINED DATA COMBINER
# ============================================================================


class RefinedDataCombiner:
    """Combines JSON files from refined_data/ into unified registry."""

    def __init__(self, docs_root: Path):
        """Initialize combiner.

        Args:
            docs_root: Path to docs/source directory
        """
        self.docs_root = docs_root
        self.registry = MathematicalRegistry()
        self.transformer = EnrichedToMathTypesTransformer()
        self.stats = {
            "objects": 0,
            "axioms": 0,
            "parameters": 0,
            "theorems": 0,
            "relationships": 0,
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

    def find_refined_data_directories(self, chapter_path: Path) -> dict[str, Path]:
        """Find refined_data subdirectories in a chapter.

        Only looks for refined_data as direct children of section directories,
        avoiding outdated data scattered elsewhere in the tree.

        Args:
            chapter_path: Path to chapter directory

        Returns:
            Dict mapping entity type to directory path
        """
        refined_dirs = {}

        # Look for section directories (direct children only)
        for section_dir in chapter_path.iterdir():
            if not section_dir.is_dir():
                continue

            # Check for refined_data as immediate child of section directory
            refined_data_dir = section_dir / "refined_data"
            if not refined_data_dir.exists():
                continue

            # Check for entity type subdirectories within refined_data
            for entity_type in ["objects", "axioms", "parameters", "theorems", "relationships"]:
                entity_dir = refined_data_dir / entity_type
                if entity_dir.exists():
                    # Store with full path for uniqueness
                    key = f"{section_dir.name}_{entity_type}"
                    refined_dirs[key] = entity_dir

        return refined_dirs

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

    def process_refined_directory(self, dir_path: Path, dir_type: str):
        """Process all JSON files in a refined_data directory.

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
                with open(json_file) as f:
                    data = json.load(f)

                # Transform using enriched_to_math_types transformer
                transformed = None
                if dir_type == "objects":
                    transformed = self.transformer.transform_enriched_object(data, json_file)
                    if transformed:
                        self.registry.add(transformed)
                        self.stats["objects"] += 1

                elif dir_type == "axioms":
                    transformed = self.transformer.transform_enriched_axiom(data, json_file)
                    if transformed:
                        self.registry.add(transformed)
                        self.stats["axioms"] += 1

                elif dir_type == "parameters":
                    transformed = self.transformer.transform_enriched_parameter(data, json_file)
                    if transformed:
                        self.registry.add(transformed)
                        self.stats["parameters"] += 1

                elif dir_type == "theorems":
                    transformed = self.transformer.transform_enriched_theorem(data, json_file)
                    if transformed:
                        self.registry.add(transformed)
                        self.stats["theorems"] += 1

                        # Also add embedded relationships
                        for rel in transformed.relations_established:
                            try:
                                # Validate relationship
                                is_valid = self.validate_relationship(rel)
                                if not is_valid:
                                    print(f"    ⚠️  Invalid relationship in {json_file.name}")

                                # Add to registry
                                self.registry.add(rel)
                                self.stats["relationships"] += 1
                            except ValueError as e:
                                # Relationship might already exist
                                if "Duplicate ID" not in str(e):
                                    raise

                elif dir_type == "relationships":
                    # Handle standalone relationship files (if any)
                    # These should already be in proper format
                    if "source" in data and isinstance(data["source"], dict):
                        data["source"] = SourceLocation.model_validate(data["source"])
                    rel = Relationship.model_validate(data)
                    self.registry.add(rel)
                    self.stats["relationships"] += 1

            except Exception as e:
                error_msg = f"Error loading {json_file.name}: {e}"
                print(f"    ⚠️  {error_msg}")
                self.stats["errors"].append(error_msg)

    def process_chapter(self, chapter_path: Path):
        """Process all refined_data in a chapter.

        Args:
            chapter_path: Path to chapter directory
        """
        chapter_name = chapter_path.name
        print(f"\n{'=' * 70}")
        print(f"Processing Chapter: {chapter_name}")
        print(f"{'=' * 70}")

        self.stats["chapters_processed"].add(chapter_name)

        # Find all refined_data subdirectories
        refined_dirs = self.find_refined_data_directories(chapter_path)

        if not refined_dirs:
            print("  ⚠️  No refined_data directories found")
            return

        # Process in order: objects, axioms, parameters, theorems, relationships
        # (theorems depend on objects/axioms/parameters)
        # Group directories by entity type to process all sections' objects first, then all axioms, etc.
        for entity_type in ["objects", "axioms", "parameters", "theorems", "relationships"]:
            # Find all directories for this entity type across all sections
            matching_dirs = {
                k: v for k, v in refined_dirs.items() if k.endswith(f"_{entity_type}")
            }
            for dir_path in matching_dirs.values():
                self.process_refined_directory(dir_path, entity_type)

    def combine_all(self):
        """Main method to combine all chapters."""
        print("=" * 70)
        print("COMBINING REFINED DATA INTO UNIFIED REGISTRY")
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

        # Transformation statistics
        if self.transformer.report.warnings:
            print(f"\n⚠️  Transformation Warnings: {len(self.transformer.report.warnings)}")
            for warning in self.transformer.report.warnings[:5]:
                print(f"  - {warning}")
            if len(self.transformer.report.warnings) > 5:
                print(f"  ... and {len(self.transformer.report.warnings) - 5} more warnings")

        if self.stats["errors"]:
            print(f"\n❌ Errors Encountered: {len(self.stats['errors'])}")
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
        print("  python -m fragile.proofs.tools.build_refined_registry \\")
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
        description="Build registry from refined (enriched) data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build registry from all refined_data directories
  python -m fragile.proofs.tools.build_refined_registry \\
    --docs-root docs/source \\
    --output refined_registry

  # Build with custom docs root
  python -m fragile.proofs.tools.build_refined_registry \\
    --docs-root /path/to/docs/source \\
    --output my_refined_registry
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
        default="refined_registry",
        help="Output directory name (default: refined_registry)",
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
    combiner = RefinedDataCombiner(docs_root)
    combiner.combine_all()

    # Save registry
    combiner.save_registry(output_dir)

    # Exit with error code if there were errors
    if combiner.stats["errors"] or combiner.transformer.report.errors:
        total_errors = len(combiner.stats["errors"]) + len(combiner.transformer.report.errors)
        print(f"\n❌ Processing completed with {total_errors} errors")
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("✅ COMPLETE!")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
