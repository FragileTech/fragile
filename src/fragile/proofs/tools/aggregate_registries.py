#!/usr/bin/env python3
"""
Aggregate Per-Document Registries into Combined Registry.

This script:
1. Scans registries/per_document/ for all available documents
2. Loads all per-document registries of specified type (refined or pipeline)
3. Merges entities into a single combined registry
4. Checks for duplicate labels and warns if found
5. Saves to registries/combined/{type}/

Usage:
    # Aggregate refined registries
    python -m fragile.proofs.tools.aggregate_registries \\
        --type refined \\
        --per-document-root registries/per_document \\
        --output registries/combined/refined

    # Aggregate pipeline registries
    python -m fragile.proofs.tools.aggregate_registries \\
        --type pipeline \\
        --per-document-root registries/per_document \\
        --output registries/combined/pipeline

Version: 1.0.0
"""

import argparse
from pathlib import Path
import sys
from typing import List, Set

from fragile.proofs.registry.registry import MathematicalRegistry
from fragile.proofs.registry.storage import (
    load_registry_from_directory,
    save_registry_to_directory,
)


# ============================================================================
# REGISTRY AGGREGATOR
# ============================================================================


class RegistryAggregator:
    """Aggregates multiple per-document registries into a single combined registry."""

    def __init__(self, per_document_root: Path, registry_type: str):
        """Initialize aggregator.

        Args:
            per_document_root: Path to registries/per_document directory
            registry_type: Type of registry to aggregate ("refined" or "pipeline")
        """
        self.per_document_root = per_document_root
        self.registry_type = registry_type
        self.combined_registry = MathematicalRegistry()
        self.stats = {
            "documents_processed": [],
            "objects": 0,
            "axioms": 0,
            "parameters": 0,
            "theorems": 0,
            "relationships": 0,
            "duplicates": [],
        }

    def find_available_documents(self) -> list[Path]:
        """Find all documents with registries of specified type.

        Returns:
            List of paths to document directories containing the specified registry type
        """
        documents = []

        if not self.per_document_root.exists():
            print(f"⚠️  Per-document root not found: {self.per_document_root}")
            return documents

        # Scan for document directories
        for doc_dir in self.per_document_root.iterdir():
            if not doc_dir.is_dir():
                continue

            # Check if registry type exists for this document
            registry_path = doc_dir / self.registry_type
            if registry_path.exists() and (registry_path / "index.json").exists():
                documents.append(doc_dir)

        return sorted(documents)

    def merge_registry(self, doc_path: Path):
        """Merge a per-document registry into the combined registry.

        Args:
            doc_path: Path to document directory (e.g., registries/per_document/01_fragile_gas_framework)
        """
        registry_path = doc_path / self.registry_type
        doc_name = doc_path.name

        print(f"\nMerging {doc_name} ({self.registry_type})...")
        print(f"  Loading from: {registry_path}")

        # Load registry
        try:
            registry = load_registry_from_directory(MathematicalRegistry, registry_path)
        except Exception as e:
            print(f"  ❌ Error loading registry: {e}")
            return

        # Track what we're about to add
        objects = registry.get_all_objects()
        axioms = registry.get_all_axioms()
        parameters = registry.get_all_parameters()
        theorems = registry.get_all_theorems()
        relationships = registry.get_all_relationships()

        print(
            f"  Found: {len(objects)} objects, {len(axioms)} axioms, "
            f"{len(theorems)} theorems, {len(relationships)} relationships"
        )

        # Check for duplicates before adding
        existing_labels = {obj.label for obj in self.combined_registry.get_all_objects()}
        existing_labels.update(ax.label for ax in self.combined_registry.get_all_axioms())
        existing_labels.update(thm.label for thm in self.combined_registry.get_all_theorems())

        duplicates_found = []
        for entity in objects + axioms + theorems:
            if entity.label in existing_labels:
                duplicates_found.append(entity.label)

        if duplicates_found:
            print(f"  ⚠️  Found {len(duplicates_found)} duplicate labels:")
            for label in duplicates_found[:5]:
                print(f"      - {label}")
            if len(duplicates_found) > 5:
                print(f"      ... and {len(duplicates_found) - 5} more")
            self.stats["duplicates"].extend(duplicates_found)
            print("  ⚠️  Skipping duplicates (first occurrence wins)")

        # Add entities (skip duplicates)
        added_count = 0
        for entity in objects + axioms + parameters + theorems + relationships:
            if entity.label not in existing_labels:
                try:
                    self.combined_registry.add(entity)
                    added_count += 1
                except Exception as e:
                    print(f"  ⚠️  Error adding {entity.label}: {e}")

        # Update statistics
        self.stats["documents_processed"].append(doc_name)
        self.stats["objects"] += len([obj for obj in objects if obj.label not in duplicates_found])
        self.stats["axioms"] += len([ax for ax in axioms if ax.label not in duplicates_found])
        self.stats["theorems"] += len([
            thm for thm in theorems if thm.label not in duplicates_found
        ])
        self.stats["parameters"] += len(parameters)
        self.stats["relationships"] += len(relationships)

        print(f"  ✓ Merged {added_count} entities (skipped {len(duplicates_found)} duplicates)")

    def aggregate_all(self):
        """Main method to aggregate all per-document registries."""
        print("=" * 70)
        print(f"AGGREGATING {self.registry_type.upper()} REGISTRIES")
        print("=" * 70)
        print(f"\nPer-document root: {self.per_document_root}")
        print(f"Registry type: {self.registry_type}")

        # Find all available documents
        documents = self.find_available_documents()
        print(f"\nFound {len(documents)} documents with {self.registry_type} registries:")
        for doc in documents:
            print(f"  - {doc.name}")

        if not documents:
            print("\n⚠️  No documents found to aggregate!")
            return

        # Merge each document
        for doc_path in documents:
            self.merge_registry(doc_path)

        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        """Print aggregation statistics."""
        print("\n" + "=" * 70)
        print("AGGREGATION STATISTICS")
        print("=" * 70)

        print(f"\n✓ Documents Processed: {len(self.stats['documents_processed'])}")
        for doc in self.stats["documents_processed"]:
            print(f"  - {doc}")

        total_items = (
            self.stats["objects"]
            + self.stats["axioms"]
            + self.stats["parameters"]
            + self.stats["theorems"]
            + self.stats["relationships"]
        )

        print(f"\n✓ Total Items in Combined Registry: {total_items}")
        print(f"  - Objects:        {self.stats['objects']}")
        print(f"  - Axioms:         {self.stats['axioms']}")
        print(f"  - Parameters:     {self.stats['parameters']}")
        print(f"  - Theorems:       {self.stats['theorems']}")
        print(f"  - Relationships:  {self.stats['relationships']}")

        if self.stats["duplicates"]:
            print(f"\n⚠️  Duplicates Found: {len(self.stats['duplicates'])} total")
            print("    (First occurrence kept, others skipped)")


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate per-document registries into combined registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate refined registries
  python -m fragile.proofs.tools.aggregate_registries \\
      --type refined \\
      --per-document-root registries/per_document \\
      --output registries/combined/refined

  # Aggregate pipeline registries
  python -m fragile.proofs.tools.aggregate_registries \\
      --type pipeline \\
      --per-document-root registries/per_document \\
      --output registries/combined/pipeline
        """,
    )

    parser.add_argument(
        "--type",
        choices=["refined", "pipeline"],
        required=True,
        help="Type of registry to aggregate (refined or pipeline)",
    )

    parser.add_argument(
        "--per-document-root",
        type=Path,
        default=Path("registries/per_document"),
        help="Path to per-document registries root (default: registries/per_document)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for combined registry",
    )

    args = parser.parse_args()

    # Run aggregation
    aggregator = RegistryAggregator(args.per_document_root, args.type)
    aggregator.aggregate_all()

    # Save combined registry
    if (
        aggregator.combined_registry.get_all_objects()
        or aggregator.combined_registry.get_all_theorems()
    ):
        print(f"\n{'=' * 70}")
        print(f"SAVING COMBINED REGISTRY TO: {args.output}")
        print("=" * 70)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_registry_to_directory(aggregator.combined_registry, args.output)
        print(f"\n✓ Combined {args.type} registry saved successfully!")
    else:
        print("\n⚠️  No entities to save (empty registry)")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ AGGREGATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
