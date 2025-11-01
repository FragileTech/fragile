#!/usr/bin/env python3
"""
Build Registry from Pipeline-Ready JSON Files.

This is the fastest registry builder - it loads pre-validated pipeline JSON files
that have already been transformed to math_types format.

Use this when:
- Loading from pipeline_data/ directories (output of enriched_to_math_types.py)
- Loading from previously saved registries
- You need fastest loading performance

Usage:
    python -m fragile.mathster.tools.build_pipeline_registry \\
        --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \\
        --output pipeline_registry

Version: 1.0.0
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set

from mathster.core.article_system import SourceLocation
from mathster.core.math_types import (
    Attribute,
    Axiom,
    MathematicalObject,
    Parameter,
    Relationship,
    TheoremBox,
)
from mathster.registry.registry import MathematicalRegistry
from mathster.registry.storage import save_registry_to_directory


# ============================================================================
# PIPELINE DATA LOADER
# ============================================================================


class PipelineDataLoader:
    """Loads pre-validated pipeline JSON into registry."""

    def __init__(self, pipeline_dir: Path):
        """Initialize loader.

        Args:
            pipeline_dir: Path to pipeline_data directory
        """
        self.pipeline_dir = pipeline_dir
        self.registry = MathematicalRegistry()
        self.stats = {
            "objects": 0,
            "axioms": 0,
            "parameters": 0,
            "theorems": 0,
            "relationships": 0,
            "errors": [],
        }

    def load_objects(self):
        """Load all MathematicalObject JSON files."""
        objects_dir = self.pipeline_dir / "objects"
        if not objects_dir.exists():
            print(f"⚠️  Objects directory not found: {objects_dir}")
            return

        print(f"\nLoading objects from: {objects_dir.relative_to(self.pipeline_dir.parent)}")
        json_files = list(objects_dir.glob("*.json"))
        print(f"Found {len(json_files)} files")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle source location
                if "source" in data and isinstance(data["source"], dict):
                    data["source"] = SourceLocation.model_validate(data["source"])

                obj = MathematicalObject.model_validate(data)
                self.registry.add(obj)
                self.stats["objects"] += 1

            except Exception as e:
                error_msg = f"Error loading {json_file.name}: {e}"
                print(f"  ✗ {error_msg}")
                self.stats["errors"].append(error_msg)

    def load_axioms(self):
        """Load all Axiom JSON files."""
        axioms_dir = self.pipeline_dir / "axioms"
        if not axioms_dir.exists():
            print(f"⚠️  Axioms directory not found: {axioms_dir}")
            return

        print(f"\nLoading axioms from: {axioms_dir.relative_to(self.pipeline_dir.parent)}")
        json_files = list(axioms_dir.glob("*.json"))
        print(f"Found {len(json_files)} files")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle source location
                if "source" in data and isinstance(data["source"], dict):
                    data["source"] = SourceLocation.model_validate(data["source"])

                axiom = Axiom.model_validate(data)
                self.registry.add(axiom)
                self.stats["axioms"] += 1

            except Exception as e:
                error_msg = f"Error loading {json_file.name}: {e}"
                print(f"  ✗ {error_msg}")
                self.stats["errors"].append(error_msg)

    def load_parameters(self):
        """Load all Parameter JSON files."""
        params_dir = self.pipeline_dir / "parameters"
        if not params_dir.exists():
            print(f"⚠️  Parameters directory not found: {params_dir}")
            return

        print(f"\nLoading parameters from: {params_dir.relative_to(self.pipeline_dir.parent)}")
        json_files = list(params_dir.glob("*.json"))
        print(f"Found {len(json_files)} files")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                param = Parameter.model_validate(data)
                self.registry.add(param)
                self.stats["parameters"] += 1

            except Exception as e:
                error_msg = f"Error loading {json_file.name}: {e}"
                print(f"  ✗ {error_msg}")
                self.stats["errors"].append(error_msg)

    def load_theorems(self):
        """Load all TheoremBox JSON files."""
        theorems_dir = self.pipeline_dir / "theorems"
        if not theorems_dir.exists():
            print(f"⚠️  Theorems directory not found: {theorems_dir}")
            return

        print(f"\nLoading theorems from: {theorems_dir.relative_to(self.pipeline_dir.parent)}")
        json_files = list(theorems_dir.glob("*.json"))
        print(f"Found {len(json_files)} files")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle source location
                if "source" in data and isinstance(data["source"], dict):
                    data["source"] = SourceLocation.model_validate(data["source"])

                # Handle attributes_added
                if "attributes_added" in data:
                    attributes = []
                    for attr_data in data["attributes_added"]:
                        if isinstance(attr_data, dict):
                            if "source" in attr_data and isinstance(attr_data["source"], dict):
                                attr_data["source"] = SourceLocation.model_validate(
                                    attr_data["source"]
                                )
                            attributes.append(Attribute.model_validate(attr_data))
                    data["attributes_added"] = attributes

                # Handle relations_established
                if "relations_established" in data:
                    relations = []
                    for rel_data in data["relations_established"]:
                        if isinstance(rel_data, dict):
                            if "source" in rel_data and isinstance(rel_data["source"], dict):
                                rel_data["source"] = SourceLocation.model_validate(
                                    rel_data["source"]
                                )
                            rel = Relationship.model_validate(rel_data)
                            relations.append(rel)
                            # Add relationship to registry
                            try:
                                self.registry.add(rel)
                                self.stats["relationships"] += 1
                            except ValueError:
                                # Relationship might already exist
                                pass
                    data["relations_established"] = relations

                thm = TheoremBox.model_validate(data)
                self.registry.add(thm)
                self.stats["theorems"] += 1

            except Exception as e:
                error_msg = f"Error loading {json_file.name}: {e}"
                print(f"  ✗ {error_msg}")
                self.stats["errors"].append(error_msg)

    def load_relationships(self):
        """Load standalone Relationship JSON files (if any)."""
        rels_dir = self.pipeline_dir / "relationships"
        if not rels_dir.exists():
            # Relationships directory is optional
            return

        print(f"\nLoading relationships from: {rels_dir.relative_to(self.pipeline_dir.parent)}")
        json_files = list(rels_dir.glob("*.json"))
        if not json_files:
            return

        print(f"Found {len(json_files)} files")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle source location
                if "source" in data and isinstance(data["source"], dict):
                    data["source"] = SourceLocation.model_validate(data["source"])

                rel = Relationship.model_validate(data)
                try:
                    self.registry.add(rel)
                    self.stats["relationships"] += 1
                except ValueError:
                    # Relationship might already be added from theorems
                    pass

            except Exception as e:
                error_msg = f"Error loading {json_file.name}: {e}"
                print(f"  ✗ {error_msg}")
                self.stats["errors"].append(error_msg)

    def load_all(self):
        """Load all pipeline JSON files."""
        print("=" * 70)
        print("LOADING PIPELINE DATA INTO REGISTRY")
        print("=" * 70)
        print(f"\nPipeline directory: {self.pipeline_dir}")

        # Load in order: objects, axioms, parameters, then theorems
        # (theorems depend on objects/axioms/parameters)
        self.load_objects()
        self.load_axioms()
        self.load_parameters()
        self.load_theorems()
        self.load_relationships()

        # Print statistics
        self.print_statistics()

    def print_statistics(self):
        """Print final statistics."""
        print("\n" + "=" * 70)
        print("LOADING STATISTICS")
        print("=" * 70)

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

        if self.stats["errors"]:
            print(f"\n⚠️  Errors Encountered: {len(self.stats['errors'])}")
            print("\nFirst 10 errors:")
            for error in self.stats["errors"][:10]:
                print(f"  - {error}")
            if len(self.stats["errors"]) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more errors")

    def save_registry(self, output_dir: Path):
        """Save registry to directory.

        Args:
            output_dir: Output directory for registry
        """
        print("\n" + "=" * 70)
        print(f"SAVING REGISTRY TO: {output_dir}")
        print("=" * 70)

        save_registry_to_directory(self.registry, output_dir)

        print("\n✓ Registry saved successfully!")
        print("\nTo visualize with dashboard:")
        print("  panel serve src/fragile/mathster/proof_pipeline_dashboard.py --show")
        print("\nTo regenerate:")
        print("  python -m fragile.mathster.tools.build_pipeline_registry \\")
        print(f"    --pipeline-dir {self.pipeline_dir} \\")
        print(f"    --output {output_dir}")
        print("\nOr load programmatically:")
        print("  from fragile.mathster import load_registry_from_directory, MathematicalRegistry")
        print(f"  registry = load_registry_from_directory(MathematicalRegistry, '{output_dir}')")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build registry from pipeline-ready JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build registry from Chapter 1 pipeline data
  python -m fragile.mathster.tools.build_pipeline_registry \\
    --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \\
    --output pipeline_registry

  # Build from multiple chapters (run separately and merge)
  python -m fragile.mathster.tools.build_pipeline_registry \\
    --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \\
    --output chapter1_registry
        """,
    )
    parser.add_argument(
        "--pipeline-dir",
        "-p",
        type=str,
        required=True,
        help="Path to pipeline_data/ directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="pipeline_registry",
        help="Output directory name (default: pipeline_registry)",
    )

    args = parser.parse_args()

    # Setup paths
    pipeline_dir = Path(args.pipeline_dir)
    output_dir = Path(args.output)

    if not pipeline_dir.exists():
        print(f"Error: Pipeline directory not found: {pipeline_dir}")
        sys.exit(1)

    # Create loader and process
    loader = PipelineDataLoader(pipeline_dir)
    loader.load_all()

    # Save registry
    loader.save_registry(output_dir)

    # Exit with error code if there were errors
    if loader.stats["errors"]:
        print(f"\n❌ Loading completed with {len(loader.stats['errors'])} errors")
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("✅ COMPLETE!")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
