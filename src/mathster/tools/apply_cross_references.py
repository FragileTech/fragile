#!/usr/bin/env python3
"""
Apply cross-reference analysis results to entity JSON files.

Reads analysis results and updates the JSON files with:
- input_objects
- input_axioms
- input_parameters
- output_type
- relations_established
"""

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Dict, Optional


def apply_analysis_to_entity(entity_file: Path, analysis: dict) -> bool:
    """Apply analysis results to entity file."""

    # Load entity
    with open(entity_file) as f:
        entity = json.load(f)

    # Initialize fields if missing
    if "input_objects" not in entity:
        entity["input_objects"] = []
    if "input_axioms" not in entity:
        entity["input_axioms"] = []
    if "input_parameters" not in entity:
        entity["input_parameters"] = []
    if "output_type" not in entity:
        entity["output_type"] = None
    if "relations_established" not in entity:
        entity["relations_established"] = []

    # Merge with existing (preserve existing data, add new)
    entity["input_objects"] = list(
        set(entity.get("input_objects", []) + analysis.get("input_objects", []))
    )
    entity["input_axioms"] = list(
        set(entity.get("input_axioms", []) + analysis.get("input_axioms", []))
    )
    entity["input_parameters"] = list(
        set(entity.get("input_parameters", []) + analysis.get("input_parameters", []))
    )

    # Update output_type if provided and not already set
    if analysis.get("output_type") and not entity.get("output_type"):
        entity["output_type"] = analysis["output_type"]

    # Merge relations_established
    entity["relations_established"] = list(
        set(entity.get("relations_established", []) + analysis.get("relations_established", []))
    )

    # Save updated entity
    with open(entity_file, "w") as f:
        json.dump(entity, f, indent=2)

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Apply cross-reference analysis")
    parser.add_argument("results_file", type=Path, help="JSON file with analysis results")
    parser.add_argument("raw_data_dir", type=Path, help="Path to raw_data directory")

    args = parser.parse_args()

    # Load results
    with open(args.results_file) as f:
        results = json.load(f)

    raw_data_dir = Path(args.raw_data_dir)

    print(f"Applying {len(results)} analysis results...")
    print()

    applied = 0
    errors = 0
    stats = defaultdict(int)

    for label, analysis in results.items():
        # Find entity file
        entity_file = None

        for subdir in ["theorems", "lemmas", "propositions", "corollaries"]:
            candidate = raw_data_dir / subdir / f"{label}.json"
            if candidate.exists():
                entity_file = candidate
                break

        if not entity_file:
            print(f"  ERROR: Entity file not found for {label}")
            errors += 1
            continue

        try:
            # Apply analysis
            if apply_analysis_to_entity(entity_file, analysis):
                applied += 1

                # Track statistics
                stats["objects"] += len(analysis.get("input_objects", []))
                stats["axioms"] += len(analysis.get("input_axioms", []))
                stats["parameters"] += len(analysis.get("input_parameters", []))
                stats["relations"] += len(analysis.get("relations_established", []))

                print(f"  ✓ {label}")

        except Exception as e:
            print(f"  ERROR applying {label}: {e}")
            errors += 1

    print()
    print("=" * 80)
    print(f"Applied: {applied}")
    print(f"Errors: {errors}")
    print()
    print("Dependencies filled:")
    print(f"  Objects: {stats['objects']}")
    print(f"  Axioms: {stats['axioms']}")
    print(f"  Parameters: {stats['parameters']}")
    print(f"  Relations: {stats['relations']}")


if __name__ == "__main__":
    main()
