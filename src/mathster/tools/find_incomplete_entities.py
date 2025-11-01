"""Find incomplete entities in refined_data directory.

This tool scans refined entities and identifies those with missing or incomplete fields,
outputting a JSON file that can be used by complete_refinement.py to fix them.
"""

import argparse
from collections import defaultdict
import json
import operator
from pathlib import Path
import sys
from typing import Any

from mathster.tools.validation import (
    AxiomValidator,
    EquationValidator,
    ObjectValidator,
    ParameterValidator,
    ProofValidator,
    RemarkValidator,
    TheoremValidator,
)


def find_incomplete_entities(refined_dir: Path, output_file: Path) -> dict[str, Any]:
    """Find incomplete entities in refined_data directory.

    Args:
        refined_dir: Path to refined_data directory
        output_file: Path to output JSON file

    Returns:
        Dictionary with incomplete entity information
    """
    print(f"Scanning refined data directory: {refined_dir}")
    print()

    # Map entity types to their validators
    validator_map = {
        "theorems": TheoremValidator(strict=False),
        "axioms": AxiomValidator(strict=False),
        "objects": ObjectValidator(strict=False),
        "parameters": ParameterValidator(strict=False),
        "mathster": ProofValidator(strict=False),
        "remarks": RemarkValidator(strict=False),
        "equations": EquationValidator(strict=False),
    }

    # Track incomplete entities
    incomplete_entities: dict[str, list] = defaultdict(list)
    statistics = {
        "total_entities": 0,
        "incomplete_entities": 0,
        "entities_by_type": {},
        "missing_fields_count": defaultdict(int),
        "warning_count": defaultdict(int),
    }

    # Scan each entity type
    for entity_type, validator in validator_map.items():
        entity_dir = refined_dir / entity_type
        if not entity_dir.exists():
            print(f"  Skipping {entity_type}: directory not found")
            continue

        json_files = list(entity_dir.glob("*.json"))
        if not json_files:
            print(f"  Skipping {entity_type}: no JSON files")
            continue

        print(f"  Scanning {entity_type}: {len(json_files)} files...", end=" ")

        entity_incomplete_count = 0
        for json_file in json_files:
            statistics["total_entities"] += 1

            # Validate entity
            result = validator.validate_file(json_file)

            # Check if entity is incomplete
            if result.errors or result.warnings:
                # Load entity data
                with open(json_file) as f:
                    data = json.load(f)

                label = data.get("label") or data.get("proof_id") or json_file.stem

                incomplete_info = {
                    "file": str(json_file.relative_to(refined_dir)),
                    "label": label,
                    "entity_type": entity_type.rstrip("s"),
                    "errors": [],
                    "warnings": [],
                    "missing_fields": [],
                    "data": data,  # Include current data for reference
                }

                # Collect errors
                for error in result.errors:
                    incomplete_info["errors"].append({
                        "field": error.field,
                        "message": error.message,
                        "severity": error.severity,
                    })
                    if error.field:
                        incomplete_info["missing_fields"].append(error.field)
                        statistics["missing_fields_count"][error.field] += 1

                # Collect warnings
                for warning in result.warnings:
                    incomplete_info["warnings"].append({
                        "field": warning.field,
                        "message": warning.message,
                        "suggestion": warning.suggestion,
                    })
                    if warning.field:
                        statistics["warning_count"][warning.field] += 1

                incomplete_entities[entity_type].append(incomplete_info)
                entity_incomplete_count += 1
                statistics["incomplete_entities"] += 1

        statistics["entities_by_type"][entity_type] = {
            "total": len(json_files),
            "incomplete": entity_incomplete_count,
            "complete": len(json_files) - entity_incomplete_count,
        }

        print(f"{entity_incomplete_count} incomplete")

    # Generate summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total entities scanned: {statistics['total_entities']}")
    print(f"Incomplete entities: {statistics['incomplete_entities']}")
    print(
        f"Completion rate: {(statistics['total_entities'] - statistics['incomplete_entities']) / statistics['total_entities'] * 100:.1f}%"
    )
    print()

    if statistics["missing_fields_count"]:
        print("Top missing fields:")
        sorted_fields = sorted(
            statistics["missing_fields_count"].items(), key=operator.itemgetter(1), reverse=True
        )
        for field, count in sorted_fields[:10]:
            print(f"  - {field}: {count} entities")
        print()

    if statistics["warning_count"]:
        print("Top warning fields:")
        sorted_warnings = sorted(
            statistics["warning_count"].items(), key=operator.itemgetter(1), reverse=True
        )
        for field, count in sorted_warnings[:10]:
            print(f"  - {field}: {count} entities")
        print()

    # Save to JSON
    output_data = {
        "refined_dir": str(refined_dir),
        "statistics": {
            "total_entities": statistics["total_entities"],
            "incomplete_entities": statistics["incomplete_entities"],
            "completion_rate": (
                (statistics["total_entities"] - statistics["incomplete_entities"])
                / statistics["total_entities"]
                * 100
                if statistics["total_entities"] > 0
                else 0
            ),
            "entities_by_type": statistics["entities_by_type"],
            "missing_fields_count": dict(statistics["missing_fields_count"]),
            "warning_count": dict(statistics["warning_count"]),
        },
        "incomplete_entities": dict(incomplete_entities),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Incomplete entities report saved to: {output_file}")
    print()

    return output_data


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Find incomplete entities in refined_data directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find incomplete entities
  python -m fragile.mathster.tools.find_incomplete_entities \\
    --refined-dir docs/source/.../refined_data/

  # Specify output file
  python -m fragile.mathster.tools.find_incomplete_entities \\
    --refined-dir docs/source/.../refined_data/ \\
    --output incomplete_entities.json
        """,
    )

    parser.add_argument(
        "--refined-dir",
        type=Path,
        required=True,
        help="Path to refined_data directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("incomplete_entities.json"),
        help="Output JSON file path (default: incomplete_entities.json)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.refined_dir.exists():
        print(f"Error: refined_dir does not exist: {args.refined_dir}", file=sys.stderr)
        sys.exit(1)

    # Find incomplete entities
    try:
        result = find_incomplete_entities(args.refined_dir, args.output)

        # Exit with status based on results
        if result["statistics"]["incomplete_entities"] > 0:
            print(f"Found {result['statistics']['incomplete_entities']} incomplete entities.")
            print("Use complete_refinement.py to fix them automatically.")
            sys.exit(0)  # Not an error - just informational
        else:
            print("✅ All entities are complete!")
            sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
