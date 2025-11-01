"""Command-line interface for entity validation."""

import argparse
from pathlib import Path
import sys

from mathster.tools.validation.entity_validators import (
    AxiomValidator,
    EquationValidator,
    ObjectValidator,
    ParameterValidator,
    ProofValidator,
    RemarkValidator,
    TheoremValidator,
)
from mathster.tools.validation.framework_validator import FrameworkValidator
from mathster.tools.validation.relationship_validator import RelationshipValidator
from mathster.tools.validation.schema_validator import SchemaValidator
from mathster.tools.validation.validation_report import ValidationReport


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate refined mathematical entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Schema validation only (fast)
  python -m fragile.mathster.tools.validation \\
    --refined-dir docs/source/.../refined_data/ \\
    --mode schema

  # Complete validation (includes relationships and framework)
  python -m fragile.mathster.tools.validation \\
    --refined-dir docs/source/.../refined_data/ \\
    --mode complete \\
    --output-report validation_report.md

  # Validate specific entity types
  python -m fragile.mathster.tools.validation \\
    --refined-dir docs/source/.../refined_data/ \\
    --entity-types theorems axioms \\
    --mode schema

  # Strict mode (warnings as errors)
  python -m fragile.mathster.tools.validation \\
    --refined-dir docs/source/.../refined_data/ \\
    --mode complete \\
    --strict
        """,
    )

    parser.add_argument(
        "--refined-dir",
        type=Path,
        required=True,
        help="Path to refined_data directory",
    )

    parser.add_argument(
        "--mode",
        choices=["schema", "relationships", "framework", "complete"],
        default="schema",
        help="Validation mode (default: schema)",
    )

    parser.add_argument(
        "--entity-types",
        nargs="+",
        choices=["theorems", "axioms", "objects", "parameters", "mathster", "remarks", "equations"],
        help="Specific entity types to validate (default: all)",
    )

    parser.add_argument(
        "--output-report",
        type=Path,
        help="Output path for validation report (markdown)",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output path for validation report (JSON)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("docs/glossary.md"),
        help="Path to glossary.md for framework validation",
    )

    parser.add_argument(
        "--validation-mode",
        choices=["refined", "pipeline"],
        default="refined",
        help="Schema validation mode: 'refined' for refined_data/, 'pipeline' for pipeline_data/ (default: refined)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.refined_dir.exists():
        print(f"Error: refined_dir does not exist: {args.refined_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine entity types to validate
    entity_types = args.entity_types or [
        "theorems",
        "axioms",
        "objects",
        "parameters",
        "mathster",
        "remarks",
        "equations",
    ]

    # Run validation
    print(f"Running validation in '{args.mode}' mode...")
    print(f"Entity types: {', '.join(entity_types)}")
    print(f"Refined directory: {args.refined_dir}")
    print()

    if args.mode == "schema":
        result = validate_schema(args.refined_dir, entity_types, args.strict, args.validation_mode)
    elif args.mode == "relationships":
        result = validate_relationships(args.refined_dir, entity_types, args.strict)
    elif args.mode == "framework":
        result = validate_framework(args.refined_dir, entity_types, args.strict, args.glossary)
    elif args.mode == "complete":
        result = validate_complete(args.refined_dir, entity_types, args.strict, args.glossary)
    else:
        print(f"Error: Unknown mode '{args.mode}'", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report = ValidationReport(result, args.refined_dir, args.mode)

    # Print summary to console
    report.print_summary()

    # Save reports if requested
    if args.output_report:
        report.save_markdown(args.output_report)
        print(f"Markdown report saved to: {args.output_report}")

    if args.output_json:
        report.save_json(args.output_json)
        print(f"JSON report saved to: {args.output_json}")

    # Exit with appropriate code
    sys.exit(0 if result.is_valid else 1)


def validate_schema(
    refined_dir: Path, entity_types: list[str], strict: bool, validation_mode: str = "refined"
):
    """Run schema validation.

    Args:
        refined_dir: Directory containing entity files
        entity_types: List of entity types to validate
        strict: Whether to treat warnings as errors
        validation_mode: "refined" for refined_data/, "pipeline" for pipeline_data/
    """
    from mathster.tools.validation.base_validator import ValidationResult

    aggregated_result = ValidationResult(is_valid=True, entity_count=0)

    # Use SchemaValidator with appropriate mode
    print(f"  Using {validation_mode} schema validation mode")
    schema_validator = SchemaValidator(strict=strict, mode=validation_mode)

    for entity_type in entity_types:
        entity_dir = refined_dir / entity_type
        if not entity_dir.exists():
            print(f"  Skipping {entity_type}: directory not found")
            continue

        print(f"  Validating {entity_type}...", end=" ")
        result = schema_validator.validate_directory(entity_dir, pattern="*.json")
        print(
            f"{result.entity_count} entities, {len(result.errors)} errors, {len(result.warnings)} warnings"
        )

        aggregated_result.merge(result)

    return aggregated_result


def validate_relationships(refined_dir: Path, entity_types: list[str], strict: bool):
    """Run relationship validation."""
    from mathster.tools.validation.base_validator import ValidationResult

    print("  Building entity registry...")
    validator = RelationshipValidator(strict=strict, refined_dir=refined_dir)

    print(f"  Loaded {len(validator.entity_registry)} entities")

    aggregated_result = ValidationResult(is_valid=True, entity_count=0)

    for entity_type in entity_types:
        entity_dir = refined_dir / entity_type
        if not entity_dir.exists():
            continue

        print(f"  Validating {entity_type} relationships...", end=" ")
        result = validator.validate_directory(entity_dir)
        aggregated_result.merge(result)
        print(
            f"{result.entity_count} entities, {result.error_count} errors, {result.warning_count} warnings"
        )

    # Check for circular dependencies
    print("  Checking for circular dependencies...")
    cycles = validator.detect_circular_dependencies()
    if cycles:
        print(f"    Found {len(cycles)} circular dependency chains!")
        for cycle in cycles[:3]:  # Show first 3
            print(f"      {' → '.join(cycle)}")
        aggregated_result.metadata["circular_dependencies"] = len(cycles)

    return aggregated_result


def validate_framework(
    refined_dir: Path, entity_types: list[str], strict: bool, glossary_path: Path
):
    """Run framework consistency validation."""
    from mathster.tools.validation.base_validator import ValidationResult

    print(f"  Loading glossary from: {glossary_path}")
    validator = FrameworkValidator(strict=strict, glossary_path=glossary_path)

    aggregated_result = ValidationResult(is_valid=True, entity_count=0)

    for entity_type in entity_types:
        entity_dir = refined_dir / entity_type
        if not entity_dir.exists():
            continue

        print(f"  Validating {entity_type} framework consistency...", end=" ")
        result = validator.validate_directory(entity_dir)
        aggregated_result.merge(result)
        print(
            f"{result.entity_count} entities, {result.error_count} errors, {result.warning_count} warnings"
        )

    return aggregated_result


def validate_complete(
    refined_dir: Path, entity_types: list[str], strict: bool, glossary_path: Path
):
    """Run complete validation (schema + relationships + framework)."""
    from mathster.tools.validation.base_validator import ValidationResult

    print("\n=== PHASE 1: Schema Validation ===\n")
    schema_result = validate_schema(refined_dir, entity_types, strict)

    print("\n=== PHASE 2: Relationship Validation ===\n")
    relationship_result = validate_relationships(refined_dir, entity_types, strict)

    print("\n=== PHASE 3: Framework Validation ===\n")
    framework_result = validate_framework(refined_dir, entity_types, strict, glossary_path)

    # Merge all results
    aggregated_result = ValidationResult(is_valid=True, entity_count=0)
    aggregated_result.merge(schema_result)
    aggregated_result.merge(relationship_result)
    aggregated_result.merge(framework_result)

    return aggregated_result


if __name__ == "__main__":
    main()
