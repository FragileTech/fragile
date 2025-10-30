#!/usr/bin/env python3
"""
Validate pipeline_data files and generate diagnostic report.

This tool validates all JSON files in pipeline_data/ against the schemas
in fragile.proofs.core.math_types.py and reports validation errors.

Usage:
    python -m fragile.proofs.tools.validate_pipeline_data \\
        --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \\
        --output-report validation_errors.md
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.core.math_types import (
    Axiom,
    MathematicalObject,
    Parameter,
    TheoremBox,
)


class PipelineDataValidator:
    """Validates pipeline_data files against Pydantic schemas."""

    def __init__(self, pipeline_dir: Path):
        """Initialize validator.

        Args:
            pipeline_dir: Path to pipeline_data directory
        """
        self.pipeline_dir = pipeline_dir
        self.errors: list[dict[str, Any]] = []
        self.stats = defaultdict(int)

    def validate_file(self, file_path: Path, model_class) -> tuple[bool, str | None]:
        """Validate a single JSON file.

        Args:
            file_path: Path to JSON file
            model_class: Pydantic model class to validate against

        Returns:
            (is_valid, error_message) tuple
        """
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Handle source location if present
            if "source" in data and isinstance(data["source"], dict):
                # Check if source has data
                if any(data["source"].values()):
                    data["source"] = SourceLocation.model_validate(data["source"])
                else:
                    data["source"] = None

            # Validate against model
            model_class.model_validate(data)
            return True, None

        except ValidationError as e:
            # Format error message
            error_details = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                msg = error["msg"]
                input_val = error.get("input")
                error_details.append(f"  - {loc}: {msg}")
                if input_val is not None and len(str(input_val)) < 100:
                    error_details.append(f"    Input: {input_val}")

            return False, "\n".join(error_details)

        except Exception as e:
            return False, f"  Unexpected error: {e!s}"

    def validate_directory(self, entity_type: str, model_class) -> None:
        """Validate all files in a directory.

        Args:
            entity_type: Entity type name (e.g., "axioms", "objects")
            model_class: Pydantic model class to validate against
        """
        entity_dir = self.pipeline_dir / entity_type
        if not entity_dir.exists():
            print(f"  ⚠️  Directory not found: {entity_type}/")
            return

        json_files = list(entity_dir.glob("*.json"))
        if not json_files:
            print(f"  ⚠️  No JSON files in {entity_type}/")
            return

        print(f"\n📂 Validating {entity_type}/ ({len(json_files)} files)...")

        valid_count = 0
        error_count = 0

        for json_file in json_files:
            self.stats["total_files"] += 1
            is_valid, error_msg = self.validate_file(json_file, model_class)

            if is_valid:
                valid_count += 1
                self.stats[f"{entity_type}_valid"] += 1
            else:
                error_count += 1
                self.stats[f"{entity_type}_errors"] += 1
                self.errors.append({
                    "file": str(json_file.relative_to(self.pipeline_dir)),
                    "entity_type": entity_type,
                    "error": error_msg,
                })

        print(f"  ✓ Valid: {valid_count}")
        if error_count > 0:
            print(f"  ✗ Errors: {error_count}")

    def validate_all(self) -> None:
        """Validate all entity types."""
        print("=" * 70)
        print("PIPELINE DATA VALIDATION")
        print("=" * 70)
        print(f"\nPipeline directory: {self.pipeline_dir}\n")

        # Validate each entity type
        self.validate_directory("axioms", Axiom)
        self.validate_directory("objects", MathematicalObject)
        self.validate_directory("parameters", Parameter)
        self.validate_directory("theorems", TheoremBox)

    def print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        total = self.stats["total_files"]
        total_valid = sum(v for k, v in self.stats.items() if k.endswith("_valid"))
        total_errors = sum(v for k, v in self.stats.items() if k.endswith("_errors"))

        print(f"\nTotal files validated: {total}")
        print(f"  ✓ Valid:  {total_valid} ({total_valid / total * 100:.1f}%)")
        print(f"  ✗ Errors: {total_errors} ({total_errors / total * 100:.1f}%)")

        # Breakdown by entity type
        entity_types = ["axioms", "objects", "parameters", "theorems"]
        print("\nBreakdown by entity type:")
        for entity_type in entity_types:
            valid = self.stats.get(f"{entity_type}_valid", 0)
            errors = self.stats.get(f"{entity_type}_errors", 0)
            if valid + errors > 0:
                total_type = valid + errors
                print(
                    f"  {entity_type:12s}: {valid:3d} valid, {errors:3d} errors (of {total_type})"
                )

    def generate_markdown_report(self, output_path: Path) -> None:
        """Generate markdown validation report.

        Args:
            output_path: Path to output markdown file
        """
        lines = []

        # Header
        lines.append("# Pipeline Data Validation Report")
        lines.append("")
        lines.append(f"**Pipeline Directory**: `{self.pipeline_dir}`")
        lines.append(f"**Total Files**: {self.stats['total_files']}")
        lines.append(
            f"**Valid Files**: {sum(v for k, v in self.stats.items() if k.endswith('_valid'))}"
        )
        lines.append(f"**Files with Errors**: {len(self.errors)}")
        lines.append("")

        # Summary table
        lines.append("## Summary by Entity Type")
        lines.append("")
        lines.append("| Entity Type | Valid | Errors | Total | Success Rate |")
        lines.append("|-------------|-------|--------|-------|--------------|")

        entity_types = ["axioms", "objects", "parameters", "theorems"]
        for entity_type in entity_types:
            valid = self.stats.get(f"{entity_type}_valid", 0)
            errors = self.stats.get(f"{entity_type}_errors", 0)
            total = valid + errors
            if total > 0:
                rate = valid / total * 100
                lines.append(f"| {entity_type} | {valid} | {errors} | {total} | {rate:.1f}% |")

        lines.append("")

        # Error details
        if self.errors:
            lines.extend((
                "## Validation Errors",
                "",
                f"Found **{len(self.errors)} files** with validation errors:",
                "",
            ))

            # Group errors by type
            errors_by_type = defaultdict(list)
            for error in self.errors:
                errors_by_type[error["entity_type"]].append(error)

            for entity_type in entity_types:
                if entity_type not in errors_by_type:
                    continue

                errors = errors_by_type[entity_type]
                lines.extend((f"### {entity_type.title()} ({len(errors)} errors)", ""))

                for i, error in enumerate(errors, 1):
                    lines.extend((
                        f"#### {i}. `{error['file']}`",
                        "",
                        "```",
                        error["error"],
                        "```",
                        "",
                    ))

            # Common error patterns
            lines.extend(("## Common Error Patterns", ""))

            error_patterns = defaultdict(list)
            for error in self.errors:
                # Extract error type from message
                if "String should have at least 1 character" in error["error"]:
                    error_patterns["Empty string fields"].append(error["file"])
                elif "String should match pattern" in error["error"]:
                    error_patterns["Invalid label format"].append(error["file"])
                elif (
                    "Input should be a valid string" in error["error"] and "None" in error["error"]
                ):
                    error_patterns["Null required field"].append(error["file"])
                elif "Input should be a valid dictionary" in error["error"]:
                    error_patterns["Invalid object type"].append(error["file"])
                elif "Input should be" in error["error"] and "enum" in error["error"].lower():
                    error_patterns["Invalid enum value"].append(error["file"])
                else:
                    error_patterns["Other validation errors"].append(error["file"])

            for pattern, files in sorted(error_patterns.items(), key=lambda x: -len(x[1])):
                lines.extend((f"### {pattern} ({len(files)} files)", ""))
                for file in sorted(files)[:10]:  # Show first 10
                    lines.append(f"- `{file}`")
                if len(files) > 10:
                    lines.append(f"- ... and {len(files) - 10} more")
                lines.append("")

        else:
            lines.extend(("## ✅ All Files Valid", "", "No validation errors found!"))

        # Write report
        output_path.write_text("\n".join(lines))
        print(f"\n📄 Markdown report saved to: {output_path}")

    def generate_json_report(self, output_path: Path) -> None:
        """Generate JSON validation report.

        Args:
            output_path: Path to output JSON file
        """
        report = {
            "pipeline_dir": str(self.pipeline_dir),
            "statistics": dict(self.stats),
            "total_errors": len(self.errors),
            "errors": self.errors,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📄 JSON report saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate pipeline_data files and generate diagnostic report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate and show summary
  python -m fragile.proofs.tools.validate_pipeline_data \\
    --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data

  # Generate detailed markdown report
  python -m fragile.proofs.tools.validate_pipeline_data \\
    --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \\
    --output-report validation_errors.md

  # Generate JSON report for programmatic access
  python -m fragile.proofs.tools.validate_pipeline_data \\
    --pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \\
    --output-json validation_errors.json
        """,
    )

    parser.add_argument(
        "--pipeline-dir", "-p", type=Path, required=True, help="Path to pipeline_data directory"
    )

    parser.add_argument("--output-report", "-r", type=Path, help="Output path for markdown report")

    parser.add_argument("--output-json", "-j", type=Path, help="Output path for JSON report")

    args = parser.parse_args()

    # Validate pipeline directory exists
    if not args.pipeline_dir.exists():
        print(f"Error: Pipeline directory not found: {args.pipeline_dir}")
        sys.exit(1)

    # Run validation
    validator = PipelineDataValidator(args.pipeline_dir)
    validator.validate_all()
    validator.print_summary()

    # Generate reports
    if args.output_report:
        validator.generate_markdown_report(args.output_report)

    if args.output_json:
        validator.generate_json_report(args.output_json)

    # Exit with error code if validation errors found
    if validator.errors:
        print(f"\n❌ Validation completed with {len(validator.errors)} errors")
        sys.exit(1)
    else:
        print("\n✅ All files valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
