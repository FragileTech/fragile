#!/usr/bin/env python3
"""
Comprehensive Source Location Validation Tool.

Performs strict validation of source_location fields in entity JSON files.

Requirements (User Specified):
- document_id: Required, must match pattern [0-9]{2}_[a-z_]+
- file_path: Required, must be valid path
- section: Required, must be X.Y.Z format (only numbers and dots, NO symbols/text)
- line_range: Required, must be [int, int] with start >= 1, start <= end
- directive_label: Required if entity has label_text
- equation: Optional (can be None)
- url_fragment: Optional (can be None)

Usage:
    # Validate single file
    python scripts/validate_all_source_locations.py --file entity.json

    # Validate directory
    python scripts/validate_all_source_locations.py --directory raw_data/

    # Validate all documents
    python scripts/validate_all_source_locations.py --all-documents docs/source/

    # Strict mode (exit with error if any validation fails)
    python scripts/validate_all_source_locations.py --all-documents docs/source/ --strict
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys


@dataclass
class ValidationError:
    """Single validation error."""

    field: str
    issue_type: str  # MISSING, INVALID, OUT_OF_BOUNDS
    message: str
    current_value: str | None = None


@dataclass
class ValidationResult:
    """Result of validating a single entity."""

    file_path: Path
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    def add_error(
        self, field: str, issue_type: str, message: str, current_value: str | None = None
    ):
        """Add a validation error."""
        self.errors.append(ValidationError(field, issue_type, message, current_value))
        self.valid = False


def validate_source_location(
    entity: dict, file_path: Path, markdown_path: Path | None = None
) -> ValidationResult:
    """
    Strictly validate source_location field against requirements.

    Args:
        entity: Entity dictionary
        file_path: Path to entity JSON file
        markdown_path: Optional path to markdown file for line_range validation

    Returns:
        ValidationResult with detailed errors for ALL missing/invalid attributes
    """
    result = ValidationResult(file_path=file_path, valid=True)

    # Check source_location field exists
    if "source_location" not in entity:
        result.add_error("source_location", "MISSING", "source_location field missing entirely")
        return result

    loc = entity["source_location"]

    if loc is None:
        result.add_error("source_location", "MISSING", "source_location is null")
        return result

    # 1. VALIDATE DOCUMENT_ID (Required)
    document_id = loc.get("document_id")
    if not document_id:
        result.add_error("document_id", "MISSING", "document_id is missing or empty")
    elif not isinstance(document_id, str):
        result.add_error(
            "document_id",
            "INVALID",
            f"document_id must be string, got {type(document_id)}",
            str(document_id),
        )
    elif not re.match(r"^[0-9]{2}_[a-z_]+$", document_id):
        result.add_error(
            "document_id",
            "INVALID",
            "document_id must match pattern [0-9]{2}_[a-z_]+",
            document_id,
        )

    # 2. VALIDATE FILE_PATH (Required)
    file_path_val = loc.get("file_path")
    if not file_path_val:
        result.add_error("file_path", "MISSING", "file_path is missing or empty")
    elif not isinstance(file_path_val, str):
        result.add_error(
            "file_path",
            "INVALID",
            f"file_path must be string, got {type(file_path_val)}",
            str(file_path_val),
        )

    # 3. VALIDATE SECTION (Required, strict X.Y.Z format)
    section = loc.get("section")
    if section is None or section == "":
        result.add_error("section", "MISSING", "section is missing or empty (REQUIRED)")
    elif not isinstance(section, str):
        result.add_error(
            "section", "INVALID", f"section must be string, got {type(section)}", str(section)
        )
    elif not re.match(r"^\d+(\.\d+)*$", section):
        # Check for common invalid patterns
        issues = []
        if "§" in section:
            issues.append("contains § symbol")
        if section.lower().startswith("section "):
            issues.append('contains "Section " prefix')
        if ". " in section or ": " in section:
            issues.append("contains title after dot/colon")
        if not section.replace(".", "").isdigit():
            issues.append("contains non-numeric characters")

        issue_desc = ", ".join(issues) if issues else "invalid format"
        result.add_error(
            "section",
            "INVALID",
            f"section must be X.Y.Z format (only digits and dots), {issue_desc}",
            section,
        )

    # 4. VALIDATE LINE_RANGE (Required, strict [int, int] format)
    line_range = loc.get("line_range")
    if line_range is None:
        result.add_error("line_range", "MISSING", "line_range is missing (REQUIRED)")
    elif not isinstance(line_range, list):
        result.add_error(
            "line_range",
            "INVALID",
            f"line_range must be list, got {type(line_range)}",
            str(line_range),
        )
    elif len(line_range) != 2:
        result.add_error(
            "line_range",
            "INVALID",
            f"line_range must have exactly 2 elements, got {len(line_range)}",
            str(line_range),
        )
    # Validate elements
    elif not all(isinstance(x, int) for x in line_range):
        types = [type(x).__name__ for x in line_range]
        result.add_error(
            "line_range",
            "INVALID",
            f"line_range must contain integers, got {types}",
            str(line_range),
        )
    else:
        start, end = line_range
        if start < 1:
            result.add_error(
                "line_range",
                "INVALID",
                f"line_range start must be >= 1, got {start}",
                str(line_range),
            )
        if end < start:
            result.add_error(
                "line_range",
                "INVALID",
                f"line_range end must be >= start, got start={start}, end={end}",
                str(line_range),
            )

        # Validate against markdown file if provided
        if markdown_path and markdown_path.exists():
            try:
                markdown_content = markdown_path.read_text(encoding="utf-8")
                max_lines = len(markdown_content.splitlines())
                if end > max_lines:
                    result.add_error(
                        "line_range",
                        "OUT_OF_BOUNDS",
                        f"line_range end {end} exceeds file length {max_lines}",
                        str(line_range),
                    )
            except Exception:
                # Can't validate, but don't fail validation
                pass

    # 5. VALIDATE DIRECTIVE_LABEL (Required if entity has label_text)
    directive_label = loc.get("directive_label")
    entity_has_label = bool(entity.get("label_text"))

    if entity_has_label:
        if not directive_label:
            result.add_error(
                "directive_label",
                "MISSING",
                f'directive_label is missing but entity has label_text: {entity.get("label_text")}',
            )
        elif not isinstance(directive_label, str):
            result.add_error(
                "directive_label",
                "INVALID",
                f"directive_label must be string, got {type(directive_label)}",
                str(directive_label),
            )
        elif not re.match(r"^[a-z][a-z0-9-]*$", directive_label):
            result.add_error(
                "directive_label",
                "INVALID",
                "directive_label must match pattern [a-z][a-z0-9-]*",
                directive_label,
            )

    # 6. VALIDATE EQUATION (Optional, but if present must be string)
    equation = loc.get("equation")
    if equation is not None and not isinstance(equation, str):
        result.add_error(
            "equation",
            "INVALID",
            f"equation must be string or None, got {type(equation)}",
            str(equation),
        )

    # 7. VALIDATE URL_FRAGMENT (Optional, but if present must be string)
    url_fragment = loc.get("url_fragment")
    if url_fragment is not None and not isinstance(url_fragment, str):
        result.add_error(
            "url_fragment",
            "INVALID",
            f"url_fragment must be string or None, got {type(url_fragment)}",
            str(url_fragment),
        )

    return result


def validate_directory(directory: Path, markdown_path: Path | None = None) -> dict:
    """
    Validate all JSON files in a directory.

    Args:
        directory: Directory to scan
        markdown_path: Optional markdown file for line_range validation

    Returns:
        Statistics dictionary
    """
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors_by_field": defaultdict(int),
        "errors_by_type": defaultdict(int),
        "invalid_files": [],
    }

    results = []

    for json_file in directory.rglob("*.json"):
        # Skip report files
        if "report" in json_file.name.lower():
            continue

        stats["total"] += 1

        try:
            with open(json_file, encoding="utf-8") as f:
                entity = json.load(f)

            result = validate_source_location(entity, json_file, markdown_path)
            results.append(result)

            if result.valid:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
                stats["invalid_files"].append(result)

                for error in result.errors:
                    stats["errors_by_field"][error.field] += 1
                    stats["errors_by_type"][error.issue_type] += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            stats["invalid"] += 1

    return stats, results


def print_validation_report(
    stats: dict, results: list[ValidationResult], show_details: bool = True, max_errors: int = 20
):
    """Print comprehensive validation report."""
    print(f"\n{'=' * 80}")
    print("SOURCE LOCATION VALIDATION REPORT")
    print(f"{'=' * 80}")
    print(f"Total files validated: {stats['total']}")
    print(
        f"  ✓ Valid:             {stats['valid']} ({100 * stats['valid'] / stats['total']:.1f}%)"
    )
    print(
        f"  ✗ Invalid:           {stats['invalid']} ({100 * stats['invalid'] / stats['total']:.1f}%)"
    )

    if stats["invalid"] > 0:
        print(f"\n{'─' * 80}")
        print("ERRORS BY FIELD:")
        print(f"{'─' * 80}")
        for field, count in sorted(stats["errors_by_field"].items(), key=lambda x: -x[1]):
            print(f"  {field:20} {count:5} errors")

        print(f"\n{'─' * 80}")
        print("ERRORS BY TYPE:")
        print(f"{'─' * 80}")
        for error_type, count in sorted(stats["errors_by_type"].items(), key=lambda x: -x[1]):
            print(f"  {error_type:20} {count:5} errors")

        if show_details and stats["invalid_files"]:
            print(f"\n{'─' * 80}")
            print(f"DETAILED ERRORS (showing first {max_errors}):")
            print(f"{'─' * 80}")

            for i, result in enumerate(stats["invalid_files"][:max_errors], 1):
                print(f"\n{i}. {result.file_path.name}")
                for error in result.errors:
                    symbol = "✗" if error.issue_type in {"MISSING", "INVALID"} else "⚠"
                    print(f"   {symbol} {error.field}: {error.issue_type}")
                    print(f"      {error.message}")
                    if error.current_value:
                        print(f"      Current value: '{error.current_value}'")

            if stats["invalid"] > max_errors:
                print(f"\n... and {stats['invalid'] - max_errors} more files with errors")

    print(f"\n{'=' * 80}")
    if stats["valid"] == stats["total"]:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print(f"✗ {stats['invalid']} FILES NEED FIXING")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate source_location fields with strict requirements"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=Path, help="Validate single JSON file")
    group.add_argument("--directory", "-d", type=Path, help="Validate all JSON files in directory")
    group.add_argument(
        "--all-documents", "-a", type=Path, help="Validate all documents in docs/source/"
    )

    parser.add_argument(
        "--markdown", "-m", type=Path, help="Markdown file for line_range validation"
    )
    parser.add_argument(
        "--strict", "-s", action="store_true", help="Exit with error if any validation fails"
    )
    parser.add_argument(
        "--show-details", action="store_true", default=True, help="Show detailed error messages"
    )
    parser.add_argument(
        "--max-errors", type=int, default=20, help="Maximum errors to show in detail"
    )
    parser.add_argument("--output", "-o", type=Path, help="Output JSON report to file")

    args = parser.parse_args()

    if args.file:
        # Validate single file
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        with open(args.file, encoding="utf-8") as f:
            entity = json.load(f)

        result = validate_source_location(entity, args.file, args.markdown)

        print(f"File: {args.file}")
        if result.valid:
            print("Status: ✓ VALID")
        else:
            print("Status: ✗ INVALID")
            print("\nErrors:")
            for error in result.errors:
                print(f"  ✗ {error.field}: {error.issue_type} - {error.message}")
                if error.current_value:
                    print(f"     Current: '{error.current_value}'")

        sys.exit(0 if result.valid else 1)

    elif args.directory:
        # Validate directory
        if not args.directory.exists():
            print(f"Error: Directory not found: {args.directory}")
            sys.exit(1)

        stats, results = validate_directory(args.directory, args.markdown)
        print_validation_report(stats, results, args.show_details, args.max_errors)

        if args.output:
            report = {
                "stats": dict(stats),
                "invalid_files": [
                    {
                        "file": str(r.file_path),
                        "errors": [
                            {
                                "field": e.field,
                                "type": e.issue_type,
                                "message": e.message,
                                "value": e.current_value,
                            }
                            for e in r.errors
                        ],
                    }
                    for r in stats["invalid_files"]
                ],
            }
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"\nReport written to: {args.output}")

        sys.exit(0 if stats["invalid"] == 0 else (1 if args.strict else 0))

    elif args.all_documents:
        # Validate all documents
        if not args.all_documents.exists():
            print(f"Error: Directory not found: {args.all_documents}")
            sys.exit(1)

        total_stats = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "errors_by_field": defaultdict(int),
            "errors_by_type": defaultdict(int),
            "invalid_files": [],
        }

        document_reports = {}

        # Find all raw_data directories
        for raw_data_dir in args.all_documents.rglob("raw_data"):
            if not raw_data_dir.is_dir():
                continue

            document_dir = raw_data_dir.parent
            document_id = document_dir.name

            # Find markdown file
            markdown_file = document_dir.parent / f"{document_id}.md"

            print(f"\nValidating {document_id}...")
            stats, results = validate_directory(
                raw_data_dir, markdown_file if markdown_file.exists() else None
            )

            document_reports[document_id] = stats

            # Aggregate stats
            total_stats["total"] += stats["total"]
            total_stats["valid"] += stats["valid"]
            total_stats["invalid"] += stats["invalid"]
            for field, count in stats["errors_by_field"].items():
                total_stats["errors_by_field"][field] += count
            for error_type, count in stats["errors_by_type"].items():
                total_stats["errors_by_type"][error_type] += count
            total_stats["invalid_files"].extend(stats["invalid_files"])

        # Print aggregate report
        print_validation_report(total_stats, [], args.show_details, args.max_errors)

        # Print per-document summary
        print(f"\n{'=' * 80}")
        print("PER-DOCUMENT SUMMARY:")
        print(f"{'=' * 80}")
        for doc_id, stats in sorted(document_reports.items()):
            status = "✓" if stats["invalid"] == 0 else "✗"
            print(f"{status} {doc_id:40} {stats['valid']:4}/{stats['total']:4} valid")

        if args.output:
            report = {
                "total_stats": {
                    k: dict(v) if isinstance(v, defaultdict) else v for k, v in total_stats.items()
                },
                "document_reports": {
                    k: dict(v) if isinstance(v, dict) else v for k, v in document_reports.items()
                },
            }
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"\nReport written to: {args.output}")

        sys.exit(0 if total_stats["invalid"] == 0 else (1 if args.strict else 0))


if __name__ == "__main__":
    main()
