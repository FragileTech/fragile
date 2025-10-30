#!/usr/bin/env python3
"""
Batch Source Location Fix Tool.

Fixes source_location fields in entity JSON files to meet strict requirements:
1. Normalizes section format to X.Y.Z (no symbols/text)
2. Finds line_range using text matching
3. Extracts directive_label from entity
4. Ensures all required fields are present
5. Validates and saves

Usage:
    # Fix single document
    python scripts/batch_fix_source_locations.py --document docs/source/.../01_fragile_gas_framework

    # Fix all documents
    python scripts/batch_fix_source_locations.py --all-documents docs/source/

    # Dry run (preview changes)
    python scripts/batch_fix_source_locations.py --document docs/... --dry-run
"""

import argparse
import json
from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import normalization function
from normalize_section_format import normalize_section

from fragile.proofs.tools.line_finder import get_file_line_count, validate_line_range
from fragile.proofs.tools.source_location_enricher import (
    extract_directive_label_from_entity,
    find_entity_location,
)


def fix_source_location(
    entity_path: Path, markdown_path: Path, document_id: str, dry_run: bool = False
) -> tuple[bool, list]:
    """
    Fix source_location in a single entity JSON file.

    Args:
        entity_path: Path to entity JSON
        markdown_path: Path to source markdown
        document_id: Document ID
        dry_run: If True, don't write changes

    Returns:
        Tuple of (success, errors_fixed)
    """
    try:
        # Load entity
        with open(entity_path, encoding="utf-8") as f:
            entity = json.load(f)

        # Read markdown
        markdown_content = markdown_path.read_text(encoding="utf-8")
        max_lines = get_file_line_count(markdown_content)

        # Get or create source_location
        if "source_location" not in entity:
            entity["source_location"] = {}

        loc = entity["source_location"]
        errors_fixed = []

        # FIX 1: Ensure document_id
        if not loc.get("document_id"):
            loc["document_id"] = document_id
            errors_fixed.append("Added document_id")

        # FIX 2: Ensure file_path
        if not loc.get("file_path"):
            loc["file_path"] = str(markdown_path)
            errors_fixed.append("Added file_path")

        # FIX 3: Normalize section from raw entity
        section = loc.get("section")

        # Try to get section from raw entity if not in source_location
        if not section and "source_section" in entity:
            section = entity["source_section"]

        # Normalize section format
        if section:
            normalized = normalize_section(section)
            if normalized != section:
                loc["section"] = normalized
                errors_fixed.append(f'Normalized section: "{section}" → "{normalized}"')
            elif normalized:
                loc["section"] = normalized
        # Try to extract from source_section
        elif "source_section" in entity:
            normalized = normalize_section(entity["source_section"])
            if normalized:
                loc["section"] = normalized
                errors_fixed.append(f'Extracted section from source_section: "{normalized}"')

        # FIX 4: Find line_range if missing
        line_range = loc.get("line_range")
        if not line_range or not validate_line_range(
            tuple(line_range) if isinstance(line_range, list) else line_range, max_lines
        ):
            # Use enricher to find location
            found_loc = find_entity_location(
                entity_data=entity,
                markdown_content=markdown_content,
                document_id=document_id,
                file_path=str(markdown_path),
            )

            if found_loc and found_loc.line_range:
                loc["line_range"] = list(found_loc.line_range)  # Ensure list format
                errors_fixed.append(f"Found line_range: {list(found_loc.line_range)}")

                # Also update section if found and not already set
                if found_loc.section and not loc.get("section"):
                    loc["section"] = found_loc.section
                    errors_fixed.append(
                        f'Extracted section from found location: "{found_loc.section}"'
                    )

                # Update directive_label if found
                if found_loc.directive_label and not loc.get("directive_label"):
                    loc["directive_label"] = found_loc.directive_label
                    errors_fixed.append(
                        f'Extracted directive_label: "{found_loc.directive_label}"'
                    )

        # FIX 5: Extract directive_label from entity
        if not loc.get("directive_label"):
            label = extract_directive_label_from_entity(entity)
            if label:
                loc["directive_label"] = label
                errors_fixed.append(f'Extracted directive_label from entity: "{label}"')

        # FIX 6: Ensure equation is None (not missing)
        if "equation" not in loc:
            loc["equation"] = None

        # FIX 7: Generate url_fragment from directive_label if missing
        if not loc.get("url_fragment") and loc.get("directive_label"):
            loc["url_fragment"] = f"#{loc['directive_label']}"
            errors_fixed.append(f'Generated url_fragment: "{loc["url_fragment"]}"')

        # Save if changes made and not dry run
        if errors_fixed and not dry_run:
            with open(entity_path, "w", encoding="utf-8") as f:
                json.dump(entity, f, indent=2, ensure_ascii=False)

        return (True, errors_fixed)

    except Exception as e:
        return (False, [f"Error: {e!s}"])


def fix_directory(directory: Path, markdown_path: Path, document_id: str, dry_run: bool = False):
    """
    Fix all entity JSON files in a directory.

    Args:
        directory: Directory containing raw_data/ subdirectories
        markdown_path: Path to source markdown
        document_id: Document ID
        dry_run: If True, don't write changes
    """
    stats = {"total": 0, "fixed": 0, "already_valid": 0, "failed": 0, "total_fixes": 0}

    changes = []

    for json_file in directory.rglob("*.json"):
        stats["total"] += 1

        success, errors_fixed = fix_source_location(json_file, markdown_path, document_id, dry_run)

        if not success:
            stats["failed"] += 1
            changes.append((json_file, False, errors_fixed))
        elif errors_fixed:
            stats["fixed"] += 1
            stats["total_fixes"] += len(errors_fixed)
            changes.append((json_file, True, errors_fixed))
        else:
            stats["already_valid"] += 1

    # Print report
    print(f"\n{'=' * 80}")
    print(f"BATCH FIX REPORT: {document_id}")
    print(f"{'=' * 80}")
    print(f"Total files processed: {stats['total']}")
    print(f"  ✓ Fixed:             {stats['fixed']} ({stats['total_fixes']} individual fixes)")
    print(f"  ✓ Already valid:     {stats['already_valid']}")
    print(f"  ✗ Failed:            {stats['failed']}")

    if changes:
        print(f"\n{'─' * 80}")
        print("CHANGES MADE:")
        print(f"{'─' * 80}")

        # Show successful fixes first
        success_changes = [(f, fixes) for f, success, fixes in changes if success]
        for file_path, errors_fixed in success_changes[:10]:  # Show first 10
            print(f"\n✓ {file_path.name}")
            for fix in errors_fixed:
                print(f"   - {fix}")

        if len(success_changes) > 10:
            print(f"\n... and {len(success_changes) - 10} more files fixed")

        # Show failures
        failed_changes = [(f, fixes) for f, success, fixes in changes if not success]
        if failed_changes:
            print(f"\n{'─' * 80}")
            print("FAILED FIXES:")
            print(f"{'─' * 80}")
            for file_path, errors in failed_changes[:10]:
                print(f"\n✗ {file_path.name}")
                for error in errors:
                    print(f"   - {error}")

    if dry_run:
        print(f"\n{'=' * 80}")
        print("DRY RUN: No files were modified")
        print(f"{'=' * 80}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch fix source_location fields to meet strict requirements"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--document", "-d", type=Path, help="Fix single document directory")
    group.add_argument(
        "--all-documents", "-a", type=Path, help="Fix all documents in docs/source/"
    )

    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON report to file")

    args = parser.parse_args()

    if args.document:
        # Fix single document
        if not args.document.exists():
            print(f"Error: Directory not found: {args.document}")
            sys.exit(1)

        document_id = args.document.name
        raw_data_dir = args.document / "raw_data"
        markdown_file = args.document.parent / f"{document_id}.md"

        if not raw_data_dir.exists():
            print(f"Error: raw_data directory not found: {raw_data_dir}")
            sys.exit(1)

        if not markdown_file.exists():
            print(f"Error: Markdown file not found: {markdown_file}")
            sys.exit(1)

        stats = fix_directory(raw_data_dir, markdown_file, document_id, args.dry_run)

        sys.exit(0 if stats["failed"] == 0 else 1)

    elif args.all_documents:
        # Fix all documents
        if not args.all_documents.exists():
            print(f"Error: Directory not found: {args.all_documents}")
            sys.exit(1)

        total_stats = {"total": 0, "fixed": 0, "already_valid": 0, "failed": 0, "total_fixes": 0}

        document_reports = {}

        # Find all raw_data directories
        for raw_data_dir in sorted(args.all_documents.rglob("raw_data")):
            if not raw_data_dir.is_dir():
                continue

            document_dir = raw_data_dir.parent
            document_id = document_dir.name

            # Find markdown file
            markdown_file = document_dir.parent / f"{document_id}.md"

            if not markdown_file.exists():
                print(f"Warning: Markdown file not found for {document_id}, skipping")
                continue

            print(f"\nFixing {document_id}...")
            stats = fix_directory(raw_data_dir, markdown_file, document_id, args.dry_run)

            document_reports[document_id] = stats

            # Aggregate stats
            total_stats["total"] += stats["total"]
            total_stats["fixed"] += stats["fixed"]
            total_stats["already_valid"] += stats["already_valid"]
            total_stats["failed"] += stats["failed"]
            total_stats["total_fixes"] += stats["total_fixes"]

        # Print aggregate report
        print(f"\n{'=' * 80}")
        print("AGGREGATE FIX REPORT")
        print(f"{'=' * 80}")
        print(f"Total files processed: {total_stats['total']}")
        print(
            f"  ✓ Fixed:             {total_stats['fixed']} ({total_stats['total_fixes']} individual fixes)"
        )
        print(f"  ✓ Already valid:     {total_stats['already_valid']}")
        print(f"  ✗ Failed:            {total_stats['failed']}")

        # Print per-document summary
        print(f"\n{'=' * 80}")
        print("PER-DOCUMENT SUMMARY:")
        print(f"{'=' * 80}")
        for doc_id, stats in sorted(document_reports.items()):
            status = "✓" if stats["failed"] == 0 else "✗"
            print(f"{status} {doc_id:40} {stats['fixed']:4} fixed, {stats['failed']:4} failed")

        if args.output:
            args.output.write_text(json.dumps(document_reports, indent=2, ensure_ascii=False))
            print(f"\nReport written to: {args.output}")

        sys.exit(0 if total_stats["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
