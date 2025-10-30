#!/usr/bin/env python3
"""
Enrich Line Ranges from Jupyter Book Directives.

For entities with labels but missing line_range, finds the exact location
by searching for Jupyter Book directives like :::{prf:theorem} label-name.

Also extracts section from the found line_range if section is missing.

Usage:
    python scripts/enrich_line_ranges_from_directives.py \
        --document docs/source/.../01_fragile_gas_framework

    python scripts/enrich_line_ranges_from_directives.py \
        --all-documents docs/source/
"""

import argparse
import json
from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def infer_directive_type_from_label(label: str) -> str | None:
    """Infer Jupyter Book directive type from label prefix."""
    if label.startswith("thm-"):
        return "theorem"
    if label.startswith("lem-"):
        return "lemma"
    if label.startswith(("def-", "obj-")):
        return "definition"
    if label.startswith(("ax-", "axiom-")):
        return "axiom"
    if label.startswith("proof-"):
        return "proof"
    if label.startswith(("rem-", "remark-")):
        return "remark"
    if label.startswith("prop-"):
        return "property"
    if label.startswith("cor-"):
        return "corollary"
    return None  # Let find_directive_lines try all types


def extract_section_from_line_range(
    line_range: tuple[int, int], section_map: list[tuple[int, int, str]]
) -> str | None:
    """Find which section a line range falls within."""
    start, _end = line_range
    for sec_start, sec_end, section in section_map:
        if sec_start <= start <= sec_end:
            return section
    return None


def build_section_map(markdown_path: Path) -> list[tuple[int, int, str]]:
    """Build map of (start_line, end_line, section_number) from headers."""
    import re

    content = markdown_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    sections = []
    current_section = None
    current_start = 1

    for i, line in enumerate(lines, start=1):
        if line.startswith("#"):
            # Extract section number from header
            text = line.lstrip("#").strip()
            match = re.match(r"^(\d+(?:\.\d+)*)", text)
            if match:
                section_num = match.group(1)

                # Close previous section
                if current_section:
                    sections.append((current_start, i - 1, current_section))

                # Start new section
                current_section = section_num
                current_start = i

    # Close final section
    if current_section:
        sections.append((current_start, len(lines), current_section))

    return sections


def find_directive_by_label_line(
    markdown_content: str, directive_label: str
) -> tuple[int, int] | None:
    """
    Find directive where label is on separate line after opening.

    Format:
        :::{prf:theorem} Title
        :label: thm-main
        content
        :::
    """
    import re

    lines = markdown_content.splitlines()

    # Find lines with :label: directive_label
    for i, line in enumerate(lines):
        if re.match(rf"^:label:\s+{re.escape(directive_label)}\s*$", line.strip()):
            # Found label line, look backwards for directive opening
            if i > 0:
                prev_line = lines[i - 1].strip()
                if re.match(r"^:::+\s*\{prf:\w+\}", prev_line):
                    # Found directive, now find closing :::

                    for j in range(i + 1, len(lines)):
                        if re.match(r"^:::+\s*$", lines[j].strip()):
                            return (
                                i,
                                j + 1,
                            )  # 1-indexed (i is 0-indexed, so i = line i+1 in 1-indexed)

                    # No closing found, return to end
                    return (i, len(lines))

    return None


def enrich_entity_with_directive(
    entity_path: Path,
    markdown_content: str,
    section_map: list[tuple[int, int, str]],
    dry_run: bool = False,
) -> tuple[bool, str | None]:
    """
    Find line_range using directive matching.

    Returns:
        (success, error_message)
    """
    try:
        with open(entity_path, encoding="utf-8") as f:
            entity = json.load(f)

        # Check if already has line_range
        sl = entity.get("source_location", {})
        if sl and sl.get("line_range"):
            return (True, None)  # Already enriched

        # Get label
        label = entity.get("label") or entity.get("label_text")
        if not label:
            return (False, "No label field")

        # Try to find directive with label on separate line
        line_range = find_directive_by_label_line(markdown_content, label)

        if not line_range:
            return (False, f"Directive not found for label: {label}")

        # Update source_location
        if "source_location" not in entity:
            entity["source_location"] = {}

        entity["source_location"]["line_range"] = list(line_range)
        entity["source_location"]["directive_label"] = label

        # Extract section if missing
        if not entity["source_location"].get("section"):
            section = extract_section_from_line_range(line_range, section_map)
            if section:
                entity["source_location"]["section"] = section

        # Generate url_fragment if missing
        if not entity["source_location"].get("url_fragment"):
            entity["source_location"]["url_fragment"] = f"#{label}"

        # Save
        if not dry_run:
            with open(entity_path, "w", encoding="utf-8") as f:
                json.dump(entity, f, indent=2, ensure_ascii=False)

        return (True, None)

    except Exception as e:
        return (False, f"Error: {e!s}")


def enrich_directory(
    directory: Path, markdown_path: Path, document_id: str, dry_run: bool = False
):
    """Enrich all entities in directory using directive matching."""

    # Build section map
    section_map = build_section_map(markdown_path)

    # Load markdown
    markdown_content = markdown_path.read_text(encoding="utf-8")

    stats = {
        "total": 0,
        "already_had_line_range": 0,
        "enriched": 0,
        "no_label": 0,
        "not_found": 0,
        "failed": 0,
    }

    changes = []
    failures = []

    for json_file in sorted(directory.rglob("*.json")):
        if "report" in json_file.name.lower():
            continue

        stats["total"] += 1

        # Check if needs enrichment
        try:
            with open(json_file, encoding="utf-8") as f:
                entity = json.load(f)

            sl = entity.get("source_location", {})
            if sl and sl.get("line_range"):
                stats["already_had_line_range"] += 1
                continue

            if not (entity.get("label") or entity.get("label_text")):
                stats["no_label"] += 1
                continue

        except Exception as e:
            stats["failed"] += 1
            failures.append((json_file.name, f"Read error: {e}"))
            continue

        # Try to enrich
        success, error = enrich_entity_with_directive(
            json_file, markdown_content, section_map, dry_run
        )

        if success and error is None:
            stats["enriched"] += 1
            changes.append(json_file.name)
        elif error and "not found" in error.lower():
            stats["not_found"] += 1
            failures.append((json_file.name, error))
        else:
            stats["failed"] += 1
            if error:
                failures.append((json_file.name, error))

    # Print report
    print(f"\n{'=' * 80}")
    print(f"DIRECTIVE-BASED ENRICHMENT REPORT: {document_id}")
    print(f"{'=' * 80}")
    print(f"Total files processed: {stats['total']}")
    print(f"  ✓ Already had line_range: {stats['already_had_line_range']}")
    print(f"  ✓ Enriched: {stats['enriched']}")
    print(f"  - No label: {stats['no_label']}")
    print(f"  ✗ Directive not found: {stats['not_found']}")
    print(f"  ✗ Failed: {stats['failed']}")

    if changes:
        print(f"\n{'─' * 80}")
        print("ENRICHED FILES (showing first 20):")
        print(f"{'─' * 80}")
        for name in changes[:20]:
            print(f"  ✓ {name}")
        if len(changes) > 20:
            print(f"\n... and {len(changes) - 20} more files")

    if failures:
        print(f"\n{'─' * 80}")
        print("FAILURES (showing first 20):")
        print(f"{'─' * 80}")
        for name, error in failures[:20]:
            print(f"  ✗ {name}")
            print(f"     {error}")
        if len(failures) > 20:
            print(f"\n... and {len(failures) - 20} more failures")

    if dry_run:
        print(f"\n{'=' * 80}")
        print("DRY RUN: No files were modified")
        print(f"{'=' * 80}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Enrich line_range using Jupyter Book directive matching"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--document", "-d", type=Path, help="Document directory")
    group.add_argument("--all-documents", "-a", type=Path, help="Process all documents")

    parser.add_argument("--dry-run", action="store_true", help="Preview changes")

    args = parser.parse_args()

    if args.document:
        if not args.document.exists():
            print(f"Error: Directory not found: {args.document}")
            sys.exit(1)

        document_id = args.document.name
        raw_data_dir = args.document / "raw_data"
        markdown_file = args.document.parent / f"{document_id}.md"

        if not raw_data_dir.exists():
            print(f"Error: raw_data not found: {raw_data_dir}")
            sys.exit(1)

        if not markdown_file.exists():
            print(f"Error: Markdown not found: {markdown_file}")
            sys.exit(1)

        stats = enrich_directory(raw_data_dir, markdown_file, document_id, args.dry_run)

        sys.exit(0 if stats["failed"] == 0 else 1)

    elif args.all_documents:
        if not args.all_documents.exists():
            print(f"Error: Directory not found: {args.all_documents}")
            sys.exit(1)

        total_stats = {
            "total": 0,
            "already_had_line_range": 0,
            "enriched": 0,
            "no_label": 0,
            "not_found": 0,
            "failed": 0,
        }

        for raw_data_dir in sorted(args.all_documents.rglob("raw_data")):
            if not raw_data_dir.is_dir():
                continue

            document_dir = raw_data_dir.parent
            document_id = document_dir.name
            markdown_file = document_dir.parent / f"{document_id}.md"

            if not markdown_file.exists():
                print(f"Warning: Markdown not found for {document_id}, skipping")
                continue

            print(f"\nProcessing {document_id}...")
            stats = enrich_directory(raw_data_dir, markdown_file, document_id, args.dry_run)

            # Aggregate
            for key in total_stats:
                total_stats[key] += stats[key]

        # Print aggregate
        print(f"\n{'=' * 80}")
        print("AGGREGATE REPORT")
        print(f"{'=' * 80}")
        print(f"Total files: {total_stats['total']}")
        print(f"  ✓ Enriched: {total_stats['enriched']}")
        print(f"  - Already had line_range: {total_stats['already_had_line_range']}")
        print(f"  - No label: {total_stats['no_label']}")
        print(f"  ✗ Not found: {total_stats['not_found']}")
        print(f"  ✗ Failed: {total_stats['failed']}")

        sys.exit(0 if total_stats["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
