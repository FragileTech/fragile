#!/usr/bin/env python3
"""
Add Missing directive_label Fields.

For files with label but missing directive_label, copies label to directive_label.
"""

import argparse
import json
from pathlib import Path


def add_missing_directive_labels(directory: Path, dry_run: bool = False):
    """Add directive_label for files that have label but missing directive_label."""

    stats = {"total": 0, "already_had": 0, "added": 0, "no_label": 0}

    added = []

    for json_file in sorted(directory.rglob("*.json")):
        if "report" in json_file.name.lower():
            continue

        stats["total"] += 1

        with open(json_file, encoding="utf-8") as f:
            entity = json.load(f)

        sl = entity.get("source_location", {})

        # Check if already has directive_label
        if sl.get("directive_label"):
            stats["already_had"] += 1
            continue

        # Get label
        label = entity.get("label") or entity.get("label_text")
        if not label:
            stats["no_label"] += 1
            continue

        # Add directive_label
        if "source_location" not in entity:
            entity["source_location"] = {}

        entity["source_location"]["directive_label"] = label
        stats["added"] += 1
        added.append((json_file.name, label))

        # Save
        if not dry_run:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(entity, f, indent=2, ensure_ascii=False)

    # Report
    print(f"\n{'=' * 80}")
    print("ADD DIRECTIVE_LABEL REPORT")
    print(f"{'=' * 80}")
    print(f"Total files: {stats['total']}")
    print(f"  ✓ Already had: {stats['already_had']}")
    print(f"  ✓ Added: {stats['added']}")
    print(f"  - No label: {stats['no_label']}")

    if added:
        print(f"\n{'─' * 80}")
        print("ADDED (showing first 30):")
        print(f"{'─' * 80}")
        for fname, label in added[:30]:
            print(f"  ✓ {fname:60} → {label}")
        if len(added) > 30:
            print(f"\n... and {len(added) - 30} more")

    if dry_run:
        print(f"\n{'=' * 80}")
        print("DRY RUN: No files modified")
        print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    add_missing_directive_labels(args.directory, args.dry_run)


if __name__ == "__main__":
    main()
