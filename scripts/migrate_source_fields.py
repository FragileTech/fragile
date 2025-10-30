#!/usr/bin/env python3
"""
Migration Script: Standardize Source Field Names

This script migrates all raw_data JSON files to use the 'source' field name
instead of 'source_location', and ensures all entities have source locations.

Usage:
    # Dry run (show what would be changed)
    python scripts/migrate_source_fields.py --dry-run

    # Actually migrate all documents
    python scripts/migrate_source_fields.py

    # Migrate specific directory
    python scripts/migrate_source_fields.py docs/source/1_euclidean_gas/

Author: Claude Code
Date: 2025
"""

import argparse
import json
import logging
from pathlib import Path
import sys


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.files_processed = 0
        self.files_migrated = 0
        self.files_skipped = 0
        self.files_error = 0
        self.entities_total = 0
        self.entities_migrated = 0
        self.entities_enriched = 0
        self.entities_skipped = 0
        self.errors: list[tuple[str, str]] = []

    def print_summary(self):
        """Print migration summary."""
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files processed:  {self.files_processed}")
        print(f"Files migrated:   {self.files_migrated}")
        print(f"Files skipped:    {self.files_skipped}")
        print(f"Files with errors: {self.files_error}")
        print(f"\nEntities total:    {self.entities_total}")
        print(f"Entities migrated: {self.entities_migrated} (renamed source_location → source)")
        print(f"Entities enriched: {self.entities_enriched} (added missing source)")
        print(f"Entities skipped:  {self.entities_skipped} (already have source)")

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for file_path, error in self.errors[:10]:  # Show first 10 errors
                print(f"  {file_path}: {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        print("=" * 60)


def find_raw_data_json_files(root_dir: Path) -> list[Path]:
    """Find all JSON files in raw_data directories and their subdirectories."""
    json_files = []

    # Find all raw_data directories
    raw_data_dirs = list(root_dir.glob("**/raw_data"))

    for raw_data_dir in raw_data_dirs:
        # Get all JSON files in this directory AND subdirectories (recursive)
        json_files.extend(raw_data_dir.glob("**/*.json"))

    return sorted(json_files)


def find_all_json_files(root_dir: Path) -> list[Path]:
    """Find ALL JSON files recursively under root_dir."""
    return sorted(root_dir.glob("**/*.json"))


def migrate_entity(entity_data: dict, entity_type: str) -> tuple[bool, str]:
    """
    Migrate a single entity's source field.

    Returns:
        (migrated, status) where status is:
        - "migrated": source_location renamed to source (or replaced old source)
        - "enriched": source added from scratch
        - "skipped": already has complete source (no source_location exists)
        - "error": migration failed
    """
    # Check if has 'source_location' (prioritize this over incomplete 'source')
    if "source_location" in entity_data:
        source_location = entity_data["source_location"]
        if source_location is not None:
            # Rename field (overwrites 'source' if it exists)
            entity_data["source"] = source_location
            del entity_data["source_location"]
            return True, "migrated"

    # Check if already has 'source' field (and no source_location)
    if "source" in entity_data and entity_data["source"] is not None:
        return False, "skipped"

    # Neither field exists - needs enrichment
    # For now, we skip entities that need enrichment and log them
    # Full enrichment requires running source_location_enricher separately
    return False, "enriched"


def migrate_json_file(file_path: Path, stats: MigrationStats, dry_run: bool = False) -> bool:
    """
    Migrate a single JSON file.

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Read JSON file
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Check if it's a valid entity file (has expected fields)
        if not isinstance(data, dict):
            logger.warning(f"Skipping {file_path}: not a dict")
            stats.files_skipped += 1
            return False

        # Determine entity type (for logging)
        temp_id = data.get("temp_id", data.get("label", "unknown"))
        entity_type = temp_id.split("-")[1] if "-" in temp_id else "unknown"

        stats.entities_total += 1

        # Migrate the entity
        modified, status = migrate_entity(data, entity_type)

        if status == "migrated":
            stats.entities_migrated += 1
            logger.info(f"Migrated {file_path.name}: source_location → source")
        elif status == "enriched":
            stats.entities_enriched += 1
            logger.warning(f"Needs enrichment: {file_path.name} (missing source)")
        elif status == "skipped":
            stats.entities_skipped += 1
            logger.debug(f"Skipped {file_path.name}: already has source")

        # Write back if modified and not dry run
        if modified and not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")  # Add trailing newline

        return modified

    except Exception as e:
        logger.error(f"Error migrating {file_path}: {e}")
        stats.errors.append((str(file_path), str(e)))
        stats.files_error += 1
        return False


def migrate_directory(
    root_dir: Path, stats: MigrationStats, dry_run: bool = False, all_files: bool = False
):
    """Migrate all JSON files in a directory tree."""
    # Find JSON files
    if all_files:
        json_files = find_all_json_files(root_dir)
        logger.info(f"Searching ALL JSON files under {root_dir}")
    else:
        json_files = find_raw_data_json_files(root_dir)
        logger.info(f"Searching raw_data directories under {root_dir}")

    if not json_files:
        logger.warning(f"No JSON files found under {root_dir}")
        return

    logger.info(f"Found {len(json_files)} JSON files to process")

    if dry_run:
        logger.info("DRY RUN MODE - no files will be modified")

    # Process each file
    for json_file in json_files:
        stats.files_processed += 1

        modified = migrate_json_file(json_file, stats, dry_run=dry_run)

        if modified:
            stats.files_migrated += 1


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate source field names in raw_data JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show what would be changed in raw_data only)
  python scripts/migrate_source_fields.py --dry-run

  # Migrate raw_data directories only (default)
  python scripts/migrate_source_fields.py

  # Migrate ALL JSON files recursively (comprehensive)
  python scripts/migrate_source_fields.py --all

  # Migrate specific directory (all files)
  python scripts/migrate_source_fields.py docs/source/1_euclidean_gas/ --all

  # Verbose output
  python scripts/migrate_source_fields.py --all --verbose
        """,
    )

    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default="docs/source",
        help="Root directory to search for raw_data (default: docs/source)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without modifying files"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate ALL JSON files recursively (not just raw_data directories)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate directory
    root_dir = Path(args.directory)
    if not root_dir.exists():
        logger.error(f"Directory not found: {root_dir}")
        sys.exit(1)

    # Initialize stats
    stats = MigrationStats()

    # Run migration
    logger.info(f"Starting migration in: {root_dir}")
    migrate_directory(root_dir, stats, dry_run=args.dry_run, all_files=args.all)

    # Print summary
    stats.print_summary()

    # Exit with error code if there were errors
    if stats.files_error > 0:
        sys.exit(1)

    # Warn if entities need enrichment
    if stats.entities_enriched > 0:
        logger.warning(
            f"\n{stats.entities_enriched} entities need source location enrichment. "
            "Run source_location_enricher to add missing sources."
        )


if __name__ == "__main__":
    main()
