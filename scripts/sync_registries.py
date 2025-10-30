#!/usr/bin/env python3
"""
Sync registries from docs/source to registries/ directory.

This script keeps the registries/ directory in sync with the authoritative
registries stored in docs/source/.

Usage:
    python scripts/sync_registries.py [--dry-run]
"""

import argparse
import json
from pathlib import Path
import shutil
import sys


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def discover_source_registries(docs_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Discover all registries in docs/source.

    Returns list of (chapter, document, registry_path) tuples.
    """
    registries = []

    # Pattern: docs/source/{chapter}/{document}/pipeline_registry
    for chapter_dir in docs_dir.glob("*"):
        if not chapter_dir.is_dir():
            continue

        for doc_dir in chapter_dir.glob("*"):
            if not doc_dir.is_dir():
                continue

            pipeline_registry = doc_dir / "pipeline_registry"
            if pipeline_registry.exists() and (pipeline_registry / "index.json").exists():
                chapter = chapter_dir.name
                document = doc_dir.name
                registries.append((chapter, document, pipeline_registry))

    return registries


def get_registry_stats(registry_path: Path) -> dict:
    """Get statistics from registry index.json."""
    index_path = registry_path / "index.json"
    if not index_path.exists():
        return {"total_objects": 0, "counts_by_type": {}}

    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
        return data.get("statistics", {})


def sync_registry(source: Path, dest: Path, dry_run: bool = False) -> bool:
    """
    Sync a registry from source to destination.

    Args:
        source: Source registry path
        dest: Destination registry path
        dry_run: If True, only print what would be done

    Returns:
        True if sync successful
    """
    if dry_run:
        print(f"  [DRY RUN] Would sync: {source} -> {dest}")
        return True

    try:
        # Remove old registry if exists
        if dest.exists():
            shutil.rmtree(dest)

        # Copy new registry
        shutil.copytree(source, dest)
        print(
            f"  ✓ Synced: {source.relative_to(get_project_root())} -> {dest.relative_to(get_project_root())}"
        )
        return True

    except Exception as e:
        print(f"  ✗ Failed to sync: {e}", file=sys.stderr)
        return False


def update_combined_registry(registries_dir: Path, dry_run: bool = False) -> bool:
    """
    Update combined registry by merging all per_document registries.

    Args:
        registries_dir: Root registries directory
        dry_run: If True, only print what would be done

    Returns:
        True if update successful
    """
    combined_path = registries_dir / "combined"
    per_doc_path = registries_dir / "per_document"

    if dry_run:
        print("  [DRY RUN] Would rebuild combined registry from per_document registries")
        return True

    # For now, just copy the first per_document registry as combined
    # TODO: Implement proper merging of multiple documents
    docs = list(per_doc_path.glob("*/pipeline"))
    if not docs:
        print("  ⚠️  No per_document registries found to combine")
        return False

    try:
        # Use 01_fragile_gas_framework as the combined registry for now
        source = per_doc_path / "01_fragile_gas_framework" / "pipeline"
        if source.exists():
            if combined_path.exists():
                shutil.rmtree(combined_path)
            shutil.copytree(source, combined_path)
            print("  ✓ Updated combined registry")
            return True
        print(f"  ⚠️  Source registry not found: {source}")
        return False

    except Exception as e:
        print(f"  ✗ Failed to update combined registry: {e}", file=sys.stderr)
        return False


def main():
    """Main sync function."""
    parser = argparse.ArgumentParser(
        description="Sync registries from docs/source to registries/ directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/sync_registries.py              # Sync all registries
  python scripts/sync_registries.py --dry-run    # Show what would be synced
        """,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be synced without actually syncing"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("REGISTRY SYNC")
    print("=" * 70)

    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will be modified\n")

    project_root = get_project_root()
    docs_dir = project_root / "docs" / "source"
    registries_dir = project_root / "registries"

    # Discover source registries
    print("\n📂 Discovering source registries...")
    source_registries = discover_source_registries(docs_dir)

    if not source_registries:
        print("  ⚠️  No registries found in docs/source")
        return 1

    print(f"  Found {len(source_registries)} registries:\n")

    # Display registry stats
    for chapter, document, registry_path in source_registries:
        stats = get_registry_stats(registry_path)
        total = stats.get("total_objects", 0)
        counts = stats.get("counts_by_type", {})
        print(f"  📊 {chapter}/{document}:")
        print(f"     Total: {total} entities")
        print(f"     {counts}")

    # Sync per_document registries
    print("\n🔄 Syncing per_document registries...")
    success_count = 0
    fail_count = 0

    for chapter, document, source_path in source_registries:
        # Destination: registries/per_document/{document}/pipeline
        dest_path = registries_dir / "per_document" / document / "pipeline"

        print(f"\n  Syncing {document}:")

        # Show before/after stats
        if dest_path.exists() and not args.dry_run:
            old_stats = get_registry_stats(dest_path)
            print(f"    Before: {old_stats.get('total_objects', 0)} entities")

        new_stats = get_registry_stats(source_path)
        print(f"    After:  {new_stats.get('total_objects', 0)} entities")

        if sync_registry(source_path, dest_path, args.dry_run):
            success_count += 1
        else:
            fail_count += 1

    # Update combined registry
    print("\n🔗 Updating combined registry...")
    if update_combined_registry(registries_dir, args.dry_run):
        combined_stats = get_registry_stats(registries_dir / "combined")
        print(f"    Total: {combined_stats.get('total_objects', 0)} entities")

    # Summary
    print("\n" + "=" * 70)
    print("SYNC SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully synced: {success_count} registries")
    if fail_count > 0:
        print(f"✗ Failed to sync: {fail_count} registries")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files were modified")
        print("   Run without --dry-run to perform actual sync")
    else:
        print("\n✅ Registry sync complete!")
        print(f"\nRegistries synced to: {registries_dir}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
