"""Consolidate scattered extraction files into unified raw_data structure.

This tool consolidates extraction outputs from multiple section-level parsers
into a single organized raw_data directory structure.

Target structure:
    raw_data/
    ├── axioms/
    ├── definitions/
    ├── theorems/
    ├── lemmas/
    ├── propositions/
    ├── corollaries/
    ├── mathster/
    ├── equations/
    ├── parameters/
    ├── remarks/
    ├── citations/
    └── objects/
    statistics/
        └── (all statistics files)
"""

from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Dict, List, Set


# Entity type directories
ENTITY_DIRS = [
    "axioms",
    "definitions",
    "theorems",
    "lemmas",
    "propositions",
    "corollaries",
    "mathster",
    "equations",
    "parameters",
    "remarks",
    "citations",
    "objects",
]

# File patterns that indicate statistics/reports
STATS_PATTERNS = ["*statistics*.json", "*extraction*.json", "*report*.json", "*summary*.json"]


def is_statistics_file(file_path: Path) -> bool:
    """Check if file is a statistics/report file."""
    name_lower = file_path.name.lower()
    return any([
        "statistics" in name_lower,
        "extraction" in name_lower and file_path.suffix == ".json",
        "report" in name_lower and file_path.suffix == ".json",
        "summary" in name_lower and file_path.suffix == ".json",
    ])


def identify_entity_type(file_path: Path) -> str | None:
    """Identify entity type from filename or content.

    Returns:
        Entity type directory name, or None if not an entity file
    """
    name = file_path.stem.lower()

    # Check filename prefixes
    if name.startswith(("axiom-", "def-axiom-")):
        return "axioms"
    if name.startswith(("def-", "obj-")) and "axiom" not in name:
        # obj- files can be either definitions or objects, check both
        if "obj-" in name:
            return "objects"
        return "definitions"
    if name.startswith("thm-"):
        return "theorems"
    if name.startswith("lem-"):
        return "lemmas"
    if name.startswith("prop-"):
        return "propositions"
    if name.startswith("cor-"):
        return "corollaries"
    if name.startswith(("proof-", "prf-")):
        return "mathster"
    if name.startswith("param-"):
        return "parameters"
    if name.startswith(("rem-", "remark-")):
        return "remarks"
    if name.startswith(("cite-", "citation-")):
        return "citations"
    if name.startswith("eq-"):
        return "equations"

    # Try to determine from content
    if file_path.suffix == ".json":
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Check for type field
            if "type" in data:
                type_val = data["type"].lower()
                if "axiom" in type_val:
                    return "axioms"
                if "theorem" in type_val:
                    return "theorems"
                if "lemma" in type_val:
                    return "lemmas"
                if "proposition" in type_val:
                    return "propositions"
                if "corollary" in type_val:
                    return "corollaries"
                if "proof" in type_val:
                    return "mathster"
                if "definition" in type_val:
                    return "definitions"
                if "object" in type_val:
                    return "objects"
                if "parameter" in type_val:
                    return "parameters"
                if "remark" in type_val:
                    return "remarks"

            # Check label field
            if "label" in data:
                label = data["label"].lower()
                if label.startswith("axiom-"):
                    return "axioms"
                if label.startswith("thm-"):
                    return "theorems"
                if label.startswith("lem-"):
                    return "lemmas"
                if label.startswith("prop-"):
                    return "propositions"
                if label.startswith("cor-"):
                    return "corollaries"
                if label.startswith(("def-", "obj-")):
                    if "obj-" in label:
                        return "objects"
                    return "definitions"

        except (OSError, json.JSONDecodeError):
            pass

    return None


def find_entity_files(base_dir: Path, exclude_dirs: set[str]) -> dict[str, list[Path]]:
    """Find all entity JSON files in directory tree.

    Args:
        base_dir: Base directory to search
        exclude_dirs: Set of directory names to exclude

    Returns:
        Dictionary mapping entity type to list of file paths
    """
    entities = defaultdict(list)
    stats_files = []

    for json_file in base_dir.rglob("*.json"):
        # Skip if in excluded directory
        if any(excluded in json_file.parts for excluded in exclude_dirs):
            continue

        # Skip if already in target raw_data structure
        if "raw_data" in json_file.parts:
            rel_parts = json_file.relative_to(base_dir).parts
            if len(rel_parts) >= 2 and rel_parts[0] == "raw_data":
                continue

        # Check if statistics file
        if is_statistics_file(json_file):
            stats_files.append(json_file)
            continue

        # Identify entity type
        entity_type = identify_entity_type(json_file)
        if entity_type:
            entities[entity_type].append(json_file)

    entities["statistics"] = stats_files
    return dict(entities)


def consolidate_files(
    base_dir: Path, dry_run: bool = False, exclude_dirs: set[str] | None = None
) -> dict[str, any]:
    """Consolidate all entity files into raw_data structure.

    Args:
        base_dir: Base directory containing scattered files
        dry_run: If True, only report what would be done
        exclude_dirs: Set of directory names to exclude

    Returns:
        Report dictionary with consolidation statistics
    """
    if exclude_dirs is None:
        exclude_dirs = {"deprecated", "raw_data", "__pycache__"}

    # Find all entity files
    print(f"Scanning {base_dir} for entity files...")
    entities_by_type = find_entity_files(base_dir, exclude_dirs)

    # Create target directories
    raw_data_dir = base_dir / "raw_data"
    statistics_dir = base_dir / "statistics"

    if not dry_run:
        raw_data_dir.mkdir(exist_ok=True)
        statistics_dir.mkdir(exist_ok=True)
        for entity_dir in ENTITY_DIRS:
            (raw_data_dir / entity_dir).mkdir(exist_ok=True)

    # Track operations
    moved = defaultdict(list)
    skipped = defaultdict(list)
    duplicates = defaultdict(list)

    # Move files
    for entity_type, files in entities_by_type.items():
        if entity_type == "statistics":
            target_dir = statistics_dir
        else:
            target_dir = raw_data_dir / entity_type

        print(f"\nProcessing {len(files)} {entity_type} files...")

        for src_file in files:
            target_file = target_dir / src_file.name

            # Check for duplicates
            if target_file.exists():
                # Check if identical
                if src_file.resolve() == target_file.resolve():
                    skipped[entity_type].append(src_file.name)
                    continue

                # Check content
                try:
                    with open(src_file) as f1, open(target_file) as f2:
                        if f1.read() == f2.read():
                            skipped[entity_type].append(src_file.name)
                            if not dry_run:
                                src_file.unlink()  # Remove duplicate
                            continue
                except OSError:
                    pass

                # Different content - rename
                base_name = target_file.stem
                counter = 1
                while target_file.exists():
                    target_file = target_dir / f"{base_name}_dup{counter}.json"
                    counter += 1
                duplicates[entity_type].append((src_file.name, target_file.name))

            # Move file
            if dry_run:
                print(
                    f"  Would move: {src_file.relative_to(base_dir)} -> {target_file.relative_to(base_dir)}"
                )
            else:
                shutil.move(str(src_file), str(target_file))
                print(f"  Moved: {src_file.name}")

            moved[entity_type].append(src_file.name)

    # Clean up empty directories
    if not dry_run:
        cleanup_empty_dirs(base_dir, exclude_dirs)

    # Generate report
    return {
        "base_directory": str(base_dir),
        "dry_run": dry_run,
        "files_moved": {k: len(v) for k, v in moved.items()},
        "files_skipped": {k: len(v) for k, v in skipped.items()},
        "duplicates_renamed": {k: len(v) for k, v in duplicates.items()},
        "total_moved": sum(len(v) for v in moved.values()),
        "details": {
            "moved": dict(moved),
            "skipped": dict(skipped),
            "duplicates": dict(duplicates),
        },
    }


def cleanup_empty_dirs(base_dir: Path, exclude_dirs: set[str]):
    """Remove empty directories (except excluded ones)."""
    for item in base_dir.rglob("*"):
        if not item.is_dir():
            continue
        if item.name in exclude_dirs:
            continue
        if item.name.startswith("."):
            continue
        if not any(item.iterdir()):
            try:
                item.rmdir()
                print(f"  Removed empty directory: {item.relative_to(base_dir)}")
            except OSError:
                pass


def print_report(report: dict):
    """Print consolidation report."""
    print(f"\n{'=' * 80}")
    print("CONSOLIDATION REPORT")
    print(f"{'=' * 80}\n")

    print(f"Base Directory: {report['base_directory']}")
    print(f"Dry Run: {report['dry_run']}\n")

    print("Files Moved by Type:")
    print("-" * 40)
    for entity_type, count in sorted(report["files_moved"].items()):
        print(f"  {entity_type:20s}: {count:4d}")
    print(f"  {'TOTAL':20s}: {report['total_moved']:4d}\n")

    if report["files_skipped"]:
        print("Files Skipped (already in place):")
        print("-" * 40)
        for entity_type, count in sorted(report["files_skipped"].items()):
            print(f"  {entity_type:20s}: {count:4d}")
        print()

    if report["duplicates_renamed"]:
        print("Duplicate Files Renamed:")
        print("-" * 40)
        for entity_type, count in sorted(report["duplicates_renamed"].items()):
            print(f"  {entity_type:20s}: {count:4d}")
        print()

    print(f"{'=' * 80}\n")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python consolidate_raw_data.py <directory> [--dry-run]")
        print("\nExample:")
        print(
            "  python consolidate_raw_data.py docs/source/1_euclidean_gas/01_fragile_gas_framework/"
        )
        print(
            "  python consolidate_raw_data.py docs/source/1_euclidean_gas/01_fragile_gas_framework/ --dry-run"
        )
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN MODE - No files will be moved\n")

    report = consolidate_files(base_dir, dry_run=dry_run)
    print_report(report)

    # Save report
    if not dry_run:
        report_file = base_dir / "consolidation_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
