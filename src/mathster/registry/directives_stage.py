#!/usr/bin/env python3
"""
Directives Stage: Chapter-to-Registry Transformation.

This module transforms chapter-by-chapter directive extraction files into
entity-type-aggregated registry files. It provides the bridge between the
directive extraction stage and the downstream registry consumption.

Transformation:
    Input:  directives/chapter_0.json, chapter_1.json, ... (mixed entity types per chapter)
    Output: registry/directives/definition.json, theorem.json, ... (single entity type, all chapters)

Each output file contains all entities of one type across all chapters, with
metadata tracking the source chapter and section.

Usage:
    # Single document
    python -m mathster.registry.directives_stage \\
        docs/source/1_euclidean_gas/07_mean_field

    # Batch processing
    python -m mathster.registry.directives_stage \\
        docs/source --batch

    # Programmatic usage
    from mathster.registry.directives_stage import process_document, discover_documents

    # Process single document
    process_document(Path("docs/source/1_euclidean_gas/07_mean_field"))

    # Process all documents
    for doc_path in discover_documents(Path("docs/source")):
        process_document(doc_path)
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import operator
from pathlib import Path
import sys


logger = logging.getLogger(__name__)


# ============================================================================
# DOCUMENT DISCOVERY
# ============================================================================


def discover_documents(docs_root: Path, recursive: bool = True) -> list[Path]:
    """
    Discover all document directories containing directives/ folder.

    Args:
        docs_root: Root directory to search (e.g., docs/source)
        recursive: Search recursively through subdirectories

    Returns:
        List of document directory paths containing directives/
    """
    document_paths = []

    if recursive:
        # Search recursively for any directory with directives/ subfolder
        for directives_dir in docs_root.rglob("directives"):
            if directives_dir.is_dir():
                document_path = directives_dir.parent

                # Skip if this is a registry/directives folder (avoid recursion)
                if document_path.name == "registry":
                    continue

                document_paths.append(document_path)
    else:
        # Check immediate subdirectories only
        for item in docs_root.iterdir():
            if item.is_dir():
                directives_dir = item / "directives"
                if directives_dir.exists() and directives_dir.is_dir():
                    document_paths.append(item)

    return sorted(document_paths)


def extract_document_id(document_path: Path) -> str:
    """
    Extract document ID from path.

    Examples:
        docs/source/1_euclidean_gas/07_mean_field -> "07_mean_field"
        docs/source/2_geometric_gas/01_introduction -> "01_introduction"

    Args:
        document_path: Path to document directory

    Returns:
        Document identifier
    """
    return document_path.name


# ============================================================================
# CHAPTER LOADING
# ============================================================================


def load_chapter_files(directives_dir: Path) -> list[dict]:
    """
    Load all chapter JSON files from directives directory.

    Args:
        directives_dir: Path to directives/ folder

    Returns:
        List of chapter data dictionaries, sorted by chapter_index
    """
    chapter_files = sorted(directives_dir.glob("chapter_*.json"))

    if not chapter_files:
        logger.warning(f"No chapter files found in {directives_dir}")
        return []

    chapters = []
    for chapter_file in chapter_files:
        try:
            with open(chapter_file, encoding="utf-8") as f:
                data = json.load(f)
                chapters.append({
                    "data": data,
                    "filename": chapter_file.name,
                })
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {chapter_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error loading {chapter_file}: {e}")
            continue

    # Sort by chapter_index
    chapters.sort(key=lambda c: c["data"].get("chapter_index", 0))

    return chapters


# ============================================================================
# ENTITY GROUPING
# ============================================================================


def group_hints_by_type(
    chapters: list[dict],
    document_id: str,
) -> dict[str, list[dict]]:
    """
    Group all hints by directive_type across all chapters.

    Args:
        chapters: List of chapter data (from load_chapter_files)
        document_id: Document identifier

    Returns:
        Dictionary mapping directive_type -> list of enriched items
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for chapter_entry in chapters:
        chapter_data = chapter_entry["data"]
        chapter_filename = chapter_entry["filename"]

        chapter_index = chapter_data.get("chapter_index", 0)
        section_id = chapter_data.get("section_id", "")
        hints = chapter_data.get("hints", [])

        for hint in hints:
            # Skip hints without directive_type
            directive_type = hint.get("directive_type")
            if not directive_type:
                logger.warning(
                    f"Hint in {chapter_filename} missing directive_type: {hint.get('label', 'unknown')}"
                )
                continue

            # Create enriched item with registry context
            enriched_item = hint.copy()
            enriched_item["_registry_context"] = {
                "stage": "directives",
                "document_id": document_id,
                "chapter_index": chapter_index,
                "chapter_file": chapter_filename,
                "section_id": section_id,
            }

            groups[directive_type].append(enriched_item)

    return groups


# ============================================================================
# REGISTRY OUTPUT
# ============================================================================


def write_registry_file(
    output_path: Path,
    document_id: str,
    directive_type: str,
    items: list[dict],
) -> None:
    """
    Write registry file for a specific directive type.

    Args:
        output_path: Path to output JSON file
        document_id: Document identifier
        directive_type: Type of directive (definition, theorem, etc.)
        items: List of enriched directive items
    """
    registry_data = {
        "document_id": document_id,
        "stage": "directives",
        "directive_type": directive_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(items)} {directive_type} items to {output_path.name}")


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================


def process_document(
    document_path: Path,
    force: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Process a single document: transform chapter files into registry files.

    Args:
        document_path: Path to document directory
        force: Force regeneration even if registry files exist
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    document_id = extract_document_id(document_path)
    directives_dir = document_path / "directives"
    registry_dir = document_path / "registry" / "directives"

    # Validate input directory
    if not directives_dir.exists():
        logger.error(f"Directives directory not found: {directives_dir}")
        return False

    # Check if registry already exists
    if registry_dir.exists() and not force:
        existing_files = list(registry_dir.glob("*.json"))
        if existing_files:
            logger.info(
                f"Registry already exists for {document_id} "
                f"({len(existing_files)} files). Use --force to regenerate."
            )
            return True

    logger.info(f"Processing document: {document_id}")

    # Load chapter files
    chapters = load_chapter_files(directives_dir)
    if not chapters:
        logger.warning(f"No chapter files found in {directives_dir}")
        return False

    logger.info(f"Loaded {len(chapters)} chapter files")

    # Group hints by type
    groups = group_hints_by_type(chapters, document_id)

    if not groups:
        logger.warning(f"No hints found in any chapter for {document_id}")
        return False

    # Write registry files
    logger.info(f"Found {len(groups)} directive types")

    for directive_type, items in sorted(groups.items()):
        output_path = registry_dir / f"{directive_type}.json"
        write_registry_file(output_path, document_id, directive_type, items)

    # Summary
    total_items = sum(len(items) for items in groups.values())
    logger.info(f" Processed {document_id}: {total_items} items across {len(groups)} types")

    return True


def process_batch(
    docs_root: Path,
    force: bool = False,
    verbose: bool = False,
    build_unified: bool = True,
    unified_output: Path | None = None,
) -> dict[str, bool]:
    """
    Process all documents in a directory tree.

    Args:
        docs_root: Root directory containing documents
        force: Force regeneration even if registry files exist
        verbose: Enable verbose logging
        build_unified: Build unified registry after processing (default: True)
        unified_output: Output directory for unified registry (default: unified_registry)

    Returns:
        Dictionary mapping document_id -> success status
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Discover all documents
    document_paths = discover_documents(docs_root)

    if not document_paths:
        logger.error(f"No documents with directives/ folder found in {docs_root}")
        return {}

    logger.info(f"Found {len(document_paths)} documents to process")

    # Process each document
    results = {}
    for document_path in document_paths:
        document_id = extract_document_id(document_path)
        try:
            success = process_document(document_path, force=force, verbose=verbose)
            results[document_id] = success
        except Exception as e:
            logger.error(f"Error processing {document_id}: {e}")
            results[document_id] = False

    # Summary
    successful = sum(1 for v in results.values() if v)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Batch processing complete: {successful}/{len(results)} successful")
    logger.info(f"{'=' * 60}")

    # Build unified registry if requested
    if build_unified and successful > 0:
        logger.info("")
        try:
            build_unified_registry(docs_root, output_dir=unified_output, verbose=verbose)
        except Exception as e:
            logger.error(f"Failed to build unified registry: {e}")

    return results


# ============================================================================
# UNIFIED REGISTRY GENERATION
# ============================================================================


def discover_document_registries(docs_root: Path) -> list[Path]:
    """
    Discover all document registry/directives directories.

    Args:
        docs_root: Root directory to search

    Returns:
        List of registry/directives directory paths
    """
    registry_paths = []

    for registry_dir in docs_root.rglob("registry/directives"):
        if registry_dir.is_dir():
            # Skip unified/combined registry folders
            parent_name = registry_dir.parent.parent.name
            if parent_name in {"unified_registry", "combined_registry"}:
                continue

            registry_paths.append(registry_dir)

    return sorted(registry_paths)


def load_registry_files_by_type(
    registry_dirs: list[Path],
) -> dict[str, list[tuple[str, dict]]]:
    """
    Load all registry JSON files grouped by directive type.

    Args:
        registry_dirs: List of registry/directives directories

    Returns:
        Dictionary mapping directive_type -> list of (document_id, registry_data)
    """
    files_by_type: dict[str, list[tuple[str, dict]]] = defaultdict(list)

    for registry_dir in registry_dirs:
        # Extract document_id from path
        document_path = registry_dir.parent.parent
        document_id = extract_document_id(document_path)

        # Load all JSON files in this registry
        for json_file in registry_dir.glob("*.json"):
            directive_type = json_file.stem

            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    files_by_type[directive_type].append((document_id, data))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse {json_file}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue

    return files_by_type


def merge_items_by_type(
    files_by_type: dict[str, list[tuple[str, dict]]],
) -> dict[str, dict]:
    """
    Merge items by directive type across all documents.

    Args:
        files_by_type: Dictionary mapping type -> list of (document_id, data)

    Returns:
        Dictionary mapping directive_type -> unified registry data
    """
    unified_registries = {}

    for directive_type, doc_files in sorted(files_by_type.items()):
        # Collect all items from all documents
        all_items = []
        source_documents = []

        for document_id, registry_data in sorted(doc_files, key=operator.itemgetter(0)):
            items = registry_data.get("items", [])
            all_items.extend(items)
            source_documents.append(document_id)

        # Create unified registry data
        unified_registries[directive_type] = {
            "stage": "directives",
            "directive_type": directive_type,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_documents": source_documents,
            "document_count": len(source_documents),
            "total_count": len(all_items),
            "items": all_items,
        }

        logger.info(
            f"Merged {directive_type}: {len(all_items)} items from {len(source_documents)} documents"
        )

    return unified_registries


def build_unified_registry(
    docs_root: Path,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> bool:
    """
    Build unified registry from all document registries.

    Args:
        docs_root: Root directory containing document registries
        output_dir: Output directory for unified registry (default: unified_registry)
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if output_dir is None:
        # Default to unified_registry at project root
        # Traverse up to find project root (look for pyproject.toml or src/)
        current = docs_root.resolve()
        while current.parent != current:
            if (current / "pyproject.toml").exists() or (current / "src").exists():
                project_root = current
                break
            current = current.parent
        else:
            # Fallback: assume docs_root is like /path/to/project/docs/source
            project_root = (
                docs_root.parent.parent if docs_root.name == "source" else docs_root.parent
            )

        output_dir = project_root / "unified_registry"

    output_path = output_dir / "directives"

    logger.info("Building unified registry...")
    logger.info(f"Output: {output_path}")

    # Discover all document registries
    registry_dirs = discover_document_registries(docs_root)

    if not registry_dirs:
        logger.warning(f"No document registries found in {docs_root}")
        return False

    logger.info(f"Found {len(registry_dirs)} document registries")

    # Load all registry files grouped by type
    files_by_type = load_registry_files_by_type(registry_dirs)

    if not files_by_type:
        logger.warning("No registry files found")
        return False

    logger.info(f"Found {len(files_by_type)} directive types")

    # Merge items by type
    unified_registries = merge_items_by_type(files_by_type)

    # Write unified registry files
    output_path.mkdir(parents=True, exist_ok=True)

    for directive_type, registry_data in sorted(unified_registries.items()):
        output_file = output_path / f"{directive_type}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote {output_file.name}")

    # Summary
    total_items = sum(r["total_count"] for r in unified_registries.values())
    logger.info(
        f"✓ Unified registry complete: {total_items} items across {len(unified_registries)} types"
    )

    return True


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Transform chapter-by-chapter directives into entity-type registry files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single document
  %(prog)s docs/source/1_euclidean_gas/07_mean_field

  # Process all documents in tree (auto-builds unified registry)
  %(prog)s docs/source --batch

  # Skip unified registry
  %(prog)s docs/source --batch --no-unified

  # Custom unified output location
  %(prog)s docs/source --batch --unified-output combined_registry

  # Only build unified registry from existing document registries
  %(prog)s docs/source --build-unified-only

  # Force regeneration
  %(prog)s docs/source/1_euclidean_gas/07_mean_field --force

  # Verbose output
  %(prog)s docs/source --batch --verbose
        """,
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to document directory or root directory (with --batch)",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all documents in directory tree",
    )

    parser.add_argument(
        "--build-unified-only",
        action="store_true",
        help="Only build unified registry from existing document registries",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if registry files exist",
    )

    parser.add_argument(
        "--no-unified",
        action="store_true",
        help="Skip building unified registry (batch mode only)",
    )

    parser.add_argument(
        "--unified-output",
        type=Path,
        help="Output directory for unified registry (default: unified_registry at project root)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    # Validate path
    if not args.path.exists():
        logger.error(f"Path does not exist: {args.path}")
        return 1

    # Process
    try:
        # Build unified registry only
        if args.build_unified_only:
            success = build_unified_registry(
                args.path,
                output_dir=args.unified_output,
                verbose=args.verbose,
            )
            return 0 if success else 1

        # Batch processing
        if args.batch:
            build_unified = not args.no_unified
            results = process_batch(
                args.path,
                force=args.force,
                verbose=args.verbose,
                build_unified=build_unified,
                unified_output=args.unified_output,
            )
            return 0 if all(results.values()) else 1

        # Single document processing
        success = process_document(args.path, force=args.force, verbose=args.verbose)
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
