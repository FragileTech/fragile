#!/usr/bin/env python3
"""
Extract Stage: Unified Registry Builder.

The Mathster extraction agents emit structured outputs on a per-document basis
under ``<document>/registry/extract`` (for example,
``docs/source/1_euclidean_gas/07_mean_field/registry/extract/theorem.json``).
This module aggregates those files into a single ``unified_registry/extract``
workspace so downstream tooling can consume the entire corpus without manual
concatenation.

Each unified file preserves the original schema (plain lists of extraction
objects). The only change is that entries from every document are appended into
one list per entity type.

Usage:

    # Combine everything under docs/source into unified_registry/extract
    python -m mathster.registry.extract_stage docs/source

    # Custom output directory
    python -m mathster.registry.extract_stage docs/source --output-dir combined_registry
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Iterable

from mathster.registry.directives_stage import extract_document_id


logger = logging.getLogger(__name__)


# ============================================================================
# DISCOVERY
# ============================================================================


def discover_extract_registries(docs_root: Path) -> list[Path]:
    """
    Locate all ``registry/extract`` directories under a root path.

    Args:
        docs_root: Root search directory (e.g., ``docs/source``).

    Returns:
        Sorted list of directories that contain extraction outputs.
    """
    extract_dirs: list[Path] = []

    for directory in docs_root.rglob("registry/extract"):
        if not directory.is_dir():
            continue

        document_dir = directory.parent.parent

        # Skip previously unified/combined registries
        if document_dir.name in {"unified_registry", "combined_registry"}:
            continue

        extract_dirs.append(directory)

    return sorted(extract_dirs)


# ============================================================================
# LOADING / NORMALISATION
# ============================================================================


def _normalise_items(payload: object, *, source: Path, document_id: str) -> list[dict]:
    """
    Normalise a registry payload into a list of entity dictionaries.

    The per-document extract files are plain lists. For robustness we also
    accept dictionaries containing an ``items`` field, mirroring the directives
    registry structure.
    """
    if isinstance(payload, list):
        # All elements should be dictionaries; non-dicts are filtered out.
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]

    logger.warning(
        "Ignoring unsupported payload in %s (document %s)",
        source,
        document_id,
    )
    return []


def load_extract_files_by_type(
    extract_dirs: Iterable[Path],
) -> dict[str, list[tuple[str, Path, list[dict]]]]:
    """
    Load every extract JSON file and group them by entity type (filename stem).

    Returns:
        Mapping of ``entity_type`` -> list of tuples
        ``(document_id, source_path, items)``.
    """
    files_by_type: dict[str, list[tuple[str, Path, list[dict]]]] = defaultdict(list)

    for directory in extract_dirs:
        document_path = directory.parent.parent
        document_id = extract_document_id(document_path)

        for json_file in directory.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as fh:
                    data = json.load(fh)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse %s: %s", json_file, exc)
                continue
            except Exception as exc:  # pragma: no cover - filesystem errors
                logger.warning("Error reading %s: %s", json_file, exc)
                continue

            items = _normalise_items(data, source=json_file, document_id=document_id)
            entity_type = json_file.stem

            if not items:
                logger.debug(
                    "No items found in %s for %s (document %s)",
                    json_file.name,
                    entity_type,
                    document_id,
                )
                continue

            files_by_type[entity_type].append((document_id, json_file, items))

    return files_by_type


# ============================================================================
# MERGING
# ============================================================================


def merge_extract_items(
    files_by_type: dict[str, list[tuple[str, Path, list[dict]]]],
) -> dict[str, list[dict]]:
    """
    Merge per-document items into corpus-level lists grouped by entity type.
    """
    unified: dict[str, list[dict]] = {}

    for entity_type, entries in sorted(files_by_type.items()):
        aggregated: list[dict] = []

        for document_id, json_file, items in sorted(entries, key=lambda entry: entry[0]):
            aggregated.extend(items)
            logger.debug(
                "Added %s %s entries from %s (%s)",
                len(items),
                entity_type,
                document_id,
                json_file.name,
            )

        unified[entity_type] = aggregated
        logger.info(
            "Merged %s entries for %s across %s documents",
            len(aggregated),
            entity_type,
            len(entries),
        )

    return unified


# ============================================================================
# OUTPUT
# ============================================================================


def _resolve_output_directory(docs_root: Path, output_dir: Path | None) -> Path:
    """
    Determine where the unified registry should be written.
    """
    if output_dir is not None:
        return output_dir

    current = docs_root.resolve()
    project_root = current

    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "src").exists():
            project_root = current
            break
        current = current.parent
    else:
        if docs_root.name == "source":
            project_root = docs_root.parent.parent
        else:
            project_root = docs_root.parent

    return project_root / "unified_registry"


def write_unified_extracts(
    merged_items: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """
    Persist merged extract files under ``output_dir/extract``.
    """
    extract_dir = output_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "types": {},
    }

    for entity_type, items in merged_items.items():
        output_path = extract_dir / f"{entity_type}.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=2, ensure_ascii=False)

        manifest["types"][entity_type] = {"count": len(items)}
        logger.info("Wrote %s entries to %s", len(items), output_path)

    manifest_path = extract_dir / "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    logger.info("Manifest written to %s", manifest_path)


# ============================================================================
# ORCHESTRATION
# ============================================================================


def build_unified_extract_registry(
    docs_root: Path,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> bool:
    """
    Build ``unified_registry/extract`` from all per-document extract files.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    extract_dirs = discover_extract_registries(docs_root)

    if not extract_dirs:
        logger.error("No registry/extract directories found under %s", docs_root)
        return False

    logger.info("Discovered %s extract directories", len(extract_dirs))

    files_by_type = load_extract_files_by_type(extract_dirs)

    if not files_by_type:
        logger.error("No extract JSON files were discovered.")
        return False

    merged_items = merge_extract_items(files_by_type)

    if not merged_items:
        logger.error("Nothing to merge after loading extract files.")
        return False

    destination = _resolve_output_directory(docs_root, output_dir)
    write_unified_extracts(merged_items, destination)

    logger.info(
        "Unified extract registry complete: %s entity types written to %s",
        len(merged_items),
        destination / "extract",
    )
    return True


# ============================================================================
# CLI
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine document-level extraction outputs into a unified registry.",
    )
    parser.add_argument(
        "docs_root",
        type=Path,
        help="Root directory containing documents (e.g., docs/source).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional custom output directory for the unified registry.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    if not args.docs_root.exists():
        logger.error("Path does not exist: %s", args.docs_root)
        return 1

    try:
        success = build_unified_extract_registry(
            docs_root=args.docs_root,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Fatal error: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
