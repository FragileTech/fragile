#!/usr/bin/env python3
"""
Build global unified mathematical entity registry from parser, enriched, and raw_data sources.

This script creates a GLOBAL preprocessed registry that loads from:
- Parser output: chapter_N.json files (from mathster.parsing pipeline)
- Enriched output: chapter_N.json files (from mathster.enrichment pipeline)
- Raw data: individual entity JSON files (from manual extraction)

The registry stores sources SEPARATELY (not merged) to enable independent visualization
and debugging of each pipeline stage.

The registry is FLAT BY ENTITY TYPE - all definitions together, all theorems together, etc.
Entities include metadata for chapter/document filtering in the dashboard.

Output: project_root/unified_registry/ with {entity_type}s.json files

Usage:
    # Build global registry for entire corpus
    python -m mathster.tools.build_unified_registry --all

    # Build registry for specific chapter
    python -m mathster.tools.build_unified_registry docs/source/1_euclidean_gas/

    # Rebuild with verbose output
    python -m mathster.tools.build_unified_registry --all --verbose
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from mathster.tools.enriched_data_loader import EnrichedDataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ParserLoader:
    """Load and flatten entities from parser/chapter_N.json files."""

    def __init__(self):
        self.entities_loaded = 0
        self.chapters_processed = 0
        self.errors = []

    def load_from_parser_directory(self, parser_dir: Path, document_id: str) -> list[dict]:
        """
        Load all entities from parser directory.

        Args:
            parser_dir: Path to parser/ directory
            document_id: Document identifier (e.g., "07_mean_field")

        Returns:
            List of flattened entity dicts
        """
        logger.info(f"Loading from parser directory: {parser_dir}")

        if not parser_dir.exists():
            logger.warning(f"Parser directory not found: {parser_dir}")
            return []

        all_entities = []
        chapter_files = sorted(parser_dir.glob("chapter_*.json"))

        if not chapter_files:
            logger.warning(f"No chapter files found in {parser_dir}")
            return []

        for chapter_file in chapter_files:
            # Extract chapter number from filename
            chapter_num = int(chapter_file.stem.split("_")[1])

            entities = self._load_chapter_file(chapter_file, chapter_num, document_id)
            all_entities.extend(entities)
            self.chapters_processed += 1

        logger.info(f"Loaded {len(all_entities)} entities from {len(chapter_files)} chapters")
        return all_entities

    def _load_chapter_file(
        self, chapter_file: Path, chapter_num: int, document_id: str
    ) -> list[dict]:
        """
        Load and flatten a single chapter_N.json file.

        Args:
            chapter_file: Path to chapter JSON file
            chapter_num: Chapter number
            document_id: Document identifier

        Returns:
            List of flattened entities from this chapter
        """
        try:
            with open(chapter_file, encoding="utf-8") as f:
                section_data = json.load(f)
        except Exception as e:
            error = f"Failed to load {chapter_file}: {e}"
            logger.error(error)
            self.errors.append(error)
            return []

        entities = []

        # Entity types to process
        entity_types = [
            "definitions",
            "theorems",
            "proofs",
            "axioms",
            "assumptions",
            "parameters",
            "remarks",
            "citations",
        ]

        section_id = section_data.get("section_id", "")

        # Flatten each entity type
        for entity_type in entity_types:
            entity_list = section_data.get(entity_type, [])

            for entity in entity_list:
                # Convert to dict if it's a Pydantic model
                if hasattr(entity, "model_dump"):
                    entity_dict = entity.model_dump()
                elif isinstance(entity, dict):
                    entity_dict = entity.copy()
                else:
                    logger.warning(f"Unexpected entity format: {type(entity)}")
                    continue

                # Add dashboard metadata
                entity_dict["_entity_type"] = entity_type.rstrip("s")  # "definitions" → "definition"
                entity_dict["_source_type"] = "parser"
                entity_dict["_section_id"] = section_id
                entity_dict["_chapter_number"] = chapter_num
                entity_dict["_parser_file"] = str(chapter_file.name)
                entity_dict["_document_id"] = document_id
                entity_dict["_chapter"] = None  # Will be set by builder

                # Extract line number for sorting
                source = entity_dict.get("source", {})
                line_range = source.get("line_range")

                if line_range and isinstance(line_range, dict):
                    lines = line_range.get("lines", [])
                    if lines and isinstance(lines, list) and len(lines) > 0:
                        if isinstance(lines[0], list) and len(lines[0]) >= 1:
                            entity_dict["_line_number"] = lines[0][0]
                        else:
                            entity_dict["_line_number"] = float("inf")
                    else:
                        entity_dict["_line_number"] = float("inf")
                else:
                    entity_dict["_line_number"] = float("inf")

                entities.append(entity_dict)
                self.entities_loaded += 1

        return entities


class RawDataLoader:
    """Load entities from raw_data/ directories (existing structure)."""

    def __init__(self):
        self.entities_loaded = 0
        self.errors = []

    def load_from_raw_data_directory(self, raw_data_dir: Path, document_id: str) -> list[dict]:
        """
        Load all entities from raw_data directory.

        Args:
            raw_data_dir: Path to raw_data/ directory
            document_id: Document identifier

        Returns:
            List of entity dicts
        """
        logger.info(f"Loading from raw_data directory: {raw_data_dir}")

        if not raw_data_dir.exists():
            logger.warning(f"Raw data directory not found: {raw_data_dir}")
            return []

        all_entities = []

        for entity_type_dir in sorted(raw_data_dir.iterdir()):
            if not entity_type_dir.is_dir() or entity_type_dir.name.startswith("."):
                continue

            entity_type = entity_type_dir.name.rstrip("s")  # "theorems" → "theorem"

            for json_file in entity_type_dir.glob("*.json"):
                # Skip metadata files
                if json_file.name in {
                    "refinement_report.json",
                    "object_refinement_report.json",
                    "object_fix_report.json",
                }:
                    continue

                try:
                    with open(json_file, encoding="utf-8") as f:
                        entity_dict = json.load(f)

                    # Add metadata
                    entity_dict["_entity_type"] = entity_type
                    entity_dict["_source_type"] = "raw_data"
                    entity_dict["_section_id"] = None
                    entity_dict["_chapter_number"] = None
                    entity_dict["_raw_data_file"] = str(json_file.name)
                    entity_dict["_document_id"] = document_id
                    entity_dict["_chapter"] = None  # Will be set by builder

                    # Extract line number
                    source = entity_dict.get("source", {})
                    line_range = source.get("line_range")

                    if line_range and isinstance(line_range, dict):
                        lines = line_range.get("lines", [])
                        if lines and isinstance(lines, list) and len(lines) > 0:
                            if isinstance(lines[0], list) and len(lines[0]) >= 1:
                                entity_dict["_line_number"] = lines[0][0]
                            else:
                                entity_dict["_line_number"] = float("inf")
                        else:
                            entity_dict["_line_number"] = float("inf")
                    else:
                        entity_dict["_line_number"] = float("inf")

                    all_entities.append(entity_dict)
                    self.entities_loaded += 1

                except Exception as e:
                    error = f"Failed to load {json_file}: {e}"
                    logger.warning(error)
                    self.errors.append(error)

        logger.info(f"Loaded {len(all_entities)} entities from raw_data")
        return all_entities


class RegistryMerger:
    """Merge entities from multiple sources with deduplication."""

    def __init__(self, deduplicate: bool = True):
        self.deduplicate = deduplicate
        self.entities_merged = 0
        self.duplicates_found = 0
        self.conflicts = []

    def merge(
        self, parser_entities: list[dict], raw_data_entities: list[dict]
    ) -> list[dict]:
        """
        Merge entities from parser and raw_data sources.

        Args:
            parser_entities: Entities from parser/
            raw_data_entities: Entities from raw_data/

        Returns:
            Merged entity list
        """
        logger.info(f"Merging {len(parser_entities)} parser + {len(raw_data_entities)} raw_data entities")

        if not self.deduplicate:
            # Simple concatenation
            merged = parser_entities + raw_data_entities
            logger.info(f"No deduplication: {len(merged)} total entities")
            return merged

        # Deduplicate by label (prefer raw_data over parser)
        entities_by_label = {}

        # Add parser entities first
        for entity in parser_entities:
            label = entity.get("label")
            if label:
                entities_by_label[label] = entity

        # Add/replace with raw_data entities (higher priority)
        for entity in raw_data_entities:
            label = entity.get("label")
            if label:
                if label in entities_by_label:
                    # Conflict: entity exists in both sources
                    self.duplicates_found += 1
                    existing = entities_by_label[label]

                    # Track conflict
                    self.conflicts.append(
                        {
                            "label": label,
                            "parser_type": existing.get("_entity_type"),
                            "raw_data_type": entity.get("_entity_type"),
                            "resolution": "Prefer raw_data",
                        }
                    )

                    # Mark entity as existing in both sources
                    entity["_in_both_sources"] = True
                    entity["_parser_version"] = existing

                entities_by_label[label] = entity

        merged = list(entities_by_label.values())
        logger.info(
            f"After deduplication: {len(merged)} entities ({self.duplicates_found} duplicates resolved)"
        )

        return merged


class UnifiedRegistryBuilder:
    """Build unified registry from parser, enriched, and raw_data sources."""

    def __init__(self, deduplicate: bool = True):
        self.deduplicate = deduplicate
        self.parser_loader = ParserLoader()
        self.enriched_data_loader = EnrichedDataLoader()
        self.raw_data_loader = RawDataLoader()
        self.merger = RegistryMerger(deduplicate=deduplicate)

    def build_for_document(self, document_path: Path) -> dict[str, Any]:
        """
        Build unified registry for a single document.

        Loads from THREE sources and stores them SEPARATELY (not merged):
        - Parser: docs/source/{chapter}/parser/chapter_N.json
        - Enriched: docs/source/{chapter}/{document}/enriched/chapter_N.json
        - Raw data: docs/source/{chapter}/{document}/raw_data/{entity_type}/*.json

        Args:
            document_path: Path to markdown document

        Returns:
            Registry metadata dict
        """
        logger.info(f"Building unified registry for: {document_path}")

        if not document_path.exists():
            logger.error(f"Document not found: {document_path}")
            return {}

        # Determine paths
        parent_dir = document_path.parent
        doc_name = document_path.stem
        parser_dir = parent_dir / "parser"
        enriched_dir = parent_dir / doc_name / "enriched"
        raw_data_dir = parent_dir / doc_name / "raw_data"

        # Load from THREE sources
        parser_entities = []
        enriched_entities = []
        raw_data_entities = []

        if parser_dir.exists():
            parser_entities = self.parser_loader.load_from_parser_directory(parser_dir, doc_name)
        else:
            logger.warning(f"No parser directory found at {parser_dir}")

        if enriched_dir.exists():
            enriched_entities = self.enriched_data_loader.load_from_enriched_directory(
                enriched_dir, doc_name
            )
        else:
            logger.info(f"No enriched directory found at {enriched_dir}")

        if raw_data_dir.exists():
            raw_data_entities = self.raw_data_loader.load_from_raw_data_directory(
                raw_data_dir, doc_name
            )
        else:
            logger.warning(f"No raw_data directory found at {raw_data_dir}")

        # Store sources SEPARATELY (not merged)
        # Group each source by entity type
        parser_by_type = self._group_by_entity_type(parser_entities)
        enriched_by_type = self._group_by_entity_type(enriched_entities)
        raw_data_by_type = self._group_by_entity_type(raw_data_entities)

        # Create output directory
        output_dir = parent_dir / doc_name / "unified_registry"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all entity types across all sources
        all_entity_types = set(parser_by_type.keys()) | set(enriched_by_type.keys()) | set(raw_data_by_type.keys())

        # Save entity files with sources stored separately
        for entity_type in sorted(all_entity_types):
            parser_entities_typed = parser_by_type.get(entity_type, [])
            enriched_entities_typed = enriched_by_type.get(entity_type, [])
            raw_data_entities_typed = raw_data_by_type.get(entity_type, [])

            # Create combined structure with separate sources
            entity_file_data = {
                "parsed": parser_entities_typed,
                "enriched": enriched_entities_typed,
                "raw_data": raw_data_entities_typed,
            }

            output_file = output_dir / f"{entity_type}s.json"
            with open(output_file, "w") as f:
                json.dump(entity_file_data, f, indent=2)

            total_count = len(parser_entities_typed) + len(enriched_entities_typed) + len(raw_data_entities_typed)
            logger.info(
                f"  Saved {entity_type}s: {len(parser_entities_typed)} parsed, "
                f"{len(enriched_entities_typed)} enriched, {len(raw_data_entities_typed)} raw_data "
                f"({total_count} total)"
            )

        # Calculate total entities across all sources
        total_entities = len(parser_entities) + len(enriched_entities) + len(raw_data_entities)

        # Calculate by_type statistics
        by_type_stats = {}
        for entity_type in all_entity_types:
            by_type_stats[entity_type] = {
                "parsed": len(parser_by_type.get(entity_type, [])),
                "enriched": len(enriched_by_type.get(entity_type, [])),
                "raw_data": len(raw_data_by_type.get(entity_type, [])),
                "total": (
                    len(parser_by_type.get(entity_type, []))
                    + len(enriched_by_type.get(entity_type, []))
                    + len(raw_data_by_type.get(entity_type, []))
                ),
            }

        # Generate metadata
        metadata = {
            "document_id": doc_name,
            "document_path": str(document_path),
            "build_timestamp": datetime.now().isoformat(),
            "sources": {
                "parser_dir": str(parser_dir) if parser_dir.exists() else None,
                "enriched_dir": str(enriched_dir) if enriched_dir.exists() else None,
                "raw_data_dir": str(raw_data_dir) if raw_data_dir.exists() else None,
            },
            "statistics": {
                "total_entities": total_entities,
                "parser_entities": len(parser_entities),
                "enriched_entities": len(enriched_entities),
                "raw_data_entities": len(raw_data_entities),
                "parser_chapters_processed": self.parser_loader.chapters_processed,
                "enriched_chapters_processed": self.enriched_data_loader.chapters_processed,
                "by_type": by_type_stats,
            },
            "errors": (
                self.parser_loader.errors
                + self.enriched_data_loader.errors
                + self.raw_data_loader.errors
            ),
        }

        # Save metadata
        metadata_file = output_dir / "registry_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✓ Registry built: {output_dir}")
        logger.info(f"  Total entities: {metadata['statistics']['total_entities']}")
        logger.info(f"  Parsed: {metadata['statistics']['parser_entities']}")
        logger.info(f"  Enriched: {metadata['statistics']['enriched_entities']}")
        logger.info(f"  Raw data: {metadata['statistics']['raw_data_entities']}")

        return metadata

    def build_for_chapter(self, chapter_dir: Path) -> dict[str, Any]:
        """
        Build unified registry for all documents in a chapter.

        Args:
            chapter_dir: Path to chapter directory (e.g., docs/source/1_euclidean_gas/)

        Returns:
            Combined registry metadata
        """
        logger.info(f"Building unified registry for chapter: {chapter_dir}")

        if not chapter_dir.exists():
            logger.error(f"Chapter directory not found: {chapter_dir}")
            return {}

        all_metadata = []

        # Find all markdown files (documents) in chapter
        for md_file in sorted(chapter_dir.glob("*.md")):
            metadata = self.build_for_document(md_file)
            if metadata:
                all_metadata.append(metadata)

        # Summary
        total_entities = sum(m["statistics"]["total_entities"] for m in all_metadata)

        logger.info(f"\n✓ Chapter registry complete:")
        logger.info(f"  Documents processed: {len(all_metadata)}")
        logger.info(f"  Total entities: {total_entities}")

        return {"documents": all_metadata, "total_entities": total_entities}

    def build_global_registry(self, docs_source_dir: Path, project_root: Path) -> dict[str, Any]:
        """
        Build global registry for entire corpus (FLAT BY ENTITY TYPE).

        Processes all documents across all chapters and creates a single global registry
        at project_root/unified_registry/ with flat structure by entity type.

        Args:
            docs_source_dir: Path to docs/source/ directory
            project_root: Path to project root

        Returns:
            Global registry metadata
        """
        logger.info(f"Building GLOBAL unified registry for corpus: {docs_source_dir}")
        logger.info(f"Output location: {project_root / 'unified_registry'}")

        all_entities = []
        documents_processed = []

        # Discover all chapters
        for chapter_dir in sorted(docs_source_dir.iterdir()):
            if not chapter_dir.is_dir() or chapter_dir.name.startswith("."):
                continue

            chapter_name = chapter_dir.name
            logger.info(f"\nProcessing chapter: {chapter_name}")

            # Process all markdown documents in chapter
            for md_file in sorted(chapter_dir.glob("*.md")):
                doc_name = md_file.stem
                logger.info(f"  Document: {doc_name}")

                # Paths
                parser_dir = chapter_dir / "parser"
                enriched_dir = chapter_dir / doc_name / "enriched"
                raw_data_dir = chapter_dir / doc_name / "raw_data"

                # Load entities from THREE sources (keep separate)
                parser_entities = []
                enriched_entities = []
                raw_data_entities = []

                if parser_dir.exists():
                    parser_entities = self.parser_loader.load_from_parser_directory(
                        parser_dir, doc_name
                    )
                    # Add chapter metadata
                    for e in parser_entities:
                        e["_chapter"] = chapter_name

                if enriched_dir.exists():
                    enriched_entities = self.enriched_data_loader.load_from_enriched_directory(
                        enriched_dir, doc_name
                    )
                    # Add chapter metadata
                    for e in enriched_entities:
                        e["_chapter"] = chapter_name

                if raw_data_dir.exists():
                    raw_data_entities = self.raw_data_loader.load_from_raw_data_directory(
                        raw_data_dir, doc_name
                    )
                    # Add chapter metadata
                    for e in raw_data_entities:
                        e["_chapter"] = chapter_name

                # Store all entities (SEPARATE, not merged)
                all_entities.extend(parser_entities)
                all_entities.extend(enriched_entities)
                all_entities.extend(raw_data_entities)

                documents_processed.append(
                    {
                        "chapter": chapter_name,
                        "document": doc_name,
                        "entities": len(parser_entities) + len(enriched_entities) + len(raw_data_entities),
                        "parser": len(parser_entities),
                        "enriched": len(enriched_entities),
                        "raw_data": len(raw_data_entities),
                    }
                )

        # Group entities by type AND source (NESTED format)
        # First, separate by source type
        parsed_entities = [e for e in all_entities if e.get("_source_type") == "parser"]
        enriched_entities = [e for e in all_entities if e.get("_source_type") == "enriched"]
        raw_data_entities = [e for e in all_entities if e.get("_source_type") == "raw_data"]

        # Then group each source by entity type
        parsed_by_type = self._group_by_entity_type(parsed_entities)
        enriched_by_type = self._group_by_entity_type(enriched_entities)
        raw_data_by_type = self._group_by_entity_type(raw_data_entities)

        # Get all entity types across all sources
        all_entity_types = set(parsed_by_type.keys()) | set(enriched_by_type.keys()) | set(raw_data_by_type.keys())

        # Create global output directory
        output_dir = project_root / "unified_registry"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save entity files in NESTED format
        for entity_type in sorted(all_entity_types):
            parsed_typed = parsed_by_type.get(entity_type, [])
            enriched_typed = enriched_by_type.get(entity_type, [])
            raw_data_typed = raw_data_by_type.get(entity_type, [])

            # Create nested structure
            entity_file_data = {
                "parsed": parsed_typed,
                "enriched": enriched_typed,
                "raw_data": raw_data_typed,
            }

            output_file = output_dir / f"{entity_type}s.json"
            with open(output_file, "w") as f:
                json.dump(entity_file_data, f, indent=2)

            total_count = len(parsed_typed) + len(enriched_typed) + len(raw_data_typed)
            logger.info(
                f"  Saved {entity_type}s: {len(parsed_typed)} parsed, "
                f"{len(enriched_typed)} enriched, {len(raw_data_typed)} raw_data "
                f"({total_count} total)"
            )

        # Calculate statistics by type
        by_type_stats = {}
        for entity_type in all_entity_types:
            by_type_stats[entity_type] = {
                "parsed": len(parsed_by_type.get(entity_type, [])),
                "enriched": len(enriched_by_type.get(entity_type, [])),
                "raw_data": len(raw_data_by_type.get(entity_type, [])),
                "total": (
                    len(parsed_by_type.get(entity_type, []))
                    + len(enriched_by_type.get(entity_type, []))
                    + len(raw_data_by_type.get(entity_type, []))
                ),
            }

        # Generate global metadata
        metadata = {
            "registry_type": "global",
            "build_timestamp": datetime.now().isoformat(),
            "corpus_root": str(docs_source_dir),
            "statistics": {
                "total_entities": len(all_entities),
                "total_parsed": len(parsed_entities),
                "total_enriched": len(enriched_entities),
                "total_raw_data": len(raw_data_entities),
                "documents_processed": len(documents_processed),
                "chapters_processed": len(set(d["chapter"] for d in documents_processed)),
                "by_type": by_type_stats,
                "by_chapter": self._count_by_chapter(all_entities),
            },
            "documents": documents_processed,
            "errors": (
                self.parser_loader.errors
                + self.enriched_data_loader.errors
                + self.raw_data_loader.errors
            ),
        }

        # Save metadata
        metadata_file = output_dir / "registry_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"✓ GLOBAL Registry built: {output_dir}")
        logger.info(f"  Total entities: {metadata['statistics']['total_entities']}")
        logger.info(f"  Documents: {metadata['statistics']['documents_processed']}")
        logger.info(f"  Chapters: {metadata['statistics']['chapters_processed']}")
        logger.info(f"  Parsed: {metadata['statistics']['total_parsed']}")
        logger.info(f"  Enriched: {metadata['statistics']['total_enriched']}")
        logger.info(f"  Raw data: {metadata['statistics']['total_raw_data']}")
        logger.info(f"{'='*80}")

        return metadata

    def _count_by_chapter(self, entities: list[dict]) -> dict[str, int]:
        """Count entities by chapter."""
        counts = {}
        for entity in entities:
            chapter = entity.get("_chapter", "unknown")
            counts[chapter] = counts.get(chapter, 0) + 1
        return counts

    def _group_by_entity_type(self, entities: list[dict]) -> dict[str, list[dict]]:
        """
        Group entities by type.

        Args:
            entities: List of entity dicts

        Returns:
            Dict mapping entity type to entity list
        """
        grouped = {}

        for entity in entities:
            entity_type = entity.get("_entity_type", "unknown")

            if entity_type not in grouped:
                grouped[entity_type] = []

            grouped[entity_type].append(entity)

        return grouped


def build_unified_registry(
    path: Path | None = None,
    all_corpus: bool = False,
    deduplicate: bool = True,
    project_root: Path | None = None,
) -> dict:
    """
    Build unified registry from parser and raw_data sources.

    Args:
        path: Path to document, chapter, or docs/source directory (None if --all)
        all_corpus: If True, build global registry for entire corpus
        deduplicate: If True, resolve duplicates (prefer raw_data)
        project_root: Path to project root (for output location)

    Returns:
        Registry metadata dict
    """
    builder = UnifiedRegistryBuilder(deduplicate=deduplicate)

    # Determine project root
    if project_root is None:
        if path:
            # Walk up to find project root (contains src/, docs/, etc.)
            current = path.resolve()
            while current.parent != current:
                if (current / "src").exists() and (current / "docs").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                logger.error("Could not find project root")
                return {}
        else:
            # Assume current directory is project root
            project_root = Path.cwd()

    if all_corpus:
        # Build GLOBAL registry for entire corpus
        docs_source_dir = project_root / "docs" / "source"
        return builder.build_global_registry(docs_source_dir, project_root)
    elif path and path.is_file():
        # Single document (old behavior - per-document registry)
        return builder.build_for_document(path)
    elif path and path.is_dir():
        # Chapter directory (old behavior)
        return builder.build_for_chapter(path)
    else:
        logger.error(f"Invalid path or missing --all flag")
        return {}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build global unified mathematical entity registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build GLOBAL registry for entire corpus (recommended)
  python -m mathster.tools.build_unified_registry --all

  # Build registry for specific chapter
  python -m mathster.tools.build_unified_registry docs/source/1_euclidean_gas/

  # Build with verbose output
  python -m mathster.tools.build_unified_registry --all --verbose

Output (global mode):
  Creates project_root/unified_registry/ directory with:
  - definitions.json (ALL definitions from ALL documents)
  - theorems.json (ALL theorems from ALL documents)
  - parameters.json (ALL parameters - includes parser extractions!)
  - {entity_type}s.json (one per type)
  - registry_metadata.json (build info and statistics)
        """,
    )

    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to markdown document or chapter directory (optional if --all)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Build GLOBAL registry for entire corpus (saves to project_root/unified_registry/)",
    )

    parser.add_argument(
        "--deduplicate",
        dest="deduplicate",
        action="store_true",
        default=True,
        help="Deduplicate entities by label (default: True, prefer raw_data)",
    )

    parser.add_argument(
        "--no-deduplicate",
        dest="deduplicate",
        action="store_false",
        help="Keep all entities from both sources (no deduplication)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build registry
    metadata = build_unified_registry(
        path=args.path.resolve() if args.path else None,
        all_corpus=args.all,
        deduplicate=args.deduplicate,
    )

    if metadata:
        print("\n" + "=" * 80)
        print("✓ Unified registry build complete!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ Registry build failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
