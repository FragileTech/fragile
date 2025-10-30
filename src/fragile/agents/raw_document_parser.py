"""
Raw Document Parser Agent - Stage 1 Extraction.

Autonomous agent that extracts raw mathematical content from MyST markdown documents
into staging JSON files following the Extract-then-Enrich pipeline.

Stage 1: Raw Extraction
- Input: MyST markdown document
- Process: LLM-based verbatim transcription
- Output: Individual raw JSON files per entity type
- Output directory: <source_dir>/raw_data/

Architecture:
    1. Split document into sections
    2. Extract directive hints (hybrid parsing)
    3. Call LLM for each section
    4. Merge sections into single StagingDocument
    5. Export individual JSON files with temp IDs
    6. Generate statistics

Output Structure:
    <source_dir>/
    ├── raw_data/
    │   ├── definitions/
    │   │   ├── raw-def-001.json
    │   │   └── ...
    │   ├── theorems/
    │   ├── axioms/
    │   ├── proofs/
    │   ├── equations/
    │   ├── parameters/
    │   ├── remarks/
    │   └── citations/
    └── statistics/
        └── raw_statistics.json

Usage:
    from fragile.agents.raw_document_parser import RawDocumentParser

    parser = RawDocumentParser("docs/source/1_euclidean_gas/03_cloning.md")
    result = parser.extract()

    # Or use convenience function:
    from fragile.agents.raw_document_parser import extract_document
    result = extract_document("docs/source/1_euclidean_gas/03_cloning.md")
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path

from fragile.proofs.llm.pipeline_orchestration import (
    merge_sections,
    process_sections_parallel,
)
from fragile.proofs.staging_types import StagingDocument
from fragile.proofs.tools import split_into_sections


logger = logging.getLogger(__name__)


class RawDocumentParser:
    """
    Stage 1 extraction agent that processes MyST markdown into raw staging JSON.

    This parser performs verbatim extraction of mathematical entities using LLM-based
    transcription and exports them to individual JSON files following the staging_types
    schema.
    """

    def __init__(
        self,
        source_path: Path | str,
        output_dir: Path | str | None = None,
        model: str = "claude-sonnet-4",
    ):
        """
        Initialize raw document parser.

        Args:
            source_path: Path to MyST markdown file
            output_dir: Custom output directory (default: auto-detect from source)
            model: LLM model to use for extraction
        """
        self.source_path = Path(source_path)
        self.model = model

        # Validate source exists
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")

        if not self.source_path.is_file():
            raise ValueError(f"Source must be a file, not directory: {self.source_path}")

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self._auto_detect_output_dir()

        # Storage
        self.staging_doc: StagingDocument | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def _auto_detect_output_dir(self) -> Path:
        """
        Auto-detect output directory from source path.

        Pattern: docs/source/chapter/document.md -> docs/source/chapter/document/
        """
        # Get parent directory and stem (filename without extension)
        parent = self.source_path.parent
        stem = self.source_path.stem
        return parent / stem

    def extract(self) -> dict:
        """
        Perform Stage 1 raw extraction.

        Returns:
            Dictionary with extraction results and statistics

        Workflow:
            1. Read markdown file
            2. Split into sections
            3. Process each section with LLM
            4. Merge sections
            5. Export to individual JSON files
            6. Generate statistics
        """
        logger.info("=" * 60)
        logger.info("🚀 Document Parser - Stage 1: Raw Extraction")
        logger.info("=" * 60)
        logger.info(f"   Source: {self.source_path}")
        logger.info(f"   Output: {self.output_dir}")
        logger.info("")

        self.start_time = datetime.now()

        try:
            # Step 1: Read document
            logger.info("📄 Reading document...")
            markdown_text = self.source_path.read_text(encoding="utf-8")

            # Step 2: Split into sections
            logger.info("📋 Splitting into sections...")
            sections = split_into_sections(markdown_text)
            logger.info(f"   Found {len(sections)} sections")
            logger.info("")

            # Step 3: Process sections with LLM
            logger.info("Stage 1: Raw Extraction")
            staging_docs = process_sections_parallel(sections=sections, model=self.model)

            # Step 4: Merge sections
            logger.info("")
            logger.info("📦 Merging sections...")
            self.staging_doc = merge_sections(staging_docs)
            logger.info(f"   Total entities: {self.staging_doc.total_entities}")
            logger.info("")

            # Step 5: Export individual JSON files
            logger.info("💾 Exporting individual JSON files...")
            self._export_raw_entities()

            # Step 6: Generate statistics
            logger.info("")
            logger.info("📊 Generating statistics...")
            stats = self._generate_statistics()

            self.end_time = datetime.now()
            elapsed = (self.end_time - self.start_time).total_seconds()

            logger.info("")
            logger.info("=" * 60)
            logger.info("✅ Raw extraction complete!")
            logger.info("=" * 60)
            logger.info(f"   Output: {self.output_dir}/raw_data/")
            logger.info(f"   Time: {elapsed:.1f} seconds")
            logger.info("")

            return {
                "status": "success",
                "output_dir": str(self.output_dir),
                "statistics": stats,
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            raise

    def _export_raw_entities(self):
        """
        Export raw entities to individual JSON files.

        Creates directory structure:
            raw_data/
                definitions/raw-def-001.json, ...
                theorems/raw-thm-001.json, ...
                axioms/raw-axiom-001.json, ...
                proofs/raw-proof-001.json, ...
                equations/raw-eq-001.json, ...
                parameters/raw-param-001.json, ...
                remarks/raw-remark-001.json, ...
                citations/raw-cite-001.json, ...
        """
        if not self.staging_doc:
            msg = "No staging document to export (run extract() first)"
            raise ValueError(msg)

        # Create base directory
        raw_data_dir = self.output_dir / "raw_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Export each entity type
        self._export_entity_type(
            entities=self.staging_doc.definitions, subdir="definitions", raw_data_dir=raw_data_dir
        )

        self._export_entity_type(
            entities=self.staging_doc.theorems, subdir="theorems", raw_data_dir=raw_data_dir
        )

        self._export_entity_type(
            entities=self.staging_doc.axioms, subdir="axioms", raw_data_dir=raw_data_dir
        )

        self._export_entity_type(
            entities=self.staging_doc.proofs, subdir="proofs", raw_data_dir=raw_data_dir
        )

        self._export_entity_type(
            entities=self.staging_doc.equations, subdir="equations", raw_data_dir=raw_data_dir
        )

        self._export_entity_type(
            entities=self.staging_doc.parameters, subdir="parameters", raw_data_dir=raw_data_dir
        )

        self._export_entity_type(
            entities=self.staging_doc.remarks, subdir="remarks", raw_data_dir=raw_data_dir
        )

        self._export_entity_type(
            entities=self.staging_doc.citations, subdir="citations", raw_data_dir=raw_data_dir
        )

    def _export_entity_type(self, entities: list, subdir: str, raw_data_dir: Path):
        """
        Export a list of entities to a subdirectory.

        Args:
            entities: List of Pydantic models (RawDefinition, RawTheorem, etc.)
            subdir: Subdirectory name (e.g., "definitions", "theorems")
            raw_data_dir: Parent raw_data directory
        """
        # Always create subdirectory (even if empty)
        subdir_path = raw_data_dir / subdir
        subdir_path.mkdir(exist_ok=True)

        if not entities:
            logger.info(f"   ✓ {subdir}/: 0 files")
            return

        # Export each entity
        for entity in entities:
            # Get temp_id from entity
            temp_id = entity.temp_id
            filename = f"{temp_id}.json"
            filepath = subdir_path / filename

            # Export as JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(entity.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        logger.info(f"   ✓ {subdir}/: {len(entities)} files")

    def _generate_statistics(self) -> dict:
        """
        Generate extraction statistics.

        Returns:
            Statistics dictionary
        """
        if not self.staging_doc:
            msg = "No staging document (run extract() first)"
            raise ValueError(msg)

        elapsed = None
        if self.start_time and self.end_time:
            elapsed = (self.end_time - self.start_time).total_seconds()

        stats = {
            "source_file": str(self.source_path),
            "processing_stage": "raw_extraction",
            "entities_extracted": {
                "definitions": len(self.staging_doc.definitions),
                "theorems": len(self.staging_doc.theorems),
                "axioms": len(self.staging_doc.axioms),
                "proofs": len(self.staging_doc.proofs),
                "equations": len(self.staging_doc.equations),
                "parameters": len(self.staging_doc.parameters),
                "remarks": len(self.staging_doc.remarks),
                "citations": len(self.staging_doc.citations),
            },
            "total_entities": self.staging_doc.total_entities,
            "extraction_time_seconds": elapsed,
            "output_directory": str(self.output_dir / "raw_data"),
            "timestamp": datetime.now().isoformat(),
        }

        # Write statistics to file
        stats_dir = self.output_dir / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)

        stats_file = stats_dir / "raw_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"   ✓ Statistics written to: {stats_file}")

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def extract_document(
    source_path: Path | str, output_dir: Path | str | None = None, model: str = "claude-sonnet-4"
) -> dict:
    """
    Convenience function for raw document extraction.

    Args:
        source_path: Path to MyST markdown file
        output_dir: Custom output directory (default: auto-detect)
        model: LLM model to use

    Returns:
        Extraction results dictionary

    Examples:
        >>> result = extract_document("docs/source/1_euclidean_gas/03_cloning.md")
        >>> print(result["statistics"]["total_entities"])
        135
    """
    parser = RawDocumentParser(source_path=source_path, output_dir=output_dir, model=model)
    return parser.extract()


def extract_multiple_documents(
    source_paths: list[Path | str],
    output_dirs: list[Path | str] | None = None,
    model: str = "claude-sonnet-4",
) -> list[dict]:
    """
    Extract multiple documents in sequence.

    Args:
        source_paths: List of document paths
        output_dirs: Optional list of custom output directories
        model: LLM model to use

    Returns:
        List of extraction results
    """
    results = []

    for i, source_path in enumerate(source_paths):
        output_dir = output_dirs[i] if output_dirs else None
        result = extract_document(source_path=source_path, output_dir=output_dir, model=model)
        results.append(result)

    return results
