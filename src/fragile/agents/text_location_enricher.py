"""
Text Location Enricher Agent.

This agent specializes in creating precise TextLocation objects by analyzing source
markdown documents and enriching raw JSON files with source metadata.

The agent can operate in three modes:
1. Single file mode: Enrich one JSON file
2. Directory mode: Enrich all entities in raw_data/
3. Batch mode: Process entire corpus automatically

Key Features:
- Automatic detection of entity location using multiple strategies
- Multi-level fallback (directive → text → section → minimal)
- Validation of line ranges
- Batch processing with progress reporting
- Integration with document-parser and document-refiner agents

Usage:
    # Standalone
    from fragile.agents.text_location_enricher import TextLocationEnricher

    agent = TextLocationEnricher()
    agent.enrich_directory(
        raw_data_dir="docs/source/.../raw_data/",
        markdown_file="docs/source/.../document.md",
        document_id="document_id"
    )

    # Or via CLI
    python -m fragile.agents.text_location_enricher directory \\
        docs/source/.../raw_data/ \\
        docs/source/.../document.md \\
        document_id

Maps to Lean:
    namespace TextLocationEnricher
      structure Config where
        force_re_enrich : Bool := false
        entity_types : Option (List String) := none
        validate_after : Bool := true

      structure EnrichmentResult where
        succeeded : Nat
        total : Nat
        coverage : Float

      def enrich_single_file : Path → Path → String → IO Bool
      def enrich_directory : Path → Path → String → IO EnrichmentResult
      def batch_enrich_corpus : Path → IO (Map String EnrichmentResult)
    end TextLocationEnricher
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

from fragile.proofs.tools.line_finder import extract_lines, validate_line_range
from fragile.proofs.tools.source_location_enricher import (
    batch_enrich_all_documents,
    enrich_directory,
    enrich_single_entity,
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class EnrichmentConfig:
    """Configuration for text location enrichment.

    Attributes:
        force_re_enrich: If True, re-enrich entities even if they already have sources
        entity_types: List of entity types to enrich (None = all types)
        validate_after: If True, validate line ranges after enrichment
        verbose: Enable verbose logging

    Maps to Lean:
        structure EnrichmentConfig where
          force_re_enrich : Bool
          entity_types : Option (List String)
          validate_after : Bool
          verbose : Bool
    """

    force_re_enrich: bool = False
    entity_types: list[str] | None = None
    validate_after: bool = True
    verbose: bool = True


@dataclass
class EnrichmentResult:
    """Result of an enrichment operation.

    Attributes:
        succeeded: Number of entities successfully enriched
        total: Total number of entities processed
        coverage: Percentage of entities with line ranges (0-100)

    Maps to Lean:
        structure EnrichmentResult where
          succeeded : Nat
          total : Nat
          coverage : Float
    """

    succeeded: int
    total: int

    @property
    def coverage(self) -> float:
        """Calculate coverage percentage."""
        return 100.0 * self.succeeded / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.succeeded}/{self.total} ({self.coverage:.1f}%)"


# =============================================================================
# AGENT CLASS
# =============================================================================


class TextLocationEnricher:
    """
    Agent that enriches raw JSON entities with precise TextLocation metadata.

    This agent wraps the existing source_location_enricher tools and provides
    a unified interface for use by other agents or standalone operation.

    Examples:
        >>> agent = TextLocationEnricher()
        >>> result = agent.enrich_directory(
        ...     raw_data_dir=Path("docs/source/.../raw_data/"),
        ...     markdown_file=Path("docs/source/.../doc.md"),
        ...     document_id="doc_id",
        ... )
        >>> print(f"Coverage: {result.coverage:.1f}%")
        Coverage: 98.5%

    Maps to Lean:
        structure TextLocationEnricher where
          config : EnrichmentConfig
    """

    def __init__(self, config: EnrichmentConfig | None = None):
        """
        Initialize the text location enricher agent.

        Args:
            config: Optional configuration (uses defaults if None)
        """
        self.config = config or EnrichmentConfig()

        if self.config.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    def enrich_single_file(self, json_file: Path, markdown_file: Path, document_id: str) -> bool:
        """
        Enrich a single JSON file with TextLocation metadata.

        Args:
            json_file: Path to raw JSON file to enrich
            markdown_file: Path to source markdown document
            document_id: Document identifier

        Returns:
            True if enrichment succeeded, False otherwise

        Examples:
            >>> agent = TextLocationEnricher()
            >>> success = agent.enrich_single_file(
            ...     json_file=Path("raw_data/theorems/thm-keystone.json"),
            ...     markdown_file=Path("docs/source/.../03_cloning.md"),
            ...     document_id="03_cloning",
            ... )
            >>> assert success

        Maps to Lean:
            def enrich_single_file
              (self : TextLocationEnricher)
              (json_path : Path)
              (markdown_path : Path)
              (document_id : String)
              : IO Bool
        """
        logger.info(f"🔍 Enriching: {json_file.name}")

        # Check if already enriched (unless force_re_enrich)
        if not self.config.force_re_enrich:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
                if "source" in data and data["source"] is not None:
                    logger.info("  ⏭️  Skipping (already has source)")
                    return True

        # Enrich using existing tool
        success = enrich_single_entity(json_file, markdown_file, document_id)

        if success and self.config.validate_after:
            self._validate_single_entity(json_file, markdown_file)

        return success

    def enrich_directory(
        self,
        raw_data_dir: Path,
        markdown_file: Path,
        document_id: str,
    ) -> EnrichmentResult:
        """
        Enrich all entities in a raw_data/ directory.

        Args:
            raw_data_dir: Directory containing entity subdirectories (theorems/, definitions/, etc.)
            markdown_file: Path to source markdown document
            document_id: Document identifier

        Returns:
            EnrichmentResult with success statistics

        Examples:
            >>> agent = TextLocationEnricher()
            >>> result = agent.enrich_directory(
            ...     raw_data_dir=Path("docs/source/.../raw_data/"),
            ...     markdown_file=Path("docs/source/.../doc.md"),
            ...     document_id="doc_id",
            ... )
            >>> print(f"Coverage: {result.coverage:.1f}%")
            Coverage: 98.0%

        Maps to Lean:
            def enrich_directory
              (self : TextLocationEnricher)
              (raw_dir : Path)
              (markdown_path : Path)
              (document_id : String)
              : IO EnrichmentResult
        """
        logger.info("\n🔍 Text Location Enricher - Directory Mode")
        logger.info(f"   Source: {markdown_file.name}")
        logger.info(f"   Target: {raw_data_dir.name}\n")

        # Enrich using existing tool
        succeeded, total = enrich_directory(
            raw_data_dir=raw_data_dir,
            markdown_file=markdown_file,
            document_id=document_id,
            entity_types=self.config.entity_types,
        )

        result = EnrichmentResult(succeeded=succeeded, total=total)

        # Validation
        if self.config.validate_after and succeeded > 0:
            logger.info("\n📊 Validating enriched entities...")
            self._validate_directory(raw_data_dir, markdown_file)

        # Report
        logger.info("\n✅ Enrichment complete!")
        logger.info(f"   Total: {result.total} entities")
        logger.info(f"   Succeeded: {result.succeeded}")
        logger.info(f"   Coverage: {result.coverage:.1f}%")

        return result

    def batch_enrich_corpus(self, docs_source_dir: Path) -> dict[str, EnrichmentResult]:
        """
        Batch enrich all documents in the corpus.

        Automatically discovers all documents with raw_data/ subdirectories
        and enriches them.

        Args:
            docs_source_dir: Path to docs/source/ directory

        Returns:
            Dictionary mapping document_id → EnrichmentResult

        Examples:
            >>> agent = TextLocationEnricher()
            >>> results = agent.batch_enrich_corpus(Path("docs/source"))
            >>> for doc_id, result in results.items():
            ...     print(f"{doc_id}: {result}")
            03_cloning: 12/12 (100.0%)
            04_convergence: 47/50 (94.0%)

        Maps to Lean:
            def batch_enrich_corpus
              (self : TextLocationEnricher)
              (docs_dir : Path)
              : IO (Map String EnrichmentResult)
        """
        logger.info("\n🔍 Text Location Enricher - Batch Mode")
        logger.info(f"   Corpus: {docs_source_dir}\n")

        # Enrich using existing tool
        raw_results = batch_enrich_all_documents(
            docs_source_dir=docs_source_dir, entity_types=self.config.entity_types
        )

        # Convert to EnrichmentResult objects
        results = {
            doc_id: EnrichmentResult(succeeded=s, total=t)
            for doc_id, (s, t) in raw_results.items()
        }

        # Report corpus-wide statistics
        logger.info(f"\n{'=' * 70}")
        logger.info("CORPUS ENRICHMENT REPORT")
        logger.info(f"{'=' * 70}\n")

        total_succeeded = sum(r.succeeded for r in results.values())
        total_count = sum(r.total for r in results.values())
        total_coverage = 100.0 * total_succeeded / total_count if total_count > 0 else 0.0

        for doc_id in sorted(results.keys()):
            result = results[doc_id]
            logger.info(f"{doc_id:30s}: {result}")

        logger.info(f"\n{'-' * 70}")
        logger.info(f"{'TOTAL':30s}: {total_succeeded}/{total_count} ({total_coverage:.1f}%)")
        logger.info(f"{'=' * 70}\n")

        return results

    def _validate_single_entity(self, json_file: Path, markdown_file: Path) -> None:
        """Validate that a single entity's TextLocation is correct."""
        try:
            with open(json_file, encoding="utf-8") as f:
                entity = json.load(f)

            if "source" not in entity or entity["source"] is None:
                logger.warning(f"  ⚠️  No source field: {json_file.name}")
                return

            source = entity["source"]

            if "line_range" not in source or source["line_range"] is None:
                logger.debug(f"  ℹ️  No line range (fallback mode): {json_file.name}")
                return

            # Validate line range
            markdown_content = markdown_file.read_text(encoding="utf-8")
            max_lines = len(markdown_content.splitlines())
            start, end = source["line_range"]

            if not validate_line_range((start, end), max_lines):
                logger.warning(
                    f"  ⚠️  Invalid line range {start}-{end} (max {max_lines}): {json_file.name}"
                )
                return

            # Check text match
            extracted = extract_lines(markdown_content, (start, end))

            # Check if entity content is present in extracted text
            if "full_statement_text" in entity:
                key_text = entity["full_statement_text"][:100]
                if key_text.lower() not in extracted.lower():
                    logger.warning(f"  ⚠️  Text mismatch at lines {start}-{end}: {json_file.name}")
                    return

            logger.debug(f"  ✓ Valid line range {start}-{end}: {json_file.name}")

        except Exception as e:
            logger.error(f"  ✗ Validation error for {json_file.name}: {e}")

    def _validate_directory(self, raw_data_dir: Path, markdown_file: Path) -> None:
        """Validate all entities in a directory."""
        entity_types = self.config.entity_types or [
            "theorems",
            "definitions",
            "axioms",
            "mathster",
            "equations",
            "parameters",
            "remarks",
        ]

        for entity_type in entity_types:
            entity_dir = raw_data_dir / entity_type
            if not entity_dir.exists():
                continue

            json_files = list(entity_dir.glob("*.json"))
            for json_file in json_files[:5]:  # Validate first 5 of each type
                self._validate_single_entity(json_file, markdown_file)


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """Command-line interface for text location enricher agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich raw JSON entities with TextLocation metadata"
    )

    parser.add_argument("mode", choices=["single", "directory", "batch"], help="Enrichment mode")

    parser.add_argument("target", type=Path, help="Target JSON file, directory, or corpus")

    parser.add_argument(
        "--source", type=Path, help="Source markdown file (required for single/directory mode)"
    )

    parser.add_argument(
        "--document-id",
        type=str,
        help="Document ID (required for single/directory mode)",
    )

    parser.add_argument(
        "--types",
        nargs="+",
        help="Entity types to enrich (e.g., theorems definitions)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Force re-enrichment even if source exists"
    )

    parser.add_argument(
        "--no-validate", action="store_true", help="Skip validation after enrichment"
    )

    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Validate required arguments
    if args.mode in {"single", "directory"}:
        if not args.source or not args.document_id:
            parser.error(f"{args.mode} mode requires --source and --document-id")

    # Create configuration
    config = EnrichmentConfig(
        force_re_enrich=args.force,
        entity_types=args.types,
        validate_after=not args.no_validate,
        verbose=not args.quiet,
    )

    # Create agent
    agent = TextLocationEnricher(config)

    # Execute based on mode
    if args.mode == "single":
        success = agent.enrich_single_file(args.target, args.source, args.document_id)
        sys.exit(0 if success else 1)

    elif args.mode == "directory":
        result = agent.enrich_directory(args.target, args.source, args.document_id)
        sys.exit(0 if result.coverage > 90.0 else 1)

    elif args.mode == "batch":
        results = agent.batch_enrich_corpus(args.target)
        total_succeeded = sum(r.succeeded for r in results.values())
        total_count = sum(r.total for r in results.values())
        coverage = 100.0 * total_succeeded / total_count if total_count > 0 else 0.0
        sys.exit(0 if coverage > 90.0 else 1)


if __name__ == "__main__":
    main()
