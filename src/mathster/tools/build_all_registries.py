#!/usr/bin/env python3
"""
Build All Registries Automatically (Document-Agnostic).

This master script:
1. Discovers all documents with refined_data/ automatically
2. Transforms refined_data → pipeline_data (if needed)
3. Builds per-document registries (refined + pipeline)
4. Aggregates all per-document registries into combined registries

Usage:
    # Build everything automatically
    python -m fragile.mathster.tools.build_all_registries --docs-root docs/source

    # Custom output directory
    python -m fragile.mathster.tools.build_all_registries \\
        --docs-root docs/source \\
        --output-root registries

    # Skip aggregation (only build per-document)
    python -m fragile.mathster.tools.build_all_registries \\
        --docs-root docs/source \\
        --skip-aggregate

Version: 1.0.0
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
from typing import List, Tuple


# ============================================================================
# DOCUMENT DISCOVERY
# ============================================================================


def discover_all_documents(docs_root: Path) -> list[tuple[str, str, Path]]:
    """Discover all documents with refined_data directories.

    Args:
        docs_root: Path to docs/source directory

    Returns:
        List of (chapter, document, refined_data_path) tuples
    """
    documents = []
    chapter_pattern = re.compile(r"^\d+_\w+$")

    # Find all chapters
    for chapter_path in docs_root.iterdir():
        if not chapter_path.is_dir():
            continue

        # Check if matches chapter pattern
        if not chapter_pattern.match(chapter_path.name):
            continue

        # Look for numbered document directories within chapter
        for doc_path in chapter_path.iterdir():
            if not doc_path.is_dir():
                continue

            # Check for refined_data directory
            refined_data_path = doc_path / "refined_data"
            if refined_data_path.exists():
                chapter = chapter_path.name
                document = doc_path.name
                documents.append((chapter, document, refined_data_path))

    return sorted(documents)


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================


class RegistryPipelineOrchestrator:
    """Orchestrates the complete registry building pipeline."""

    def __init__(self, docs_root: Path, output_root: Path, skip_aggregate: bool = False):
        """Initialize orchestrator.

        Args:
            docs_root: Path to docs/source directory
            output_root: Path to registries output root
            skip_aggregate: If True, skip combined registry aggregation
        """
        self.docs_root = docs_root
        self.output_root = output_root
        self.skip_aggregate = skip_aggregate
        self.stats = {
            "documents_discovered": 0,
            "transforms_run": 0,
            "refined_registries_built": 0,
            "pipeline_registries_built": 0,
            "errors": [],
        }

    def transform_refined_to_pipeline(self, chapter: str, document: str) -> bool:
        """Transform refined_data → pipeline_data for a document.

        Args:
            chapter: Chapter name (e.g., '1_euclidean_gas')
            document: Document name (e.g., '01_fragile_gas_framework')

        Returns:
            True if successful, False otherwise
        """
        input_path = self.docs_root / chapter / document / "refined_data"
        output_path = self.docs_root / chapter / document / "pipeline_data"

        # Skip if pipeline_data already exists
        if output_path.exists():
            print("    ⏭  Pipeline data already exists, skipping transformation")
            return True

        print("    Transforming refined_data → pipeline_data...")

        # Run enriched_to_math_types
        cmd = [
            sys.executable,
            "-m",
            "fragile.mathster.tools.enriched_to_math_types",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("    ✓ Transformation completed")
            self.stats["transforms_run"] += 1
            return True
        except subprocess.CalledProcessError as e:
            print(f"    ❌ Transformation failed: {e}")
            print(f"    Error output: {e.stderr}")
            self.stats["errors"].append(f"{document}: transformation failed")
            return False

    def build_refined_registry(self, chapter: str, document: str) -> bool:
        """Build refined registry for a document.

        Args:
            chapter: Chapter name
            document: Document name

        Returns:
            True if successful, False otherwise
        """
        output_path = self.output_root / "per_document" / document / "refined"

        print("    Building refined registry...")

        # Run build_refined_registry for this specific document
        self.docs_root / chapter / document / "refined_data"
        cmd = [
            sys.executable,
            "-m",
            "fragile.mathster.tools.build_refined_registry",
            "--docs-root",
            str(self.docs_root / chapter),  # Pass chapter as docs-root
            "--output",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"    ✓ Refined registry built: {output_path}")
            self.stats["refined_registries_built"] += 1
            return True
        except subprocess.CalledProcessError as e:
            print(f"    ❌ Build failed: {e}")
            print(f"    Error output: {e.stderr}")
            self.stats["errors"].append(f"{document}: refined registry build failed")
            return False

    def build_pipeline_registry(self, chapter: str, document: str) -> bool:
        """Build pipeline registry for a document.

        Args:
            chapter: Chapter name
            document: Document name

        Returns:
            True if successful, False otherwise
        """
        input_path = self.docs_root / chapter / document / "pipeline_data"
        output_path = self.output_root / "per_document" / document / "pipeline"

        # Skip if pipeline_data doesn't exist
        if not input_path.exists():
            print("    ⚠️  Pipeline data not found, skipping pipeline registry build")
            return False

        print("    Building pipeline registry...")

        # Run build_pipeline_registry
        cmd = [
            sys.executable,
            "-m",
            "fragile.mathster.tools.build_pipeline_registry",
            "--pipeline-dir",
            str(input_path),
            "--output",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"    ✓ Pipeline registry built: {output_path}")
            self.stats["pipeline_registries_built"] += 1
            return True
        except subprocess.CalledProcessError as e:
            print(f"    ❌ Build failed: {e}")
            print(f"    Error output: {e.stderr}")
            self.stats["errors"].append(f"{document}: pipeline registry build failed")
            return False

    def aggregate_combined_registries(self):
        """Aggregate all per-document registries into combined registries."""
        if self.skip_aggregate:
            print("\n⏭  Skipping combined registry aggregation (--skip-aggregate)")
            return

        print("\n" + "=" * 70)
        print("AGGREGATING COMBINED REGISTRIES")
        print("=" * 70)

        per_document_root = self.output_root / "per_document"

        # Aggregate refined registries
        print("\n📚 Aggregating refined registries...")
        refined_output = self.output_root / "combined" / "refined"
        cmd = [
            sys.executable,
            "-m",
            "fragile.mathster.tools.aggregate_registries",
            "--type",
            "refined",
            "--per-document-root",
            str(per_document_root),
            "--output",
            str(refined_output),
        ]

        try:
            subprocess.run(cmd, capture_output=False, text=True, check=True)
            print(f"✓ Combined refined registry created: {refined_output}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Refined aggregation failed: {e}")
            self.stats["errors"].append("combined refined aggregation failed")

        # Aggregate pipeline registries
        print("\n📚 Aggregating pipeline registries...")
        pipeline_output = self.output_root / "combined" / "pipeline"
        cmd = [
            sys.executable,
            "-m",
            "fragile.mathster.tools.aggregate_registries",
            "--type",
            "pipeline",
            "--per-document-root",
            str(per_document_root),
            "--output",
            str(pipeline_output),
        ]

        try:
            subprocess.run(cmd, capture_output=False, text=True, check=True)
            print(f"✓ Combined pipeline registry created: {pipeline_output}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline aggregation failed: {e}")
            self.stats["errors"].append("combined pipeline aggregation failed")

    def process_document(self, chapter: str, document: str, refined_data_path: Path):
        """Process a single document through the complete pipeline.

        Args:
            chapter: Chapter name
            document: Document name
            refined_data_path: Path to refined_data directory
        """
        print(f"\n{'=' * 70}")
        print(f"Processing: {chapter}/{document}")
        print("=" * 70)

        # Step 1: Transform refined → pipeline
        self.transform_refined_to_pipeline(chapter, document)

        # Step 2: Build refined registry
        self.build_refined_registry(chapter, document)

        # Step 3: Build pipeline registry
        self.build_pipeline_registry(chapter, document)

    def build_all(self):
        """Main method to build all registries."""
        print("=" * 70)
        print("BUILD ALL REGISTRIES (DOCUMENT-AGNOSTIC)")
        print("=" * 70)
        print(f"\nDocs root: {self.docs_root}")
        print(f"Output root: {self.output_root}")

        # Discover all documents
        documents = discover_all_documents(self.docs_root)
        self.stats["documents_discovered"] = len(documents)

        print(f"\n✓ Discovered {len(documents)} documents with refined_data:")
        for chapter, document, refined_path in documents:
            print(f"  - {chapter}/{document}")

        if not documents:
            print("\n⚠️  No documents found to process!")
            return

        # Process each document
        for chapter, document, refined_data_path in documents:
            self.process_document(chapter, document, refined_data_path)

        # Aggregate combined registries
        self.aggregate_combined_registries()

        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        """Print final statistics."""
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)

        print(f"\n✓ Documents Discovered:           {self.stats['documents_discovered']}")
        print(f"✓ Transforms Run:                 {self.stats['transforms_run']}")
        print(f"✓ Refined Registries Built:       {self.stats['refined_registries_built']}")
        print(f"✓ Pipeline Registries Built:      {self.stats['pipeline_registries_built']}")

        if self.stats["errors"]:
            print(f"\n⚠️  Errors: {len(self.stats['errors'])}")
            for error in self.stats["errors"]:
                print(f"    - {error}")
        else:
            print("\n✅ No errors!")


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build all registries automatically (document-agnostic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automatically discovers all documents and builds registries.

Examples:
  # Build everything (discover all documents, build per-document, aggregate combined)
  python -m fragile.mathster.tools.build_all_registries --docs-root docs/source

  # Custom output location
  python -m fragile.mathster.tools.build_all_registries \\
      --docs-root docs/source \\
      --output-root registries

  # Skip combined registry aggregation
  python -m fragile.mathster.tools.build_all_registries \\
      --docs-root docs/source \\
      --skip-aggregate
        """,
    )

    parser.add_argument(
        "--docs-root",
        type=Path,
        required=True,
        help="Path to docs/source directory",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("registries"),
        help="Path to registries output root (default: registries)",
    )

    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip combined registry aggregation (only build per-document)",
    )

    args = parser.parse_args()

    # Run pipeline
    orchestrator = RegistryPipelineOrchestrator(
        args.docs_root,
        args.output_root,
        args.skip_aggregate,
    )
    orchestrator.build_all()

    # Exit with error code if errors occurred
    if orchestrator.stats["errors"]:
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ ALL REGISTRIES BUILT SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
