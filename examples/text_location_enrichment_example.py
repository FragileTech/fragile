"""
Example: Text Location Enrichment Workflow

This example demonstrates how to use the TextLocationEnricher agent to add
precise source location metadata to raw mathematical entities extracted from
markdown documents.

The agent can be used in three modes:
1. Single file mode - Enrich one JSON file
2. Directory mode - Enrich all entities in raw_data/
3. Batch mode - Process entire corpus automatically

This example shows all three modes with complete workflow from extraction
through enrichment to validation.
"""

from pathlib import Path

from fragile.agents import (
    EnrichmentConfig,
    extract_document,
    RawDocumentParser,
    TextLocationEnricher,
)


# =============================================================================
# EXAMPLE 1: Single File Enrichment
# =============================================================================


def example_single_file():
    """Enrich a single theorem JSON file with source location."""
    print("=" * 70)
    print("EXAMPLE 1: Single File Enrichment")
    print("=" * 70)

    # Setup paths
    json_file = Path("docs/source/1_euclidean_gas/03_cloning/raw_data/theorems/thm-keystone.json")
    markdown_file = Path("docs/source/1_euclidean_gas/03_cloning.md")
    document_id = "03_cloning"

    # Create agent with default configuration
    agent = TextLocationEnricher()

    # Enrich single file
    success = agent.enrich_single_file(
        json_file=json_file, markdown_file=markdown_file, document_id=document_id
    )

    if success:
        print("\n✅ Single file enrichment successful!")
        print(f"   File: {json_file.name}")
        print(f"   Source: {markdown_file.name}\n")
    else:
        print("\n❌ Single file enrichment failed!\n")


# =============================================================================
# EXAMPLE 2: Directory Enrichment
# =============================================================================


def example_directory():
    """Enrich all entities in a raw_data/ directory."""
    print("=" * 70)
    print("EXAMPLE 2: Directory Enrichment")
    print("=" * 70)

    # Setup paths
    raw_data_dir = Path("docs/source/1_euclidean_gas/03_cloning/raw_data")
    markdown_file = Path("docs/source/1_euclidean_gas/03_cloning.md")
    document_id = "03_cloning"

    # Create agent with custom configuration
    config = EnrichmentConfig(
        force_re_enrich=False,  # Skip already-enriched entities
        entity_types=["theorems", "definitions", "axioms"],  # Only these types
        validate_after=True,  # Validate line ranges after enrichment
        verbose=True,  # Enable progress reporting
    )

    agent = TextLocationEnricher(config)

    # Enrich directory
    result = agent.enrich_directory(
        raw_data_dir=raw_data_dir,
        markdown_file=markdown_file,
        document_id=document_id,
    )

    print("\n📊 Enrichment Statistics:")
    print(f"   Total entities: {result.total}")
    print(f"   Succeeded: {result.succeeded}")
    print(f"   Coverage: {result.coverage:.1f}%\n")


# =============================================================================
# EXAMPLE 3: Batch Corpus Enrichment
# =============================================================================


def example_batch():
    """Enrich entire corpus automatically."""
    print("=" * 70)
    print("EXAMPLE 3: Batch Corpus Enrichment")
    print("=" * 70)

    # Setup corpus root
    docs_source_dir = Path("docs/source")

    # Create agent
    config = EnrichmentConfig(
        entity_types=["theorems", "definitions", "axioms"],
        validate_after=False,  # Skip validation for speed
        verbose=True,
    )

    agent = TextLocationEnricher(config)

    # Batch enrich corpus
    results = agent.batch_enrich_corpus(docs_source_dir=docs_source_dir)

    # Print summary
    print("\n📊 Corpus-Wide Statistics:")
    for doc_id, result in sorted(results.items()):
        print(f"   {doc_id:30s}: {result}")

    total_succeeded = sum(r.succeeded for r in results.values())
    total_count = sum(r.total for r in results.values())
    total_coverage = 100.0 * total_succeeded / total_count if total_count > 0 else 0.0

    print(f"\n   {'TOTAL':30s}: {total_succeeded}/{total_count} ({total_coverage:.1f}%)\n")


# =============================================================================
# EXAMPLE 4: Complete Workflow (Extract → Enrich → Validate)
# =============================================================================


def example_complete_workflow():
    """Demonstrate complete workflow from extraction to enrichment."""
    print("=" * 70)
    print("EXAMPLE 4: Complete Workflow (Extract → Enrich → Validate)")
    print("=" * 70)

    # Step 1: Extract raw entities from markdown
    print("\n📄 Step 1: Extracting raw entities...")
    markdown_file = Path("docs/source/1_euclidean_gas/03_cloning.md")

    extraction_result = extract_document(markdown_file)

    print(f"   ✅ Extracted {extraction_result.total_entities} entities")
    print(f"   Output: {extraction_result.raw_data_dir}\n")

    # Step 2: Enrich with source locations
    print("🔍 Step 2: Enriching with source locations...")

    agent = TextLocationEnricher(
        EnrichmentConfig(
            validate_after=True,
            verbose=True,
        )
    )

    enrich_result = agent.enrich_directory(
        raw_data_dir=extraction_result.raw_data_dir,
        markdown_file=markdown_file,
        document_id=extraction_result.document_id,
    )

    print(f"\n   ✅ Enriched {enrich_result.succeeded}/{enrich_result.total} entities")
    print(f"   Coverage: {enrich_result.coverage:.1f}%\n")

    # Step 3: Ready for Stage 2 (document-refiner)
    print("🚀 Step 3: Ready for Stage 2 transformation!")
    print(f"   Raw data with sources: {extraction_result.raw_data_dir}")
    print("   Next: Use document-refiner agent for semantic enrichment\n")


# =============================================================================
# EXAMPLE 5: Integration with Other Agents
# =============================================================================


def example_agent_integration():
    """Show how other agents can invoke TextLocationEnricher."""
    print("=" * 70)
    print("EXAMPLE 5: Integration with Other Agents")
    print("=" * 70)

    # Simulate document-parser invoking text-location-enricher
    print("\n📄 document-parser: Extracting entities...")

    markdown_file = Path("docs/source/1_euclidean_gas/03_cloning.md")

    # Stage 1: Extract
    parser = RawDocumentParser()
    extraction_result = parser.parse_document(markdown_file)

    print(f"   ✅ Extracted {extraction_result.total_entities} entities\n")

    # Stage 1.5: Automatically enrich with locations
    print("🔍 document-parser: Auto-enriching with source locations...")

    enricher = TextLocationEnricher(EnrichmentConfig(validate_after=True, verbose=False))

    enrich_result = enricher.enrich_directory(
        raw_data_dir=extraction_result.raw_data_dir,
        markdown_file=markdown_file,
        document_id=extraction_result.document_id,
    )

    print(f"   ✅ Auto-enriched {enrich_result.succeeded}/{enrich_result.total} entities")
    print(f"   Coverage: {enrich_result.coverage:.1f}%\n")

    # Stage 2: Ready for document-refiner
    print("✅ document-parser: Extraction + Enrichment complete!")
    print("   Raw data ready for Stage 2 (document-refiner)\n")


# =============================================================================
# EXAMPLE 6: CLI Usage
# =============================================================================


def example_cli_usage():
    """Show command-line interface usage."""
    print("=" * 70)
    print("EXAMPLE 6: CLI Usage")
    print("=" * 70)

    print("\n📋 Command-Line Interface Examples:\n")

    print("# Single file mode:")
    print(
        "python -m fragile.agents.text_location_enricher single \\\n"
        "    docs/source/.../raw_data/theorems/thm-keystone.json \\\n"
        "    --source docs/source/.../03_cloning.md \\\n"
        "    --document-id 03_cloning\n"
    )

    print("# Directory mode:")
    print(
        "python -m fragile.agents.text_location_enricher directory \\\n"
        "    docs/source/.../raw_data/ \\\n"
        "    --source docs/source/.../03_cloning.md \\\n"
        "    --document-id 03_cloning \\\n"
        "    --types theorems definitions axioms\n"
    )

    print("# Batch mode (entire corpus):")
    print(
        "python -m fragile.agents.text_location_enricher batch \\\n"
        "    docs/source/ \\\n"
        "    --types theorems definitions axioms\n"
    )

    print("# Force re-enrichment:")
    print(
        "python -m fragile.agents.text_location_enricher directory \\\n"
        "    docs/source/.../raw_data/ \\\n"
        "    --source docs/source/.../03_cloning.md \\\n"
        "    --document-id 03_cloning \\\n"
        "    --force\n"
    )

    print("# Quiet mode (suppress verbose output):")
    print(
        "python -m fragile.agents.text_location_enricher batch \\\n"
        "    docs/source/ \\\n"
        "    --quiet\n"
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("TEXT LOCATION ENRICHMENT EXAMPLES")
    print("=" * 70 + "\n")

    # Run examples (comment out any you don't want to run)
    # example_single_file()
    # example_directory()
    # example_batch()
    # example_complete_workflow()
    # example_agent_integration()
    example_cli_usage()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
