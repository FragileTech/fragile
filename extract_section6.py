#!/usr/bin/env python3
"""
Extract Section 6 from 01_fragile_gas_framework.md.

Extracts mathematical entities from Section 6 (Algorithm Space and Distance Measurement)
using the raw document parser.
"""

import json
import logging
from pathlib import Path

from fragile.proofs.llm.pipeline_orchestration import process_section
from fragile.proofs.tools import (
    DocumentSection,
    extract_jupyter_directives,
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    # Read Section 6 (lines 1212-1241)
    source_file = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
    )

    logger.info("=" * 60)
    logger.info("🚀 Extracting Section 6: Algorithm Space and Distance Measurement")
    logger.info("=" * 60)
    logger.info(f"   Source: {source_file}")
    logger.info("")

    # Read the entire file
    with open(source_file, encoding="utf-8") as f:
        lines = f.readlines()

    # Extract Section 6 (lines 1212-1241, zero-indexed as 1211-1240)
    section_lines = lines[1211:1241]
    section_content = "".join(section_lines)

    logger.info("📄 Section 6 content:")
    logger.info(f"   Lines: 1212-1241 ({len(section_lines)} lines)")
    logger.info(f"   Characters: {len(section_content)}")
    logger.info("")

    # Extract directive hints FIRST
    logger.info("🔍 Extracting directive hints...")
    directives = extract_jupyter_directives(section_content)
    logger.info(f"   Found {len(directives)} directives:")
    for d in directives:
        logger.info(f"      - {d.directive_type}: {d.label} (lines {d.start_line}-{d.end_line})")
    logger.info("")

    # Create DocumentSection with directives
    section = DocumentSection(
        section_id="§6",
        title="Algorithm Space and Distance Measurement",
        level=2,
        start_line=1212,
        end_line=1241,
        content=section_content,
        directives=directives,
    )

    # Process section with LLM
    logger.info("🤖 Processing with LLM (claude-sonnet-4)...")
    logger.info("")

    staging_doc = process_section(section=section, model="claude-sonnet-4")

    logger.info("")
    logger.info("📊 Extraction results:")
    logger.info(f"   Definitions: {len(staging_doc.definitions)}")
    logger.info(f"   Theorems: {len(staging_doc.theorems)}")
    logger.info(f"   Axioms: {len(staging_doc.axioms)}")
    logger.info(f"   Proofs: {len(staging_doc.proofs)}")
    logger.info(f"   Equations: {len(staging_doc.equations)}")
    logger.info(f"   Parameters: {len(staging_doc.parameters)}")
    logger.info(f"   Remarks: {len(staging_doc.remarks)}")
    logger.info(f"   Citations: {len(staging_doc.citations)}")
    logger.info(f"   TOTAL: {staging_doc.total_entities}")
    logger.info("")

    # Create output directory
    output_dir = Path("/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework")
    raw_data_dir = output_dir / "raw_data"

    # Export entities to individual JSON files
    logger.info("💾 Exporting individual JSON files...")

    entity_types = [
        ("definitions", staging_doc.definitions, "raw-def"),
        ("theorems", staging_doc.theorems, "raw-thm"),
        ("axioms", staging_doc.axioms, "raw-axiom"),
        ("proofs", staging_doc.proofs, "raw-proof"),
        ("equations", staging_doc.equations, "raw-eq"),
        ("parameters", staging_doc.parameters, "raw-param"),
        ("remarks", staging_doc.remarks, "raw-remark"),
        ("citations", staging_doc.citations, "raw-cite"),
    ]

    total_files = 0

    for entity_dir, entities, prefix in entity_types:
        if not entities:
            continue

        dir_path = raw_data_dir / entity_dir
        dir_path.mkdir(parents=True, exist_ok=True)

        for idx, entity in enumerate(entities, start=1):
            # Assign temp_id if not present
            if not hasattr(entity, "temp_id") or not entity.temp_id:
                entity.temp_id = f"{prefix}-{idx:03d}"

            file_path = dir_path / f"{entity.temp_id}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entity.dict(), f, indent=2, ensure_ascii=False)

            total_files += 1

        logger.info(f"   ✓ {entity_dir}/: {len(entities)} files")

    # Generate statistics
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "source_file": str(source_file),
        "section": "§6 - Algorithm Space and Distance Measurement",
        "line_range": "1212-1241",
        "processing_stage": "raw_extraction",
        "entities_extracted": {
            "definitions": len(staging_doc.definitions),
            "theorems": len(staging_doc.theorems),
            "axioms": len(staging_doc.axioms),
            "proofs": len(staging_doc.proofs),
            "equations": len(staging_doc.equations),
            "parameters": len(staging_doc.parameters),
            "remarks": len(staging_doc.remarks),
            "citations": len(staging_doc.citations),
        },
        "total_entities": staging_doc.total_entities,
        "total_files": total_files,
        "output_directory": str(raw_data_dir),
    }

    stats_file = stats_dir / "section6_raw_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("📈 Statistics saved to:")
    logger.info(f"   {stats_file}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Section 6 extraction complete!")
    logger.info("=" * 60)
    logger.info(f"   Total entities: {staging_doc.total_entities}")
    logger.info(f"   Total files: {total_files}")
    logger.info(f"   Output: {raw_data_dir}/")
    logger.info("")

    return stats


if __name__ == "__main__":
    main()
