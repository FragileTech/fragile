#!/usr/bin/env python3
"""
Extract Section 7 (Swarm Measuring) from 01_fragile_gas_framework.md

This script extracts mathematical entities from lines 1242-1480 and saves them
to the appropriate subdirectories.
"""

from datetime import datetime
import json
import logging
from pathlib import Path

from fragile.proofs.llm.pipeline_orchestration import process_section
from fragile.proofs.tools import extract_jupyter_directives
from fragile.proofs.tools.directive_parser import DocumentSection


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_section_7():
    """Extract Section 7 (Swarm Measuring) from the framework document."""

    # Paths
    source_file = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md"
    )
    output_base = Path(
        "/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework"
    )

    logger.info(f"Reading section from {source_file}")

    # Read the specific section (lines 1242-1480)
    with open(source_file, encoding="utf-8") as f:
        lines = f.readlines()

    section_lines = lines[1241:1480]  # 0-indexed, so subtract 1
    section_content = "".join(section_lines)

    logger.info(f"Extracted {len(section_lines)} lines from Section 7")
    logger.info(f"Section length: {len(section_content)} characters")

    # Extract directive hints from this section first
    directives = extract_jupyter_directives(section_content, section_id="§7")
    logger.info(f"Found {len(directives)} directive hints:")
    for d in directives:
        logger.info(f"  - {d.directive_type}: {d.label}")

    # Create a DocumentSection object
    section = DocumentSection(
        section_id="§7",
        title="Swarm Measuring",
        level=2,  # Section 7 is level 2 (##)
        content=section_content,
        start_line=1242,
        end_line=1480,
        directives=directives,
    )

    # Process the section with LLM
    logger.info("Processing section with LLM...")
    staging_doc = process_section(section=section, model="claude-sonnet-4")

    logger.info("Section processed successfully")

    # Create output directories
    output_dirs = {
        "definitions": output_base / "definitions",
        "theorems": output_base / "theorems",
        "axioms": output_base / "axioms",
        "proofs": output_base / "proofs",
        "equations": output_base / "equations",
        "parameters": output_base / "parameters",
        "remarks": output_base / "remarks",
        "citations": output_base / "citations",
        "objects": output_base / "objects",
    }

    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Export individual JSON files
    stats = {
        "source_file": str(source_file),
        "section": "Section 7 (Swarm Measuring)",
        "lines": "1242-1480",
        "processing_stage": "raw_extraction",
        "entities_extracted": {},
        "extraction_time": datetime.now().isoformat(),
    }

    # Export definitions
    if staging_doc.definitions:
        logger.info(f"Exporting {len(staging_doc.definitions)} definitions...")
        for i, raw_def in enumerate(staging_doc.definitions, 1):
            file_path = output_dirs["definitions"] / f"{raw_def.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_def.model_dump(), f, indent=2)
        stats["entities_extracted"]["definitions"] = len(staging_doc.definitions)

    # Export theorems (includes lemmas, propositions, corollaries)
    if staging_doc.theorems:
        logger.info(f"Exporting {len(staging_doc.theorems)} theorems...")
        for raw_thm in staging_doc.theorems:
            file_path = output_dirs["theorems"] / f"{raw_thm.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_thm.model_dump(), f, indent=2)
        stats["entities_extracted"]["theorems"] = len(staging_doc.theorems)

    # Export axioms
    if staging_doc.axioms:
        logger.info(f"Exporting {len(staging_doc.axioms)} axioms...")
        for raw_axiom in staging_doc.axioms:
            file_path = output_dirs["axioms"] / f"{raw_axiom.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_axiom.model_dump(), f, indent=2)
        stats["entities_extracted"]["axioms"] = len(staging_doc.axioms)

    # Export proofs
    if staging_doc.proofs:
        logger.info(f"Exporting {len(staging_doc.proofs)} proofs...")
        for raw_proof in staging_doc.proofs:
            file_path = output_dirs["proofs"] / f"{raw_proof.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_proof.model_dump(), f, indent=2)
        stats["entities_extracted"]["proofs"] = len(staging_doc.proofs)

    # Export equations
    if staging_doc.equations:
        logger.info(f"Exporting {len(staging_doc.equations)} equations...")
        for raw_eq in staging_doc.equations:
            file_path = output_dirs["equations"] / f"{raw_eq.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_eq.model_dump(), f, indent=2)
        stats["entities_extracted"]["equations"] = len(staging_doc.equations)

    # Export parameters
    if staging_doc.parameters:
        logger.info(f"Exporting {len(staging_doc.parameters)} parameters...")
        for raw_param in staging_doc.parameters:
            file_path = output_dirs["parameters"] / f"{raw_param.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_param.model_dump(), f, indent=2)
        stats["entities_extracted"]["parameters"] = len(staging_doc.parameters)

    # Export remarks
    if staging_doc.remarks:
        logger.info(f"Exporting {len(staging_doc.remarks)} remarks...")
        for raw_remark in staging_doc.remarks:
            file_path = output_dirs["remarks"] / f"{raw_remark.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_remark.model_dump(), f, indent=2)
        stats["entities_extracted"]["remarks"] = len(staging_doc.remarks)

    # Export citations
    if staging_doc.citations:
        logger.info(f"Exporting {len(staging_doc.citations)} citations...")
        for raw_cite in staging_doc.citations:
            file_path = output_dirs["citations"] / f"{raw_cite.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_cite.model_dump(), f, indent=2)
        stats["entities_extracted"]["citations"] = len(staging_doc.citations)

    # Export objects (if any mathematical objects are defined)
    if staging_doc.objects:
        logger.info(f"Exporting {len(staging_doc.objects)} objects...")
        for raw_obj in staging_doc.objects:
            file_path = output_dirs["objects"] / f"{raw_obj.temp_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(raw_obj.model_dump(), f, indent=2)
        stats["entities_extracted"]["objects"] = len(staging_doc.objects)

    # Calculate total
    stats["total_entities"] = sum(stats["entities_extracted"].values())

    # Save statistics
    stats_file = output_base / "section7_extraction_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total entities extracted: {stats['total_entities']}")
    for entity_type, count in stats["entities_extracted"].items():
        logger.info(f"  - {entity_type}: {count}")
    logger.info(f"\nOutput directory: {output_base}")
    logger.info(f"Statistics saved to: {stats_file}")
    logger.info(f"{'=' * 60}\n")

    return stats


if __name__ == "__main__":
    try:
        stats = extract_section_7()
        print("\n✓ Extraction completed successfully!")
        print(f"✓ Total entities: {stats['total_entities']}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise
