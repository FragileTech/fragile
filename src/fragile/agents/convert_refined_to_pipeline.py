"""
Convert refined_data (Stage 2) to pipeline_data (Stage 3).

This script transforms semantically enriched entities from refined_data/
to framework-compatible schema in pipeline_data/, ensuring compliance with
fragile.proofs.core.math_types.py
"""

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


def load_json(file_path: Path) -> dict[str, Any] | None:
    """Load JSON file, return None on error."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error loading {file_path}: {e}", file=sys.stderr)
        return None


def save_json(file_path: Path, data: dict[str, Any]) -> bool:
    """Save JSON file with pretty formatting."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"  Error saving {file_path}: {e}", file=sys.stderr)
        return False


def add_source_fields(source: dict[str, Any]) -> dict[str, Any] | None:
    """Add equation and url_fragment fields to source object.

    Returns None if source has no valid data (all None/null values).
    """
    if source is None:
        return None

    # Check if source has any non-None values
    has_data = any([
        source.get("document_id"),
        source.get("file_path"),
        source.get("section"),
        source.get("directive_label"),
    ])

    if not has_data:
        return None

    return {
        "document_id": source.get("document_id"),
        "file_path": source.get("file_path"),
        "section": source.get("section"),
        "directive_label": source.get("directive_label"),
        "equation": source.get("equation"),  # Usually null
        "line_range": source.get("line_range"),
        "url_fragment": source.get("url_fragment"),  # Usually null
    }


def convert_axiom(refined: dict[str, Any]) -> dict[str, Any]:
    """Convert axiom from refined_data to pipeline_data schema."""
    # Get source from refined data (could be "source" or "source_location")
    source = refined.get("source_location") or refined.get("source", {})

    return {
        "label": refined["label"],
        "statement": refined.get("statement", ""),
        "mathematical_expression": refined.get("mathematical_expression", ""),
        "foundational_framework": refined.get("foundational_framework", "Fragile Gas Framework"),
        "chapter": refined.get("chapter"),
        "document": refined.get("document"),
        "name": refined.get("name"),
        "core_assumption": refined.get("core_assumption"),
        "parameters": refined.get("parameters"),
        "condition": refined.get("condition"),
        "failure_mode_analysis": refined.get("failure_mode_analysis"),
        "source": add_source_fields(source),
    }


def convert_definition_to_object(refined: dict[str, Any]) -> dict[str, Any] | None:
    """Convert definition from refined_data to object in pipeline_data schema."""
    label = refined.get("label")
    if not label:
        print(
            f"  Warning: Missing label in definition: {refined.get('name', 'unknown')}",
            file=sys.stderr,
        )
        return None

    # Convert def-* to obj-* (handle double obj-obj- prefix)
    if label.startswith("obj-obj-"):
        object_label = label[4:]  # Remove extra obj- prefix
    elif label.startswith("def-"):
        object_label = label.replace("def-", "obj-", 1)
    elif label.startswith("obj-"):
        object_label = label
    else:
        object_label = f"obj-{label}"

    # Get source from refined data (could be "source" or "source_location")
    source = refined.get("source_location") or refined.get("source", {})

    # Extract mathematical expression from formal_statement
    formal_stmt = refined.get("formal_statement", "")
    math_expr = formal_stmt or refined.get("name", "Mathematical object")

    # Build result with required fields
    result = {
        "label": object_label,
        "name": refined.get("name", "Unnamed object"),
        "mathematical_expression": math_expr,
        "object_type": refined.get("object_type", "structure"),
        "current_attributes": [],  # Attributes added by theorems
        "attribute_history": [],
        "tags": refined.get("tags", []),
        "source": add_source_fields(source),
        "chapter": refined.get("chapter"),
        "document": refined.get("document"),
    }

    # Only add definition_label if original label was def-*
    # (standalone objects don't have a formal definition)
    if label.startswith("def-"):
        result["definition_label"] = label

    return result


def convert_property_to_attribute(prop: dict[str, Any]) -> dict[str, Any]:
    """Convert property object to attribute object (rename fields)."""
    # Get source if present (could be "source" or "source_location")
    source = prop.get("source_location") or prop.get("source", {})
    source_location = add_source_fields(source) if source else None

    return {
        "label": prop.get("label", ""),
        "expression": prop.get("expression", ""),
        "object_label": prop.get("object_label", ""),
        "established_by": prop.get("established_by", ""),
        "timestamp": prop.get("timestamp"),
        "can_be_refined": prop.get("can_be_refined", False),
        "refinements": prop.get("refinements", []),
        "source": source_location,
    }


def convert_theorem(refined: dict[str, Any]) -> dict[str, Any]:
    """Convert theorem/lemma from refined_data to pipeline_data schema."""
    label = refined["label"]
    statement_type = refined.get("statement_type", "theorem")

    # Get source (could be "source" or "source_location")
    source = refined.get("source_location") or refined.get("source", {})

    # Convert properties_added to attributes_added
    properties_added = refined.get("properties_added", [])
    attributes_added = [convert_property_to_attribute(prop) for prop in properties_added]

    # Convert properties_required to attributes_required
    properties_required = refined.get("properties_required", {})
    attributes_required = properties_required  # Same structure, just renamed

    return {
        "label": label,
        "name": refined.get("name", ""),
        "statement_type": statement_type,
        "source": add_source_fields(source),
        "chapter": refined.get("chapter"),
        "document": refined.get("document"),
        "proof": refined.get("proof"),
        "proof_status": refined.get("proof_status", "unproven"),
        "input_objects": refined.get("input_objects", []),
        "input_axioms": refined.get("input_axioms", []),
        "input_parameters": refined.get("input_parameters", []),
        "attributes_required": attributes_required,
        "internal_lemmas": refined.get("internal_lemmas", []),
        "internal_propositions": refined.get("internal_propositions", []),
        "lemma_dag_edges": refined.get("lemma_dag_edges", []),
        "output_type": refined.get("output_type", "Property"),
        "attributes_added": attributes_added,
        "relations_established": refined.get("relations_established", []),
        "natural_language_statement": refined.get("natural_language_statement"),
        "assumptions": refined.get("assumptions", []),
        "conclusion": refined.get("conclusion"),
        "equation_label": refined.get("equation_label"),
        "uses_definitions": refined.get("uses_definitions", []),
        "validation_errors": refined.get("validation_errors", []),
        "raw_fallback": refined.get("raw_fallback"),
    }


def extract_parameters(all_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract unique parameters from all entities' input_parameters fields."""
    param_symbols: set[str] = set()

    for entity in all_files:
        params = entity.get("input_parameters", [])
        param_symbols.update(params)

    # Create Parameter objects (minimal schema)
    parameters = []
    for symbol in sorted(param_symbols):
        parameters.append({
            "label": f"param-{symbol.lower().replace('_', '-')}",
            "name": symbol,
            "symbol": symbol,
            "parameter_type": "real",  # Default, would need inference
            "domain": None,
            "constraints": None,
        })

    return parameters


def main():
    """Main conversion function."""
    # Paths
    refined_dir = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data")
    pipeline_dir = Path("docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data")

    # Statistics
    stats = defaultdict(int)
    stats["total_files"] = 0
    stats["converted"] = 0
    stats["failed"] = 0
    stats["skipped"] = 0

    all_entities = []

    # 1. Convert axioms
    print("\n=== Converting Axioms ===")
    axiom_files = list((refined_dir / "axioms").glob("*.json"))
    existing_axioms = {f.stem for f in (pipeline_dir / "axioms").glob("*.json")}

    for axiom_file in axiom_files:
        stats["total_files"] += 1

        if axiom_file.stem in existing_axioms:
            print(f"  Skipping {axiom_file.name} (already exists)")
            stats["skipped"] += 1
            continue

        refined = load_json(axiom_file)
        if refined is None:
            stats["failed"] += 1
            continue

        all_entities.append(refined)
        pipeline = convert_axiom(refined)

        output_path = pipeline_dir / "axioms" / axiom_file.name
        if save_json(output_path, pipeline):
            print(f"  ✓ Converted {axiom_file.name}")
            stats["converted"] += 1
        else:
            stats["failed"] += 1

    # 2. Convert definitions → objects
    print("\n=== Converting Definitions → Objects ===")
    def_files = list((refined_dir / "definitions").glob("*.json"))
    existing_objects = {f.stem for f in (pipeline_dir / "objects").glob("*.json")}

    for def_file in def_files:
        stats["total_files"] += 1

        refined = load_json(def_file)
        if refined is None:
            stats["failed"] += 1
            continue

        all_entities.append(refined)
        pipeline = convert_definition_to_object(refined)

        if pipeline is None:
            print(f"  Error: Failed to convert {def_file.name}")
            stats["failed"] += 1
            continue

        # Check if object already exists
        if pipeline["label"] in existing_objects:
            print(f"  Skipping {def_file.name} → {pipeline['label']} (already exists)")
            stats["skipped"] += 1
            continue

        output_path = pipeline_dir / "objects" / f"{pipeline['label']}.json"
        if save_json(output_path, pipeline):
            print(f"  ✓ Converted {def_file.name} → {pipeline['label']}")
            stats["converted"] += 1
        else:
            stats["failed"] += 1

    # 3. Convert standalone objects
    print("\n=== Converting Standalone Objects ===")
    obj_files = list((refined_dir / "objects").glob("*.json"))

    for obj_file in obj_files:
        stats["total_files"] += 1

        refined = load_json(obj_file)
        if refined is None:
            stats["failed"] += 1
            continue

        all_entities.append(refined)

        # Check if object already has obj-* label (not def-*)
        if refined.get("label", "").startswith("obj-"):
            label = refined["label"]
            if label in existing_objects:
                print(f"  Skipping {obj_file.name} (already exists)")
                stats["skipped"] += 1
                continue

            # Same transformation as definition → object
            pipeline = convert_definition_to_object(refined)
            if pipeline is None:
                print(f"  Error: Failed to convert {obj_file.name}")
                stats["failed"] += 1
                continue

            output_path = pipeline_dir / "objects" / f"{label}.json"

            if save_json(output_path, pipeline):
                print(f"  ✓ Converted {obj_file.name}")
                stats["converted"] += 1
            else:
                stats["failed"] += 1
        else:
            # Treat as definition
            pipeline = convert_definition_to_object(refined)
            if pipeline is None:
                print(f"  Error: Failed to convert {obj_file.name}")
                stats["failed"] += 1
                continue

            if pipeline["label"] in existing_objects:
                print(f"  Skipping {obj_file.name} → {pipeline['label']} (already exists)")
                stats["skipped"] += 1
                continue

            output_path = pipeline_dir / "objects" / f"{pipeline['label']}.json"
            if save_json(output_path, pipeline):
                print(f"  ✓ Converted {obj_file.name} → {pipeline['label']}")
                stats["converted"] += 1
            else:
                stats["failed"] += 1

    # 4. Convert lemmas → theorems
    print("\n=== Converting Lemmas → Theorems ===")
    lemma_files = list((refined_dir / "lemmas").glob("*.json"))
    existing_theorems = {f.stem for f in (pipeline_dir / "theorems").glob("*.json")}

    for lemma_file in lemma_files:
        stats["total_files"] += 1

        if lemma_file.stem in existing_theorems:
            print(f"  Skipping {lemma_file.name} (already exists)")
            stats["skipped"] += 1
            continue

        refined = load_json(lemma_file)
        if refined is None:
            stats["failed"] += 1
            continue

        all_entities.append(refined)
        pipeline = convert_theorem(refined)

        output_path = pipeline_dir / "theorems" / lemma_file.name
        if save_json(output_path, pipeline):
            print(f"  ✓ Converted {lemma_file.name}")
            stats["converted"] += 1
        else:
            stats["failed"] += 1

    # 5. Convert theorems
    print("\n=== Converting Theorems ===")
    thm_files = list((refined_dir / "theorems").glob("*.json"))

    for thm_file in thm_files:
        stats["total_files"] += 1

        if thm_file.stem in existing_theorems:
            print(f"  Skipping {thm_file.name} (already exists)")
            stats["skipped"] += 1
            continue

        refined = load_json(thm_file)
        if refined is None:
            stats["failed"] += 1
            continue

        all_entities.append(refined)
        pipeline = convert_theorem(refined)

        output_path = pipeline_dir / "theorems" / thm_file.name
        if save_json(output_path, pipeline):
            print(f"  ✓ Converted {thm_file.name}")
            stats["converted"] += 1
        else:
            stats["failed"] += 1

    # 6. Extract parameters
    print("\n=== Extracting Parameters ===")
    parameters = extract_parameters(all_entities)
    print(f"  Found {len(parameters)} unique parameters")

    for param in parameters:
        stats["total_files"] += 1
        output_path = pipeline_dir / "parameters" / f"{param['label']}.json"

        if output_path.exists():
            print(f"  Skipping {param['label']} (already exists)")
            stats["skipped"] += 1
            continue

        if save_json(output_path, param):
            print(f"  ✓ Created {param['label']}")
            stats["converted"] += 1
        else:
            stats["failed"] += 1

    # 7. Generate transformation report
    print("\n=== Generating Transformation Report ===")
    report = {
        "timestamp": "2025-01-28",
        "source_directory": str(refined_dir),
        "target_directory": str(pipeline_dir),
        "statistics": dict(stats),
        "entity_counts": {
            "axioms": len(list((pipeline_dir / "axioms").glob("*.json"))),
            "objects": len(list((pipeline_dir / "objects").glob("*.json"))),
            "theorems": len(list((pipeline_dir / "theorems").glob("*.json"))),
            "parameters": len(list((pipeline_dir / "parameters").glob("*.json"))),
        },
    }

    report_path = pipeline_dir / "transformation_report_refined_to_pipeline.json"
    if save_json(report_path, report):
        print(f"  ✓ Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRANSFORMATION COMPLETE")
    print("=" * 60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"  Converted: {stats['converted']}")
    print(f"  Skipped (already exists): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print("\nFinal entity counts:")
    for entity_type, count in report["entity_counts"].items():
        print(f"  {entity_type}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
