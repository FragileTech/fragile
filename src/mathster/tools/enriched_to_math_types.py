#!/usr/bin/env python3
"""
Transform Enriched Data to Pipeline Math Types.

This module transforms enriched JSON (from document-refiner) into pipeline-ready
math_types format suitable for use in MathematicalRegistry.

Transformations:
- Enriched Axiom → math_types.Axiom
- Enriched Object → math_types.MathematicalObject
- Enriched Theorem → math_types.TheoremBox
- Enriched Parameter → math_types.Parameter
- ParameterBox → math_types.Parameter (uses type_conversions)
- EquationBox, RemarkBox → semantic linking (update existing entities)

Usage:
    python -m fragile.mathster.tools.enriched_to_math_types \\
        --input docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data \\
        --output docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data

Version: 1.0.0
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import warnings

from mathster.core.article_system import SourceLocation
from mathster.core.enriched_types import EquationBox, ParameterBox, RemarkBox
from mathster.core.math_types import (
    Attribute,
    Axiom,
    MathematicalObject,
    ObjectType,
    Parameter,
    ParameterType,
    Relationship,
    TheoremBox,
    TheoremOutputType,
)
from mathster.core.type_conversions import parameter_box_to_parameter
from mathster.tools.registry_builders_common import (
    create_source_location,
    preprocess_attributes_added,
    preprocess_lemma_edges,
    preprocess_relations_established,
)


# ============================================================================
# TRANSFORMATION REPORT
# ============================================================================


class TransformationReport:
    """Track transformation statistics and errors."""

    def __init__(self):
        self.objects_transformed = 0
        self.axioms_transformed = 0
        self.parameters_transformed = 0
        self.theorems_transformed = 0
        self.relationships_created = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, error: str):
        """Add error message."""
        self.errors.append(error)
        warnings.warn(error)

    def add_warning(self, warning: str):
        """Add warning message."""
        self.warnings.append(warning)

    def print_summary(self):
        """Print transformation summary."""
        print("\n" + "=" * 70)
        print("TRANSFORMATION SUMMARY")
        print("=" * 70)
        print("\n✓ Entities Transformed:")
        print(f"  - Objects:        {self.objects_transformed}")
        print(f"  - Axioms:         {self.axioms_transformed}")
        print(f"  - Parameters:     {self.parameters_transformed}")
        print(f"  - Theorems:       {self.theorems_transformed}")
        print(f"  - Relationships:  {self.relationships_created}")

        total = (
            self.objects_transformed
            + self.axioms_transformed
            + self.parameters_transformed
            + self.theorems_transformed
        )
        print(f"  - TOTAL:          {total}")

        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more warnings")

        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors[:5]:
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more errors")


# ============================================================================
# ENRICHED TO MATH TYPES TRANSFORMER
# ============================================================================


class EnrichedToMathTypesTransformer:
    """Transform enriched_data → pipeline math_types."""

    def __init__(self):
        self.report = TransformationReport()

    def transform_enriched_object(self, data: dict, json_path: Path) -> MathematicalObject | None:
        """Transform enriched Object JSON → MathematicalObject.

        Args:
            data: Enriched object JSON data
            json_path: Path to JSON file (for error reporting)

        Returns:
            MathematicalObject or None if transformation fails
        """
        try:
            # Extract required fields
            label = data.get("label")
            if not label:
                self.report.add_error(f"{json_path.name}: Missing label")
                return None

            # Map enriched fields to math_types fields
            obj_data = {
                "label": label,
                "name": data.get("name", label),
                "mathematical_expression": data.get("mathematical_expression", ""),
                "object_type": data.get("object_type", "set"),
                "current_attributes": data.get("current_attributes", []),
                "attribute_history": data.get("attribute_history", []),
                "tags": data.get("tags", []),
                "chapter": data.get("chapter"),
                "document": data.get("document"),
                "definition_label": data.get("definition_label"),
            }

            # Add source if present
            source = data.get("source")
            if source:
                if isinstance(source, dict):
                    obj_data["source"] = SourceLocation.model_validate(source)
                else:
                    obj_data["source"] = source
            else:
                # Try to infer from chapter/document
                chapter = data.get("chapter")
                document = data.get("document")
                if chapter and document:
                    file_path = f"docs/source/{chapter}/{document}.md"
                    obj_data["source"] = create_source_location(
                        chapter, document, file_path, label
                    )

            # Validate and create
            obj = MathematicalObject.model_validate(obj_data)
            self.report.objects_transformed += 1
            return obj

        except Exception as e:
            self.report.add_error(f"{json_path.name}: {e}")
            return None

    def transform_enriched_axiom(self, data: dict, json_path: Path) -> Axiom | None:
        """Transform enriched Axiom JSON → Axiom.

        Args:
            data: Enriched axiom JSON data
            json_path: Path to JSON file (for error reporting)

        Returns:
            Axiom or None if transformation fails
        """
        try:
            # Extract required fields
            label = data.get("label")
            if not label:
                self.report.add_error(f"{json_path.name}: Missing label")
                return None

            # Map enriched fields to math_types fields
            axiom_data = {
                "label": label,
                "name": data.get("name"),
                "statement": data.get("statement", ""),
                "mathematical_expression": data.get("mathematical_expression", ""),
                "foundational_framework": data.get(
                    "foundational_framework", "Fragile Gas Framework"
                ),
                "chapter": data.get("chapter"),
                "document": data.get("document"),
                "core_assumption": data.get("core_assumption"),
                "parameters": data.get("parameters"),
                "condition": data.get("condition"),
                "failure_mode_analysis": data.get("failure_mode_analysis"),
            }

            # Add source if present
            source = data.get("source")
            if source:
                if isinstance(source, dict):
                    axiom_data["source"] = SourceLocation.model_validate(source)
                else:
                    axiom_data["source"] = source

            # Validate and create
            axiom = Axiom.model_validate(axiom_data)
            self.report.axioms_transformed += 1
            return axiom

        except Exception as e:
            self.report.add_error(f"{json_path.name}: {e}")
            return None

    def transform_enriched_parameter(self, data: dict, json_path: Path) -> Parameter | None:
        """Transform enriched Parameter JSON → Parameter.

        Handles both enriched Parameter and ParameterBox formats.

        Args:
            data: Enriched parameter JSON data
            json_path: Path to JSON file (for error reporting)

        Returns:
            Parameter or None if transformation fails
        """
        try:
            # Check if this is a ParameterBox (has 'domain' field)
            if "domain" in data:
                # Use type_conversions for ParameterBox
                param_box = ParameterBox.model_validate(data)
                param = parameter_box_to_parameter(param_box)
                self.report.parameters_transformed += 1
                return param

            # Otherwise, transform enriched Parameter
            label = data.get("label")
            if not label:
                self.report.add_error(f"{json_path.name}: Missing label")
                return None

            # Map enriched fields to math_types fields
            param_data = {
                "label": label,
                "name": data.get("name", label),
                "symbol": data.get("symbol", label),
                "parameter_type": data.get("parameter_type", "real"),
                "constraints": data.get("constraints"),
                "default_value": data.get("default_value"),
                "chapter": data.get("chapter"),
                "document": data.get("document"),
            }

            # Validate and create
            param = Parameter.model_validate(param_data)
            self.report.parameters_transformed += 1
            return param

        except Exception as e:
            self.report.add_error(f"{json_path.name}: {e}")
            return None

    def transform_enriched_theorem(self, data: dict, json_path: Path) -> TheoremBox | None:
        """Transform enriched Theorem JSON → TheoremBox.

        Args:
            data: Enriched theorem JSON data
            json_path: Path to JSON file (for error reporting)

        Returns:
            TheoremBox or None if transformation fails
        """
        try:
            # Extract required fields
            label = data.get("label")
            if not label:
                self.report.add_error(f"{json_path.name}: Missing label")
                return None

            # Extract source location
            source = data.get("source")
            if source and isinstance(source, dict):
                source = SourceLocation.model_validate(source)
            elif not source:
                # Try to infer from chapter/document
                chapter = data.get("chapter")
                document = data.get("document")
                if chapter and document:
                    file_path = f"docs/source/{chapter}/{document}.md"
                    source = create_source_location(chapter, document, file_path, label)

            # Preprocess attributes_added
            props_raw = data.get("attributes_added", [])
            props_processed = preprocess_attributes_added(props_raw)
            attributes_added = []
            for prop_data in props_processed:
                try:
                    if source and "source" not in prop_data:
                        prop_data["source"] = source
                    prop = Attribute.model_validate(prop_data)
                    attributes_added.append(prop)
                except Exception as e:
                    self.report.add_warning(f"{label}: Error parsing attribute: {e}")

            # Preprocess relations_established
            rels_raw = data.get("relations_established", [])
            input_objects = data.get("input_objects", [])
            rels_processed = preprocess_relations_established(
                rels_raw, label, input_objects, source
            )
            relations_established = []
            for rel_data in rels_processed:
                try:
                    if "source" in rel_data and isinstance(rel_data["source"], dict):
                        rel_data["source"] = SourceLocation.model_validate(rel_data["source"])
                    rel = Relationship.model_validate(rel_data)
                    relations_established.append(rel)
                    self.report.relationships_created += 1
                except Exception as e:
                    self.report.add_warning(f"{label}: Error parsing relationship: {e}")

            # Preprocess lemma_dag_edges
            edges_raw = data.get("lemma_dag_edges", [])
            lemma_dag_edges = preprocess_lemma_edges(edges_raw, label)

            # Map enriched fields to math_types fields
            thm_data = {
                "label": label,
                "name": data.get("name", label),
                "statement": data.get("statement", ""),
                "proof_sketch": data.get("proof_sketch"),
                "assumptions": data.get("assumptions", []),
                "conclusion": data.get("conclusion"),
                "input_objects": input_objects,
                "input_axioms": data.get("input_axioms", []),
                "input_parameters": data.get("input_parameters", []),
                "attributes_required": data.get("attributes_required", {}),
                "internal_lemmas": data.get("internal_lemmas", []),
                "internal_propositions": data.get("internal_propositions", []),
                "lemma_dag_edges": lemma_dag_edges,
                "output_type": data.get("output_type", "Property"),
                "attributes_added": attributes_added,
                "relations_established": relations_established,
                "source": source,
            }

            # Add optional fields if present
            if data.get("proof"):
                thm_data["proof"] = data["proof"]

            # Validate and create
            thm = TheoremBox.model_validate(thm_data)
            self.report.theorems_transformed += 1
            return thm

        except Exception as e:
            self.report.add_error(f"{json_path.name}: {e}")
            return None

    def transform_directory(
        self, input_dir: Path, output_dir: Path, entity_types: list[str] | None = None
    ):
        """Transform entire enriched_data/ directory to pipeline_data/.

        Args:
            input_dir: Path to refined_data/ directory
            output_dir: Path to output pipeline_data/ directory
            entity_types: List of entity types to transform (default: all)
        """
        if entity_types is None:
            entity_types = ["objects", "axioms", "parameters", "theorems"]

        print("=" * 70)
        print("TRANSFORMING ENRICHED DATA TO PIPELINE MATH TYPES")
        print("=" * 70)
        print(f"\nInput:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"\nEntity types: {', '.join(entity_types)}")

        # Create output directories
        for entity_type in entity_types:
            (output_dir / entity_type).mkdir(parents=True, exist_ok=True)

        # Transform each entity type
        for entity_type in entity_types:
            input_subdir = input_dir / entity_type
            output_subdir = output_dir / entity_type

            if not input_subdir.exists():
                print(f"\n⚠️  Skipping {entity_type}: input directory not found")
                continue

            print(f"\n{'=' * 70}")
            print(f"Transforming {entity_type}...")
            print(f"{'=' * 70}")

            json_files = list(input_subdir.glob("*.json"))
            print(f"Found {len(json_files)} files")

            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    # Transform based on entity type
                    transformed = None
                    if entity_type == "objects":
                        transformed = self.transform_enriched_object(data, json_file)
                    elif entity_type == "axioms":
                        transformed = self.transform_enriched_axiom(data, json_file)
                    elif entity_type == "parameters":
                        transformed = self.transform_enriched_parameter(data, json_file)
                    elif entity_type == "theorems":
                        transformed = self.transform_enriched_theorem(data, json_file)

                    # Save transformed data
                    if transformed:
                        output_file = output_subdir / json_file.name
                        with open(output_file, "w") as f:
                            json.dump(transformed.model_dump(mode="json"), f, indent=2)
                        print(f"  ✓ {json_file.name}")
                    else:
                        print(f"  ✗ {json_file.name} (transformation failed)")

                except Exception as e:
                    self.report.add_error(f"{json_file.name}: {e}")
                    print(f"  ✗ {json_file.name}: {e}")

        # Print final report
        self.report.print_summary()

        # Write report to file
        report_file = output_dir / "transformation_report.json"
        with open(report_file, "w") as f:
            json.dump(
                {
                    "objects_transformed": self.report.objects_transformed,
                    "axioms_transformed": self.report.axioms_transformed,
                    "parameters_transformed": self.report.parameters_transformed,
                    "theorems_transformed": self.report.theorems_transformed,
                    "relationships_created": self.report.relationships_created,
                    "errors": self.report.errors,
                    "warnings": self.report.warnings,
                },
                f,
                indent=2,
            )

        print(f"\n✓ Transformation report saved to: {report_file}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transform enriched data to pipeline math_types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform Chapter 1 refined data
  python -m fragile.mathster.tools.enriched_to_math_types \\
    --input docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data \\
    --output docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data

  # Transform only objects and theorems
  python -m fragile.mathster.tools.enriched_to_math_types \\
    --input docs/source/1_euclidean_gas/02_euclidean_gas/refined_data \\
    --output docs/source/1_euclidean_gas/02_euclidean_gas/pipeline_data \\
    --types objects theorems
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to refined_data/ directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output pipeline_data/ directory",
    )
    parser.add_argument(
        "--types",
        "-t",
        nargs="+",
        default=["objects", "axioms", "parameters", "theorems"],
        help="Entity types to transform (default: all)",
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Create transformer and process
    transformer = EnrichedToMathTypesTransformer()
    transformer.transform_directory(
        input_dir=input_dir, output_dir=output_dir, entity_types=args.types
    )

    # Exit with error code if there were errors
    if transformer.report.errors:
        print(f"\n❌ Transformation completed with {len(transformer.report.errors)} errors")
        sys.exit(1)
    else:
        print("\n✅ Transformation completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
