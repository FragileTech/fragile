"""
Transform document-parser output to pipeline schema (fragile.mathster.core.pipeline_types).

This script converts the deep dependency extraction JSON to fully compliant Pydantic models.
"""

import json
from pathlib import Path
import re
import sys
from typing import Any

from fragile.proofs.core.pipeline_types import (
    Axiom,
    MathematicalObject,
    ObjectType,
    TheoremBox,
    TheoremOutputType,
)


def extract_math_expression(directive: dict[str, Any]) -> str:
    """Extract the first substantial mathematical expression from a directive."""
    # Try first_math field
    first_math = directive.get("first_math", "")
    if first_math and len(first_math) > 10:
        return first_math[:500]  # Truncate to reasonable length

    # Try to extract from content
    content = directive.get("content", "")
    if content:
        # Look for $$ blocks
        math_blocks = re.findall(r"\$\$(.*?)\$\$", content, re.DOTALL)
        if math_blocks:
            return math_blocks[0].strip()[:500]

        # Look for inline math
        inline_math = re.findall(r"\$(.*?)\$", content)
        if inline_math:
            return inline_math[0].strip()[:500]

    # Fallback: use title or first line of content
    title = directive.get("title", "")
    if title:
        return title[:200]

    return content[:200] if content else "Mathematical object"


def infer_object_type(directive: dict[str, Any]) -> ObjectType:
    """Infer ObjectType from directive title and content."""
    title = directive.get("title", "").lower()
    content = directive.get("content", "").lower()
    combined = f"{title} {content}"

    if any(word in combined for word in ["operator", "map", "transformation", "projection"]):
        return ObjectType.OPERATOR
    if any(word in combined for word in ["space", "metric", "topology"]):
        return ObjectType.SPACE
    if any(word in combined for word in ["distribution", "measure", "probability"]):
        return ObjectType.MEASURE
    if any(word in combined for word in ["function", "functional"]):
        return ObjectType.FUNCTION
    if any(word in combined for word in ["field", "vector field"]):
        return ObjectType.FIELD
    return ObjectType.STRUCTURE


def infer_theorem_output_type(directive: dict[str, Any]) -> TheoremOutputType:
    """Infer TheoremOutputType from directive title and content."""
    title = directive.get("title", "").lower()
    content = directive.get("content", "").lower()
    combined = f"{title} {content}"

    if any(word in combined for word in ["exists", "existence", "there exists"]):
        return TheoremOutputType.EXISTENCE
    if any(word in combined for word in ["unique", "uniqueness"]):
        return TheoremOutputType.UNIQUENESS
    if any(
        word in combined
        for word in ["continuous", "lipschitz", "smooth", "differentiable", "measurable"]
    ):
        return TheoremOutputType.PROPERTY
    if any(word in combined for word in ["converge", "convergence", "limit"]):
        return TheoremOutputType.PROPERTY
    if any(word in combined for word in ["equivalent", "equivalence", "if and only if"]):
        return TheoremOutputType.EQUIVALENCE
    if any(word in combined for word in ["embedding", "embeds"]):
        return TheoremOutputType.EMBEDDING
    if any(word in combined for word in ["approximation", "approximate", "error bound"]):
        return TheoremOutputType.APPROXIMATION
    if any(word in combined for word in ["decomposition", "decompose"]):
        return TheoremOutputType.DECOMPOSITION
    if any(word in combined for word in ["extension", "extends"]):
        return TheoremOutputType.EXTENSION
    if any(word in combined for word in ["reduction", "reduces to"]):
        return TheoremOutputType.REDUCTION
    return TheoremOutputType.PROPERTY  # default


def extract_tags(directive: dict[str, Any]) -> list[str]:
    """Extract relevant tags from directive."""
    tags = []

    title = directive.get("title", "").lower()
    label = directive.get("label", "").lower()

    # Framework tags
    if "euclidean" in title or "euclidean" in label:
        tags.append("euclidean-gas")
    if "geometric" in title or "geometric" in label:
        tags.append("geometric-gas")
    if "adaptive" in title or "adaptive" in label:
        tags.append("adaptive-gas")

    # Topic tags
    if "cloning" in title or "clone" in label:
        tags.append("cloning")
    if "kinetic" in title or "kinetic" in label:
        tags.append("kinetic")
    if "sasaki" in title or "sasaki" in label:
        tags.append("sasaki-metric")
    if "regularit" in title:
        tags.append("regularity")
    if "convergence" in title:
        tags.append("convergence")
    if "mean-field" in title or "meanfield" in label:
        tags.append("mean-field")

    return tags


def transform_definition_to_object(directive: dict[str, Any]) -> MathematicalObject | None:
    """Transform a definition directive to a MathematicalObject."""
    try:
        label = directive["label"]

        # Convert def-* to obj-*
        object_label = (
            label.replace("def-", "obj-", 1) if label.startswith("def-") else f"obj-{label}"
        )

        # Extract mathematical expression (REQUIRED)
        math_expr = extract_math_expression(directive)
        if not math_expr:
            print(
                f"  Warning: No mathematical expression found for {label}, using title",
                file=sys.stderr,
            )
            math_expr = directive.get("title", "Mathematical object")

        return MathematicalObject(
            label=object_label,
            name=directive.get("title", "Unnamed object"),
            mathematical_expression=math_expr,
            object_type=infer_object_type(directive),
            current_properties=[],  # Properties added by theorems
            property_history=[],
            tags=extract_tags(directive),
        )
    except Exception as e:
        print(
            f"  Error transforming definition {directive.get('label', 'unknown')}: {e}",
            file=sys.stderr,
        )
        return None


def transform_axiom(directive: dict[str, Any]) -> Axiom | None:
    """Transform an axiom directive to an Axiom."""
    try:
        math_expr = extract_math_expression(directive)
        if not math_expr:
            math_expr = directive.get("title", "Axiom")

        return Axiom(
            label=directive["label"],
            statement=directive.get("title", "Axiom"),
            mathematical_expression=math_expr,
            foundational_framework="Fragile Gas Framework",
        )
    except Exception as e:
        print(
            f"  Error transforming axiom {directive.get('label', 'unknown')}: {e}", file=sys.stderr
        )
        return None


def transform_theorem_to_box(directive: dict[str, Any]) -> TheoremBox | None:
    """Transform a theorem/lemma/proposition to a TheoremBox."""
    try:
        label = directive["label"]

        # Extract inputs from dependencies
        deps = directive.get("dependencies", {})
        explicit_deps = deps.get("explicit", [])
        implicit_deps = deps.get("implicit", [])

        input_objects = []
        input_axioms = []

        # Process explicit dependencies
        for dep in explicit_deps:
            target = dep.get("target_label", "")
            if target.startswith("def-"):
                # Convert def-* to obj-*
                input_objects.append(target.replace("def-", "obj-", 1))
            elif target.startswith("obj-"):
                input_objects.append(target)
            elif target.startswith("axiom-"):
                input_axioms.append(target)

        # Process implicit dependencies
        for dep in implicit_deps:
            target = dep.get("target_label", "")
            if target.startswith("def-"):
                input_objects.append(target.replace("def-", "obj-", 1))
            elif target.startswith("obj-"):
                input_objects.append(target)
            elif target.startswith("axiom-"):
                input_axioms.append(target)

        # Process assumptions
        for assumption in deps.get("assumptions", []):
            target = assumption.get("target_label", "")
            if target.startswith("axiom-"):
                input_axioms.append(target)

        # Remove duplicates and filter invalid labels
        input_objects = list({obj for obj in input_objects if obj.startswith("obj-")})
        input_axioms = list({ax for ax in input_axioms if ax.startswith("axiom-")})

        # Get mathematical statement
        math_statement = directive.get("content", "")
        if not math_statement:
            # Try to construct from title and first math
            math_statement = directive.get("title", "")
            first_math = directive.get("first_math", "")
            if first_math:
                math_statement += f"\n\n$$\n{first_math}\n$$"

        return TheoremBox(
            label=label,
            name=directive.get("title", ""),
            statement_type="theorem",  # Will be auto-detected from label
            input_objects=input_objects,
            input_axioms=input_axioms,
            input_parameters=[],  # Would need additional extraction
            properties_required={},  # Would need deeper analysis
            output_type=infer_theorem_output_type(directive),
            properties_added=[],  # Properties would be added during proof execution
            relations_established=[],
            mathematical_statement=math_statement[:1000]
            if math_statement
            else directive.get("title", ""),
        )
    except Exception as e:
        print(
            f"  Error transforming theorem {directive.get('label', 'unknown')}: {e}",
            file=sys.stderr,
        )
        import traceback

        traceback.print_exc()
        return None


def transform_extraction_to_pipeline_schema(extraction_file: Path, output_file: Path):
    """
    Transform document-parser extraction to pipeline schema format.

    Args:
        extraction_file: Path to deep_dependency_analysis.json
        output_file: Path to save pipeline-compatible JSON
    """
    with open(extraction_file, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both dict and list formats for directives
    directives_raw = data.get("directives", [])
    if isinstance(directives_raw, dict):
        # Convert dict to list of directives
        directives = list(directives_raw.values())
    else:
        directives = directives_raw

    # Categorize and transform directives
    objects = []
    axioms = []
    theorems = []
    lemmas = []
    propositions = []
    algorithms = []
    other = []

    for directive in directives:
        label = directive.get("label", "")
        dtype = directive.get("type", "")

        if label.startswith("def-") or dtype == "definition":
            obj = transform_definition_to_object(directive)
            if obj:
                objects.append(obj.model_dump())

        elif label.startswith("axiom-") or dtype == "axiom":
            axiom = transform_axiom(directive)
            if axiom:
                axioms.append(axiom.model_dump())

        elif label.startswith("thm-") or dtype == "theorem":
            thm = transform_theorem_to_box(directive)
            if thm:
                theorems.append(thm.model_dump())

        elif label.startswith("lem-") or dtype == "lemma":
            lem = transform_theorem_to_box(directive)
            if lem:
                lemmas.append(lem.model_dump())

        elif label.startswith("prop-") or dtype == "proposition":
            prop = transform_theorem_to_box(directive)
            if prop:
                propositions.append(prop.model_dump())

        elif dtype in {"algorithm", "alg"}:
            # Algorithms are stored but not part of theorem pipeline
            algorithms.append({
                "label": label,
                "title": directive.get("title", ""),
                "content": directive.get("content", "")[:500],
            })

        else:
            # Store other types for reference (mathster, remarks, etc.)
            other.append({
                "label": label,
                "type": dtype,
                "title": directive.get("title", ""),
            })

    # Build pipeline-compatible structure
    pipeline_data = {
        "document": data.get("document", ""),
        "extraction_mode": "pipeline-schema-compliant",
        "schema_version": "2.0.0",
        "framework": "Fragile Gas Framework",
        "statistics": {
            "total_directives": len(directives),
            "mathematical_objects": len(objects),
            "axioms": len(axioms),
            "theorems": len(theorems),
            "lemmas": len(lemmas),
            "propositions": len(propositions),
            "algorithms": len(algorithms),
            "other": len(other),
        },
        "mathematical_objects": objects,
        "axioms": axioms,
        "theorems": theorems,
        "lemmas": lemmas,
        "propositions": propositions,
        "algorithms": algorithms,
        "other_directives": other,
        "metadata": {
            "source_extraction": str(extraction_file),
            "total_lines": data.get("metadata", {}).get("total_lines", 0),
        },
    }

    # Save to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_data, f, indent=2)

    print(
        f"✓ Transformed {len(directives)} directives from {extraction_file.parent.parent.name}/{extraction_file.parent.name}"
    )
    print(
        f"    Objects: {len(objects)}, Axioms: {len(axioms)}, "
        + f"Theorems: {len(theorems)}, Lemmas: {len(lemmas)}, Props: {len(propositions)}"
    )

    return pipeline_data


def main():
    """Transform all document extractions to pipeline schema."""
    docs = [
        "docs/source/1_euclidean_gas/01_fragile_gas_framework",
        "docs/source/1_euclidean_gas/02_euclidean_gas",
        "docs/source/1_euclidean_gas/03_cloning",
        "docs/source/1_euclidean_gas/05_kinetic_contraction",
        "docs/source/2_geometric_gas/11_geometric_gas",
        "docs/source/2_geometric_gas/13_geometric_gas_c3_regularity",
        "docs/source/2_geometric_gas/16_convergence_mean_field",
    ]

    all_stats = {
        "total_objects": 0,
        "total_axioms": 0,
        "total_theorems": 0,
        "total_lemmas": 0,
        "total_propositions": 0,
    }

    for doc_path in docs:
        extraction_file = Path(doc_path) / "data" / "deep_dependency_analysis.json"
        output_file = Path(doc_path) / "data" / "pipeline_schema.json"

        if extraction_file.exists():
            try:
                result = transform_extraction_to_pipeline_schema(extraction_file, output_file)
                stats = result["statistics"]
                all_stats["total_objects"] += stats["mathematical_objects"]
                all_stats["total_axioms"] += stats["axioms"]
                all_stats["total_theorems"] += stats["theorems"]
                all_stats["total_lemmas"] += stats["lemmas"]
                all_stats["total_propositions"] += stats["propositions"]
            except Exception as e:
                print(f"✗ Error processing {doc_path}: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
        else:
            print(f"✗ Skipping {doc_path}: extraction file not found")

    print("\n" + "=" * 70)
    print("PIPELINE SCHEMA TRANSFORMATION COMPLETE")
    print("=" * 70)
    print(f"Total mathematical objects: {all_stats['total_objects']}")
    print(f"Total axioms: {all_stats['total_axioms']}")
    print(f"Total theorems: {all_stats['total_theorems']}")
    print(f"Total lemmas: {all_stats['total_lemmas']}")
    print(f"Total propositions: {all_stats['total_propositions']}")
    print("\nAll files saved as: docs/source/.../data/pipeline_schema.json")


if __name__ == "__main__":
    main()
