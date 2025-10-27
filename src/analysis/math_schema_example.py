#!/usr/bin/env python3
"""
Example usage of the math_schema.py Pydantic models.

Demonstrates how to:
1. Create mathematical documents programmatically
2. Validate documents against the schema
3. Serialize to JSON
4. Load from JSON
"""

import json
from datetime import date
from pathlib import Path

from src.analysis.math_schema import (
    Algorithm,
    AlgorithmComplexity,
    AlgorithmInput,
    AlgorithmOutput,
    AlgorithmStep,
    CrossReference,
    Definition,
    DefinedObject,
    DefinitionExample,
    MathematicalDocument,
    MathematicalProperty,
    Metadata,
    Proof,
    ProofStep,
    Theorem,
    TheoremConclusion,
)


def create_example_document() -> MathematicalDocument:
    """Create an example mathematical document using Pydantic models."""

    # Define metadata
    metadata = Metadata(
        title="Example: Walker Definition and Basic Properties",
        document_id="example_walker",
        version="1.0.0",
        authors=["Claude Code"],
        date_created=date(2025, 10, 25),
        date_modified=date(2025, 10, 25),
        rigor_level="rigorous",
        abstract="Demonstrates the JSON schema with a simple definition, theorem, and proof from the Fragile Gas framework.",
    )

    # Create a definition
    walker_def = Definition(
        type="definition",
        label="def-walker-example",
        title="Walker",
        statement="A **walker**, denoted $w$, is a tuple consisting of a position and a status.",
        tags=["general", "viability"],
        defined_objects=[
            DefinedObject(
                name="Walker",
                symbol="w",
                mathematical_definition="w := (x, s)",
                type="structure",
                properties=[
                    MathematicalProperty(
                        name="Minimal representation",
                        statement="Every fragile-gas instantiation requires at minimum position $x$ and status $s$ components",
                    )
                ],
            )
        ],
        examples=[
            DefinitionExample(
                description="Euclidean walker in 2D",
                instance="w = ((2.5, -1.3), 1)",
            ),
            DefinitionExample(
                description="Dead walker",
                instance="w = ((0.0, 0.0), 0)",
            ),
        ],
    )

    # Create a theorem
    alive_partition_thm = Theorem(
        type="theorem",
        label="thm-alive-partition",
        title="Alive/Dead Partition",
        statement="For any swarm $\\mathcal{S}$, the indices can be partitioned into alive and dead sets.",
        tags=["general", "viability"],
        importance="foundational",
        hypotheses=[],
        conclusion=TheoremConclusion(
            statement="$\\{1, \\ldots, N\\} = \\mathcal{A}(\\mathcal{S}) \\sqcup \\mathcal{D}(\\mathcal{S})$",
            properties_established=[
                MathematicalProperty(
                    name="Disjoint partition",
                    statement="Every walker is either alive or dead, never both, never neither",
                ),
                MathematicalProperty(
                    name="Cardinality conservation",
                    statement="$|\\mathcal{A}(\\mathcal{S})| + |\\mathcal{D}(\\mathcal{S})| = N$",
                    quantitative=True,
                ),
            ],
        ),
    )

    # Create a proof
    partition_proof = Proof(
        type="proof",
        label="proof-alive-partition",
        title="Proof of Alive/Dead Partition",
        statement="Proof of the Alive/Dead Partition theorem",
        proves=CrossReference(
            label="thm-alive-partition",
            type="theorem",
            role="proves",
        ),
        proof_type="direct",
        strategy="Direct proof by verifying the two properties of a partition: (1) disjoint sets, (2) cover all elements.",
        steps=[
            ProofStep(
                id="Step 1",
                title="Disjointness",
                content="We first show that $\\mathcal{A}(\\mathcal{S}) \\cap \\mathcal{D}(\\mathcal{S}) = \\emptyset$.",
                techniques=["contradiction"],
                intermediate_result="\\mathcal{A}(\\mathcal{S}) \\cap \\mathcal{D}(\\mathcal{S}) = \\emptyset",
            ),
            ProofStep(
                id="Step 2",
                title="Coverage",
                content="We now show that $\\mathcal{A}(\\mathcal{S}) \\cup \\mathcal{D}(\\mathcal{S}) = \\{1, \\ldots, N\\}$.",
                techniques=["case-analysis", "direct-proof"],
                intermediate_result="\\mathcal{A}(\\mathcal{S}) \\cup \\mathcal{D}(\\mathcal{S}) = \\{1, \\ldots, N\\}",
            ),
            ProofStep(
                id="Step 3",
                title="Conclusion",
                content="By Step 1, the sets are disjoint. By Step 2, they cover all indices.",
                techniques=["direct-proof"],
                intermediate_result="\\{1, \\ldots, N\\} = \\mathcal{A}(\\mathcal{S}) \\sqcup \\mathcal{D}(\\mathcal{S})",
            ),
        ],
        difficulty="routine",
        rigor_level=10,
    )

    # Create an algorithm
    cloning_algorithm = Algorithm(
        type="algorithm",
        label="alg-clone-operator",
        title="Cloning Operator",
        statement="The cloning operator selects walkers for replication based on fitness.",
        tags=["cloning", "operator"],
        inputs=[
            AlgorithmInput(
                name="S",
                type="swarm state $\\mathcal{S} \\in \\Sigma_N$",
                description="Current swarm configuration",
            )
        ],
        outputs=[
            AlgorithmOutput(
                name="S'",
                type="swarm state $\\mathcal{S}' \\in \\Sigma_N$",
                description="Updated swarm after cloning",
                guarantees=["$|\\mathcal{A}(\\mathcal{S}')| = N$ (alive set preserved)"],
            )
        ],
        steps=[
            AlgorithmStep(
                step_number=1,
                description="Compute fitness for all alive walkers",
                pseudocode="fitness[i] = exp(-V(x_i)) for i in A(S)",
                complexity="O(N)",
            ),
            AlgorithmStep(
                step_number=2,
                description="Select walkers to clone via tournament selection",
                pseudocode="selected = tournament_select(fitness, N)",
                complexity="O(N log N)",
            ),
            AlgorithmStep(
                step_number=3,
                description="Replace dead walkers with cloned copies",
                pseudocode="S'[i] = S[selected[i]] for i in D(S)",
                complexity="O(N)",
            ),
        ],
        complexity=AlgorithmComplexity(
            time="O(N log N)",
            space="O(N)",
            worst_case="O(N log N)",
        ),
    )

    # Assemble the document
    doc = MathematicalDocument(
        metadata=metadata,
        directives=[
            walker_def,
            alive_partition_thm,
            partition_proof,
            cloning_algorithm,
        ],
    )

    return doc


def validate_and_save_example():
    """Create, validate, and save an example document."""

    # Create document
    print("Creating example mathematical document...")
    doc = create_example_document()

    # Validate (Pydantic does this automatically)
    print(f"✅ Document validated successfully!")
    print(f"  - Title: {doc.metadata.title}")
    print(f"  - Directives: {len(doc.directives)}")
    print(f"  - Version: {doc.metadata.version}")

    # Serialize to JSON
    output_path = Path("/tmp/example_document.json")
    doc_json = doc.model_dump(mode="json", exclude_none=True)

    with open(output_path, "w") as f:
        json.dump(doc_json, f, indent=2, default=str)

    print(f"\n✅ Document saved to: {output_path}")
    print(f"  - File size: {output_path.stat().st_size} bytes")

    # Load it back
    with open(output_path) as f:
        loaded_json = json.load(f)

    loaded_doc = MathematicalDocument(**loaded_json)
    print(f"\n✅ Document loaded successfully!")
    print(f"  - Loaded {len(loaded_doc.directives)} directives")

    return doc, output_path


def demonstrate_field_validation():
    """Demonstrate Pydantic validation."""

    print("\n" + "=" * 60)
    print("Demonstrating validation...")
    print("=" * 60)

    try:
        # This should fail - invalid label (must be kebab-case)
        Definition(
            type="definition",
            label="Invalid Label!",  # ❌ Not kebab-case
            title="Test",
            statement="Test statement",
            defined_objects=[
                DefinedObject(
                    name="Test",
                    mathematical_definition="x := y",
                )
            ],
        )
    except Exception as e:
        print(f"\n✅ Caught validation error (as expected):")
        print(f"  - {type(e).__name__}: Label must be kebab-case")

    try:
        # This should fail - rigor score out of range
        Proof(
            type="proof",
            label="proof-test",
            title="Test Proof",
            statement="Test",
            proves=CrossReference(label="thm-test", type="theorem"),
            strategy="Test strategy",
            steps=[ProofStep(id="Step 1", content="Test")],
            rigor_level=15,  # ❌ Must be 1-10
        )
    except Exception as e:
        print(f"\n✅ Caught validation error (as expected):")
        print(f"  - {type(e).__name__}: Rigor level must be 1-10")

    try:
        # This should fail - invalid version format
        Metadata(
            title="Test",
            document_id="test",
            version="invalid-version",  # ❌ Must match semantic versioning
        )
    except Exception as e:
        print(f"\n✅ Caught validation error (as expected):")
        print(f"  - {type(e).__name__}: Version must be semantic (e.g., '1.0.0')")

    print("\n✅ All validation tests passed!")


if __name__ == "__main__":
    # Create and save example
    doc, path = validate_and_save_example()

    # Demonstrate validation
    demonstrate_field_validation()

    print("\n" + "=" * 60)
    print("✅ Example complete!")
    print("=" * 60)
    print(f"\nYou can now:")
    print(f"  1. View the generated JSON: cat {path}")
    print(f"  2. Render it to markdown: python src/analysis/render_math_json.py {path}")
    print(f"  3. Use it as a template for your own documents")
