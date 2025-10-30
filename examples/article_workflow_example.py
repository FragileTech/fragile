"""
Example: Article System Workflow

Demonstrates the Article system for linking mathematical objects to source documentation.

This example shows:
1. Registering articles in the ArticleRegistry
2. Creating mathematical objects with source locations
3. Querying the registry by label, tag, and chapter
4. Generating glossary.md from the registry
5. Export/import functionality
"""

from datetime import datetime
from pathlib import Path

from fragile.proofs import (
    Article,
    create_simple_object,
    create_simple_theorem,
    get_article_registry,
    Property,
    SourceLocation,
    SourceLocationBuilder,
)


def main():
    """Demonstrate Article system workflow."""

    print("=" * 80)
    print("Article System Workflow Example")
    print("=" * 80)

    # Get the singleton registry
    registry = get_article_registry()

    # =========================================================================
    # Step 1: Register articles representing mathematical documents
    # =========================================================================
    print("\n1. Registering Articles")
    print("-" * 80)

    # Article for Euclidean Gas framework document
    euclidean_gas_article = Article(
        document_id="02_euclidean_gas",
        title="Euclidean Gas: Langevin Dynamics with Cloning",
        file_path="docs/source/1_euclidean_gas/02_euclidean_gas.md",
        chapter=1,
        section_number="2",
        tags=["euclidean-gas", "langevin", "cloning", "framework"],
        version="1.0.0",
        last_modified=datetime.now(),
        abstract="Defines the Euclidean Gas algorithm combining Langevin dynamics with a cloning operator.",
        contains_labels=[
            "def-euclidean-gas",
            "thm-euclidean-convergence",
            "lem-kinetic-mixing",
            "prop-momentum-conservation",
        ],
    )
    registry.register_article(euclidean_gas_article)
    print(f"✓ Registered: {euclidean_gas_article.title}")

    # Article for cloning mechanism
    cloning_article = Article(
        document_id="03_cloning",
        title="Cloning Operator and the Keystone Principle",
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        chapter=1,
        section_number="3",
        tags=["cloning", "measurement", "fitness", "keystone"],
        version="1.0.0",
        last_modified=datetime.now(),
        abstract="Details the cloning operator and establishes the Keystone Principle.",
        contains_labels=[
            "def-cloning-operator",
            "thm-keystone-principle",
            "lem-fitness-monotonicity",
            "prop-clone-invariance",
        ],
    )
    registry.register_article(cloning_article)
    print(f"✓ Registered: {cloning_article.title}")

    # Article for geometric gas
    geometric_article = Article(
        document_id="11_geometric_gas",
        title="Geometric Gas on Riemannian Manifolds",
        file_path="docs/source/2_geometric_gas/11_geometric_gas.md",
        chapter=2,
        section_number="1",
        tags=["geometric-gas", "riemannian", "manifold", "curvature"],
        version="1.0.0",
        last_modified=datetime.now(),
        abstract="Extends the Gas framework to Riemannian manifolds with intrinsic geometry.",
        contains_labels=[
            "def-geometric-gas",
            "thm-manifold-convergence",
            "lem-curvature-bound",
        ],
    )
    registry.register_article(geometric_article)
    print(f"✓ Registered: {geometric_article.title}")

    # =========================================================================
    # Step 2: Create mathematical objects with source locations
    # =========================================================================
    print("\n2. Creating Mathematical Objects with Source Locations")
    print("-" * 80)

    # Example: Theorem with source location using builder
    euclidean_convergence_source = SourceLocationBuilder.from_jupyter_directive(
        document_id="02_euclidean_gas",
        file_path="docs/source/1_euclidean_gas/02_euclidean_gas.md",
        directive_label="thm-euclidean-convergence",
        section="Convergence Theory",
    )

    euclidean_convergence_thm = create_simple_theorem(
        label="thm-euclidean-convergence",
        statement="Under the stated axioms, the Euclidean Gas converges exponentially fast to a unique quasi-stationary distribution.",
        tags=["convergence", "qsd", "exponential"],
        source=euclidean_convergence_source,
    )
    print(f"✓ Created theorem: {euclidean_convergence_thm.label}")
    print(
        f"  Source: {euclidean_convergence_source.file_path}#{euclidean_convergence_source.directive_label}"
    )

    # Example: Mathematical object with source
    state_space_source = SourceLocationBuilder.from_jupyter_directive(
        document_id="02_euclidean_gas",
        file_path="docs/source/1_euclidean_gas/02_euclidean_gas.md",
        directive_label="def-state-space",
        section="Mathematical Foundations",
    )

    state_space_obj = create_simple_object(
        label="state-space-euclidean",
        description="State space X = ℝ^d for Euclidean Gas",
        tags=["state-space", "euclidean"],
        source=state_space_source,
    )
    print(f"✓ Created object: {state_space_obj.label}")

    # Example: Property with source and equation reference
    kinetic_property_source = SourceLocation(
        document_id="02_euclidean_gas",
        file_path="docs/source/1_euclidean_gas/02_euclidean_gas.md",
        section="Langevin Dynamics",
        directive_label="def-kinetic-operator",
        equation="eq:langevin-sde",
        url_fragment="#def-kinetic-operator",
    )

    kinetic_property = Property(
        label="prop-kinetic-ergodic",
        description="The kinetic operator Ψ_kin is ergodic with respect to the Gibbs measure",
        tags=["kinetic", "ergodic", "langevin"],
        source=kinetic_property_source,
    )
    print(f"✓ Created property: {kinetic_property.label}")
    print(f"  Equation: {kinetic_property_source.equation}")

    # Example: Object without source (programmatically generated)
    runtime_obj = create_simple_object(
        label="runtime-parameter-N",
        description="Number of walkers (programmatically determined)",
        tags=["parameter", "runtime"],
        source=None,  # No documentation source - pure runtime object
    )
    print(f"✓ Created runtime object: {runtime_obj.label} (no source)")

    # =========================================================================
    # Step 3: Query the registry
    # =========================================================================
    print("\n3. Querying the ArticleRegistry")
    print("-" * 80)

    # Query by label
    article = registry.get_article_for_label("thm-euclidean-convergence")
    if article:
        print(f"✓ Found article for label 'thm-euclidean-convergence': {article.title}")

    # Query by tag
    cloning_articles = registry.get_articles_by_tag("cloning")
    print(f"✓ Found {len(cloning_articles)} article(s) tagged 'cloning':")
    for art in cloning_articles:
        print(f"  - {art.title}")

    # Query by chapter
    chapter1_articles = registry.get_articles_by_chapter(1)
    print(f"✓ Found {len(chapter1_articles)} article(s) in Chapter 1:")
    for art in chapter1_articles:
        print(f"  - {art.title}")

    # Get all articles
    all_articles = registry.get_all_articles()
    print(f"✓ Total articles in registry: {len(all_articles)}")

    # Get statistics
    stats = registry.get_statistics()
    print("\n✓ Registry Statistics:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Total labels: {stats['total_labels']}")
    print(f"  Total tags: {stats['total_tags']}")
    print(f"  Tags: {', '.join(stats['tags'])}")

    # =========================================================================
    # Step 4: Generate glossary.md
    # =========================================================================
    print("\n4. Generating Glossary Markdown")
    print("-" * 80)

    glossary_md = registry.generate_glossary_markdown()

    # Preview first 500 characters
    print("✓ Generated glossary.md (preview):")
    print("-" * 40)
    print(glossary_md[:500])
    print("...")
    print("-" * 40)

    # Optionally write to file
    glossary_path = Path("examples/generated_glossary.md")
    glossary_path.write_text(glossary_md)
    print(f"✓ Written to: {glossary_path}")

    # =========================================================================
    # Step 5: Export and import
    # =========================================================================
    print("\n5. Export and Import")
    print("-" * 80)

    # Export to JSON
    export_path = Path("examples/article_registry_export.json")
    registry.export_to_json(export_path)
    print(f"✓ Exported registry to: {export_path}")

    # Clear registry
    original_count = len(registry.get_all_articles())
    registry.clear()
    print(f"✓ Cleared registry ({original_count} → {len(registry.get_all_articles())} articles)")

    # Import from JSON
    registry.import_from_json(export_path)
    restored_count = len(registry.get_all_articles())
    print(f"✓ Imported registry ({restored_count} articles restored)")

    # Verify
    assert restored_count == original_count, "Import/export mismatch!"
    print("✓ Verification passed: export/import cycle successful")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
The Article system provides:

1. **SourceLocation**: Links objects to their documentation locations
   - Document ID, file path, section, directive label
   - Equation references and line ranges
   - URL fragment generation

2. **Article**: Represents mathematical documents
   - Metadata: title, chapter, tags, version
   - Contains list of labels defined in the document
   - Tracks last modification time

3. **ArticleRegistry**: Centralized management
   - Register and index articles
   - Query by label, tag, chapter
   - Generate glossary.md automatically
   - Export/import to JSON

4. **SourceLocationBuilder**: Convenience helpers
   - from_jupyter_directive()
   - from_file_line_range()
   - from_equation_reference()

5. **Integration**: All core types support optional source field
   - TheoremBox, ProofBox, MathematicalObject
   - Property, Relationship
   - Optional - objects can exist without documentation links
    """)

    print("✓ Article workflow demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
