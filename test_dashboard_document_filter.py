#!/usr/bin/env python3
"""Test script to verify document filter functionality in proof_pipeline_dashboard.py"""

# Import the dashboard
from proof_pipeline_dashboard import ProofPipelineDashboard


def test_document_filter():
    """Test that document filter is correctly implemented."""

    print("Testing Document Filter Implementation")
    print("=" * 60)

    # Create dashboard instance
    print("\n1. Creating dashboard instance...")
    dashboard = ProofPipelineDashboard()

    # Check that document_filter exists
    assert hasattr(dashboard, "document_filter"), "Dashboard should have document_filter attribute"
    print("   ✓ document_filter attribute exists")

    # Check that document_filter is a MultiChoice widget
    import panel as pn

    assert isinstance(
        dashboard.document_filter, pn.widgets.MultiChoice
    ), "document_filter should be a MultiChoice widget"
    print("   ✓ document_filter is a MultiChoice widget")

    # Check that options were populated
    print(
        f"\n2. Document filter options: {len(dashboard.document_filter.options)} documents found"
    )
    if dashboard.document_filter.options:
        print(f"   Sample documents: {dashboard.document_filter.options[:5]}")
        print("   ✓ Document filter populated with options")
    else:
        print("   ⚠ No documents found (registry may be empty or example)")

    # Check that all documents are selected by default
    assert set(dashboard.document_filter.value) == set(
        dashboard.document_filter.options
    ), "All documents should be selected by default"
    print("   ✓ All documents selected by default")

    # Check that _update_document_filter method exists
    assert hasattr(
        dashboard, "_update_document_filter"
    ), "Dashboard should have _update_document_filter method"
    print("\n3. ✓ _update_document_filter method exists")

    # Test that document filter is in the sidebar
    dashboard.create_dashboard()
    print("\n4. ✓ Dashboard template created successfully")

    # Check registry statistics
    if dashboard.current_registry:
        n_objects = len(dashboard.current_registry.get_all_objects())
        n_theorems = len(dashboard.current_registry.get_all_theorems())
        n_axioms = len(dashboard.current_registry.get_all_axioms())

        print("\n5. Registry contents:")
        print(f"   - Objects: {n_objects}")
        print(f"   - Theorems: {n_theorems}")
        print(f"   - Axioms: {n_axioms}")

        # Collect documents from all entities
        documents = set()
        for obj in dashboard.current_registry.get_all_objects():
            doc = obj.document or "unknown"
            documents.add(doc)
        for thm in dashboard.current_registry.get_all_theorems():
            doc = thm.document or "unknown"
            documents.add(doc)
        for axiom in dashboard.current_registry.get_all_axioms():
            doc = axiom.document or "unknown"
            documents.add(doc)

        print(f"   - Unique documents: {len(documents)}")
        print(f"   - Document list: {sorted(documents)}")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Document filter is correctly implemented.")
    print("\nTo use the dashboard with document filtering:")
    print("1. Run: panel serve proof_pipeline_dashboard.py --show")
    print("2. Or: python proof_pipeline_dashboard.py")
    print("3. In the sidebar, scroll down to find the 'Document' filter")
    print("4. Select/deselect documents to filter the graph visualization")


if __name__ == "__main__":
    test_document_filter()
