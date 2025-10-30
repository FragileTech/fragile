#!/usr/bin/env python3
"""
Example usage of the Theorem Graph Dashboard.

This script demonstrates different ways to use and customize the dashboard.
"""

import networkx as nx

from analysis.theorem_graph_dashboard import (
    build_networkx_graph,
    compute_graph_statistics,
    filter_graph,
    load_graph_data,
    TheoremGraphDashboard,
)


def example_1_basic_dashboard():
    """Example 1: Launch the full interactive dashboard.

    Note: This example shows how to create the dashboard programmatically.
    To actually launch it, use:
        panel serve src/analysis/theorem_graph_dashboard.py --show
    """
    print("Example 1: Basic Dashboard")
    print("=" * 60)

    dashboard = TheoremGraphDashboard()
    template = dashboard.create_dashboard()

    print("Dashboard created!")
    print("To launch it, run:")
    print("  panel serve src/analysis/theorem_graph_dashboard.py --show")

    # Make it servable for panel serve
    return template.servable()


def example_2_programmatic_analysis():
    """Example 2: Programmatic graph analysis without GUI."""
    print("\nExample 2: Programmatic Analysis")
    print("=" * 60)

    # Load data
    data = load_graph_data()
    G = build_networkx_graph(data)

    print(f"Total theorems: {G.number_of_nodes()}")
    print(f"Total dependencies: {G.number_of_edges()}")

    # Find high-impact theorems (many dependents)
    reverse_deps = data["reverse_dependencies"]
    high_impact = sorted(
        [(node, len(deps)) for node, deps in reverse_deps.items()],
        key=lambda x: -x[1],
    )[:10]

    print("\nTop 10 High-Impact Theorems (most dependents):")
    for i, (label, n_deps) in enumerate(high_impact, 1):
        attrs = G.nodes[label]
        print(f"{i:2d}. {label:40s} ({n_deps:3d} dependents) - {attrs['recommendation']}")

    # Find theorems ready for publication
    accept_theorems = [
        (node, attrs) for node, attrs in G.nodes(data=True) if attrs["recommendation"] == "Accept"
    ]

    print(f"\n✅ Theorems ready for publication: {len(accept_theorems)}")
    for node, attrs in accept_theorems[:5]:
        print(f"   - {node}: {attrs['title'][:60]}...")


def example_3_chapter_analysis():
    """Example 3: Per-chapter quality analysis."""
    print("\nExample 3: Chapter Analysis")
    print("=" * 60)

    data = load_graph_data()
    G = build_networkx_graph(data)

    chapters = ["1_euclidean_gas", "2_geometric_gas", "3_brascamp_lieb"]

    for chapter in chapters:
        # Filter to chapter
        G_chapter = filter_graph(G, chapters=[chapter])
        stats = compute_graph_statistics(G_chapter)

        print(f"\n{chapter}:")
        print(f"  Theorems: {stats['total_nodes']}")
        print(f"  Avg Rigor: {stats['avg_rigor']}/10")
        print(f"  Avg Strategy: {stats['avg_strategy']}/10")
        print("  By recommendation:")
        for rec, count in sorted(stats["by_recommendation"].items()):
            print(f"    - {rec}: {count}")


def example_4_find_proof_gaps():
    """Example 4: Identify critical proof gaps."""
    print("\nExample 4: Critical Proof Gaps")
    print("=" * 60)

    data = load_graph_data()
    G = build_networkx_graph(data)
    reverse_deps = data["reverse_dependencies"]

    # Find theorems that:
    # 1. Need proof (status = needs_proof)
    # 2. Have high impact (many dependents)
    # 3. Are theorems/lemmas (not definitions)

    gaps = []
    for node, attrs in G.nodes(data=True):
        if (
            attrs["status"] == "needs_proof"
            and attrs["type"] in {"Theorem", "Lemma"}
            and len(reverse_deps.get(node, [])) > 0
        ):
            gaps.append((node, attrs, len(reverse_deps.get(node, []))))

    # Sort by impact
    gaps.sort(key=lambda x: -x[2])

    print(f"Found {len(gaps)} critical proof gaps\n")
    print("Top 10 high-priority theorems to prove:")
    for i, (label, attrs, n_deps) in enumerate(gaps[:10], 1):
        print(f"{i:2d}. {label:40s}")
        print(f"    Impact: {n_deps} theorems depend on this")
        print(f"    Type: {attrs['type']}")
        print(f"    Chapter: {attrs['chapter']}")
        print(f"    Rigor: {attrs['rigor']}/10, Strategy: {attrs['strategy']}/10")
        print(f"    Location: {attrs['document']}:{attrs['line']}")
        print()


def example_5_dependency_chains():
    """Example 5: Analyze dependency chains."""
    print("\nExample 5: Dependency Chain Analysis")
    print("=" * 60)

    data = load_graph_data()
    G = build_networkx_graph(data)

    # Find longest dependency chains
    if nx.is_directed_acyclic_graph(G):
        print("Graph is a DAG - can compute longest paths")
        longest_path_length = nx.dag_longest_path_length(G)
        print(f"Longest dependency chain: {longest_path_length} theorems")
    else:
        print("Graph contains cycles - computing strongly connected components")
        sccs = list(nx.strongly_connected_components(G))
        print(f"Number of strongly connected components: {len(sccs)}")

        # Find largest SCC
        largest_scc = max(sccs, key=len)
        if len(largest_scc) > 1:
            print(f"Largest cycle contains {len(largest_scc)} theorems:")
            for node in list(largest_scc)[:5]:
                print(f"  - {node}")

    # Find theorems with no dependencies (foundational)
    foundational = [node for node in G.nodes() if G.out_degree(node) == 0]
    print(f"\nFoundational theorems (no dependencies): {len(foundational)}")
    for node in foundational[:5]:
        attrs = G.nodes[node]
        print(f"  - {node}: {attrs['type']} ({attrs['recommendation']})")


def example_6_filtered_dashboard():
    """Example 6: Launch dashboard with pre-applied filters.

    Note: This example shows how to create a pre-filtered dashboard.
    The filters are reactive, so they apply automatically without needing
    to call _on_filter_apply().

    To launch it, save this as a separate script and run:
        panel serve your_script.py --show
    """
    print("\nExample 6: Filtered Dashboard")
    print("=" * 60)

    # Create dashboard
    dashboard = TheoremGraphDashboard()

    # Pre-configure filters to show only high-quality theorems
    # Note: Filters are reactive, so changes apply automatically
    dashboard.recommendation_filter.value = ["Accept", "Minor revision"]
    dashboard.rigor_slider.value = (6, 10)
    dashboard.strategy_slider.value = (6, 10)

    print("Dashboard configured to show only high-quality theorems")
    print("Filters will apply automatically when dashboard loads")

    template = dashboard.create_dashboard()
    return template.servable()


def example_7_export_subgraph():
    """Example 7: Export a filtered subgraph for external analysis."""
    print("\nExample 7: Export Subgraph")
    print("=" * 60)

    data = load_graph_data()
    G = build_networkx_graph(data)

    # Filter to only proven theorems in Chapter 1
    G_filtered = filter_graph(
        G, chapters=["1_euclidean_gas"], statuses=["proven", "has_complete_proof"]
    )

    print(f"Filtered graph: {G_filtered.number_of_nodes()} nodes")

    # Export as GraphML (can be opened in Gephi, Cytoscape, etc.)
    output_file = "euclidean_gas_proven.graphml"
    nx.write_graphml(G_filtered, output_file)
    print(f"Exported to {output_file}")

    # Export as DOT (can be rendered with Graphviz)
    output_dot = "euclidean_gas_proven.dot"
    nx.drawing.nx_pydot.write_dot(G_filtered, output_dot)
    print(f"Exported to {output_dot}")

    # Export as edge list
    output_edges = "euclidean_gas_proven_edges.csv"
    nx.write_edgelist(G_filtered, output_edges)
    print(f"Exported to {output_edges}")


def main():
    """Run all examples (except GUI-launching ones by default)."""
    print("Theorem Graph Dashboard - Examples")
    print("=" * 60)

    # Run non-GUI examples
    example_2_programmatic_analysis()
    example_3_chapter_analysis()
    example_4_find_proof_gaps()
    example_5_dependency_chains()

    print("\n" + "=" * 60)
    print("To run GUI examples, uncomment in the script:")
    print("  - example_1_basic_dashboard()")
    print("  - example_6_filtered_dashboard()")
    print("\n" + "=" * 60)

    # Uncomment to launch GUI:
    # example_1_basic_dashboard()
    # example_6_filtered_dashboard()


if __name__ == "__main__":
    main()
