"""
Data loader and transformer for math_schema.json-compliant documents.

This module provides functions to load mathematical documentation in the
math_schema.json format and transform it into NetworkX graphs suitable
for visualization in the theorem dependency graph dashboard.
"""

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


def load_math_schema_document(json_path: str | Path) -> dict[str, Any]:
    """
    Load and parse a JSON file compliant with math_schema.json.

    Args:
        json_path: Path to the JSON file to load

    Returns:
        Dictionary with keys:
        - metadata: Document metadata (title, authors, version, etc.)
        - directives: List of mathematical directives (axioms, theorems, etc.)
        - dependency_graph: Graph structure with nodes and edges
        - constants_glossary: Glossary of mathematical constants
        - notation_index: Index of mathematical notation

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    path = Path(json_path)
    if not path.exists():
        msg = f"JSON file not found: {json_path}"
        raise FileNotFoundError(msg)

    with path.open() as f:
        data = json.load(f)

    return {
        "metadata": data.get("metadata", {}),
        "directives": data.get("directives", []),
        "dependency_graph": data.get("dependency_graph", {"nodes": [], "edges": []}),
        "constants_glossary": data.get("constants_glossary", {}),
        "notation_index": data.get("notation_index", {}),
    }


def build_networkx_graph(document: dict[str, Any]) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from a math_schema document.

    The graph is constructed from the directives and dependency_graph sections.
    Each directive becomes a node with type-specific attributes.

    Node attributes:
        - label: Unique identifier (e.g., 'ax:lipschitz-fields')
        - type: Directive type (axiom, theorem, lemma, etc.)
        - title: Human-readable title
        - statement: Mathematical statement
        - tags: List of tags for categorization
        - category: Category (for axioms)
        - importance: Importance level (for theorems/lemmas/propositions)
        - rigor_level: Rigor score 1-10 (for proofs)
        - difficulty: Difficulty level (for proofs)
        - proof_type: Type of proof (for proofs)
        - + other type-specific fields

    Edge attributes:
        - relationship: Type of dependency (requires, uses, extends, proves, etc.)

    Args:
        document: Parsed document from load_math_schema_document()

    Returns:
        NetworkX DiGraph with nodes and edges representing the mathematical structure
    """
    G = nx.DiGraph()

    directives = document.get("directives", [])
    dependency_graph = document.get("dependency_graph", {})

    # Add nodes from directives
    for directive in directives:
        label = directive.get("label")
        if not label:
            continue

        # Common attributes for all directive types
        node_attrs = {
            "label": label,
            "type": directive.get("type"),
            "title": directive.get("title", ""),
            "statement": directive.get("statement", ""),
            "tags": directive.get("tags", []),
            "source": directive.get("source", ""),
        }

        # Type-specific attributes
        directive_type = directive.get("type")

        if directive_type == "axiom":
            node_attrs.update({
                "category": directive.get("category", ""),
                "axiomatic_parameters": directive.get("axiomatic_parameters", []),
                "rationale": directive.get("rationale", ""),
                "failure_modes": directive.get("failure_modes", []),
            })

        elif directive_type in ["theorem", "lemma", "proposition"]:
            node_attrs.update({
                "importance": directive.get("importance", ""),
                "hypotheses": directive.get("hypotheses", []),
                "conclusion": directive.get("conclusion", ""),
                "proof_sketch": directive.get("proof_sketch", ""),
            })

        elif directive_type == "corollary":
            node_attrs.update({
                "follows_from": directive.get("follows_from", []),
                "importance": directive.get("importance", ""),
            })

        elif directive_type == "definition":
            node_attrs.update({
                "mathematical_definition": directive.get("mathematical_definition", ""),
                "domain": directive.get("domain", ""),
                "properties": directive.get("properties", []),
            })

        elif directive_type == "proof":
            node_attrs.update({
                "proves": directive.get("proves", ""),
                "rigor_level": directive.get("rigor_level"),
                "difficulty": directive.get("difficulty", ""),
                "proof_type": directive.get("proof_type", ""),
                "dependencies": directive.get("dependencies", []),
                "key_steps": directive.get("key_steps", []),
            })

        elif directive_type == "algorithm":
            node_attrs.update({
                "inputs": directive.get("inputs", []),
                "outputs": directive.get("outputs", []),
                "complexity": directive.get("complexity", {}),
                "pseudocode": directive.get("pseudocode", ""),
            })

        elif directive_type in ["remark", "observation"]:
            node_attrs.update({
                "relates_to": directive.get("relates_to", []),
                "significance": directive.get("significance", ""),
            })

        elif directive_type == "conjecture":
            node_attrs.update({
                "confidence": directive.get("confidence", ""),
                "supporting_evidence": directive.get("supporting_evidence", []),
                "counterexamples": directive.get("counterexamples", []),
            })

        elif directive_type == "example":
            node_attrs.update({
                "demonstrates": directive.get("demonstrates", []),
                "example_content": directive.get("example_content", ""),
            })

        elif directive_type == "property":
            node_attrs.update({
                "property_of": directive.get("property_of", ""),
                "conditions": directive.get("conditions", []),
            })

        G.add_node(label, **node_attrs)

    # Add edges from dependency_graph
    edges = dependency_graph.get("edges", [])
    for edge in edges:
        source = edge.get("from")
        target = edge.get("to")
        relationship = edge.get("relationship", "depends_on")

        if source and target:
            # Only add edge if both nodes exist
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, relationship=relationship)

    return G


def compute_statistics(G: nx.DiGraph, document: dict[str, Any]) -> dict[str, Any]:
    """
    Compute statistical summaries of the graph and document.

    Args:
        G: NetworkX graph built from document
        document: Parsed document from load_math_schema_document()

    Returns:
        Dictionary with statistics:
        - total_nodes: Total number of nodes
        - total_edges: Total number of edges
        - by_type: Count of nodes by directive type
        - by_category: Count of axioms by category
        - by_importance: Count of theorems/lemmas/propositions by importance
        - by_tags: Count of nodes by tag
        - document_info: Metadata about the document
    """
    stats = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "by_type": {},
        "by_category": {},
        "by_importance": {},
        "by_tags": {},
        "document_info": document.get("metadata", {}),
    }

    # Count by type
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        stats["by_type"][node_type] = stats["by_type"].get(node_type, 0) + 1

        # Count by category (for axioms)
        if node_type == "axiom":
            category = attrs.get("category", "uncategorized")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

        # Count by importance (for theorems/lemmas/propositions)
        if node_type in ["theorem", "lemma", "proposition"]:
            importance = attrs.get("importance", "unspecified")
            stats["by_importance"][importance] = stats["by_importance"].get(importance, 0) + 1

        # Count by tags
        for tag in attrs.get("tags", []):
            stats["by_tags"][tag] = stats["by_tags"].get(tag, 0) + 1

    return stats


def filter_graph(
    G: nx.DiGraph,
    types: list[str] | None = None,
    categories: list[str] | None = None,
    importance_levels: list[str] | None = None,
    tags: list[str] | None = None,
    include_dependencies: bool = True,
) -> nx.DiGraph:
    """
    Filter the graph based on various criteria.

    Args:
        G: NetworkX graph to filter
        types: List of directive types to include (e.g., ['axiom', 'theorem'])
        categories: List of categories to include (for axioms)
        importance_levels: List of importance levels (for theorems/lemmas/propositions)
        tags: List of tags - include nodes that have ANY of these tags
        include_dependencies: If True, also include nodes that are dependencies
                            of the filtered nodes

    Returns:
        Filtered NetworkX DiGraph (subgraph of original)
    """
    # Start with all nodes
    nodes_to_include = set(G.nodes())

    # Filter by type
    if types is not None and len(types) > 0:
        nodes_to_include = {
            n for n in nodes_to_include if G.nodes[n].get("type") in types
        }

    # Filter by category (for axioms)
    if categories is not None and len(categories) > 0:
        nodes_to_include = {
            n
            for n in nodes_to_include
            if G.nodes[n].get("type") == "axiom"
            and G.nodes[n].get("category") in categories
        }

    # Filter by importance (for theorems/lemmas/propositions)
    if importance_levels is not None and len(importance_levels) > 0:
        nodes_to_include = {
            n
            for n in nodes_to_include
            if G.nodes[n].get("type") in ["theorem", "lemma", "proposition"]
            and G.nodes[n].get("importance") in importance_levels
        }

    # Filter by tags (include if node has ANY of the specified tags)
    if tags is not None and len(tags) > 0:
        nodes_to_include = {
            n
            for n in nodes_to_include
            if any(tag in G.nodes[n].get("tags", []) for tag in tags)
        }

    # Include dependencies if requested
    if include_dependencies:
        # Add all predecessors and successors
        extended_nodes = set(nodes_to_include)
        for node in nodes_to_include:
            extended_nodes.update(G.predecessors(node))
            extended_nodes.update(G.successors(node))
        nodes_to_include = extended_nodes

    # Create subgraph
    return G.subgraph(nodes_to_include).copy()


def get_all_unique_values(G: nx.DiGraph) -> dict[str, list[str]]:
    """
    Extract all unique values for filterable attributes from the graph.

    This is useful for populating filter widget options dynamically.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary with keys:
        - types: All unique directive types
        - categories: All unique categories (from axioms)
        - importance_levels: All unique importance levels
        - tags: All unique tags
    """
    types = set()
    categories = set()
    importance_levels = set()
    tags = set()

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type")
        if node_type:
            types.add(node_type)

        if node_type == "axiom":
            category = attrs.get("category")
            if category:
                categories.add(category)

        if node_type in ["theorem", "lemma", "proposition"]:
            importance = attrs.get("importance")
            if importance:
                importance_levels.add(importance)

        for tag in attrs.get("tags", []):
            tags.add(tag)

    return {
        "types": sorted(list(types)),
        "categories": sorted(list(categories)),
        "importance_levels": sorted(list(importance_levels)),
        "tags": sorted(list(tags)),
    }


def graph_to_dataframes(G: nx.DiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert NetworkX graph to pandas DataFrames for nodes and edges.

    This format is convenient for use with HoloViews and Panel.

    Args:
        G: NetworkX graph

    Returns:
        Tuple of (nodes_df, edges_df)
        - nodes_df: DataFrame with node attributes
        - edges_df: DataFrame with edge attributes
    """
    # Create nodes DataFrame
    nodes_data = []
    for node, attrs in G.nodes(data=True):
        node_dict = {"node_id": node, **attrs}
        nodes_data.append(node_dict)

    nodes_df = pd.DataFrame(nodes_data)

    # Create edges DataFrame
    edges_data = []
    for source, target, attrs in G.edges(data=True):
        edge_dict = {"source": source, "target": target, **attrs}
        edges_data.append(edge_dict)

    edges_df = pd.DataFrame(edges_data) if edges_data else pd.DataFrame(
        columns=["source", "target", "relationship"]
    )

    return nodes_df, edges_df
