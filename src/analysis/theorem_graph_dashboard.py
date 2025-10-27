#!/usr/bin/env python3
"""
Interactive Mathematical Documentation Graph Dashboard.

Visualizes mathematical directives (axioms, theorems, lemmas, etc.) from
math_schema.json-compliant documents using shaolin's interactive graph
capabilities with HoloViews and Panel.

Features:
- Supports 13 directive types from math_schema.json
- Type-specific filtering (axiom, theorem, lemma, proposition, corollary, etc.)
- Category-based filtering for axioms
- Importance-based filtering for theorems/lemmas/propositions
- Tag-based filtering
- Type-specific node details display
- Reactive updates (no apply button needed)

Usage:
    python src/analysis/theorem_graph_dashboard.py
    # Or with panel serve:
    panel serve src/analysis/theorem_graph_dashboard.py --show
"""

from pathlib import Path

import holoviews as hv
import networkx as nx
import numpy as np
import panel as pn

from analysis.load_math_schema import (
    build_networkx_graph,
    get_all_unique_values,
    load_math_schema_document,
)
from analysis.render_math_json import MathDocumentRenderer, RenderOptions
from fragile.shaolin.graph import (
    InteractiveGraph,
    create_graphviz_layout,
    edges_as_df,
    nodes_as_df,
    select_index,
)

hv.extension("bokeh")
pn.extension()


# Color schemes for directive types (13 types)
TYPE_COLORS = {
    "axiom": "#1f77b4",  # Blue
    "theorem": "#d62728",  # Red
    "lemma": "#2ca02c",  # Green
    "proposition": "#ff7f0e",  # Orange
    "corollary": "#9467bd",  # Purple
    "definition": "#8c564b",  # Brown
    "proof": "#e377c2",  # Pink
    "algorithm": "#7f7f7f",  # Gray
    "remark": "#bcbd22",  # Yellow-green
    "observation": "#17becf",  # Cyan
    "conjecture": "#ff9896",  # Light red
    "example": "#98df8a",  # Light green
    "property": "#c5b0d5",  # Light purple
}

# Color schemes for axiom categories
CATEGORY_COLORS = {
    "environmental": "#2ecc71",  # Green
    "algorithmic-dynamics": "#3498db",  # Blue
    "regularity": "#e74c3c",  # Red
    "learnability": "#f39c12",  # Orange
    "measurement": "#9b59b6",  # Purple
    "kinetic": "#1abc9c",  # Turquoise
}

# Color schemes for importance levels
IMPORTANCE_COLORS = {
    "foundational": "#f1c40f",  # Gold
    "main-result": "#e74c3c",  # Red
    "technical": "#3498db",  # Blue
    "auxiliary": "#95a5a6",  # Gray
    "routine": "#bdc3c7",  # Light gray
}


class TheoremGraphDashboard:
    """Interactive dashboard for mathematical documentation graph visualization."""

    def __init__(self, json_path: str = "cloning_complete.json"):
        """Initialize dashboard.

        Args:
            json_path: Path to math_schema.json-compliant file
        """
        # Load data
        print(f"Loading mathematical documentation from {json_path}...")
        self.document = load_math_schema_document(json_path)
        self.full_graph = build_networkx_graph(self.document)

        print(
            f"Loaded {self.full_graph.number_of_nodes()} nodes, "
            f"{self.full_graph.number_of_edges()} edges"
        )

        # Get document metadata
        self.metadata = self.document.get("metadata", {})
        print(f"Document: {self.metadata.get('title', 'Untitled')}")
        print(f"Document ID: {self.metadata.get('document_id', 'unknown')}")

        # Compute default layout (neato - spring model)
        print("Computing default layout (neato)...")
        default_layout = create_graphviz_layout(
            self.full_graph,
            top_to_bottom=False,
            prog="neato",  # Spring model - good for showing clusters
        )
        print("Layout computed!")

        # Initialize layout cache
        self.layout_cache = {
            "neato": default_layout,  # Pre-computed default
        }
        self.current_layout_name = "neato"

        # Create base dataframes using shaolin's functions
        # (nodes_as_df creates proper structure with 'index' column expected by HoloViews)
        self.df_nodes_full = nodes_as_df(self.full_graph, default_layout)

        # Handle edges (edges_as_df doesn't handle empty graphs, so check first)
        if self.full_graph.number_of_edges() > 0:
            self.df_edges_full = edges_as_df(self.full_graph, data=True)
        else:
            # Create empty edges dataframe with proper columns
            import pandas as pd
            self.df_edges_full = pd.DataFrame(columns=["source", "target"])

        # Convert categorical columns to categorical dtype
        categorical_cols = ["type"]
        for col in categorical_cols:
            if col in self.df_nodes_full.columns:
                self.df_nodes_full[col] = self.df_nodes_full[col].astype("category")

        # Add category column (for axioms, empty for others)
        if "category" not in self.df_nodes_full.columns:
            self.df_nodes_full["category"] = ""
        self.df_nodes_full["category"] = self.df_nodes_full["category"].fillna("").astype("category")

        # Add importance column (for theorems/lemmas/propositions, empty for others)
        if "importance" not in self.df_nodes_full.columns:
            self.df_nodes_full["importance"] = ""
        self.df_nodes_full["importance"] = self.df_nodes_full["importance"].fillna("").astype("category")

        # Get all unique values for filters
        self.unique_values = get_all_unique_values(self.full_graph)

        # Create filter widgets
        self._create_filter_widgets()

        # Create reactive components
        self._create_reactive_components()

        # Set up tap interaction for node details
        # This will be populated by bind_to_stream in _create_reactive_components
        self.node_details_view = None

    def _create_filter_widgets(self):
        """Create filter control widgets."""
        # Type filter (13 directive types)
        all_types = self.unique_values["types"]
        self.type_filter = pn.widgets.MultiChoice(
            name="Directive Type",
            options=all_types,
            value=all_types,
            width=280,
            description="Filter by directive type (axiom, theorem, lemma, etc.)",
        )

        # Category filter (for axioms)
        all_categories = self.unique_values["categories"]
        self.category_filter = pn.widgets.MultiChoice(
            name="Axiom Category",
            options=all_categories if all_categories else ["(no categories)"],
            value=all_categories,
            width=280,
            description="Filter axioms by category",
        )

        # Importance filter (for theorems/lemmas/propositions)
        all_importance = self.unique_values["importance_levels"]
        self.importance_filter = pn.widgets.MultiChoice(
            name="Importance Level",
            options=all_importance if all_importance else ["(no importance levels)"],
            value=all_importance,
            width=280,
            description="Filter theorems/lemmas by importance",
        )

        # Tag filter
        all_tags = self.unique_values["tags"]
        self.tags_filter = pn.widgets.MultiChoice(
            name="Tags",
            options=all_tags if all_tags else ["(no tags)"],
            value=[],  # Start empty - show all by default
            width=280,
            description="Filter by tags (shows nodes with ANY selected tag)",
        )

        # Include dependencies toggle
        self.include_deps_toggle = pn.widgets.Checkbox(
            name="Include Dependencies",
            value=True,
            width=280,
        )

        # Layout selector
        self.layout_selector = pn.widgets.Select(
            name="Graph Layout Algorithm",
            options={
                "Spring Model (neato)": "neato",
                "Hierarchical (dot)": "dot",
                "Force-Directed (sfdp)": "sfdp",
                "Spectral (NetworkX)": "spectral",
                "Circular (NetworkX)": "circular",
            },
            value="neato",  # Default layout
            width=280,
            description="Algorithm for positioning nodes",
        )

        # Reset button
        self.reset_filter_button = pn.widgets.Button(
            name="Reset All Filters",
            button_type="warning",
            width=280,
        )
        self.reset_filter_button.on_click(self._on_filter_reset)

    def _compute_node_alpha(
        self,
        types: list[str],
        categories: list[str],
        importance_levels: list[str],
        tags: list[str],
        include_deps: bool,
    ):
        """Compute alpha transparency values for nodes based on filters.

        Args:
            types: List of directive types to show
            categories: List of categories to show (for axioms)
            importance_levels: List of importance levels (for theorems/lemmas/propositions)
            tags: List of tags to show
            include_deps: If True, also show dependencies

        Returns:
            Series of alpha values (1.0 = visible, 0.01 = filtered out)
        """
        import pandas as pd

        df = self.df_nodes_full.copy()

        # Start with all nodes visible
        alpha = pd.Series(1.0, index=df.index)

        # Track which nodes pass filters
        passes_filter = pd.Series(True, index=df.index)

        # Filter by type
        if types:
            passes_filter &= df["type"].isin(types)

        # Filter by category (only applies to axioms)
        if categories and "category" in df.columns:
            # Only filter axioms by category
            axiom_mask = df["type"] == "axiom"
            category_mask = df["category"].isin(categories)
            # Axioms must match category, non-axioms pass automatically
            passes_filter &= ~axiom_mask | category_mask

        # Filter by importance (only applies to theorems/lemmas/propositions)
        if importance_levels and "importance" in df.columns:
            # Only filter theorems/lemmas/propositions by importance
            theorem_types = ["theorem", "lemma", "proposition"]
            theorem_mask = df["type"].isin(theorem_types)
            importance_mask = df["importance"].isin(importance_levels)
            # Theorems must match importance, non-theorems pass automatically
            passes_filter &= ~theorem_mask | importance_mask

        # Filter by tags (show if has ANY of the selected tags)
        if tags:
            # Check if node has any of the selected tags
            def has_any_tag(node_tags):
                if not node_tags or (isinstance(node_tags, float) and np.isnan(node_tags)):
                    return False
                return any(tag in node_tags for tag in tags)

            if "tags" in df.columns:
                tag_mask = df["tags"].apply(has_any_tag)
                passes_filter &= tag_mask

        # Get nodes that pass filters
        visible_nodes = set(df.index[passes_filter])

        # Include dependencies if requested
        if include_deps and visible_nodes:
            extended_nodes = set(visible_nodes)
            for node in visible_nodes:
                extended_nodes.update(self.full_graph.predecessors(node))
                extended_nodes.update(self.full_graph.successors(node))
            visible_nodes = extended_nodes

        # Set alpha: 1.0 for visible, 0.01 for filtered out
        alpha[~df.index.isin(visible_nodes)] = 0.01

        return alpha

    def _get_or_compute_layout(self, layout_name: str) -> dict[str, tuple[float, float]]:
        """Get layout from cache or compute it on-demand.

        Args:
            layout_name: Name of layout algorithm

        Returns:
            Dictionary mapping node labels to (x, y) positions
        """
        # Return from cache if available
        if layout_name in self.layout_cache:
            return self.layout_cache[layout_name]

        # Compute new layout
        print(f"Computing {layout_name} layout (this may take 2-5 seconds)...")

        if layout_name in ["dot", "neato", "fdp", "sfdp", "circo", "twopi"]:
            # Graphviz layouts
            layout = create_graphviz_layout(
                self.full_graph,
                top_to_bottom=False,
                prog=layout_name,
            )
        elif layout_name == "spectral":
            # NetworkX spectral layout
            layout = nx.spectral_layout(self.full_graph)
        elif layout_name == "circular":
            # NetworkX circular layout
            layout = nx.circular_layout(self.full_graph)
        elif layout_name == "spring":
            # NetworkX spring layout (slower)
            layout = nx.spring_layout(self.full_graph, k=0.5, iterations=50)
        else:
            # Fallback to neato (default)
            print(f"Unknown layout '{layout_name}', falling back to neato")
            layout = self.layout_cache["neato"]

        # Cache for future use
        self.layout_cache[layout_name] = layout
        print(f"{layout_name} layout computed and cached!")

        return layout

    def _create_reactive_components(self):
        """Create reactive graph, controls, and statistics views."""
        # Reactive graph view (plot only)
        self.graph_view = pn.bind(
            self._create_graph_plot,
            types=self.type_filter,
            categories=self.category_filter,
            importance_levels=self.importance_filter,
            tags=self.tags_filter,
            include_deps=self.include_deps_toggle,
            layout=self.layout_selector,
        )

        # Reactive controls view
        self.controls_view = pn.bind(
            self._create_graph_controls,
            types=self.type_filter,
            categories=self.category_filter,
            importance_levels=self.importance_filter,
            tags=self.tags_filter,
            include_deps=self.include_deps_toggle,
            layout=self.layout_selector,
        )

        # Reactive statistics view
        self.stats_view = pn.bind(
            self._format_statistics,
            types=self.type_filter,
            categories=self.category_filter,
            importance_levels=self.importance_filter,
            tags=self.tags_filter,
            include_deps=self.include_deps_toggle,
        )

    def _create_graph_plot(
        self,
        types: list[str],
        categories: list[str],
        importance_levels: list[str],
        tags: list[str],
        include_deps: bool,
        layout: str,
    ):
        """Create interactive graph plot with current filter settings."""
        # Get or compute layout
        layout_positions = self._get_or_compute_layout(layout)

        # Update df_nodes_full with new positions
        for node, (x, y) in layout_positions.items():
            if node in self.df_nodes_full.index:
                self.df_nodes_full.loc[node, "x"] = x
                self.df_nodes_full.loc[node, "y"] = y

        # Compute alpha values based on filters
        alpha_series = self._compute_node_alpha(
            types, categories, importance_levels, tags, include_deps
        )

        # Create dataframe with alpha column
        df_nodes = self.df_nodes_full.copy()
        df_nodes["alpha_filter"] = alpha_series

        # Convert list/dict columns to strings (HoloViews can't handle complex types)
        for col in df_nodes.columns:
            # Check first non-null value
            sample = df_nodes[col].dropna().iloc[0] if len(df_nodes[col].dropna()) > 0 else None
            if isinstance(sample, list):
                # Convert lists to comma-separated strings
                df_nodes[col] = df_nodes[col].apply(
                    lambda x: ', '.join(str(i) for i in x) if isinstance(x, list) else x
                )
            elif isinstance(sample, dict):
                # Convert dicts to string representation
                df_nodes[col] = df_nodes[col].apply(
                    lambda x: str(x) if isinstance(x, dict) else x
                )

        # Define columns to ignore (filter to only existing columns)
        ignore_cols_candidates = (
            "label",  # Duplicate of index
            "statement",  # Too verbose
            "source",  # Source reference (keep for details view)
            "tags",  # Better shown in details
            "axiomatic_parameters",  # Too complex for mapping
            "failure_modes",  # Too complex
            "hypotheses",  # Too complex (theorems/lemmas/propositions only)
            "key_steps",  # Too complex (proofs only)
            "properties",  # Too complex (definitions only)
            "inputs",  # Too complex (algorithms only)
            "outputs",  # Too complex (algorithms only)
        )
        # Only ignore columns that actually exist in the dataframe
        ignore_node_cols = tuple(col for col in ignore_cols_candidates if col in df_nodes.columns)

        # Create InteractiveGraph
        ig = InteractiveGraph(
            df_nodes=df_nodes,
            df_edges=self.df_edges_full,
            ignore_node_cols=ignore_node_cols,
            n_cols=3,
        )

        # Store reference for node tap handling
        self._current_ig = ig
        self._current_df_nodes = df_nodes

        # Set up node details view using shaolin's bind_to_stream
        # This properly handles tap-to-index conversion
        self.node_details_view = ig.bind_to_stream(self._on_node_select)

        # Return ONLY the graph plot
        return pn.pane.HoloViews(ig.dmap, height=700, sizing_mode="stretch_width")

    def _create_graph_controls(
        self,
        types: list[str],
        categories: list[str],
        importance_levels: list[str],
        tags: list[str],
        include_deps: bool,
        layout: str,
    ):
        """Create visualization controls for the graph."""
        # Check if InteractiveGraph has been created
        if not hasattr(self, "_current_ig") or self._current_ig is None:
            return pn.pane.Markdown(
                "**Graph loading...**\n\n*Controls will appear once the graph is ready.*",
                styles={"color": "#666", "font-style": "italic"},
            )

        # Return the controls
        return pn.Column(
            pn.pane.Markdown(
                """**Visual Property Mapping**

Use the controls below to customize how directive properties are displayed:
- **Node Color**: Map to `type`, `category`, `importance`
- **Node Size**: Map to any numeric property
- **Node Alpha**: Map to any numeric property (transparency)

**Examples:**
- Color by `type` to see different directive types (axiom, theorem, lemma, etc.)
- Color by `category` to see axiom categories (environmental, algorithmic, etc.)
- Color by `importance` to see theorem importance (foundational, main-result, etc.)
                """,
                styles={
                    "font-size": "0.85em",
                    "color": "#333",
                    "background": "#f0f8ff",
                    "padding": "10px",
                    "border-radius": "5px",
                },
            ),
            self._current_ig.layout(),
            sizing_mode="stretch_width",
        )

    @select_index()
    def _on_node_select(self, ix, df):
        """Handle node selection using shaolin's select_index decorator.

        Args:
            ix: Index of the selected node (computed by select_index from tap coordinates)
            df: The dataframe containing node data
        """
        try:
            if ix is None or df is None or df.empty:
                return pn.pane.Markdown(
                    "**Click a node to see details**\n\n"
                    "*Tip: Click on any directive in the graph above*",
                    width=480,
                )

            # Get the node label from the dataframe index
            node_label = df.index[ix]

            # Render node details
            return self._render_node_details(node_label)

        except Exception as e:
            print(f"ERROR in _on_node_select: {e}")
            import traceback
            traceback.print_exc()
            return pn.pane.Markdown(
                f"**Error displaying node details**\n\n```\n{e}\n```",
                width=480,
            )

    def _get_node_details_panel(self):
        """Get the current node details panel.

        Returns the reactive node_details_view if available, otherwise a default message.
        """
        if self.node_details_view is not None:
            return self.node_details_view
        else:
            return pn.pane.Markdown(
                "**Click a node to see details**\n\n"
                "*Tip: Click on any directive in the graph above*",
                width=480,
            )

    def _render_node_details(self, node_label: str):
        """Render node details using render_math_json renderer.

        Returns:
            pn.pane.Markdown: A Panel Markdown pane with the rendered node details
        """
        try:
            if node_label not in self.full_graph:
                return pn.pane.Markdown(
                    f"**Node not found:** `{node_label}`",
                    width=480,
                )

            # Get all node attributes (these are the directive fields)
            attrs = dict(self.full_graph.nodes[node_label])
            attrs["label"] = node_label

            # Create minimal document structure for rendering single directive
            doc = {
                "metadata": {
                    "title": f"Directive: {node_label}",
                    "document_id": self.metadata.get("document_id", "unknown"),
                },
                "directives": [attrs],
                "dependency_graph": {
                    "nodes": [{"label": node_label, "type": attrs.get("type")}],
                    "edges": [],
                },
            }

            # Configure renderer options
            options = RenderOptions(
                include_graph=False,
                jupyter_book=False,
            )

            # Render to markdown
            renderer = MathDocumentRenderer(doc, options)
            markdown = renderer.render()

            # Return Panel Markdown pane
            return pn.pane.Markdown(markdown, width=480)

        except Exception as e:
            print(f"ERROR in _render_node_details: {e}")
            import traceback
            traceback.print_exc()
            return pn.pane.Markdown(
                f"**Error rendering node details**\n\n```\n{e}\n```",
                width=480,
            )

    def _format_statistics(
        self,
        types: list[str],
        categories: list[str],
        importance_levels: list[str],
        tags: list[str],
        include_deps: bool,
    ) -> pn.pane.Markdown:
        """Format graph statistics showing visible/total counts."""
        import pandas as pd

        # Compute which nodes are visible
        alpha = self._compute_node_alpha(types, categories, importance_levels, tags, include_deps)
        visible_indices = alpha[alpha > 0.5].index

        # Count statistics
        total_nodes = self.full_graph.number_of_nodes()
        visible_nodes = len(visible_indices)
        total_edges = self.full_graph.number_of_edges()
        visible_edges = sum(
            1
            for u, v in self.full_graph.edges()
            if u in visible_indices and v in visible_indices
        )

        # Count by type
        type_counts_visible = {}
        type_counts_total = {}
        for node in self.full_graph.nodes():
            node_type = self.full_graph.nodes[node].get("type", "unknown")
            type_counts_total[node_type] = type_counts_total.get(node_type, 0) + 1
            if node in visible_indices:
                type_counts_visible[node_type] = type_counts_visible.get(node_type, 0) + 1

        # Count by category (for axioms)
        category_counts_visible = {}
        category_counts_total = {}
        for node in self.full_graph.nodes():
            if self.full_graph.nodes[node].get("type") == "axiom":
                category = self.full_graph.nodes[node].get("category", "uncategorized")
                category_counts_total[category] = category_counts_total.get(category, 0) + 1
                if node in visible_indices:
                    category_counts_visible[category] = category_counts_visible.get(category, 0) + 1

        # Count by importance (for theorems/lemmas/propositions)
        importance_counts_visible = {}
        importance_counts_total = {}
        for node in self.full_graph.nodes():
            if self.full_graph.nodes[node].get("type") in ["theorem", "lemma", "proposition"]:
                importance = self.full_graph.nodes[node].get("importance", "unspecified")
                importance_counts_total[importance] = importance_counts_total.get(importance, 0) + 1
                if node in visible_indices:
                    importance_counts_visible[importance] = importance_counts_visible.get(
                        importance, 0
                    ) + 1

        # Format markdown
        md = f"""
## Document Statistics

### Document Info
**Title**: {self.metadata.get('title', 'Untitled')}

**Document ID**: `{self.metadata.get('document_id', 'unknown')}`

**Version**: {self.metadata.get('version', 'unknown')}

---

### Overview
**Nodes**: **{visible_nodes}** / {total_nodes} (visible / total)

**Edges**: **{visible_edges}** / {total_edges}

---

### By Directive Type
"""

        for dtype, count in sorted(type_counts_visible.items(), key=lambda x: -x[1]):
            total = type_counts_total.get(dtype, 0)
            color = TYPE_COLORS.get(dtype, "#95a5a6")
            md += f'- <span style="color:{color}">**{dtype}**</span>: {count} / {total}\n'

        if category_counts_total:
            md += "\n---\n\n### By Axiom Category\n"
            for category, count in sorted(category_counts_visible.items(), key=lambda x: -x[1]):
                total = category_counts_total.get(category, 0)
                color = CATEGORY_COLORS.get(category, "#95a5a6")
                md += f'- <span style="color:{color}">**{category}**</span>: {count} / {total}\n'

        if importance_counts_total:
            md += "\n---\n\n### By Importance Level\n"
            for importance, count in sorted(importance_counts_visible.items(), key=lambda x: -x[1]):
                total = importance_counts_total.get(importance, 0)
                color = IMPORTANCE_COLORS.get(importance, "#95a5a6")
                md += (
                    f'- <span style="color:{color}">**{importance}**</span>: {count} / {total}\n'
                )

        return pn.pane.Markdown(md, width=480)

    def _on_filter_reset(self, event):
        """Reset all filters to defaults."""
        self.type_filter.value = self.unique_values["types"]
        self.category_filter.value = self.unique_values["categories"]
        self.importance_filter.value = self.unique_values["importance_levels"]
        self.tags_filter.value = []
        self.include_deps_toggle.value = True
        self.layout_selector.value = "neato"

    def create_dashboard(self) -> pn.Template:
        """Create the complete dashboard layout."""
        # Sidebar: Filters
        sidebar_content = [
            pn.pane.Markdown(
                "## Filters\n*Updates apply automatically*",
                styles={"font-size": "0.95em"},
                margin=(0, 0, 15, 0),
            ),
            self.type_filter,
            self.category_filter,
            self.importance_filter,
            self.tags_filter,
            self.include_deps_toggle,
            pn.layout.Divider(),
            # Layout selector
            pn.pane.Markdown(
                "## Layout Algorithm\n*May take 2-5s to compute new layouts*",
                styles={"font-size": "0.95em"},
                margin=(10, 0, 10, 0),
            ),
            self.layout_selector,
            pn.layout.Divider(),
            self.reset_filter_button,
            pn.pane.Markdown(
                "**Tip**: Filtered nodes fade to barely visible (alpha=0.01) "
                "instead of disappearing, keeping the layout stable.",
                styles={"font-size": "0.85em", "color": "#666", "margin-top": "20px"},
            ),
        ]

        # Main content
        main_content = [
            # Graph
            pn.Card(
                pn.panel(self.graph_view),
                title="Mathematical Documentation Graph",
                collapsed=False,
                sizing_mode="stretch_width",
                styles={"background": "#f8f9fa"},
            ),
            # Controls
            pn.Card(
                pn.panel(self.controls_view),
                title="🎨 Visualization Controls - Map Visual Properties",
                collapsible=True,
                collapsed=False,
                sizing_mode="stretch_width",
                styles={"background": "#fff"},
                header_background="#3498db",
            ),
            # Info row
            pn.Row(
                pn.Card(
                    pn.panel(self.stats_view),
                    title="Statistics",
                    width=500,
                    height=450,
                    scroll=True,
                    collapsed=False,
                ),
                pn.Card(
                    pn.panel(self._get_node_details_panel),
                    title="Node Details",
                    width=500,
                    height=450,
                    scroll=True,
                    collapsed=False,
                ),
                sizing_mode="stretch_width",
            ),
        ]

        # Create template
        template = pn.template.FastListTemplate(
            title=f"Mathematical Documentation: {self.metadata.get('document_id', 'Unknown')}",
            sidebar=sidebar_content,
            main=main_content,
            accent_base_color="#3498db",
            header_background="#2c3e50",
        )

        return template


def main():
    """Main entry point for panel serve."""
    dashboard = TheoremGraphDashboard()
    template = dashboard.create_dashboard()
    return template.servable()


# Create servable (executes only when module is loaded by panel serve)
main()
