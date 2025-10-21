"""Interactive visualization of the Fractal Set graph structure.

This script creates an interactive dashboard for exploring the Fractal Set
graph as it grows over time. It combines:
- GasConfig for running simulations
- FractalSet for building the graph representation
- InteractiveGraph (shaolin) for visualization

Features:
- Watch the graph grow step by step
- Color nodes by: fitness, kinetic energy, alive status, timestep
- Color edges by: edge type (CST vs IG), antisymmetric cloning potential
- Size nodes by: fitness, velocity magnitude
- Interactive selection and exploration
"""

from __future__ import annotations

import holoviews as hv
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import param

from fragile.core.fractal_set import FractalSet
from fragile.experiments.convergence_analysis import create_multimodal_potential
from fragile.experiments.gas_config_dashboard import GasConfig
from fragile.shaolin.graph import (
    create_graphviz_layout,
    InteractiveGraph,
)

# Initialize holoviews with bokeh backend
hv.extension("bokeh")


class FractalSetExplorer(param.Parameterized):
    """Interactive explorer for FractalSet graph visualization.

    This class provides a dashboard for:
    1. Configuring and running Gas simulations (GasConfig)
    2. Building the Fractal Set graph from RunHistory
    3. Visualizing the graph with interactive controls
    4. Animating graph growth over timesteps
    """

    # Display controls
    max_timestep = param.Integer(default=0, bounds=(0, 100), doc="Maximum timestep to display")
    show_cst_edges = param.Boolean(default=True, doc="Show CST (temporal) edges")
    show_ig_edges = param.Boolean(default=False, doc="Show IG (selection coupling) edges")
    layout_prog = param.ObjectSelector(
        default="physical",
        objects=["physical", "dot", "neato", "fdp", "sfdp", "circo", "twopi"],
        doc="Layout algorithm: 'physical' uses walker (x,y) positions, others use Graphviz",
    )
    top_to_bottom = param.Boolean(default=False, doc="Layout direction (top-to-bottom vs left-to-right, for Graphviz layouts only)")

    def __init__(self, **params):
        """Initialize FractalSetExplorer."""
        super().__init__(**params)

        # Create potential and GasConfig
        self.potential, _ = create_multimodal_potential(dims=2, n_gaussians=3)
        self.config = GasConfig(
            potential=self.potential,
            dims=2,
            N=20,  # Small N for readable graph
            n_steps=100,
        )

        # Data storage
        self.fractal_set: FractalSet | None = None
        self.interactive_graph: InteractiveGraph | None = None

        # Layout cache (reused across runs with same N and n_steps)
        self.layout_cache = {}  # Key: (N, n_recorded, layout_type) -> layout dict
        self.last_n_walkers = None
        self.last_n_recorded = None

        # UI components
        self.run_button = pn.widgets.Button(
            name="Run Simulation & Build Fractal Set",
            button_type="success",
        )
        self.run_button.on_click(self._on_run_clicked)

        self.status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.graph_pane = pn.pane.HoloViews(sizing_mode="stretch_both", min_height=600)

        # Watch parameters for graph updates
        self.param.watch(self._update_graph, ["max_timestep", "show_cst_edges", "show_ig_edges", "layout_prog", "top_to_bottom"])

        # Register callback from config
        self.config.add_completion_callback(self._on_simulation_complete)

    def _on_run_clicked(self, *_):
        """Handle Run button - delegate to config's run button."""
        self.config._on_run_clicked()

    def _on_simulation_complete(self, history):
        """Build Fractal Set from completed simulation."""
        self.status_pane.object = "**Building Fractal Set...**"

        # Build Fractal Set
        self.fractal_set = FractalSet(history)

        # Check if we need to recompute layouts (N or n_recorded changed)
        n_walkers = self.fractal_set.N
        n_recorded = self.fractal_set.n_recorded
        layouts_need_recompute = (
            self.last_n_walkers != n_walkers or
            self.last_n_recorded != n_recorded
        )

        if layouts_need_recompute:
            self.status_pane.object = "**Computing graph layouts (will be cached for future runs)...**"
            # Don't clear cache - keep layouts for all parameter combinations!
            self.last_n_walkers = n_walkers
            self.last_n_recorded = n_recorded
        else:
            self.status_pane.object = "**Using cached layouts from previous run...**"

        # Get or compute physical layout
        cache_key_physical = (n_walkers, n_recorded, "physical")
        if cache_key_physical in self.layout_cache:
            self.physical_layout = self.layout_cache[cache_key_physical]
        else:
            self.physical_layout = self._compute_physical_layout(self.fractal_set.graph)
            self.layout_cache[cache_key_physical] = self.physical_layout

        # Initialize graphviz layouts dict (will be populated on-demand)
        self.graphviz_layouts = {}

        # Update max_timestep bounds and START AT 0 (not max)
        self.param.max_timestep.bounds = (0, self.fractal_set.n_recorded - 1)
        self.max_timestep = 0  # Start at t=0, not at the end!

        cache_status = "cached" if not layouts_need_recompute else "computed"
        self.status_pane.object = (
            f"**Fractal Set Built!** (layouts {cache_status})\n\n"
            f"- Nodes: {self.fractal_set.total_nodes}\n"
            f"- CST edges: {self.fractal_set.num_cst_edges}\n"
            f"- IG edges: {self.fractal_set.num_ig_edges}\n"
            f"- Timesteps: {self.fractal_set.n_recorded}\n\n"
            f"Use the slider to watch the graph grow!"
        )

        # Initial graph display (will use cached layout)
        self._update_graph()

    def _compute_physical_layout(self, graph):
        """Compute layout using actual walker (x, y) positions from first 2 dimensions.

        Args:
            graph: NetworkX graph with nodes containing 'x' tensor attribute

        Returns:
            Dictionary mapping node_id -> (x, y) position
        """
        layout = {}
        for node_id, data in graph.nodes(data=True):
            x_tensor = data['x']  # [d] tensor
            # Use first two dimensions as (x, y) position
            layout[node_id] = (x_tensor[0].item(), x_tensor[1].item())
        return layout

    def _get_layout(self):
        """Get the appropriate layout based on current layout_prog setting.

        Uses cache to avoid recomputing layouts for the same N and n_recorded.

        Returns:
            Dictionary mapping node_id -> (x, y) position
        """
        if self.layout_prog == "physical":
            return self.physical_layout
        else:
            # Check if we've computed this Graphviz layout already in this session
            if self.layout_prog not in self.graphviz_layouts:
                # Check persistent cache first
                n_walkers = self.fractal_set.N
                n_recorded = self.fractal_set.n_recorded
                cache_key = (n_walkers, n_recorded, self.layout_prog)

                if cache_key in self.layout_cache:
                    self.status_pane.object = f"**Using cached {self.layout_prog} layout...**"
                    self.graphviz_layouts[self.layout_prog] = self.layout_cache[cache_key]
                else:
                    self.status_pane.object = f"**Computing {self.layout_prog} layout (will be cached)...**"
                    layout = create_graphviz_layout(
                        self.fractal_set.graph,
                        top_to_bottom=self.top_to_bottom,
                        prog=self.layout_prog,
                    )
                    self.graphviz_layouts[self.layout_prog] = layout
                    self.layout_cache[cache_key] = layout

                self.status_pane.object = (
                    f"**Fractal Set Built!**\n\n"
                    f"- Nodes: {self.fractal_set.total_nodes}\n"
                    f"- CST edges: {self.fractal_set.num_cst_edges}\n"
                    f"- IG edges: {self.fractal_set.num_ig_edges}\n"
                    f"- Timesteps: {self.fractal_set.n_recorded}\n\n"
                    f"Use the slider to watch the graph grow!"
                )
            return self.graphviz_layouts[self.layout_prog]

    def _update_graph(self, *_):
        """Update graph visualization based on current parameters."""
        if self.fractal_set is None:
            return

        # Build subgraph up to max_timestep
        subgraph = self._build_subgraph(
            max_timestep=self.max_timestep,
            show_cst=self.show_cst_edges,
            show_ig=self.show_ig_edges,
        )

        if subgraph.number_of_nodes() == 0:
            self.graph_pane.object = hv.Text(0, 0, "No nodes to display").opts(
                width=600, height=400
            )
            return

        # Get the appropriate layout (physical or Graphviz)
        full_layout = self._get_layout()

        # Filter to subgraph nodes only
        pos = {node_id: full_layout[node_id] for node_id in subgraph.nodes()}

        # Convert to DataFrames with computed attributes
        df_nodes = self._create_node_df(subgraph, pos)
        df_edges = self._create_edge_df(subgraph)

        # Create InteractiveGraph
        self.interactive_graph = InteractiveGraph(
            df_nodes=df_nodes,
            df_edges=df_edges,
            ignore_node_cols=(),  # Show all columns
            n_cols=3,
        )

        # Display graph
        self.graph_pane.object = self.interactive_graph.dmap

    def _build_subgraph(
        self,
        max_timestep: int,
        show_cst: bool = True,
        show_ig: bool = True,
    ) -> nx.DiGraph:
        """Build subgraph containing only nodes/edges up to max_timestep."""
        G = nx.DiGraph()

        # Add nodes up to max_timestep
        for walker_id in range(self.fractal_set.N):
            for t in range(min(max_timestep + 1, self.fractal_set.n_recorded)):
                node_id = (walker_id, t)
                if node_id in self.fractal_set.graph.nodes:
                    node_data = self.fractal_set.graph.nodes[node_id]
                    G.add_node(node_id, **node_data)

        # Add edges based on selection
        for u, v, data in self.fractal_set.graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            walker_u, t_u = u
            walker_v, t_v = v

            # Skip edges beyond max_timestep
            if t_u > max_timestep or t_v > max_timestep:
                continue

            # Filter by edge type
            if edge_type == "cst" and show_cst:
                G.add_edge(u, v, **data)
            elif edge_type == "ig" and show_ig:
                G.add_edge(u, v, **data)

        return G

    def _create_node_df(self, graph: nx.DiGraph, pos: dict) -> pd.DataFrame:
        """Create node DataFrame with visualization attributes.

        Note: Filters out torch.Tensor attributes to avoid DataFrame construction issues.
        """
        import torch

        # Extract nodes and filter scalar attributes only
        node_list = []
        node_id_mapping = {}  # Map tuple node IDs to string IDs
        for node_id, data in graph.nodes(data=True):
            node_dict = {}
            # Only include scalar attributes (not tensors)
            for key, val in data.items():
                if not torch.is_tensor(val):
                    # Convert numpy types to Python types
                    if hasattr(val, 'item'):
                        node_dict[key] = val.item()
                    else:
                        node_dict[key] = val
            node_list.append(node_dict)
            # Convert tuple node ID to string for index (avoid HoloViews type checking issues)
            node_id_mapping[node_id] = f"({node_id[0]},{node_id[1]})"

        # Create DataFrame with string node IDs as index
        string_node_ids = [node_id_mapping[nid] for nid in graph.nodes()]
        df = pd.DataFrame(node_list, index=string_node_ids)

        # Add positions (ensure they are float, not tensors)
        if pos is not None:
            # Convert any tensor values to Python floats and use string node IDs
            pos_clean = {}
            for node_id, (x_val, y_val) in pos.items():
                # Convert to float if it's a tensor
                if hasattr(x_val, 'item'):
                    x_val = float(x_val.item())
                else:
                    x_val = float(x_val)
                if hasattr(y_val, 'item'):
                    y_val = float(y_val.item())
                else:
                    y_val = float(y_val)
                # Use string node ID as key
                string_id = node_id_mapping[node_id]
                pos_clean[string_id] = (x_val, y_val)

            df_pos = pd.DataFrame.from_dict(pos_clean, orient="index", columns=["x", "y"])
            df = pd.concat([df, df_pos], axis=1)

        # Add computed columns for visualization
        if "E_kin" in df.columns:
            df["kinetic_energy"] = df["E_kin"]

        if "alive" in df.columns:
            df["alive_numeric"] = df["alive"].astype(float)

        if "fitness" in df.columns:
            df["fitness_viz"] = df["fitness"].fillna(0.0)

        # Timestep as numeric for coloring
        df["timestep_viz"] = df["timestep"]

        # Node label (index is now strings like "(0,5)")
        df["label"] = [f"w{row['walker_id']},t{row['timestep']}" for _, row in df.iterrows()]

        return df

    def _create_edge_df(self, graph: nx.DiGraph) -> pd.DataFrame:
        """Create edge DataFrame with visualization attributes.

        Note: Filters out torch.Tensor attributes to avoid DataFrame construction issues.
        """
        import torch

        # Extract edges and filter scalar attributes only
        edge_list = []
        for u, v, data in graph.edges(data=True):
            # Convert node IDs (tuples) to strings to avoid HoloViews type checking issues
            edge_dict = {
                "from": f"({u[0]},{u[1]})",  # Convert (walker, timestep) to string
                "to": f"({v[0]},{v[1]})"
            }
            # Only include scalar attributes (not tensors)
            for key, val in data.items():
                if not torch.is_tensor(val):
                    # Convert numpy types to Python types
                    if hasattr(val, 'item'):
                        edge_dict[key] = val.item()
                    else:
                        edge_dict[key] = val
            edge_list.append(edge_dict)

        # Create DataFrame with required columns even if empty
        if len(edge_list) == 0:
            # Empty DataFrame with required 'from' and 'to' columns
            df = pd.DataFrame(columns=["from", "to"])
        else:
            df = pd.DataFrame(edge_list)

        # Add edge type as categorical
        if "edge_type" in df.columns:
            df["edge_type_cat"] = pd.Categorical(df["edge_type"])
            df["edge_type_numeric"] = df["edge_type_cat"].cat.codes.astype(float)

        # Add cloning potential (IG edges only)
        if "V_clone" in df.columns:
            df["V_clone_viz"] = df["V_clone"].fillna(0.0)

        # Edge weight for visualization
        if "norm_Delta_x" in df.columns:
            df["displacement"] = df["norm_Delta_x"]

        return df

    def panel(self) -> pn.Column:
        """Create Panel dashboard."""
        # Config panel (left side)
        config_panel = pn.Column(
            pn.pane.Markdown("## Simulation Configuration"),
            self.config.panel(),
            sizing_mode="stretch_width",
            min_width=400,
            max_width=450,
        )

        # Display controls
        display_controls = pn.Column(
            pn.pane.Markdown("## Graph Visualization"),
            pn.Param(
                self.param,
                parameters=[
                    "max_timestep",
                    "show_cst_edges",
                    "show_ig_edges",
                    "layout_prog",
                    "top_to_bottom",
                ],
                show_name=False,
            ),
            self.status_pane,
            sizing_mode="stretch_width",
        )

        # Graph visualization (right side)
        graph_panel = pn.Column(
            display_controls,
            self.graph_pane,
            sizing_mode="stretch_both",
        )

        return pn.Row(
            config_panel,
            graph_panel,
            sizing_mode="stretch_both",
        )


def create_fractal_set_explorer() -> tuple[FractalSetExplorer, pn.Row]:
    """Factory function to create Fractal Set explorer dashboard.

    Returns:
        Tuple of (explorer, panel) for the interactive dashboard
    """
    explorer = FractalSetExplorer()
    return explorer, explorer.panel()


def main():
    """Launch the Fractal Set explorer dashboard."""
    print("=" * 80)
    print("Fractal Set Interactive Visualization")
    print("=" * 80)
    print("\nInitializing dashboard...")

    explorer, dashboard = create_fractal_set_explorer()

    print("Dashboard ready!")
    print("\nInstructions:")
    print("1. Configure simulation parameters in the left panel")
    print("2. Click 'Run Simulation & Build Fractal Set'")
    print("3. Use the slider to watch the graph grow over time")
    print("4. Toggle CST/IG edges to see different graph structures")
    print("5. Use InteractiveGraph controls to color nodes/edges")
    print("\nNode coloring options:")
    print("  - timestep_viz: Color by time")
    print("  - fitness_viz: Color by fitness value")
    print("  - kinetic_energy: Color by kinetic energy")
    print("  - alive_numeric: Color by alive status")
    print("\nEdge coloring options:")
    print("  - edge_type_numeric: CST (0) vs IG (1)")
    print("  - V_clone_viz: Antisymmetric cloning potential (IG edges)")
    print("  - displacement: Position change magnitude (CST edges)")
    print("=" * 80)

    # Serve the dashboard
    dashboard.show(port=5006, threaded=True)


if __name__ == "__main__":
    main()
