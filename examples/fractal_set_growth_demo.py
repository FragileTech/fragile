"""
Fractal Set Growth Visualization Demo

This script demonstrates how the FractalSet graph grows as the swarm evolves.
It creates an interactive visualization showing:
- Walker trajectories in position space
- FractalSet nodes as circles (one per walker per timestep)
- FractalSet edges as lines connecting nodes
- Color coding for kinetic vs cloning edges
- Animation showing growth over time

Uses HoloViz stack with Bokeh backend as per project guidelines.
"""

import holoviews as hv
from holoviews import opts
import numpy as np
import pandas as pd
import panel as pn

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)
from fragile.fractal_set import FractalSet


# Enable Bokeh backend
hv.extension("bokeh")


def run_gas_with_fractal_set(N=8, d=2, n_steps=20):
    """Run EuclideanGas and build FractalSet.

    Args:
        N: Number of walkers
        d: Spatial dimension (must be 2 for visualization)
        n_steps: Number of timesteps

    Returns:
        Tuple of (gas, result, fractal_set)
    """
    params = EuclideanGasParams(
        N=N,
        d=d,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.1),
        cloning=CloningParams(
            sigma_x=0.3,
            lambda_alg=0.1,
            alpha_restitution=0.5,
            companion_selection_method="softmax",
        ),
        device="cpu",
        dtype="float32",
    )
    gas = EuclideanGas(params)

    # Create FractalSet
    fs = FractalSet(N=N, d=d)

    # Run with fitness recording to enable cloning
    result = gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

    return gas, result, fs


def extract_fractal_set_data(fs):
    """Extract node and edge data from FractalSet for visualization.

    Args:
        fs: FractalSet instance

    Returns:
        Tuple of (nodes_df, edges_df)
    """
    # Extract node data
    nodes = []
    for node_id, node_data in fs.graph.nodes(data=True):
        walker_id, timestep = node_id
        x_pos = node_data["x"]

        nodes.append({
            "walker_id": walker_id,
            "timestep": timestep,
            "x": x_pos[0],
            "y": x_pos[1],
            "node_id": f"w{walker_id}_t{timestep}",
            "high_error": node_data.get("high_error", False),
            "alive": node_data.get("alive", True),
        })

    nodes_df = pd.DataFrame(nodes)

    # Extract edge data
    edges = []
    for u, v, edge_data in fs.graph.edges(data=True):
        u_walker, u_time = u
        v_walker, v_time = v

        # Get positions from nodes
        u_node = fs.graph.nodes[u]
        v_node = fs.graph.nodes[v]
        u_pos = u_node["x"]
        v_pos = v_node["x"]

        edge_type = edge_data.get("edge_type", "kinetic")

        edges.append({
            "source": u,
            "target": v,
            "source_walker": u_walker,
            "source_time": u_time,
            "target_walker": v_walker,
            "target_time": v_time,
            "x0": u_pos[0],
            "y0": u_pos[1],
            "x1": v_pos[0],
            "y1": v_pos[1],
            "edge_type": edge_type,
            "is_cloning": edge_type == "cloning",
        })

    edges_df = pd.DataFrame(edges)

    return nodes_df, edges_df


def create_spatial_trajectory_plot(result, N):
    """Create plot showing walker trajectories in spatial coordinates.

    Args:
        result: Result dictionary from gas.run()
        N: Number of walkers

    Returns:
        HoloViews overlay of trajectories
    """
    x_traj = result["x"].cpu().numpy()  # [T+1, N, d]
    n_steps = x_traj.shape[0] - 1

    # Create trajectory lines for each walker
    trajectories = []
    for i in range(N):
        traj_data = pd.DataFrame({
            "x": x_traj[:, i, 0],
            "y": x_traj[:, i, 1],
            "timestep": np.arange(n_steps + 1),
            "walker_id": i,
        })
        curve = hv.Curve(traj_data, kdims="x", vdims=["y", "timestep", "walker_id"])
        trajectories.append(curve)

    # Overlay all trajectories
    trajectory_overlay = hv.Overlay(trajectories).opts(
        opts.Curve(
            color="gray",
            alpha=0.3,
            line_width=1,
            tools=["hover"],
            width=600,
            height=600,
            title="Walker Trajectories in Position Space",
            xlabel="x",
            ylabel="y",
        )
    )

    # Add initial and final positions
    initial_points = pd.DataFrame({
        "x": x_traj[0, :, 0],
        "y": x_traj[0, :, 1],
        "walker_id": np.arange(N),
        "type": "initial",
    })

    final_points = pd.DataFrame({
        "x": x_traj[-1, :, 0],
        "y": x_traj[-1, :, 1],
        "walker_id": np.arange(N),
        "type": "final",
    })

    initial_scatter = hv.Scatter(
        initial_points, kdims=["x", "y"], vdims=["walker_id", "type"]
    ).opts(
        color="green",
        size=10,
        marker="o",
        alpha=0.6,
        legend_position="top_right",
        tools=["hover"],
    )

    final_scatter = hv.Scatter(final_points, kdims=["x", "y"], vdims=["walker_id", "type"]).opts(
        color="red",
        size=10,
        marker="s",
        alpha=0.6,
        legend_position="top_right",
        tools=["hover"],
    )

    return trajectory_overlay * initial_scatter * final_scatter


def create_fractal_set_graph_plot(nodes_df, edges_df, up_to_timestep=None):
    """Create plot showing FractalSet as a graph in spacetime.

    Args:
        nodes_df: DataFrame with node data
        edges_df: DataFrame with edge data
        up_to_timestep: If provided, only show data up to this timestep

    Returns:
        HoloViews overlay of graph elements
    """
    if up_to_timestep is not None:
        nodes_df = nodes_df[nodes_df["timestep"] <= up_to_timestep].copy()
        edges_df = edges_df[edges_df["target_time"] <= up_to_timestep].copy()

    if len(nodes_df) == 0:
        # Return empty plot
        return hv.Points([]).opts(width=800, height=600)

    # Create edges as Segments
    kinetic_edges = edges_df[edges_df["edge_type"] == "kinetic"]
    cloning_edges = edges_df[edges_df["edge_type"] == "cloning"]

    elements = []

    # Plot kinetic edges (blue)
    if len(kinetic_edges) > 0:
        kinetic_segments = hv.Segments(
            kinetic_edges,
            kdims=["x0", "y0", "x1", "y1"],
            vdims=["edge_type", "source_walker", "target_walker"],
        ).opts(
            color="blue",
            alpha=0.3,
            line_width=1,
            tools=["hover"],
        )
        elements.append(kinetic_segments)

    # Plot cloning edges (red)
    if len(cloning_edges) > 0:
        cloning_segments = hv.Segments(
            cloning_edges,
            kdims=["x0", "y0", "x1", "y1"],
            vdims=["edge_type", "source_walker", "target_walker"],
        ).opts(
            color="red",
            alpha=0.6,
            line_width=2,
            tools=["hover"],
        )
        elements.append(cloning_segments)

    # Plot nodes (colored by walker_id)
    nodes_scatter = hv.Points(
        nodes_df,
        kdims=["x", "y"],
        vdims=["walker_id", "timestep", "node_id", "high_error", "alive"],
    ).opts(
        color="walker_id",
        cmap="Category10",
        size=8,
        alpha=0.7,
        tools=["hover"],
        colorbar=True,
        width=800,
        height=600,
        title=f'FractalSet Graph (up to t={up_to_timestep if up_to_timestep is not None else "all"})',
        xlabel="x position",
        ylabel="y position",
    )
    elements.append(nodes_scatter)

    return hv.Overlay(elements)


def create_spacetime_plot(nodes_df, edges_df, up_to_timestep=None):
    """Create plot showing FractalSet in (x, t) spacetime coordinates.

    Args:
        nodes_df: DataFrame with node data
        edges_df: DataFrame with edge data
        up_to_timestep: If provided, only show data up to this timestep

    Returns:
        HoloViews overlay of spacetime graph
    """
    if up_to_timestep is not None:
        nodes_df = nodes_df[nodes_df["timestep"] <= up_to_timestep].copy()
        edges_df = edges_df[edges_df["target_time"] <= up_to_timestep].copy()

    if len(nodes_df) == 0:
        return hv.Points([]).opts(width=800, height=600)

    # Prepare spacetime coordinates (x, t)
    nodes_spacetime = nodes_df.copy()
    nodes_spacetime["t"] = nodes_spacetime["timestep"]

    edges_spacetime = edges_df.copy()
    edges_spacetime["t0"] = edges_spacetime["source_time"]
    edges_spacetime["t1"] = edges_spacetime["target_time"]

    elements = []

    # Plot edges
    kinetic_edges = edges_spacetime[edges_spacetime["edge_type"] == "kinetic"]
    cloning_edges = edges_spacetime[edges_spacetime["edge_type"] == "cloning"]

    if len(kinetic_edges) > 0:
        kinetic_segments = hv.Segments(
            kinetic_edges,
            kdims=["x0", "t0", "x1", "t1"],
            vdims=["edge_type", "source_walker", "target_walker"],
        ).opts(
            color="blue",
            alpha=0.3,
            line_width=1,
            tools=["hover"],
        )
        elements.append(kinetic_segments)

    if len(cloning_edges) > 0:
        cloning_segments = hv.Segments(
            cloning_edges,
            kdims=["x0", "t0", "x1", "t1"],
            vdims=["edge_type", "source_walker", "target_walker"],
        ).opts(
            color="red",
            alpha=0.6,
            line_width=2,
            tools=["hover"],
        )
        elements.append(cloning_segments)

    # Plot nodes
    nodes_scatter = hv.Points(
        nodes_spacetime, kdims=["x", "t"], vdims=["walker_id", "timestep", "node_id", "high_error"]
    ).opts(
        color="walker_id",
        cmap="Category10",
        size=6,
        alpha=0.7,
        tools=["hover"],
        colorbar=True,
        width=800,
        height=600,
        title="FractalSet Spacetime Diagram (x, t)",
        xlabel="x position",
        ylabel="Time",
    )
    elements.append(nodes_scatter)

    return hv.Overlay(elements)


def create_animated_growth(nodes_df, edges_df, n_steps):
    """Create animated visualization of FractalSet growth.

    Args:
        nodes_df: DataFrame with node data
        edges_df: DataFrame with edge data
        n_steps: Total number of timesteps

    Returns:
        HoloViews HoloMap with animation
    """
    frames = {}

    for t in range(n_steps + 1):
        frame = create_fractal_set_graph_plot(nodes_df, edges_df, up_to_timestep=t)
        frames[t] = frame

    holomap = hv.HoloMap(frames, kdims="timestep")

    return holomap.opts(
        opts.Overlay(width=800, height=600),
    )


def create_statistics_plot(fs, n_steps):
    """Create plot showing FractalSet statistics over time.

    Args:
        fs: FractalSet instance
        n_steps: Number of timesteps

    Returns:
        HoloViews layout of statistics plots
    """
    # Extract statistics
    stats = []
    for t in range(n_steps + 1):
        # Count nodes and edges up to this timestep
        nodes_at_t = [(i, ts) for i, ts in fs.graph.nodes() if ts <= t]
        edges_at_t = [(u, v) for (u, v) in fs.graph.edges() if u[1] <= t and v[1] <= t]

        # Count edge types
        n_kinetic = sum(
            1 for (u, v) in edges_at_t if fs.graph.edges[u, v].get("edge_type") == "kinetic"
        )
        n_cloning = sum(
            1 for (u, v) in edges_at_t if fs.graph.edges[u, v].get("edge_type") == "cloning"
        )

        stats.append({
            "timestep": t,
            "n_nodes": len(nodes_at_t),
            "n_edges": len(edges_at_t),
            "n_kinetic": n_kinetic,
            "n_cloning": n_cloning,
            "variance": fs.graph.graph["var_x_trajectory"][t]
            if t < len(fs.graph.graph["var_x_trajectory"])
            else 0,
        })

    stats_df = pd.DataFrame(stats)

    # Create plots
    nodes_curve = hv.Curve(stats_df, kdims="timestep", vdims="n_nodes", label="Nodes").opts(
        color="blue",
        line_width=2,
        width=400,
        height=300,
        title="Graph Growth",
        ylabel="Count",
        tools=["hover"],
    )

    edges_curve = hv.Curve(stats_df, kdims="timestep", vdims="n_edges", label="Edges").opts(
        color="green",
        line_width=2,
    )

    growth_plot = (nodes_curve * edges_curve).opts(legend_position="top_left")

    # Edge type breakdown
    kinetic_curve = hv.Curve(stats_df, kdims="timestep", vdims="n_kinetic", label="Kinetic").opts(
        color="blue",
        line_width=2,
        width=400,
        height=300,
        title="Edge Types",
        ylabel="Count",
        tools=["hover"],
    )

    cloning_curve = hv.Curve(stats_df, kdims="timestep", vdims="n_cloning", label="Cloning").opts(
        color="red",
        line_width=2,
    )

    edge_type_plot = (kinetic_curve * cloning_curve).opts(legend_position="top_left")

    # Variance plot
    variance_curve = hv.Curve(stats_df, kdims="timestep", vdims="variance").opts(
        color="purple",
        line_width=2,
        width=400,
        height=300,
        title="Position Variance",
        ylabel="Variance",
        tools=["hover"],
    )

    return hv.Layout([growth_plot, edge_type_plot, variance_curve]).cols(3)


def main():
    """Main demo function."""
    print("=" * 80)
    print("FractalSet Growth Visualization Demo")
    print("=" * 80)

    # Parameters
    N = 8  # Number of walkers
    d = 2  # Spatial dimension
    n_steps = 20  # Number of timesteps

    print(f"\nRunning EuclideanGas with N={N}, d={d}, n_steps={n_steps}...")
    _gas, result, fs = run_gas_with_fractal_set(N=N, d=d, n_steps=n_steps)

    print(
        f"FractalSet built with {fs.graph.number_of_nodes()} nodes and {fs.graph.number_of_edges()} edges"
    )

    # Get summary statistics
    stats = fs.summary_statistics()
    print("\nSummary Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Cloning events: {stats['n_cloning_events']}")
    print(f"  Initial variance: {stats['initial_variance']:.4f}")
    print(f"  Final variance: {stats['final_variance']:.4f}")
    print(f"  Variance reduction: {stats['variance_reduction']:.4f}")

    # Extract data for visualization
    print("\nExtracting graph data for visualization...")
    nodes_df, edges_df = extract_fractal_set_data(fs)

    print(f"  Nodes DataFrame: {len(nodes_df)} rows")
    print(f"  Edges DataFrame: {len(edges_df)} rows")
    print(f"  Kinetic edges: {len(edges_df[edges_df['edge_type'] == 'kinetic'])}")
    print(f"  Cloning edges: {len(edges_df[edges_df['edge_type'] == 'cloning'])}")

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. Spatial trajectories
    trajectory_plot = create_spatial_trajectory_plot(result, N)

    # 2. FractalSet graph (final state)
    graph_plot = create_fractal_set_graph_plot(nodes_df, edges_df)

    # 3. Spacetime diagram
    spacetime_plot = create_spacetime_plot(nodes_df, edges_df)

    # 4. Statistics
    stats_plots = create_statistics_plot(fs, n_steps)

    # 5. Animated growth
    print("\nCreating animation of FractalSet growth...")
    animated_growth = create_animated_growth(nodes_df, edges_df, n_steps)

    # Create dashboard layout
    print("\nCreating interactive dashboard...")

    dashboard = pn.Column(
        pn.pane.Markdown(
            "# FractalSet Growth Visualization\n\n"
            "This demo shows how the FractalSet graph grows as the swarm evolves.\n\n"
            "- **Blue lines**: Kinetic edges (self-evolution)\n"
            "- **Red lines**: Cloning edges (parent → child)\n"
            "- **Circles**: FractalSet nodes (colored by walker ID)\n"
            "- **Green squares**: Initial positions\n"
            "- **Red squares**: Final positions"
        ),
        pn.pane.Markdown("## Statistics"),
        stats_plots,
        pn.pane.Markdown("## Spatial Trajectories"),
        trajectory_plot,
        pn.pane.Markdown("## FractalSet Graph (Position Space)"),
        graph_plot,
        pn.pane.Markdown("## Spacetime Diagram (x, t)"),
        spacetime_plot,
        pn.pane.Markdown("## Animated Growth (use slider to see graph build over time)"),
        animated_growth,
    )

    # Save to HTML
    import os

    os.makedirs("data", exist_ok=True)
    output_file = "data/fractal_set_growth_demo.html"
    print(f"\nSaving interactive visualization to {output_file}...")
    dashboard.save(output_file)
    print(f"✓ Saved to {output_file}")

    print("\nVisualization complete!")
    print(f"  Open {output_file} in your web browser to view the interactive dashboard")
    print("  - Use the slider to animate graph growth")
    print("  - Hover over elements to see details")
    print("  - Blue lines = kinetic edges, Red lines = cloning edges")


if __name__ == "__main__":
    main()
