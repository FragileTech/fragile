"""
Simple FractalSet Visualization Demo

A simpler version that just prints statistics and creates a single static
visualization of the final FractalSet graph.
"""

import matplotlib


matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)
from fragile.fractal_set import FractalSet


def main():
    """Run simple FractalSet demo."""
    print("=" * 80)
    print("Simple FractalSet Visualization Demo")
    print("=" * 80)

    # Create gas with small number of walkers for clarity
    N = 6
    d = 2
    n_steps = 15

    print(f"\nSetting up EuclideanGas (N={N}, d={d}, n_steps={n_steps})...")
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

    # Run simulation
    print("Running simulation...")
    result = gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

    # Print statistics
    print(f"\n{'=' * 80}")
    print("FractalSet Statistics")
    print("=" * 80)

    stats = fs.summary_statistics()
    print("\nGraph Structure:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Nodes per timestep: {N}")
    print(f"  Expected total nodes: {N * (n_steps + 1)}")

    print("\nEdge Types:")
    print(f"  Kinetic edges (self-evolution): {stats['total_edges'] - stats['n_cloning_events']}")
    print(f"  Cloning edges (parent→child): {stats['n_cloning_events']}")

    print("\nSwarm Evolution:")
    print(f"  Initial variance: {stats['initial_variance']:.4f}")
    print(f"  Final variance: {stats['final_variance']:.4f}")
    print(
        f"  Variance reduction: {stats['variance_reduction']:.4f} ({stats['variance_reduction'] * 100:.1f}%)"
    )

    # Print parent tracking examples
    print(f"\n{'=' * 80}")
    print("Parent Tracking Examples")
    print("=" * 80)

    for t in [5, 10, n_steps]:
        parent_ids = fs.get_parent_ids(t)
        n_cloned = sum(1 for i, p in enumerate(parent_ids) if i != p)

        print(f"\nTimestep {t}:")
        print(f"  Parent IDs: {parent_ids}")
        print(f"  Walkers that cloned: {n_cloned}/{N}")

        # Show specific cloning events
        for i, p in enumerate(parent_ids):
            if i != p:
                print(f"    Walker {i} cloned from parent {p}")

    # Print walker trajectory example
    print(f"\n{'=' * 80}")
    print("Walker Trajectory Example (Walker 0)")
    print("=" * 80)

    traj = fs.get_walker_trajectory(0)
    print(f"\nTrajectory length: {len(traj['timesteps'])} timesteps")
    print(f"Initial position: {traj['positions'][0]}")
    print(f"Final position: {traj['positions'][-1]}")
    print(f"Times in high-error set: {sum(traj['high_error'])}/{len(traj['high_error'])}")

    # Print lineage
    lineage = fs.get_lineage(0, n_steps)
    print(f"\nLineage of walker 0 at t={n_steps}:")
    print(f"  Ancestors: {lineage}")
    print(f"  Generation depth: {len(lineage)}")

    # Create simple matplotlib visualization
    print(f"\n{'=' * 80}")
    print("Creating Visualization")
    print("=" * 80)

    _fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Spatial trajectories
    ax1 = axes[0]
    x_traj = result["x"].cpu().numpy()

    for i in range(N):
        ax1.plot(x_traj[:, i, 0], x_traj[:, i, 1], "-", alpha=0.3, label=f"Walker {i}")
        ax1.plot(x_traj[0, i, 0], x_traj[0, i, 1], "go", markersize=10)  # Initial
        ax1.plot(x_traj[-1, i, 0], x_traj[-1, i, 1], "rs", markersize=10)  # Final

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Walker Trajectories in Position Space")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Network graph structure
    ax2 = axes[1]

    # Create a simplified graph for visualization (only show subset)
    # Use networkx spring layout
    pos = nx.spring_layout(fs.graph, k=0.5, iterations=50)

    # Draw nodes colored by walker ID
    node_colors = []
    for node in fs.graph.nodes():
        walker_id, _timestep = node
        node_colors.append(walker_id)

    nx.draw_networkx_nodes(
        fs.graph,
        pos,
        node_color=node_colors,
        node_size=50,
        cmap="tab10",
        alpha=0.7,
        ax=ax2,
    )

    # Draw edges (kinetic in blue, cloning in red)
    kinetic_edges = [
        (u, v) for u, v in fs.graph.edges() if fs.graph.edges[u, v].get("edge_type") == "kinetic"
    ]
    cloning_edges = [
        (u, v) for u, v in fs.graph.edges() if fs.graph.edges[u, v].get("edge_type") == "cloning"
    ]

    nx.draw_networkx_edges(
        fs.graph,
        pos,
        edgelist=kinetic_edges,
        edge_color="blue",
        alpha=0.2,
        width=0.5,
        ax=ax2,
    )

    nx.draw_networkx_edges(
        fs.graph,
        pos,
        edgelist=cloning_edges,
        edge_color="red",
        alpha=0.5,
        width=1.5,
        arrows=True,
        arrowsize=5,
        ax=ax2,
    )

    ax2.set_title(
        f'FractalSet Graph Structure\n{stats["total_nodes"]} nodes, {stats["total_edges"]} edges'
    )
    ax2.axis("off")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="g",
            markersize=10,
            label="Initial positions",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="r",
            markersize=10,
            label="Final positions",
        ),
        Line2D([0], [0], color="blue", linewidth=2, alpha=0.5, label="Kinetic edges"),
        Line2D([0], [0], color="red", linewidth=2, alpha=0.5, label="Cloning edges"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()

    # Save figure
    import os

    os.makedirs("data", exist_ok=True)
    output_file = "data/fractal_set_simple_demo.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to {output_file}")

    print(f"\n{'=' * 80}")
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
