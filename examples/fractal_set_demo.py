"""
Demonstration of FractalSet integration with EuclideanGas.

This script shows how to:
1. Create a FractalSet instance
2. Run EuclideanGas with FractalSet recording
3. Query the recorded data
4. Save and load FractalSet data
"""

from pathlib import Path

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)
from fragile.fractal_set import FractalSet


def main():
    print("=" * 80)
    print("FractalSet Integration Demo")
    print("=" * 80)

    # 1. Create EuclideanGas instance
    print("\n1. Creating EuclideanGas instance...")
    params = EuclideanGasParams(
        N=20,
        d=2,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01),
        cloning=CloningParams(
            sigma_x=0.5,
            lambda_alg=0.1,
            alpha_restitution=0.5,
            companion_selection_method='softmax',
        ),
        device='cpu',
        dtype='float32',
    )
    gas = EuclideanGas(params)
    print(f"   Created EuclideanGas with N={params.N}, d={params.d}")

    # 2. Create FractalSet and run with recording
    print("\n2. Running EuclideanGas with FractalSet recording...")
    fs = FractalSet(N=params.N, d=params.d)

    n_steps = 50
    result = gas.run(n_steps=n_steps, fractal_set=fs, record_fitness=True)

    print(f"   Completed {result['final_step']} steps")
    print(f"   Initial variance: {result['var_x'][0]:.4f}")
    print(f"   Final variance: {result['var_x'][-1]:.4f}")

    # 3. Query FractalSet data
    print("\n3. Querying FractalSet data...")

    # Summary statistics
    stats = fs.summary_statistics()
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   Cloning events: {stats['n_cloning_events']}")
    print(f"   Variance reduction: {stats['variance_reduction']:.4f}")
    print(f"   Mean high-error fraction: {stats['mean_high_error_fraction']:.4f}")

    # Walker trajectory
    walker_id = 0
    traj = fs.get_walker_trajectory(walker_id)
    print(f"\n   Walker {walker_id} trajectory:")
    print(f"     - Timesteps: {len(traj['timesteps'])}")
    print(f"     - Initial position: {traj['positions'][0]}")
    print(f"     - Final position: {traj['positions'][-1]}")
    print(f"     - High-error count: {sum(traj['high_error'])}")

    # Timestep snapshot
    snapshot = fs.get_timestep_snapshot(n_steps // 2)
    print(f"\n   Snapshot at t={n_steps // 2}:")
    print(f"     - Centroid: {snapshot['centroid']}")
    print(f"     - Variance: {snapshot['variance']:.4f}")
    print(f"     - Alive walkers: {snapshot['alive_mask'].sum()}")

    # Lineage
    lineage = fs.get_lineage(walker_id, n_steps)
    print(f"\n   Lineage of walker {walker_id} at t={n_steps}:")
    print(f"     - Ancestors: {len(lineage)}")
    if lineage:
        print(f"     - Oldest ancestor: {lineage[-1]}")

    # Cloning events
    events = fs.get_cloning_events()
    print(f"\n   Cloning events:")
    print(f"     - Total: {len(events)}")
    if events:
        print(f"     - First event: parent={events[0]['parent_id']}, "
              f"child={events[0]['child_id']}, t={events[0]['timestep']}")

    # 4. Save and load FractalSet
    print("\n4. Saving and loading FractalSet...")
    output_dir = Path("data/fractal_sets")
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / "demo_fractal_set.graphml"
    fs.save(str(filepath))
    print(f"   Saved to {filepath}")

    # Load back
    fs_loaded = FractalSet.load(str(filepath))
    print(f"   Loaded from {filepath}")
    print(f"   Loaded graph has {fs_loaded.graph.number_of_nodes()} nodes")

    # Verify data matches
    stats_loaded = fs_loaded.summary_statistics()
    assert stats_loaded['total_nodes'] == stats['total_nodes']
    assert abs(stats_loaded['final_variance'] - stats['final_variance']) < 1e-6
    print("   Verification: Data matches original!")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
