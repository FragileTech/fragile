"""
Fractal Gas potential well run for Fractal Set and lattice QFT analysis.

This script runs a simple Euclidean Gas in a quadratic potential well and saves
full RunHistory data for later FractalSet construction and analysis.

Defaults:
- N=1000 walkers
- n_steps=1000
- record_every=1 (record every step)
- Quadratic well U(x) = 0.5 * alpha * ||x||^2 with alpha=0.1
- Bounds: [-bounds_extent, bounds_extent]^d with bounds_extent=10
- Balanced phase-space distance (lambda_alg=1.0)
- Calibrated QFT parameters (epsilon_d, epsilon_c, epsilon_F, nu, rho, delta_t)

Usage:
    python src/experiments/fractal_gas_potential_well.py
    python src/experiments/fractal_gas_potential_well.py --n-steps 500 --record-every 5

Notes:
- This produces large output files. Increase record_every to reduce size.
- To enable local-gauge (rho) or Hessian data for advanced analysis, edit the
  config section below.
"""

from fragile.fractalai.qft.simulation import main


if __name__ == "__main__":
    main()
