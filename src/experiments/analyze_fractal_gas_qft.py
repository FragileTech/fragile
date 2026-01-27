"""
Analyze Fractal Gas RunHistory for QFT observables and theory diagnostics.

This script loads a saved RunHistory (from fractal_gas_potential_well.py),
computes gauge/field observables, Lyapunov diagnostics, QSD variance metrics,
and optional FractalSet curvature summaries.

Usage:
    python src/experiments/analyze_fractal_gas_qft.py --history-path outputs/..._history.pt
    python src/experiments/analyze_fractal_gas_qft.py --build-fractal-set
    python src/experiments/analyze_fractal_gas_qft.py --compute-particles --build-fractal-set
"""

from fragile.fractalai.qft.analysis import main


if __name__ == "__main__":
    main()
