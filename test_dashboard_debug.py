#!/usr/bin/env python
"""Debug script to test dashboard loading."""
import sys
sys.path.insert(0, 'src')

import traceback

try:
    print("=" * 60)
    print("DASHBOARD DEBUG TEST")
    print("=" * 60)

    print("\n1. Loading extensions...")
    import holoviews as hv
    import panel as pn
    hv.extension('bokeh')
    pn.extension()
    print("   ✅ Extensions loaded")

    print("\n2. Testing benchmark visualization...")
    from fragile.core.benchmarks import MixtureOfGaussians
    benchmark = MixtureOfGaussians(dims=2, n_gaussians=3, seed=42)
    viz = benchmark.show(show_optimum=True, show_density=True, show_contours=True, n_cells=50)
    print(f"   ✅ Benchmark visualization: {type(viz).__name__}")

    print("\n3. Creating gas config...")
    from fragile.experiments.gas_config_panel import GasConfigPanel
    gas_config = GasConfigPanel(dims=2)
    print("   ✅ GasConfigPanel created")

    print("\n4. Testing background generation...")
    background = gas_config.potential.show(
        show_optimum=gas_config.show_optimum,
        show_density=gas_config.show_density,
        show_contours=gas_config.show_contours,
        n_cells=gas_config.viz_n_cells,
    )
    print(f"   ✅ Background: {type(background).__name__}")

    print("\n5. Creating dashboard...")
    from fragile.experiments.gas_visualization_dashboard import create_app
    app = create_app(dims=2)
    print(f"   ✅ Dashboard: {type(app).__name__}")

    print("\n6. Checking dashboard components...")
    print(f"   - Title: {app.title}")
    print(f"   - Sidebar items: {len(app.sidebar)}")
    print(f"   - Main items: {len(app.main)}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Dashboard should work!")
    print("=" * 60)
    print("\nTo run the dashboard:")
    print("  python -m fragile.experiments.gas_visualization_dashboard")

except Exception as e:
    print("\n" + "=" * 60)
    print("❌ ERROR OCCURRED")
    print("=" * 60)
    print(f"\nError: {e}")
    print(f"\nError type: {type(e).__name__}")
    print("\nFull traceback:")
    traceback.print_exc()
    print("\n" + "=" * 60)
    sys.exit(1)
