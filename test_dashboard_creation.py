#!/usr/bin/env python
"""Test that the QFT dashboard can be created successfully."""

import sys

import holoviews as hv
import panel as pn


# Initialize extensions (required for dashboard creation)
hv.extension("bokeh")
pn.extension()

print("Testing dashboard creation...")
print("=" * 60)

# Test standard dashboard creation
print("\n1. Testing standard dashboard creation...")
from fragile.fractalai.experiments.gas_visualization_dashboard import create_app


try:
    app_standard = create_app(dims=2)
    print("   ✓ Standard dashboard created successfully")
    print(f"   ✓ Title: {app_standard.title}")
except Exception as e:
    print(f"   ✗ Error creating standard dashboard: {e}")
    sys.exit(1)

# Test QFT dashboard creation
print("\n2. Testing QFT dashboard creation...")
from fragile.fractalai.experiments.gas_visualization_dashboard import create_qft_app


try:
    app_qft = create_qft_app()
    print("   ✓ QFT dashboard created successfully")
    print(f"   ✓ Title: {app_qft.title}")
except Exception as e:
    print(f"   ✗ Error creating QFT dashboard: {e}")
    sys.exit(1)

# Verify the title is different
print("\n3. Verifying QFT mode indicator...")
if "QFT" in app_qft.title or "Calibration" in app_qft.title:
    print(f"   ✓ QFT title correct: '{app_qft.title}'")
else:
    print(f"   ✗ QFT title doesn't indicate QFT mode: '{app_qft.title}'")
    sys.exit(1)

print("\n" + "=" * 60)
print("Dashboard creation tests passed! ✓")
print("=" * 60)
print("\nTo launch the dashboards:")
print("  Standard: python -m fragile.fractalai.experiments.gas_visualization_dashboard")
print("  QFT:      python -m fragile.fractalai.experiments.gas_visualization_dashboard --qft")
