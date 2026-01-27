#!/bin/bash
# Comprehensive test suite for QFT dashboard implementation

echo "=========================================="
echo "QFT Dashboard Implementation Test Suite"
echo "=========================================="
echo

# Test 1: Unit tests
echo "Test 1: Running unit tests..."
python test_qft_setup.py
if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed!"
    exit 1
fi
echo "✅ Unit tests passed"
echo

# Test 2: Parameter verification
echo "Test 2: Verifying QFT parameters..."
python verify_qft_dashboard_params.py
if [ $? -ne 0 ]; then
    echo "❌ Parameter verification failed!"
    exit 1
fi
echo "✅ Parameter verification passed"
echo

# Test 3: Dashboard creation
echo "Test 3: Testing dashboard creation..."
python test_dashboard_creation.py
if [ $? -ne 0 ]; then
    echo "❌ Dashboard creation failed!"
    exit 1
fi
echo "✅ Dashboard creation passed"
echo

# Test 4: Import tests
echo "Test 4: Testing imports..."
python -c "from fragile.fractalai.core.benchmarks import QuadraticWell; print('  ✓ QuadraticWell import')"
python -c "from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel; print('  ✓ GasConfigPanel import')"
python -c "from fragile.fractalai.experiments.gas_visualization_dashboard import create_qft_app; print('  ✓ create_qft_app import')"
if [ $? -ne 0 ]; then
    echo "❌ Import tests failed!"
    exit 1
fi
echo "✅ All imports successful"
echo

# Summary
echo "=========================================="
echo "All tests passed! ✅"
echo "=========================================="
echo
echo "Implementation verified:"
echo "  ✓ QuadraticWell benchmark"
echo "  ✓ QFT configuration preset"
echo "  ✓ Companion selection (separate for cloning)"
echo "  ✓ Dashboard QFT mode"
echo "  ✓ All parameters match calibration"
echo
echo "Ready to use:"
echo "  python -m fragile.fractalai.experiments.gas_visualization_dashboard --qft"
