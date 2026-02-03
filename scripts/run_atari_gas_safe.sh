#!/bin/bash
# Safe launcher for Atari Fractal Gas that uses RAM observations
# This avoids ALL OpenGL/XCB issues by not using rendering at all

set -e

echo "🎮 Atari Fractal Gas - Safe Mode (RAM observations, no rendering)"
echo

# Configure environment (belt and suspenders approach)
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_INDIRECT=0
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
export SDL_AUDIODRIVER=dummy

# Check for xvfb
if ! command -v xvfb-run &> /dev/null; then
    echo "❌ xvfb-run not found"
    echo "Install with: sudo apt-get install xvfb"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Starting simulation with RAM observations (no OpenGL needed)..."
echo "This avoids all XCB/rendering issues."
echo

# Force RAM observations and run with xvfb
xvfb-run -a \
    --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
    $PYTHON_CMD scripts/run_atari_gas_cli.py --obs-type ram "$@"
