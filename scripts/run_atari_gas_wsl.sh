#!/bin/bash
# WSL launcher for Atari Fractal Gas CLI (no dashboard)
# Handles xvfb and environment configuration automatically

set -e

echo "🎮 Atari Fractal Gas - Command Line (WSL)"
echo

# Configure OpenGL software rendering
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_INDIRECT=0
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg

# Check for xvfb
if ! command -v xvfb-run &> /dev/null; then
    echo "❌ xvfb-run not found"
    echo "Install with: sudo apt-get install xvfb"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Starting simulation with xvfb..."
echo "Environment: Software OpenGL (llvmpipe)"
echo

# Run with xvfb
xvfb-run -a \
    --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
    $PYTHON_CMD scripts/run_atari_gas_cli.py "$@"
