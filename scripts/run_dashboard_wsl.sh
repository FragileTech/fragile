#!/bin/bash
# Launcher script for Atari Dashboard on WSL with proper OpenGL configuration

set -e

echo "🚀 Atari Fractal Gas Dashboard Launcher for WSL"
echo

# Check if running on WSL
if ! grep -q Microsoft /proc/version 2>/dev/null; then
    echo "⚠️  Not running on WSL, but continuing anyway..."
fi

# Configure OpenGL software rendering
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# Disable problematic X11 extensions
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_INDIRECT=0

# Configure pyglet for headless operation
export PYGLET_HEADLESS=1

# Additional environment variables for headless operation
export SDL_VIDEODRIVER=dummy          # Headless SDL (prevents SDL from using X11)
export MPLBACKEND=Agg                 # Prevent matplotlib from using X11 backend

# Parse arguments
PORT=5006
THREADED=false
OPEN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --threaded)
            THREADED=true
            shift
            ;;
        --open)
            OPEN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --port PORT        Port to run on (default: 5006)"
            echo "  --threaded         Use multi-threaded Tornado"
            echo "  --open             Open browser automatically"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for required packages
check_package() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 not found"
        return 1
    fi
    echo "✓ $1 found"
    return 0
}

echo "Checking dependencies..."
ALL_OK=true

if ! check_package xvfb-run; then
    echo "  Install with: sudo apt-get install xvfb"
    ALL_OK=false
fi

if ! check_package python3; then
    ALL_OK=false
fi

if [ "$ALL_OK" = false ]; then
    echo
    echo "Missing dependencies. Please install them first."
    exit 1
fi

echo

# Check for mesa (software OpenGL)
if ! ldconfig -p | grep -q libGL.so; then
    echo "⚠️  Mesa/OpenGL libraries not found"
    echo "  Install with: sudo apt-get install mesa-utils libgl1-mesa-glx libgl1-mesa-dri"
    echo "  Continuing anyway..."
    echo
fi

# Build command
CMD="python src/fragile/fractalai/videogames/dashboard.py --port $PORT"
if [ "$OPEN" = true ]; then
    CMD="$CMD --open"
fi
if [ "$THREADED" = false ]; then
    CMD="$CMD"  # Single-threaded is default
else
    CMD="$CMD --threaded"
fi

echo "Starting dashboard with configuration:"
echo "  Port: $PORT"
echo "  Threaded: $THREADED"
echo "  OpenGL: Software rendering (llvmpipe)"
echo "  Display: xvfb virtual framebuffer"
echo

# Run with xvfb
xvfb-run -a \
    --server-args="-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
    $CMD
