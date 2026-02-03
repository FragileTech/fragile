#!/bin/bash
# Docker-based launcher for Atari Fractal Gas CLI
# Most reliable option - complete isolation from host XCB issues

set -e

echo "🐳 Atari Fractal Gas - Docker Mode"
echo

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found"
    echo
    echo "Install Docker:"
    echo "  - Windows/Mac: https://www.docker.com/products/docker-desktop"
    echo "  - Linux: sudo apt-get install docker.io"
    echo "  - WSL2: Use Docker Desktop for Windows with WSL2 backend"
    echo
    exit 1
fi

# Check if image exists, build if not
IMAGE_NAME="atari-gas-cli"
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "Building Docker image (this may take a few minutes)..."
    docker build -f docker/Dockerfile.atari-cli -t $IMAGE_NAME .
    echo "✓ Image built successfully"
    echo
fi

# Create results directory on host
mkdir -p results

echo "Running simulation in Docker container..."
echo

# Run container with volume mount for results
# Pass all arguments to the container
docker run --rm \
    -v "$(pwd)/results:/app/results" \
    $IMAGE_NAME "$@"

echo
echo "✓ Simulation complete!"
echo "Results saved to: $(pwd)/results/"
