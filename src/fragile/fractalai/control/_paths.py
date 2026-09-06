"""Locate native sources without importing the build command module."""

from pathlib import Path


def repository_root() -> Path:
    """Find the source checkout containing the C++ engine."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "fractal-gas-web" / "CMakeLists.txt").is_file():
            return parent
    msg = "Build from a Fragile source checkout, or set FRAGILE_CONTROL_LIBRARY."
    raise RuntimeError(msg)
