import os

import numpy
import pytest

import fragile


@pytest.fixture
def add_imports(doctest_namespace):
    """Add imports to doctest namespace. Only runs for doctests when explicitly requested."""
    doctest_namespace["np"] = numpy
    doctest_namespace["fragile"] = fragile

    # plangym is optional - only import when actually needed (lazy import)
    # Set PYGLET_HEADLESS to avoid X11 issues in WSL
    os.environ["PYGLET_HEADLESS"] = "1"
    try:
        import plangym

        doctest_namespace["plangym"] = plangym
    except ImportError:
        pass


# Set PYGLET_HEADLESS globally to prevent X11 issues in WSL
os.environ.setdefault("PYGLET_HEADLESS", "1")
