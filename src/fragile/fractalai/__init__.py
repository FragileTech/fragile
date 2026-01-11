"""Fragile: FractalAI implementation."""


def compute(*args, **kwargs):
    """Lazy import to avoid heavy optional dependencies at module import time."""
    from fragile.old_core import compute as _compute

    return _compute(*args, **kwargs)


__all__ = ["compute"]
