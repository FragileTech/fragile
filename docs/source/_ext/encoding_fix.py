"""Fix encoding issues and tweak MyST defaults for the docs build."""

import os
from typing import Iterable


def _ensure_mermaid_fence(app, config):
    """Add mermaid to the MyST fenced-as-directive list."""
    existing: Iterable[str] = getattr(config, "myst_fence_as_directive", ()) or ()
    fence_directives = sorted({*existing, "mermaid"})
    config.myst_fence_as_directive = fence_directives


def setup(app):
    """Set up encoding fix for LaTeX processing and MyST overrides."""
    # Fix encoding issue for imgmath
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"

    app.connect("config-inited", _ensure_mermaid_fence)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
