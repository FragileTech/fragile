"""Fix encoding issues for imgmath extension."""

import os


def setup(app):
    """Set up encoding fix for LaTeX processing."""
    # Fix encoding issue for imgmath
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
