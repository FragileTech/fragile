"""Reject Algorithmic Gas files in the Fragile Pages deployment bundle."""

from __future__ import annotations

import argparse
from pathlib import Path


OWNED_DIRECTORIES = {"algorithmic-gas", "euclidean-gas", "2_fractal_gas", "3_fractal_gas", "gas-demos"}
OWNED_ASSETS = {"gas-embeds.js", "gas-embeds.css"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    site = parser.parse_args().site
    if not (site / "index.html").is_file():
        parser.error(f"Pages bundle has no index.html: {site}")
    leaked = [
        path.relative_to(site).as_posix()
        for path in site.rglob("*")
        if OWNED_DIRECTORIES.intersection(path.relative_to(site).parts)
        or path.name in OWNED_ASSETS
        or path.name.startswith("2_fractal_gas-")
    ]
    if leaked:
        raise SystemExit("Algorithmic Gas content found in Fragile Pages:\n" + "\n".join(sorted(leaked)))
    print("Fragile Pages contains no Algorithmic Gas book, simulator, or static assets.")


if __name__ == "__main__":
    main()
