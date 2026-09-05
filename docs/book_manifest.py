"""Read the published book in navigation order.

The TOC is the authority for both HTML and downloadable mathematical extracts.
Research drafts and review files are never discovered by recursive directory scans.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def published_documents(book_dir: Path) -> list[Path]:
    """Resolve unique TOC documents, rejecting missing or external source paths."""
    book_dir = book_dir.resolve()
    toc = yaml.safe_load((book_dir / "_toc.yml").read_text(encoding="utf-8"))
    names: list[str] = []

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            for key in ("root", "file"):
                if key in node:
                    names.append(node[key])
            for key in ("parts", "chapters", "sections"):
                visit(node.get(key, []))

    visit(toc)
    result = []
    for name in dict.fromkeys(names):
        path = book_dir / name
        candidates = (
            [path] if path.suffix else [path.with_suffix(s) for s in (".md", ".ipynb", ".rst")]
        )
        found = next((p.resolve() for p in candidates if p.is_file()), None)
        if found is None:
            raise ValueError(f"Missing TOC document: {name}")
        if not found.is_relative_to(book_dir):
            raise ValueError(f"TOC document is outside the book: {name}")
        result.append(found)
    return result


def markdown_source(path: Path) -> str:
    """Read Markdown, including Markdown cells of a published notebook."""
    text = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return text
    cells = json.loads(text)["cells"]
    return "\n\n".join(
        "".join(cell["source"]) for cell in cells if cell.get("cell_type") == "markdown"
    )
