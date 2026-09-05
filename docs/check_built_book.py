"""Check published HTML, proof visibility, and migrated page destinations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from book_manifest import markdown_source, published_documents
from bs4 import BeautifulSoup
from check_book import STALE


def validate_html(book: Path, output: Path) -> list[str]:
    """Audit the actual publication, excluding compatibility redirect pages."""
    errors = []
    pages = {}
    for source in published_documents(book):
        name = source.relative_to(book).with_suffix("").as_posix()
        path = output / (name + ".html")
        if not path.is_file():
            errors.append(f"Missing published HTML: {name}")
            continue
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        pages[name] = soup
        article = soup.select_one("article") or soup
        if STALE.search(article.get_text(" ", strip=True)):
            errors.append(f"Obsolete terminology in rendered article: {name}")
        if "2_fractal_gas" in source.parts:
            ids = {node.get("id") for node in article.select("[id]")}
            for label in re.findall(r"^:label:\s*(\S+)", markdown_source(source), re.MULTILINE):
                if label not in ids and f"equation-{label}" not in ids:
                    errors.append(f"Source label has no rendered target: {name}#{label}")
            for formal in article.select(".proof"):
                ancestors = [formal, *formal.parents]
                if any(
                    set(node.get("class", [])) & {"feynman-prose", "feynman-added"}
                    for node in ancestors
                ):
                    errors.append(
                        f"Formal content hidden in Expert Mode: {name}#{formal.get('id', '')}"
                    )
    for name in ("searchindex.js",):
        path = output / name
        if not path.is_file():
            errors.append(f"Missing search artifact: {name}")
        elif STALE.search(path.read_text(encoding="utf-8")):
            errors.append(f"Obsolete content in search artifact: {name}")
    if (output / "source/2_hypostructure").exists():
        errors.append("Removed volume remains in HTML output; rebuild in a clean output directory")
    redirects = json.loads((book / "redirects.json").read_text())
    fragment_file = book / "redirect_fragments.json"
    fragments = json.loads(fragment_file.read_text()) if fragment_file.exists() else {}
    for old, new in redirects.items():
        page = output / (old + ".html")
        if not page.is_file():
            errors.append(f"Missing compatibility redirect: {old}")
            continue
        text = page.read_text(encoding="utf-8")
        if 'content="noindex"' not in text or "window.location.replace" not in text:
            errors.append(f"Invalid compatibility redirect: {old}")
        if new not in pages:
            errors.append(f"Unpublished redirect destination: {old} -> {new}")
            continue
        ids = {node.get("id") for node in pages[new].select("[id]")}
        for before, after in fragments.get(old, {}).items():
            if after and after not in ids:
                errors.append(f"Missing redirect anchor: {old}#{before} -> {new}#{after}")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    book = args.book.resolve()
    output = (args.output or book / "_build/html").resolve()
    errors = validate_html(book, output)
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(f"Built-book validation failed: {len(errors)} issues")
    print("Published HTML, Expert Mode proofs, search, and compatibility redirects pass.")


if __name__ == "__main__":
    main()
