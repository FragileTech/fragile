"""Validate published sources and mathematical download artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from book_manifest import markdown_source, published_documents


STALE = re.compile(r"hypostruct|hypopermits|3_fractal_gas|vol_3_references", re.IGNORECASE)
LABEL = re.compile(r"^\(([^\s]+)\)=\s*$|^\s*:label:\s*(\S+)", re.MULTILINE)


def reference_target(text: str) -> str:
    return text.rsplit("<", 1)[-1].rstrip(">").strip()


def validate(book_dir: Path, check_downloads: bool = True) -> list[str]:
    documents = published_documents(book_dir)
    sources = {p: markdown_source(p) for p in documents}
    labels = defaultdict(list)
    for path, source in sources.items():
        for first, second in LABEL.findall(source):
            labels[first or second].append(path)
    errors = []
    inventory_path = book_dir / "migrations/volume2_inventory.json"
    removed_labels = (
        set(json.loads(inventory_path.read_text())["removed_volume_labels"])
        if inventory_path.exists()
        else set()
    )
    for path, source in sources.items():
        for match in re.finditer(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", source):
            line = source.count("\n", 0, match.start()) + 1
            errors.append(f"{path}:{line}: control character in manuscript source")
        for match in STALE.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            errors.append(f"{path}:{line}: obsolete reference {match[0]}")
        for target in re.findall(r"\{(?:prf:ref|ref)\}`([^`]+)`", source):
            target = reference_target(target)
            if target in removed_labels and target not in labels:
                errors.append(f"{path}: reference to removed theorem {target}")
        for match in re.finditer(r"\{doc\}`([^`]+)`", source):
            target = reference_target(match[1])
            base = book_dir if target.startswith("/") else path.parent
            target_path = base / target.lstrip("/")
            candidates = (
                [target_path]
                if target_path.suffix
                else [target_path.with_suffix(s) for s in (".md", ".ipynb", ".rst")]
            )
            if not any(p.resolve() in sources for p in candidates):
                errors.append(f"{path}: unpublished or missing document {target}")
        if "2_fractal_gas" in path.parts:
            for target in re.findall(r"\]\((proofs/[^)]+)\)", source):
                errors.append(f"{path}: proof has not been integrated into a chapter: {target}")
            for target in re.findall(r"\{prf:ref\}`([^`]+)`", source):
                target = reference_target(target)
                if target not in labels:
                    errors.append(f"{path}: missing proof reference {target}")
    for label, owners in labels.items():
        if len(owners) > 1 and any("2_fractal_gas" in p.parts for p in owners):
            errors.append(f"Duplicate mathematical label {label}: {owners}")
    redirects = json.loads((book_dir / "redirects.json").read_text())
    built_names = {p.relative_to(book_dir.resolve()).with_suffix("").as_posix() for p in documents}
    for old, new in redirects.items():
        if old in built_names or new not in built_names:
            errors.append(f"Invalid redirect {old} -> {new}")
    if check_downloads:
        folder = book_dir / "_static/prompts"
        expected = {
            f"{v}-{p}.{e}"
            for v in ("1_agent", "2_fractal_gas")
            for p in ("with-proofs", "no-proofs")
            for e in ("md", "txt")
        }
        actual = {p.name for p in folder.iterdir() if p.suffix in {".md", ".txt"}}
        if expected != actual:
            errors.append(
                f"Unexpected download files: missing={expected - actual}, extra={actual - expected}"
            )
        for path in folder.iterdir():
            if path.suffix in {".md", ".txt"} and STALE.search(path.read_text()):
                errors.append(f"{path}: obsolete content in download")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-downloads", action="store_true")
    args = parser.parse_args()
    errors = validate(args.book.resolve(), not args.skip_downloads)
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(f"Book validation failed: {len(errors)} issues")
    print(
        "Published documents, proof references, redirects, and selected downloads are consistent."
    )


if __name__ == "__main__":
    main()
