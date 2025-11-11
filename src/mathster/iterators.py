from pathlib import Path
import re
from typing import Sequence


CHAPTER_FILENAME_PATTERN = re.compile(r"^\d{2}_.+\.md$")
DEFAULT_CHAPTER_ROOTS: tuple[Path, ...] = (Path("docs/source"),)


def discover_chapter_markdown_files(roots: Sequence[Path] | Path | None = None) -> list[Path]:
    """Return all chapter markdown files (pattern ``NN_chapter_name.md``)."""
    if isinstance(roots, Path):
        roots = [roots]
    roots = roots or DEFAULT_CHAPTER_ROOTS
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            if CHAPTER_FILENAME_PATTERN.match(path.name):
                matches.append(path)
    return sorted(set(matches))


def discover_document_folders(roots: Sequence[Path] | None = None) -> list[Path]:
    """Return document directories corresponding to chapter markdown files."""
    documents: list[Path] = []
    for markdown_file in discover_chapter_markdown_files(roots):
        document_dir = markdown_file.parent / markdown_file.stem
        documents.append(document_dir)
    return sorted(set(documents), key=str)


def discover_registry_folders(
    roots: Sequence[Path] | None = None,
    subfolder: str | None = None,
    document: str | None = None,
) -> list[Path]:
    """Return registry directories under document folders."""
    registries: list[Path] = []
    for document_folder in discover_document_folders(roots):
        if document and document_folder.name not in document:
            continue
        registry_dir = document_folder / "registry"
        if registry_dir.exists():
            if subfolder:
                registry_dir /= subfolder
                if not registry_dir.exists():
                    continue
            registries.append(registry_dir)
    return sorted(set(registries), key=str)
