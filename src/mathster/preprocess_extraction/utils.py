from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence


logger = logging.getLogger(__name__)

# Default document roots searched when resolving chapter directories from short names
DEFAULT_DOC_ROOTS: tuple[Path, ...] = (Path("docs/source"), Path("old_docs/source"))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_directive_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)

    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload

    if isinstance(payload, list):
        return {"items": [item for item in payload if isinstance(item, dict)]}

    raise ValueError(f"{path} does not contain a valid directive payload")


def load_extracted_items(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]

    raise ValueError(f"{path} does not contain a valid extraction payload")


def normalize_directive_template(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "items"}


def directive_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = payload.get("items", [])
    lookup: dict[str, dict[str, Any]] = {}

    for entry in items or []:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        if isinstance(label, str) and label:
            lookup[label] = entry

    return lookup


def wrap_directive_item(template: dict[str, Any] | None, item: dict[str, Any]) -> dict[str, Any]:
    if template is None:
        return item
    wrapped = dict(template)
    wrapped["items"] = [item]
    return wrapped


def resolve_document_directory(
    document: str | Path,
    *,
    roots: Sequence[Path] | None = None,
) -> Path:
    candidate = Path(document)

    def _from_markdown(path: Path) -> Path | None:
        if path.suffix.lower() != ".md":
            return None
        target = path.parent / path.stem
        return target if target.exists() else None

    if candidate.exists():
        if candidate.is_dir():
            return candidate
        if candidate.is_file():
            doc_dir = _from_markdown(candidate)
            if doc_dir is not None:
                return doc_dir
            raise FileNotFoundError(f"Provided file {candidate} is not a chapter markdown source.")

    search_roots = roots or DEFAULT_DOC_ROOTS
    relative = Path(str(document))

    if relative.suffix.lower() == ".md":
        for root in search_roots:
            markdown_candidate = root / relative
            if markdown_candidate.exists():
                doc_dir = _from_markdown(markdown_candidate)
                if doc_dir is not None:
                    return doc_dir

    for root in search_roots:
        direct_candidate = root / relative
        if direct_candidate.is_dir():
            return direct_candidate

    matches: list[Path] = []
    needle = relative.name or str(relative)
    for root in search_roots:
        for match in root.rglob(needle):
            if not match.is_dir():
                continue
            if not (match / "registry").exists():
                continue
            matches.append(match)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        pretty = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple document directories named '{needle}' were found: {pretty}. "
            "Provide a more specific path.",
        )

    raise FileNotFoundError(f"Could not resolve document directory for '{document}'.")


def resolve_extract_directory(registry_dir: Path) -> Path:
    extract_dir = registry_dir / "extract"
    if extract_dir.exists():
        return extract_dir

    alt_dir = registry_dir / "extraction"
    if alt_dir.exists():
        return alt_dir

    raise FileNotFoundError(f"No extract directory found under {registry_dir}")


def select_existing_file(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the expected files were found: " + ", ".join(str(path) for path in candidates),
    )
