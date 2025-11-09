from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from mathster.preprocess_extraction.data_models import UnifiedCorollary
from mathster.preprocess_extraction.utils import (
    directive_lookup,
    load_directive_payload,
    load_extracted_items,
    normalize_directive_template,
    resolve_document_directory,
    resolve_extract_directory,
    select_existing_file,
    wrap_directive_item,
)


logger = logging.getLogger(__name__)


def _match_directive_item(
    extracted: dict[str, Any],
    directive_map: dict[str, dict[str, Any]],
    directive_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    label = extracted.get("label")
    if isinstance(label, str) and label:
        match = directive_map.get(label)
        if match:
            return match

    title = extracted.get("title")
    if isinstance(title, str):
        for item in directive_items:
            if item.get("title") == title:
                return item

    return directive_items[0] if directive_items else None


def _build_unified_corollaries(
    directive_payload: dict[str, Any],
    extracted_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not extracted_items:
        return []

    template = normalize_directive_template(directive_payload)
    directive_items = [
        item for item in directive_payload.get("items", []) or [] if isinstance(item, dict)
    ]
    directive_map = directive_lookup(directive_payload)
    matched_labels: set[str] = set()
    missing: list[str] = []
    unified: list[dict[str, Any]] = []

    for extracted in extracted_items:
        if not isinstance(extracted, dict):
            continue

        directive_item = _match_directive_item(extracted, directive_map, directive_items)
        if directive_item is None:
            identifier = extracted.get("label") or extracted.get("title") or "<unknown>"
            missing.append(str(identifier))
            continue

        label = directive_item.get("label")
        if isinstance(label, str) and label:
            matched_labels.add(label)

        directive_argument = wrap_directive_item(template, directive_item)
        try:
            unified_obj = UnifiedCorollary.from_instances(directive_argument, extracted)
        except Exception as exc:  # pragma: no cover
            identifier = extracted.get("label") or extracted.get("title") or "<unknown>"
            logger.warning("Failed to merge corollary %s: %s", identifier, exc)
            continue
        unified.append(unified_obj.model_dump(mode="json"))

    unmatched_directives = set(directive_map) - matched_labels
    if missing:
        logger.warning(
            "Skipped %s extracted corollary/corollaries without directive matches: %s",
            len(missing),
            ", ".join(sorted(set(missing))),
        )
    if unmatched_directives:
        logger.info(
            "Directive corollaries without extraction counterparts: %s",
            ", ".join(sorted(unmatched_directives)),
        )

    return unified


def preprocess_document_corollaries(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directive_candidates = [
        registry_dir / "directives" / "corollary.json",
        registry_dir / "directives" / "corollary_raw.json",
    ]
    directives_path = select_existing_file(directive_candidates)

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_candidates = [
        extract_dir / "corollary.json",
        extract_dir / "corollary_extracted.json",
    ]
    extracted_path = select_existing_file(extracted_candidates)

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_corollaries(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "corollaries.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified corollary(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/corollary*.json and extract/corollary*.json into "
            "registry/preprocess/corollaries.json for a single document."
        ),
    )
    parser.add_argument(
        "document",
        help=(
            "Document identifier (directory, markdown file, or document name). "
            "Examples: 'docs/source/1_euclidean_gas/03_cloning', "
            "'docs/source/1_euclidean_gas/03_cloning.md', '03_cloning'."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional custom destination for the generated corollaries.json file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    try:
        output_path = preprocess_document_corollaries(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
