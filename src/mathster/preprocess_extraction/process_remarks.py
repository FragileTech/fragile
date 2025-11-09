from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from mathster.preprocess_extraction.data_models import (
    KeyPoint,
    QuantitativeNote,
    RawMetadata,
    Recommendation,
    RegistryContext,
    UnifiedRemark,
)
from mathster.preprocess_extraction.utils import (
    directive_lookup,
    load_directive_payload,
    load_extracted_items,
    resolve_document_directory,
    resolve_extract_directory,
    select_existing_file,
)


logger = logging.getLogger(__name__)


def _match_raw_remark(
    extracted: dict[str, Any],
    raw_lookup: dict[str, dict[str, Any]],
    raw_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    label = extracted.get("label")
    if isinstance(label, str) and label:
        match = raw_lookup.get(label)
        if match:
            return match

    title = extracted.get("title")
    if isinstance(title, str):
        for item in raw_items:
            if item.get("title") == title:
                return item

    return raw_items[0] if raw_items else None


def _build_unified_remarks(
    directive_payload: dict[str, Any],
    extracted_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not extracted_items:
        return []

    raw_items = [
        item for item in directive_payload.get("items", []) or [] if isinstance(item, dict)
    ]
    raw_lookup = directive_lookup(directive_payload)
    matched_labels: set[str] = set()
    missing: list[str] = []
    unified: list[dict[str, Any]] = []

    for extracted in extracted_items:
        if not isinstance(extracted, dict):
            continue

        raw_item = _match_raw_remark(extracted, raw_lookup, raw_items)
        if raw_item is None:
            identifier = extracted.get("label") or extracted.get("title") or "<unknown>"
            missing.append(str(identifier))
            continue

        label = raw_item.get("label")
        if isinstance(label, str) and label:
            matched_labels.add(label)

        try:
            unified_obj = UnifiedRemark.from_instances(raw=raw_item, extracted=extracted)
        except Exception as exc:  # pragma: no cover - defensive logging
            identifier = extracted.get("label") or extracted.get("title") or "<unknown>"
            logger.warning("Failed to merge remark %s: %s", identifier, exc)
            continue
        unified.append(unified_obj.model_dump(mode="json"))

    unmatched = set(raw_lookup) - matched_labels
    if missing:
        logger.warning(
            "Skipped %s extracted remark(s) without directive matches: %s",
            len(missing),
            ", ".join(sorted(set(missing))),
        )
    if unmatched:
        logger.info(
            "Directive remarks without extraction counterparts: %s",
            ", ".join(sorted(unmatched)),
        )

    return unified


def preprocess_document_remarks(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directive_candidates = [
        registry_dir / "directives" / "remark.json",
        registry_dir / "directives" / "remark_raw.json",
    ]
    directives_path = select_existing_file(directive_candidates)

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_candidates = [
        extract_dir / "remark.json",
        extract_dir / "remark_extracted.json",
    ]
    extracted_path = select_existing_file(extracted_candidates)

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_remarks(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "remarks.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified remark(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/remark*.json and extract/remark*.json into "
            "registry/preprocess/remarks.json for a single document."
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
        help="Optional custom destination for the generated remarks.json file.",
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
        output_path = preprocess_document_remarks(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
