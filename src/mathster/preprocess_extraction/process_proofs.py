from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

from mathster.preprocess_extraction.data_models import (
    CaseItem,
    Conclusion,
    Gap,
    KeyEquation,
    MathTool,
    ProofAssumption,
    ProofStep,
    Remark,
    Span,
    UnifiedProof,
)
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



def _build_unified_proofs(
    directive_payload: dict[str, Any],
    extracted_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not extracted_items:
        return []

    template = normalize_directive_template(directive_payload)
    directive_map = directive_lookup(directive_payload)
    unified: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    directive_labels = set(directive_map)
    extracted_labels = {
        item.get("label")
        for item in extracted_items
        if isinstance(item, dict) and isinstance(item.get("label"), str)
    }

    for extracted in extracted_items:
        label = extracted.get("label")
        if not isinstance(label, str) or not label:
            logger.warning("Skipping extracted proof without a label: %s", extracted)
            continue
        directive_item = directive_map.get(label)
        if directive_item is None:
            missing_labels.append(label)
            continue
        directive_argument = wrap_directive_item(template, directive_item)
        try:
            unified_obj = UnifiedProof.from_instances(directive_argument, extracted)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to merge proof %s: %s", label, exc)
            continue
        unified.append(unified_obj.model_dump(mode="json"))

    unmatched_directives = directive_labels - extracted_labels
    if missing_labels:
        logger.warning(
            "Skipped %s extracted proof(s) without directive matches: %s",
            len(missing_labels),
            ", ".join(sorted(set(missing_labels))),
        )
    if unmatched_directives:
        logger.info(
            "Directive proofs without extraction counterparts: %s",
            ", ".join(sorted(unmatched_directives)),
        )

    return unified


def preprocess_document_proofs(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directive_candidates = [
        registry_dir / "directives" / "proof.json",
        registry_dir / "directives" / "proof_raw.json",
    ]
    directives_path = select_existing_file(directive_candidates)

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_candidates = [
        extract_dir / "proof.json",
        extract_dir / "proof_extracted.json",
    ]
    extracted_path = select_existing_file(extracted_candidates)

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_proofs(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "proof.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified proof(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/proof*.json and extract/proof*.json into "
            "registry/preprocess/proof.json for a single document."
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
        help="Optional custom destination for the generated peroof.json file.",
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
        output_path = preprocess_document_proofs(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
