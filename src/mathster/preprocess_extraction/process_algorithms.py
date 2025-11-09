from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mathster.preprocess_extraction.utils import (
    directive_lookup,
    load_directive_payload,
    load_extracted_items,
    resolve_document_directory,
    resolve_extract_directory,
    select_existing_file,
)


logger = logging.getLogger(__name__)


# ----------------------------
# Small helper sub-models
# ----------------------------


class DocumentMetadata(BaseModel):
    document_id: str | None = None
    stage: str | None = None
    generated_at: str | None = None

    class Config:
        extra = "allow"


class AlgorithmParameter(BaseModel):
    """A flexible parameter representation. Accepts either a string name or a dict."""

    name: str
    type: str | None = None
    default: Any | None = None
    description: str | None = None

    @classmethod
    def from_any(cls, obj: str | dict[str, Any]) -> AlgorithmParameter:
        if isinstance(obj, str):
            return cls(name=obj)
        # try common shapes
        name = obj.get("name") or obj.get("id") or obj.get("param") or "param"
        return cls(
            name=name,
            type=obj.get("type"),
            default=obj.get("default"),
            description=obj.get("description"),
        )


class AlgorithmSignature(BaseModel):
    """Structured I/O signature of the algorithm."""

    input: list[str] = Field(default_factory=list)
    output: list[str] = Field(default_factory=list)
    parameters: list[AlgorithmParameter] = Field(default_factory=list)

    @field_validator("parameters", mode="before")
    @classmethod
    def _coerce_parameters(cls, v):
        if v is None:
            return []
        coerced: list[AlgorithmParameter] = []
        for item in v:
            coerced.append(AlgorithmParameter.from_any(item))
        return coerced


class AlgorithmStep(BaseModel):
    order: int
    text: str
    comment: str | None = None

    class Config:
        extra = "allow"


class GuardCondition(BaseModel):
    """Open schema: keep anything provided by the extractor."""

    condition: str | None = None
    description: str | None = None
    action: str | None = None
    severity: str | None = None

    class Config:
        extra = "allow"


class FailureMode(BaseModel):
    description: str
    impact: str | None = None

    class Config:
        extra = "allow"


class RawAlgorithm(BaseModel):
    """Verbatim raw directive slice from algorithm.json."""

    label: str | None = None
    title: str | None = None
    section: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    content_start: int | None = None
    content_end: int | None = None
    header_lines: list[int] | None = None
    raw_directive: str | None = None
    content: str | None = None  # human/pseudocode text block

    class Config:
        extra = "allow"


class ExtractedAlgorithm(BaseModel):
    """Structured fields from algorithm_extracted.json."""

    label: str | None = None
    title: str | None = None
    complexity: str | None = None
    nl_summary: str | None = None
    signature: AlgorithmSignature | None = None
    steps: list[AlgorithmStep] = Field(default_factory=list)
    guard_conditions: list[GuardCondition] = Field(default_factory=list)
    references: list[str | dict[str, Any]] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    class Config:
        extra = "allow"


# ----------------------------
# Unified model
# ----------------------------


class Algorithm(BaseModel):
    """
    Unified Algorithm model combining the raw directive and the structured extraction.
    """

    # canonical identity
    label: str
    title: str | None = None

    # high-level summary & metadata
    complexity: str | None = None
    nl_summary: str | None = None
    tags: list[str] = Field(default_factory=list)

    # signature & steps
    signature: AlgorithmSignature | None = None
    steps: list[AlgorithmStep] = Field(default_factory=list)

    # safety / quality
    guard_conditions: list[GuardCondition] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)

    # references (merged)
    references: list[str | dict[str, Any]] = Field(default_factory=list)

    # raw & extracted payloads preserved verbatim
    raw: RawAlgorithm | None = None
    extracted: ExtractedAlgorithm | None = None

    # document-level metadata (if available)
    doc_meta: DocumentMetadata | None = None

    class Config:
        extra = "allow"

    # -------------
    # Constructor
    # -------------

    @classmethod
    def from_instances(
        cls,
        raw_obj: dict[str, Any],
        extracted_obj: dict[str, Any] | list[dict[str, Any]],
        *,
        prefer_extracted_title: bool = True,
    ) -> Algorithm:
        """
        Build a unified Algorithm from:
          - raw_obj: either a raw *item* dict from algorithm.json
                     OR the whole algorithm.json document (with .items list).
          - extracted_obj: either the extracted *item* dict from algorithm_extracted.json
                           OR a list containing that dict.

        The method is defensive to minor format differences.
        """

        # --------
        # unwrap raw
        # --------
        raw_doc_level = raw_obj if "items" in raw_obj else None
        if raw_doc_level:
            items = raw_doc_level.get("items") or []
            # try to match by label from extracted if possible
            extracted_item_for_label = None
            if isinstance(extracted_obj, dict):
                extracted_item_for_label = extracted_obj.get("label")
            elif isinstance(extracted_obj, list) and extracted_obj:
                extracted_item_for_label = extracted_obj[0].get("label")

            raw_item = None
            if extracted_item_for_label:
                for it in items:
                    if it.get("label") == extracted_item_for_label:
                        raw_item = it
                        break
            if raw_item is None:
                raw_item = items[0] if items else {}
            doc_meta = DocumentMetadata(
                document_id=raw_doc_level.get("document_id"),
                stage=raw_doc_level.get("stage"),
                generated_at=raw_doc_level.get("generated_at"),
            )
        else:
            # already at item level
            raw_item = raw_obj or {}
            # try find document metadata in item or its registry context
            rc = (raw_item or {}).get("_registry_context") or {}
            doc_meta = DocumentMetadata(
                document_id=rc.get("document_id"),
                stage=rc.get("stage"),
                generated_at=raw_item.get("generated_at"),
            )

        # normalize raw slice
        raw_model = RawAlgorithm(
            label=raw_item.get("label"),
            title=raw_item.get("title"),
            section=raw_item.get("section"),
            start_line=raw_item.get("start_line"),
            end_line=raw_item.get("end_line"),
            content_start=raw_item.get("content_start"),
            content_end=raw_item.get("content_end"),
            header_lines=raw_item.get("header_lines"),
            raw_directive=raw_item.get("raw_directive"),
            content=raw_item.get("content"),
            **{
                k: v
                for k, v in raw_item.items()
                if k
                not in {
                    "label",
                    "title",
                    "section",
                    "start_line",
                    "end_line",
                    "content_start",
                    "content_end",
                    "header_lines",
                    "raw_directive",
                    "content",
                }
            },
        )

        # --------
        # unwrap extracted
        # --------
        if isinstance(extracted_obj, list):
            ext_item = extracted_obj[0] if extracted_obj else {}
        else:
            ext_item = extracted_obj or {}

        extracted_model = ExtractedAlgorithm(**ext_item)

        # --------
        # merged top-level fields
        # --------
        label = extracted_model.label or raw_model.label or "algorithm"

        title = (
            (extracted_model.title if prefer_extracted_title else None)
            or raw_model.title
            or extracted_model.title
        )

        # references: union of both sides, preserving order & uniqueness
        raw_refs = raw_item.get("references") or []
        ext_refs = extracted_model.references or []
        merged_refs: list[str | dict[str, Any]] = []
        seen = set()
        for source in (ext_refs, raw_refs):
            for r in source:
                key = r if isinstance(r, str) else str(r)
                if key not in seen:
                    merged_refs.append(r)
                    seen.add(key)

        return cls(
            label=label,
            title=title,
            complexity=extracted_model.complexity,
            nl_summary=extracted_model.nl_summary,
            tags=list(extracted_model.tags or []),
            signature=extracted_model.signature,
            steps=list(extracted_model.steps or []),
            guard_conditions=list(extracted_model.guard_conditions or []),
            failure_modes=list(extracted_model.failure_modes or []),
            references=merged_refs,
            raw=raw_model,
            extracted=extracted_model,
            doc_meta=doc_meta,
        )


def _build_unified_algorithms(
    directive_payload: dict[str, Any],
    extracted_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not extracted_items:
        return []

    raw_lookup = directive_lookup(directive_payload)
    matched: set[str] = set()
    missing: list[str] = []
    unified: list[dict[str, Any]] = []

    for extracted in extracted_items:
        if not isinstance(extracted, dict):
            continue
        label = extracted.get("label")
        if not isinstance(label, str) or not label:
            logger.warning("Skipping extracted algorithm without a label: %s", extracted)
            continue

        raw_item = raw_lookup.get(label)
        if raw_item is None:
            missing.append(label)
            continue

        try:
            unified_obj = Algorithm.from_instances(raw_item, extracted)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to merge algorithm %s: %s", label, exc)
            continue

        matched.add(label)
        unified.append(unified_obj.model_dump(mode="json"))

    unmatched = set(raw_lookup) - matched
    if missing:
        logger.warning(
            "Skipped %s extracted algorithm(s) without directive matches: %s",
            len(missing),
            ", ".join(sorted(set(missing))),
        )
    if unmatched:
        logger.info(
            "Directive algorithms without extraction counterparts: %s",
            ", ".join(sorted(unmatched)),
        )

    return unified


def preprocess_document_algorithms(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directives_path = registry_dir / "directives" / "algorithm.json"
    if not directives_path.exists():
        raise FileNotFoundError(f"Missing directives/algorithm.json in {registry_dir}")

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_candidates = [
        extract_dir / "algorithm.json",
        extract_dir / "algorithm_extracted.json",
    ]
    extracted_path = select_existing_file(extracted_candidates)

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_algorithms(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "algorithms.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified algorithm(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/algorithm.json and extract/algorithm*.json into "
            "registry/preprocess/algorithms.json for a single document."
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
        help="Optional custom destination for the generated algorithms.json file.",
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
        output_path = preprocess_document_algorithms(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
