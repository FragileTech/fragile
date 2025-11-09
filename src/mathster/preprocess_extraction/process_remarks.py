from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mathster.preprocess_extraction.utils import (
    directive_lookup,
    load_directive_payload,
    load_extracted_items,
    resolve_document_directory,
    resolve_extract_directory,
    select_existing_file,
)


try:
    # Pydantic v2
    from pydantic import ConfigDict

    _PYDANTIC_V2 = True
except Exception:
    _PYDANTIC_V2 = False


# ---------- Nested types (from the extracted representation) ----------


class KeyPoint(BaseModel):
    text: str
    latex: str | None = None
    importance: str | None = None  # e.g. "low" | "medium" | "high"


class QuantitativeNote(BaseModel):
    text: str
    latex: str | None = None


class Recommendation(BaseModel):
    text: str
    severity: str | None = None  # e.g. "low" | "medium" | "high"


# ---------- Nested types (from the raw directive) ----------


class RawMetadata(BaseModel):
    # remark.json metadata includes {"label": "...", "class": "tip" | "note" | ...}
    label: str | None = None
    class_: str | None = Field(default=None, alias="class")


class RegistryContext(BaseModel):
    stage: str | None = None
    document_id: str | None = None
    chapter_index: int | None = None
    chapter_file: str | None = None
    section_id: str | None = None


# ---------- Unified Remark ----------


class UnifiedRemark(BaseModel):
    # identifiers / headline
    label: str
    title: str | None = None
    remark_type: str | None = (
        None  # prefers extracted.remark_type, else raw.metadata.class, else raw.directive_type
    )

    # extracted-side semantics
    nl_summary: str | None = None
    key_points: list[KeyPoint] = Field(default_factory=list)
    quantitative_notes: list[QuantitativeNote] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    references: list[Any] = Field(default_factory=list)  # union of raw+extracted references

    # raw-side payload / provenance
    content: str | None = None
    section: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    header_lines: list[int] = Field(default_factory=list)
    content_start: int | None = None
    content_end: int | None = None
    raw_directive: str | None = None
    directive_type: str | None = None

    metadata: RawMetadata | None = None
    # Keep the original key when (de)serializing so this round-trips to remark.json easily
    registry_context: RegistryContext | None = Field(default=None, alias="_registry_context")

    # Pydantic config
    if _PYDANTIC_V2:
        model_config = ConfigDict(populate_by_name=True, extra="ignore")
    else:

        class Config:
            allow_population_by_field_name = True
            extra = "ignore"

    # ---------- merger helpers ----------

    @staticmethod
    def _dedupe(seq: list[Any]) -> list[Any]:
        """Deduplicate while preserving order (for tags/references)."""
        seen = set()
        out = []
        for x in seq:
            k = x if isinstance(x, str | int | float | tuple) else repr(x)
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    @classmethod
    def from_instances(
        cls, raw: dict[str, Any] | None = None, extracted: dict[str, Any] | None = None
    ) -> UnifiedRemark:
        """
        Merge a raw remark (from remark.json) and an extracted remark
        (from remark_extracted.json) into a single UnifiedRemark.
        Precedence: extracted fields override raw where overlapping.
        """
        raw = raw or {}
        extracted = extracted or {}

        # id/title/type
        label = extracted.get("label") or raw.get("label") or ""
        title = extracted.get("title") or raw.get("title")
        remark_type = (
            extracted.get("remark_type")
            or (raw.get("metadata") or {}).get("class")
            or raw.get("directive_type")
        )

        # references (normalize to strings where possible) + tags (union)
        refs_raw = raw.get("references") or []
        refs_ext = extracted.get("references") or []

        def _norm_ref(r):
            if isinstance(r, str):
                return r
            if isinstance(r, dict) and "label" in r:
                return r["label"]
            return r

        references = cls._dedupe([_norm_ref(r) for r in list(refs_ext) + list(refs_raw)])
        tags = cls._dedupe(
            list(extracted.get("tags") or [])
            + [t for t in [(raw.get("metadata") or {}).get("class"), remark_type, "remark"] if t]
        )

        return cls(
            # headline
            label=label,
            title=title,
            remark_type=remark_type,
            # extracted
            nl_summary=extracted.get("nl_summary"),
            key_points=[
                KeyPoint(**kp) for kp in extracted.get("key_points", []) if isinstance(kp, dict)
            ],
            quantitative_notes=[
                QuantitativeNote(**qn)
                for qn in extracted.get("quantitative_notes", [])
                if isinstance(qn, dict)
            ],
            recommendations=[
                Recommendation(**rec)
                for rec in extracted.get("recommendations", [])
                if isinstance(rec, dict)
            ],
            dependencies=list(extracted.get("dependencies") or []),
            tags=tags,
            references=references,
            # raw payload / provenance
            content=raw.get("content"),
            section=raw.get("section"),
            start_line=raw.get("start_line"),
            end_line=raw.get("end_line"),
            header_lines=list(raw.get("header_lines") or []),
            content_start=raw.get("content_start"),
            content_end=raw.get("content_end"),
            raw_directive=raw.get("raw_directive"),
            directive_type=raw.get("directive_type"),
            metadata=RawMetadata(**(raw.get("metadata") or {})) if raw.get("metadata") else None,
            registry_context=RegistryContext(**(raw.get("_registry_context") or {}))
            if raw.get("_registry_context")
            else None,
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
