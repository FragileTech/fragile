from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field

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


# ---------- Leaf models for the extracted representation ----------


class Condition(BaseModel):
    text: str | None = None
    latex: str | None = None


class NamedProperty(BaseModel):
    name: str | None = None
    description: str | None = None


class Parameter(BaseModel):
    symbol: str | None = None
    name: str | None = None
    description: str | None = None
    constraints: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Example(BaseModel):
    text: str | None = None
    latex: str | None = None


class Note(BaseModel):
    type: str | None = None
    text: str | None = None


# ---------- Leaf model for directive (raw) positioning ----------


class Span(BaseModel):
    start_line: int | None = None
    end_line: int | None = None
    content_start: int | None = None
    content_end: int | None = None
    header_lines: list[int] = Field(default_factory=list)


# ---------- Unified Definition model ----------


class UnifiedDefinition(BaseModel):
    """
    Unified definition merging one directive item (definition.json)
    with one extracted item (definition_extracted.json).
    """

    # Core identity
    label: str
    type: str = Field(default="definition")
    title: str | None = None  # usually from directive
    term: str | None = None  # canonical term from extracted
    object_type: str | None = None  # e.g., "set", "operator", "functionals", ...

    # Human-readable content
    nl_definition: str | None = None
    content_markdown: str | None = Field(
        default=None,
        description="Cleaned directive content (line numbers stripped).",
    )
    raw_directive: str | None = None

    # Structured content
    formal_conditions: list[Condition] = Field(default_factory=list)
    properties: list[NamedProperty] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    related_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # Positioning + provenance
    document_id: str | None = None
    section: str | None = None
    span: Span | None = None
    references: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    registry_context: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None

    # Helpful: keep both original labels if they differed
    alt_labels: list[str] = Field(default_factory=list)

    # ------------------------ Construction ------------------------

    @classmethod
    def from_instances(
        cls, directive: dict[str, Any], extracted: dict[str, Any]
    ) -> UnifiedDefinition:
        """
        Build a UnifiedDefinition from:
          - directive: a dict for ONE directive-level definition OR a container with 'items'.
          - extracted: a dict for ONE extracted definition.

        Strategy:
          * Match by 'label' when possible.
          * Pull raw text + positions from the directive item (definition.json).
          * Pull structure (term, object_type, nl_definition, conditions, properties, parameters, examples, notes, tags) from the extracted item (definition_extracted.json).
          * Clean 'content' by stripping leading line numbers like '177:' per line.
        """

        # Resolve the directive item (either the dict itself or a member of its 'items')
        directive_container = None
        directive_item = directive
        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            ex_label = extracted.get("label")
            matched = next((it for it in directive["items"] if it.get("label") == ex_label), None)
            directive_item = matched or (directive["items"][0] if directive["items"] else {})

        # Identity & title
        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""
        # Prefer directive title; fall back to extracted term/title if present
        title = directive_item.get("title") or extracted.get("title") or extracted.get("term")

        # Directive content cleanup
        raw_content = directive_item.get("content")
        content_markdown = (
            cls._strip_line_numbers(raw_content) if isinstance(raw_content, str) else None
        )

        # Span / provenance
        span = (
            Span(
                start_line=directive_item.get("start_line"),
                end_line=directive_item.get("end_line"),
                content_start=directive_item.get("content_start"),
                content_end=directive_item.get("content_end"),
                header_lines=directive_item.get("header_lines", []) or [],
            )
            if directive_item
            else None
        )

        if directive_container:
            document_id = directive_container.get("document_id")
            generated_at = directive_container.get("generated_at")
        else:
            rc = directive_item.get("_registry_context", {}) or {}
            document_id = rc.get("document_id")
            generated_at = None

        registry_context = directive_item.get("_registry_context", {}) or {}
        metadata = directive_item.get("metadata", {}) or {}
        references = directive_item.get("references", []) or []
        section = directive_item.get("section")

        # Robust helpers to coerce heterogeneous extracted lists -----------------
        def _coerce_conditions(items) -> list[Condition]:
            out: list[Condition] = []
            for c in items or []:
                if isinstance(c, dict):
                    out.append(Condition(**{k: v for k, v in c.items() if k in {"text", "latex"}}))
                elif isinstance(c, str):
                    out.append(Condition(text=c))
            return out

        def _coerce_properties(items) -> list[NamedProperty]:
            out: list[NamedProperty] = []
            for p in items or []:
                if isinstance(p, dict):
                    out.append(
                        NamedProperty(**{
                            k: v for k, v in p.items() if k in {"name", "description"}
                        })
                    )
                elif isinstance(p, str):
                    out.append(NamedProperty(name=p))
            return out

        def _coerce_parameters(items) -> list[Parameter]:
            out: list[Parameter] = []
            for p in items or []:
                if isinstance(p, dict):
                    # accept extra keys but only map the ones we model
                    allowed = {
                        k: v
                        for k, v in p.items()
                        if k in {"symbol", "name", "description", "constraints", "tags"}
                    }
                    # normalize lists
                    if isinstance(allowed.get("constraints"), str):
                        allowed["constraints"] = [allowed["constraints"]]
                    if isinstance(allowed.get("tags"), str):
                        allowed["tags"] = [allowed["tags"]]
                    out.append(Parameter(**allowed))
                elif isinstance(p, str):
                    out.append(Parameter(name=p))
            return out

        def _coerce_examples(items) -> list[Example]:
            out: list[Example] = []
            for e in items or []:
                if isinstance(e, dict):
                    out.append(Example(**{k: v for k, v in e.items() if k in {"text", "latex"}}))
                elif isinstance(e, str):
                    out.append(Example(text=e))
            return out

        def _coerce_notes(items) -> list[Note]:
            out: list[Note] = []
            for n in items or []:
                if isinstance(n, dict):
                    out.append(Note(**{k: v for k, v in n.items() if k in {"type", "text"}}))
                elif isinstance(n, str):
                    out.append(Note(text=n))
            return out

        # -----------------------------------------------------------------------

        return cls(
            label=label,
            type="definition",
            title=title,
            term=extracted.get("term") or directive_item.get("title"),
            object_type=extracted.get("object_type"),
            nl_definition=extracted.get("nl_definition"),
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
            formal_conditions=_coerce_conditions(extracted.get("formal_conditions")),
            properties=_coerce_properties(extracted.get("properties")),
            parameters=_coerce_parameters(extracted.get("parameters")),
            examples=_coerce_examples(extracted.get("examples")),
            notes=_coerce_notes(extracted.get("notes")),
            related_refs=extracted.get("related_refs", []) or [],
            tags=extracted.get("tags", []) or [],
            document_id=document_id,
            section=section,
            span=span,
            references=references,
            metadata=metadata,
            registry_context=registry_context,
            generated_at=generated_at,
            alt_labels=(
                [label_ex, label_dir] if label_ex and label_dir and label_ex != label_dir else []
            ),
        )

    # ------------------------ Helpers ------------------------

    @staticmethod
    def _strip_line_numbers(text: str | None) -> str | None:
        """Remove leading 'NNN:' prefixes that appear on each line of the directive content."""
        if not text:
            return text
        out_lines = []
        for line in text.splitlines():
            out_lines.append(re.sub(r"^\s*\d+:\s?", "", line))
        return "\n".join(out_lines).strip()


def _build_unified_definitions(
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
            logger.warning("Skipping extracted definition without a label: %s", extracted)
            continue
        directive_item = directive_map.get(label)
        if directive_item is None:
            missing_labels.append(label)
            continue
        directive_argument = wrap_directive_item(template, directive_item)
        try:
            unified_obj = UnifiedDefinition.from_instances(directive_argument, extracted)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to merge definition %s: %s", label, exc)
            continue
        unified.append(unified_obj.model_dump(mode="json"))

    unmatched_directives = directive_labels - extracted_labels
    if missing_labels:
        logger.warning(
            "Skipped %s extracted definition(s) without directive matches: %s",
            len(missing_labels),
            ", ".join(sorted(set(missing_labels))),
        )
    if unmatched_directives:
        logger.info(
            "Directive definitions without extraction counterparts: %s",
            ", ".join(sorted(unmatched_directives)),
        )

    return unified


def _select_existing_file(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the expected definition registry files were found: "
        + ", ".join(str(path) for path in candidates),
    )


def preprocess_document_definitions(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directive_candidates = [
        registry_dir / "directives" / "definition.json",
        registry_dir / "directives" / "definition_raw.json",
    ]
    directives_path = select_existing_file(directive_candidates)

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_candidates = [
        extract_dir / "definition.json",
        extract_dir / "definition_extracted.json",
    ]
    extracted_path = select_existing_file(extracted_candidates)

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_definitions(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "definition.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified definition(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/definition*.json and extract/definition*.json into "
            "registry/preprocess/definition.json for a single document."
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
        help="Optional custom destination for the generated definition.json file.",
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
        output_path = preprocess_document_definitions(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
