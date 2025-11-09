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


class BulletItem(BaseModel):
    name: str | None = None
    text: str | None = None
    latex: str | None = None


class Condition(BaseModel):
    type: str | None = None
    text: str | None = None
    latex: str | None = None


class Parameter(BaseModel):
    symbol: str | None = None
    name: str | None = None
    description: str | None = None
    constraints: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


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


# ---------- Unified Assumption model ----------


class UnifiedAssumption(BaseModel):
    """
    Unified assumption merging one directive item (assumption.json)
    with one extracted item (assumption_extracted.json).
    """

    # Core identity
    label: str
    type: str = Field(default="assumption")
    title: str | None = None

    # High-level semantics
    scope: str | None = Field(default=None, description="e.g., 'model', 'global' from extracted")
    nl_summary: str | None = None

    # Readable text / content from directive
    content_markdown: str | None = Field(
        default=None, description="Cleaned directive content (line numbers stripped)."
    )
    raw_directive: str | None = None

    # Structured content from extracted
    bullet_items: list[BulletItem] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    references: list[Any] = Field(default_factory=list)  # merged from both sides

    # Positioning + provenance (from directive container/item)
    document_id: str | None = None
    section: str | None = None
    span: Span | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    registry_context: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None

    # Keep both labels if they differed
    alt_labels: list[str] = Field(default_factory=list)

    # ------------------------ Construction ------------------------

    @classmethod
    def from_instances(
        cls, directive: dict[str, Any], extracted: dict[str, Any]
    ) -> UnifiedAssumption:
        """
        Build a UnifiedAssumption from:
          - directive: a dict for ONE directive-level assumption OR a container with 'items'.
          - extracted: a dict for ONE extracted assumption.

        Strategy:
          * Match by 'label' when possible; additionally tolerate the extracted 'assump-' prefix.
          * Pull raw text + positions from the directive item (assumption.json).
          * Pull structure (scope, nl_summary, bullet_items, conditions, parameters, notes, tags) from the extracted item (assumption_extracted.json).
          * Clean 'content' by stripping leading line numbers like '801:' per line.
        """

        # Helper: try to find the directive item that best matches the extracted label/title.
        def _match_directive_item(
            container_items: list[dict[str, Any]], ex: dict[str, Any]
        ) -> dict[str, Any]:
            if not container_items:
                return {}
            ex_label = ex.get("label") or ""
            ex_title = ex.get("title")
            # Try exact match
            for it in container_items:
                if it.get("label") == ex_label:
                    return it
            # Try removing the 'assump-' prefix used by extracted data (e.g., 'assump-assumption-...').
            if ex_label.startswith("assump-"):
                alt = ex_label[len("assump-") :]
                for it in container_items:
                    if it.get("label") == alt:
                        return it
            # Try matching by title if present
            if ex_title:
                for it in container_items:
                    if it.get("title") == ex_title:
                        return it
            # Fallback
            return container_items[0]

        directive_container = None
        directive_item = directive
        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            directive_item = _match_directive_item(directive["items"], extracted)

        # Identity & title
        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""
        title = extracted.get("title") or directive_item.get("title")

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
        section = directive_item.get("section")

        # Extracted structure
        scope = extracted.get("scope")
        nl_summary = extracted.get("nl_summary")

        def _coerce_bullets(items) -> list[BulletItem]:
            out: list[BulletItem] = []
            for b in items or []:
                if isinstance(b, dict):
                    out.append(
                        BulletItem(**{
                            k: v for k, v in b.items() if k in {"name", "text", "latex"}
                        })
                    )
                elif isinstance(b, str):
                    out.append(BulletItem(text=b))
            return out

        def _coerce_conditions(items) -> list[Condition]:
            out: list[Condition] = []
            for c in items or []:
                if isinstance(c, dict):
                    out.append(
                        Condition(**{k: v for k, v in c.items() if k in {"type", "text", "latex"}})
                    )
                elif isinstance(c, str):
                    out.append(Condition(text=c))
            return out

        def _coerce_parameters(items) -> list[Parameter]:
            out: list[Parameter] = []
            for p in items or []:
                if isinstance(p, dict):
                    allowed = {
                        k: v
                        for k, v in p.items()
                        if k in {"symbol", "name", "description", "constraints", "tags"}
                    }
                    if isinstance(allowed.get("constraints"), str):
                        allowed["constraints"] = [allowed["constraints"]]
                    if isinstance(allowed.get("tags"), str):
                        allowed["tags"] = [allowed["tags"]]
                    out.append(Parameter(**allowed))
                elif isinstance(p, str):
                    out.append(Parameter(name=p))
            return out

        def _coerce_notes(items) -> list[Note]:
            out: list[Note] = []
            for n in items or []:
                if isinstance(n, dict):
                    out.append(Note(**{k: v for k, v in n.items() if k in {"type", "text"}}))
                elif isinstance(n, str):
                    out.append(Note(text=n))
            return out

        # Merge references from both sides (deduplicated, order-preserving)
        def _merge_refs(a, b) -> list[Any]:
            out, seen = [], set()
            for r in (a or []) + (b or []):
                if str(r) not in seen:
                    out.append(r)
                    seen.add(str(r))
            return out

        bullet_items = _coerce_bullets(extracted.get("bullet_items"))
        conditions = _coerce_conditions(extracted.get("conditions"))
        parameters = _coerce_parameters(extracted.get("parameters"))
        notes = _coerce_notes(extracted.get("notes"))
        tags = extracted.get("tags", []) or []
        references = _merge_refs(extracted.get("references"), directive_item.get("references"))

        # Alt labels if mismatch
        alt_labels = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        return cls(
            label=label,
            type="assumption",
            title=title,
            scope=scope,
            nl_summary=nl_summary,
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
            bullet_items=bullet_items,
            conditions=conditions,
            parameters=parameters,
            notes=notes,
            tags=tags,
            references=references,
            document_id=document_id,
            section=section,
            span=span,
            metadata=metadata,
            registry_context=registry_context,
            generated_at=generated_at,
            alt_labels=alt_labels,
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


def _build_unified_assumptions(
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
    extracted_labels = set()
    for item in extracted_items:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if isinstance(label, str):
            extracted_labels.add(label)
            if label.startswith("assump-"):
                extracted_labels.add(label.replace("assump-", "", 1))

    for extracted in extracted_items:
        label = extracted.get("label")
        if not isinstance(label, str) or not label:
            logger.warning("Skipping extracted assumption without a label: %s", extracted)
            continue
        directive_item = directive_map.get(label) or directive_map.get(
            label.replace("assump-", "", 1)
        )
        if directive_item is None:
            missing_labels.append(label)
            continue
        directive_argument = wrap_directive_item(template, directive_item)
        try:
            unified_obj = UnifiedAssumption.from_instances(directive_argument, extracted)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to merge assumption %s: %s", label, exc)
            continue
        unified.append(unified_obj.model_dump(mode="json"))

    unmatched_directives = directive_labels - extracted_labels
    if missing_labels:
        logger.warning(
            "Skipped %s extracted assumption(s) without directive matches: %s",
            len(missing_labels),
            ", ".join(sorted(set(missing_labels))),
        )
    if unmatched_directives:
        logger.info(
            "Directive assumptions without extraction counterparts: %s",
            ", ".join(sorted(unmatched_directives)),
        )

    return unified


def _select_existing_file(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the expected assumption registry files were found: "
        + ", ".join(str(path) for path in candidates),
    )


def preprocess_document_assumptions(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directive_candidates = [
        registry_dir / "directives" / "assumption.json",
        registry_dir / "directives" / "assumption_raw.json",
    ]
    directives_path = select_existing_file(directive_candidates)

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_candidates = [
        extract_dir / "assumption.json",
        extract_dir / "assumption_extracted.json",
    ]
    extracted_path = select_existing_file(extracted_candidates)

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_assumptions(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "assumption.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified assumption(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/assumption*.json and extract/assumption*.json into "
            "registry/preprocess/assumption.json for a single document."
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
        help="Optional custom destination for the generated assumption.json file.",
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
        output_path = preprocess_document_assumptions(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
