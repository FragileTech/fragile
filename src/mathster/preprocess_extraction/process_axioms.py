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


class CoreStatement(BaseModel):
    text: str | None = None
    latex: str | None = None


class Hypothesis(BaseModel):
    text: str | None = None
    latex: str | None = None


class Implication(BaseModel):
    text: str | None = None
    latex: str | None = None


class Parameter(BaseModel):
    symbol: str | None = None
    name: str | None = None
    description: str | None = None
    constraints: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class FailureMode(BaseModel):
    description: str
    impact: str | None = None


# ---------- Leaf model for directive (raw) positioning ----------


class Span(BaseModel):
    start_line: int | None = None
    end_line: int | None = None
    content_start: int | None = None
    content_end: int | None = None
    header_lines: list[int] = Field(default_factory=list)


# ---------- Unified Axiom model ----------


class UnifiedAxiom(BaseModel):
    """
    Unified axiom merging one directive item (axiom.json)
    with one extracted item (axiom_extracted.json).
    """

    # Core identity
    label: str
    title: str | None = None
    type: str = Field(default="axiom")

    # Extracted semantics
    axiom_class: str | None = None
    nl_summary: str | None = None
    core_statement: CoreStatement | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    implications: list[Implication] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # Readable text / content from directive
    content_markdown: str | None = Field(
        default=None, description="Cleaned directive content (line numbers stripped)."
    )
    raw_directive: str | None = None

    # Cross-references (merged from both sides, deduplicated)
    references: list[Any] = Field(default_factory=list)

    # Positioning + provenance from directive
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
    def from_instances(cls, directive: dict[str, Any], extracted: dict[str, Any]) -> UnifiedAxiom:
        """
        Build a UnifiedAxiom from:
          - directive: a dict for ONE directive-level axiom OR a container with 'items'.
          - extracted: a dict for ONE extracted axiom.

        Strategy:
          * Match by 'label' when possible; fall back to 'title'.
          * Pull raw text + positions from the directive item (axiom.json).
          * Pull structure (axiom_class, nl_summary, core_statement, hypotheses, implications,
            parameters, failure_modes, tags) from the extracted item (axiom_extracted.json).
          * Clean 'content' by stripping leading line numbers like '1214:' per line.
        """

        # --------- Resolve the directive item (either the dict itself or a member of its 'items') ---------
        directive_container = None
        directive_item = directive

        def _find_directive_item(
            container: dict[str, Any], ex_label: str | None, ex_title: str | None
        ) -> dict[str, Any]:
            items = container.get("items", []) or []
            if not items:
                return {}
            # 1) Exact label match
            if ex_label:
                for it in items:
                    if it.get("label") == ex_label:
                        return it
            # 2) Title match
            if ex_title:
                for it in items:
                    if it.get("title") == ex_title:
                        return it
            # 3) Fallback
            return items[0]

        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            directive_item = _find_directive_item(
                directive, extracted.get("label"), extracted.get("title")
            )

        # --------- Identity & titles ---------
        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""
        # Prefer directive's heading; fall back to extracted title
        title = directive_item.get("title") or extracted.get("title")

        # --------- Directive content cleanup ---------
        raw_content = directive_item.get("content")
        content_markdown = (
            cls._strip_line_numbers(raw_content) if isinstance(raw_content, str) else None
        )

        # --------- Span / provenance ---------
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

        # --------- Extracted structure ---------
        core_stmt = None
        if isinstance(extracted.get("core_statement"), dict):
            cs_raw = extracted["core_statement"]
            # tolerate empty {}
            if cs_raw or any(k in cs_raw for k in ("text", "latex")):
                core_stmt = CoreStatement(**{
                    k: v for k, v in cs_raw.items() if k in {"text", "latex"}
                })

        hypotheses = [Hypothesis(**h) for h in extracted.get("hypotheses", []) or []]
        implications = [Implication(**i) for i in extracted.get("implications", []) or []]

        def _coerce_param(p: Any) -> Parameter:
            if isinstance(p, dict):
                d = {
                    k: v
                    for k, v in p.items()
                    if k in {"symbol", "name", "description", "constraints", "tags"}
                }
                if isinstance(d.get("constraints"), str):
                    d["constraints"] = [d["constraints"]]
                if isinstance(d.get("tags"), str):
                    d["tags"] = [d["tags"]]
                return Parameter(**d)
            return Parameter(name=str(p))

        parameters = [_coerce_param(p) for p in (extracted.get("parameters") or [])]

        def _coerce_failure(fm: Any) -> FailureMode:
            if isinstance(fm, dict):
                d = {k: v for k, v in fm.items() if k in {"description", "impact"}}
                # description is required; if missing but 'impact' exists, fall back
                if not d.get("description") and d.get("impact"):
                    d["description"] = str(d["impact"])
                return FailureMode(**d)
            return FailureMode(description=str(fm))

        failure_modes = [_coerce_failure(fm) for fm in (extracted.get("failure_modes") or [])]

        tags = extracted.get("tags", []) or []
        axiom_class = extracted.get("axiom_class")
        nl_summary = extracted.get("nl_summary")

        # Merge references from both sides (deduped, order-preserving)
        def _merge_unique(*lists: list[Any]) -> list[Any]:
            out: list[Any] = []
            seen = set()
            for lst in lists:
                for r in lst or []:
                    key = str(r)
                    if key not in seen:
                        out.append(r)
                        seen.add(key)
            return out

        references = _merge_unique(extracted.get("references"), directive_item.get("references"))

        # --------- Alt labels if mismatch ---------
        alt_labels: list[str] = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        return cls(
            label=label,
            title=title,
            type="axiom",
            axiom_class=axiom_class,
            nl_summary=nl_summary,
            core_statement=core_stmt,
            hypotheses=hypotheses,
            implications=implications,
            parameters=parameters,
            failure_modes=failure_modes,
            tags=tags,
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
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


# ------------------------ Script helpers ------------------------


def _match_directive_item(
    extracted: dict[str, Any],
    directive_map: dict[str, dict[str, Any]],
    directive_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    label = extracted.get("label")
    if isinstance(label, str) and label:
        item = directive_map.get(label)
        if item:
            return item

    title = extracted.get("title")
    if isinstance(title, str) and title:
        for candidate in directive_items:
            if candidate.get("title") == title:
                return candidate

    if directive_items:
        return directive_items[0]

    return None


def _build_unified_axioms(
    directive_payload: dict[str, Any],
    extracted_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not extracted_items:
        return []

    directive_items = [
        item for item in directive_payload.get("items", []) or [] if isinstance(item, dict)
    ]
    template = normalize_directive_template(directive_payload)
    directive_map = directive_lookup(directive_payload)
    directive_labels = set(directive_map)
    matched_labels: set[str] = set()
    missing_labels: list[str] = []
    unified: list[dict[str, Any]] = []

    for extracted in extracted_items:
        if not isinstance(extracted, dict):
            continue

        directive_item = _match_directive_item(extracted, directive_map, directive_items)
        if directive_item is None:
            identifier = extracted.get("label") or extracted.get("title") or "<unknown>"
            missing_labels.append(identifier)
            continue

        label = directive_item.get("label")
        if isinstance(label, str) and label:
            matched_labels.add(label)

        directive_argument = wrap_directive_item(template, directive_item)
        try:
            unified_obj = UnifiedAxiom.from_instances(directive_argument, extracted)
        except Exception as exc:  # pragma: no cover - defensive logging
            identifier = extracted.get("label") or extracted.get("title") or "<unknown>"
            logger.warning("Failed to merge axiom %s: %s", identifier, exc)
            continue

        unified.append(unified_obj.model_dump(mode="json"))

    unmatched_directives = directive_labels - matched_labels
    if missing_labels:
        logger.warning(
            "Skipped %s extracted axiom(s) without directive matches: %s",
            len(missing_labels),
            ", ".join(sorted(set(missing_labels))),
        )
    if unmatched_directives:
        logger.info(
            "Directive axioms without extraction counterparts: %s",
            ", ".join(sorted(unmatched_directives)),
        )

    return unified


def preprocess_document_axioms(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directive_candidates = [
        registry_dir / "directives" / "axiom.json",
        registry_dir / "directives" / "axiom_raw.json",
    ]
    directives_path = select_existing_file(directive_candidates)

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_candidates = [
        extract_dir / "axiom.json",
        extract_dir / "axiom_extracted.json",
    ]
    extracted_path = select_existing_file(extracted_candidates)

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_axioms(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "axiom.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified axiom(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/axiom*.json and extract/axiom*.json into "
            "registry/preprocess/axiom.json for a single document."
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
        help="Optional custom destination for the generated axiom.json file.",
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
        output_path = preprocess_document_axioms(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
