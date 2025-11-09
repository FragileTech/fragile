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


class Conclusion(BaseModel):
    text: str | None = None
    latex: str | None = None


class Assumption(BaseModel):
    text: str
    latex: str | None = None


class ProofStep(BaseModel):
    order: float | None = None
    kind: str | None = None
    text: str | None = None
    latex: str | None = None
    references: list[str] = Field(default_factory=list)
    derived_statement: str | None = None


class KeyEquation(BaseModel):
    label: str | None = None
    latex: str
    role: str | None = None


class MathTool(BaseModel):
    toolName: str | None = None
    field: str | None = None
    description: str | None = None
    roleInProof: str | None = None
    levelOfAbstraction: str | None = None
    relatedTools: list[str] = Field(default_factory=list)


class CaseItem(BaseModel):
    name: str | None = None
    condition: str | None = None
    summary: str | None = None


class Remark(BaseModel):
    type: str | None = None
    text: str | None = None


class Gap(BaseModel):
    description: str
    severity: str | None = None
    location_hint: str | None = None


# ---------- Leaf models for the directive (raw) representation ----------


class Span(BaseModel):
    start_line: int | None = None
    end_line: int | None = None
    content_start: int | None = None
    content_end: int | None = None
    header_lines: list[int] = Field(default_factory=list)


# ---------- Unified model ----------


class UnifiedProof(BaseModel):
    """
    A unified proof object that merges one directive item (proof.json)
    with one extracted item (proof_extracted.json).
    """

    # Core identity
    label: str
    title: str | None = None
    type: str = Field(default="proof")

    # Link to what the proof establishes
    proves: str | None = Field(
        default=None, description="Label of the result proved (e.g., a lemma/theorem)."
    )
    proof_type: str | None = None  # e.g., 'direct', 'probabilistic' ...
    proof_status: str | None = None  # e.g., 'complete', 'sketch'

    # Readable text / content from directive
    content_markdown: str | None = Field(
        default=None, description="Cleaned directive content with line numbers stripped."
    )
    raw_directive: str | None = None

    # Structured content from extracted
    strategy_summary: str | None = None
    conclusion: Conclusion | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    steps: list[ProofStep] = Field(default_factory=list)
    key_equations: list[KeyEquation] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)  # local labels cited by the proof
    math_tools: list[MathTool] = Field(default_factory=list)
    cases: list[CaseItem] = Field(default_factory=list)
    remarks: list[Remark] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

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
    def from_instances(cls, directive: dict[str, Any], extracted: dict[str, Any]) -> UnifiedProof:
        """
        Build a UnifiedProof from:
          - directive: a dict for ONE directive-level proof OR a container with 'items'.
          - extracted: a dict for ONE extracted proof.

        It will:
          * Match items by 'label' when possible.
          * Pull raw text + positioning from the directive item (proof.json).  # filecite marker not allowed in code
          * Pull structure (proves, steps, equations, assumptions, conclusion, etc.) from the extracted item (proof_extracted.json).
          * Strip leading 'NNN:' line numbers from directive content.
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
        title = directive_item.get("title") or extracted.get("title")

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
        conclusion = (
            Conclusion(**extracted["conclusion"])
            if isinstance(extracted.get("conclusion"), dict)
            else None
        )
        assumptions = [Assumption(**a) for a in extracted.get("assumptions", []) or []]
        steps = [ProofStep(**s) for s in extracted.get("steps", []) or []]
        key_equations = [KeyEquation(**k) for k in extracted.get("key_equations", []) or []]
        math_tools = [MathTool(**m) for m in extracted.get("math_tools", []) or []]
        cases = [CaseItem(**c) for c in extracted.get("cases", []) or []]
        remarks = [Remark(**r) for r in extracted.get("remarks", []) or []]
        gaps = [Gap(**g) for g in extracted.get("gaps", []) or []]

        # Alt labels if mismatch
        alt_labels: list[str] = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        return cls(
            label=label,
            title=title,
            type="proof",
            proves=extracted.get("proves"),
            proof_type=extracted.get("proof_type"),
            proof_status=extracted.get("proof_status"),
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
            strategy_summary=extracted.get("strategy_summary"),
            conclusion=conclusion,
            assumptions=assumptions,
            steps=steps,
            key_equations=key_equations,
            references=extracted.get("references", [])
            or (directive_item.get("references", []) or []),
            math_tools=math_tools,
            cases=cases,
            remarks=remarks,
            gaps=gaps,
            tags=extracted.get("tags", []) or [],
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
