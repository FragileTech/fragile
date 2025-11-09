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
    wrap_directive_item,
)


logger = logging.getLogger(__name__)


# ---------- Leaf models for the extracted representation ----------


class Equation(BaseModel):
    label: str | None = None
    latex: str


class Hypothesis(BaseModel):
    text: str | None = None
    latex: str | None = None


class Conclusion(BaseModel):
    text: str | None = None
    latex: str | None = None


class Variable(BaseModel):
    symbol: str | None = None
    name: str | None = None
    description: str | None = None
    constraints: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Assumption(BaseModel):
    text: str
    confidence: float | None = None


class ProofStep(BaseModel):
    kind: str | None = None
    text: str | None = None
    latex: str | None = None


class Proof(BaseModel):
    availability: str | None = None
    steps: list[ProofStep] = Field(default_factory=list)


# ---------- Leaf models for the directive (raw) representation ----------


class Span(BaseModel):
    start_line: int | None = None
    end_line: int | None = None
    content_start: int | None = None
    content_end: int | None = None
    header_lines: list[int] = Field(default_factory=list)


# ---------- Unified model ----------


class UnifiedTheorem(BaseModel):
    """
    A unified theorem object that merges one directive-style item (from theorem.json)
    with one structured/extracted item (from theorem_extracted.json).
    """

    # Core identity
    label: str
    title: str | None = None
    type: str | None = Field(
        default="theorem", description="Type from the extracted object when available."
    )

    # Readable text / content
    nl_statement: str | None = Field(
        default=None, description="Natural-language summary from the extracted object."
    )
    content_markdown: str | None = Field(
        default=None,
        description="Cleaned markdown/plaintext of the raw directive content (line numbers stripped).",
    )
    raw_directive: str | None = Field(
        default=None, description="Full raw directive block from theorem.json, if present."
    )

    # Structured math content
    equations: list[Equation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    conclusion: Conclusion | None = None
    variables: list[Variable] = Field(default_factory=list)
    implicit_assumptions: list[Assumption] = Field(default_factory=list)
    local_refs: list[str] = Field(default_factory=list)
    proof: Proof | None = None
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
    ) -> UnifiedTheorem:
        """
        Build a UnifiedTheorem from:
          - directive: a dict for ONE directive-level theorem item OR a container with 'items'.
          - extracted: a dict for ONE extracted theorem object.

        The method will:
          * Match labels when possible.
          * Pull raw text + positions from the directive item.
          * Pull structured fields (equations, hypotheses, variables, proof, tags) from the extracted object.
          * Clean 'content' by stripping leading line numbers like '615: ' per line.
        """

        # If a full directive container was passed, try to locate the item matching the extracted label.
        directive_container = None
        directive_item = directive
        if "items" in directive and isinstance(directive["items"], list):
            directive_container = directive
            # prefer matching by label
            ex_label = extracted.get("label")
            matched = next((it for it in directive["items"] if it.get("label") == ex_label), None)
            directive_item = matched or (directive["items"][0] if directive["items"] else {})

        # Prefer label from extracted; fall back to directive.
        label_ex = extracted.get("label")
        label_dir = directive_item.get("label")
        label = label_ex or label_dir or ""

        # Title preference: extracted first, then directive
        title = extracted.get("title") or directive_item.get("title")

        # Raw directive content + cleanup
        raw_content = directive_item.get("content")
        content_markdown = (
            cls._strip_line_numbers(raw_content) if isinstance(raw_content, str) else None
        )

        # Span
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

        # Provenance / registry context
        if directive_container:
            document_id = directive_container.get("document_id")
            generated_at = directive_container.get("generated_at")
        else:
            # try to recover from the item's registry context if present
            rc = directive_item.get("_registry_context", {}) or {}
            document_id = rc.get("document_id")
            generated_at = None

        # Registry + metadata
        registry_context = directive_item.get("_registry_context", {}) or {}
        metadata = directive_item.get("metadata", {}) or {}
        references = directive_item.get("references", []) or []
        section = directive_item.get("section")

        # Extracted structured fields
        eqs = [Equation(**e) for e in extracted.get("equations", []) or []]
        hyps = [Hypothesis(**h) for h in extracted.get("hypotheses", []) or []]
        concl = (
            Conclusion(**extracted["conclusion"])
            if isinstance(extracted.get("conclusion"), dict)
            else None
        )
        vars_ = [Variable(**v) for v in extracted.get("variables", []) or []]
        imps = [Assumption(**a) for a in extracted.get("implicit_assumptions", []) or []]
        local_refs = extracted.get("local_refs", []) or []
        proof_block = extracted.get("proof")
        proof = Proof(**proof_block) if isinstance(proof_block, dict) else None
        tags = extracted.get("tags", []) or []

        # Natural-language summary
        nl_stmt = extracted.get("nl_statement")

        # Alt labels if mismatch
        alt_labels = []
        if label_ex and label_dir and label_ex != label_dir:
            alt_labels = [label_ex, label_dir]

        return cls(
            label=label,
            title=title,
            type=extracted.get("type", "theorem"),
            nl_statement=nl_stmt,
            content_markdown=content_markdown,
            raw_directive=directive_item.get("raw_directive"),
            equations=eqs,
            hypotheses=hyps,
            conclusion=concl,
            variables=vars_,
            implicit_assumptions=imps,
            local_refs=local_refs,
            proof=proof,
            tags=tags,
            document_id=document_id,
            section=section,
            span=span,
            references=references,
            metadata=metadata,
            registry_context=registry_context,
            generated_at=generated_at,
            alt_labels=alt_labels,
        )

    # ------------------------ Helpers ------------------------

    @staticmethod
    def _strip_line_numbers(text: str | None) -> str | None:
        """
        Remove leading 'NNN:' prefixes that appear on each line of the directive content.
        """
        if not text:
            return text
        out_lines = []
        for line in text.splitlines():
            # Remove a leading integer + colon + optional space, e.g. "615: " or "615:"
            out_lines.append(re.sub(r"^\s*\d+:\s?", "", line))
        return "\n".join(out_lines).strip()


def _build_unified_theorems(
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
            logger.warning("Skipping extracted theorem without a label: %s", extracted)
            continue
        directive_item = directive_map.get(label)
        if directive_item is None:
            missing_labels.append(label)
            continue
        directive_argument = wrap_directive_item(template, directive_item)
        try:
            unified_obj = UnifiedTheorem.from_instances(directive_argument, extracted)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to merge theorem %s: %s", label, exc)
            continue
        unified.append(unified_obj.model_dump(mode="json"))

    unmatched_directives = directive_labels - extracted_labels
    if missing_labels:
        logger.warning(
            "Skipped %s extracted theorem(s) without directive matches: %s",
            len(missing_labels),
            ", ".join(sorted(set(missing_labels))),
        )
    if unmatched_directives:
        logger.info(
            "Directive theorems without extraction counterparts: %s",
            ", ".join(sorted(unmatched_directives)),
        )

    return unified


def preprocess_document_theorems(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directives_path = registry_dir / "directives" / "theorem.json"
    if not directives_path.exists():
        raise FileNotFoundError(f"Missing directives/theorem.json in {registry_dir}")

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_path = extract_dir / "theorem.json"
    if not extracted_path.exists():
        raise FileNotFoundError(f"Missing extract/theorem.json in {extract_dir}")

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_theorems(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "theorem.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified theorem(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/theorem.json and extract/theorem.json into "
            "registry/preprocess/theorem.json for a single document."
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
        help="Optional custom destination for the generated theorem.json file.",
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
        output_path = preprocess_document_theorems(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
