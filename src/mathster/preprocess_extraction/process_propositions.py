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


logger = logging.getLogger(__name__)


# ---------- Small nested records from the extracted side ----------


class Equation(BaseModel):
    label: str | None = None
    latex: str


class Hypothesis(BaseModel):
    text: str | None = None
    latex: str | None = None


class Variable(BaseModel):
    symbol: str | None = None
    name: str | None = None
    description: str | None = None
    constraints: list[str] | None = None
    tags: list[str] | None = None


class Conclusion(BaseModel):
    text: str | None = None
    latex: str | None = None


class ProofStep(BaseModel):
    kind: str | None = None
    text: str | None = None
    latex: str | None = None


class Proof(BaseModel):
    availability: str | None = None
    steps: list[ProofStep] = Field(default_factory=list)


# ---------- Registry/locator information from the raw side ----------


class RegistryContext(BaseModel):
    stage: str | None = None
    document_id: str | None = None
    chapter_index: int | None = None
    chapter_file: str | None = None
    section_id: str | None = None


class RawLocator(BaseModel):
    # Where this proposition lives in the source registry
    section: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    header_lines: list[int] | None = None
    content_start: int | None = None
    content_end: int | None = None

    # Verbatim blocks from the registry dump (optionally cleaned)
    content: str | None = None
    raw_directive: str | None = None

    # Raw references list (if any present in the registry item)
    references: list[Any] | None = None

    # Book-keeping about where it was found
    registry_context: RegistryContext | None = None


# ---------- The unified Proposition model ----------


class PropositionModel(BaseModel):
    # Common identifiers / headings
    type: str = Field(default="proposition", description="Canonical object type")
    label: str = Field(..., description="Stable identifier, e.g. 'prop-…'")
    title: str | None = None

    # Extracted (semantic) content
    nl_statement: str | None = None
    equations: list[Equation] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    conclusion: Conclusion | None = None
    variables: list[Variable] = Field(default_factory=list)
    implicit_assumptions: list[dict] | None = None  # keep as dicts to preserve 'confidence' etc.
    local_refs: list[str] | None = None
    proof: Proof | None = None
    tags: list[str] = Field(default_factory=list)

    # Raw (registry) sidecar info
    raw: RawLocator = Field(default_factory=RawLocator)

    # ---------- Builders & helpers ----------

    @staticmethod
    def _strip_numbered_prefixes(text: str | None) -> str | None:
        """
        Remove leading 'NNN: ' prefixes that appear at the start of each line
        in the registry 'content' and 'raw_directive' blobs.
        """
        if not text:
            return text
        lines = []
        for ln in text.splitlines():
            # Remove a leading integer and a colon (e.g., '123: ') if present
            if ln and ln[0].isdigit():
                i = 0
                while i < len(ln) and ln[i].isdigit():
                    i += 1
                # expect a colon after digits
                if i < len(ln) and ln[i] == ":":
                    # also remove following single space, if present
                    j = i + 1
                    if j < len(ln) and ln[j] == " ":
                        j += 1
                    ln = ln[j:]
            lines.append(ln)
        return "\n".join(lines)

    @classmethod
    def from_instances(
        cls, raw_item: dict, extracted_item: dict, *, clean_numbered_content: bool = True
    ) -> PropositionModel:
        """
        Create a unified PropositionModel by merging:
          - raw_item: a single item dict from proposition.json["items"]
          - extracted_item: a single dict from proposition_extracted.json
        If labels disagree, a ValueError is raised.
        """
        # --- Validate identity / label
        raw_label = raw_item.get("label") or (raw_item.get("metadata") or {}).get("label")
        ext_label = extracted_item.get("label")
        if not raw_label or not ext_label or raw_label != ext_label:
            raise ValueError(
                f"Label mismatch or missing: raw='{raw_label}' vs extracted='{ext_label}'"
            )

        # --- Title preference: prefer extracted if present, else raw
        title = extracted_item.get("title") or raw_item.get("title")

        # --- Build extracted content blocks
        equations = [Equation(**e) for e in extracted_item.get("equations", []) or []]
        hypotheses = [Hypothesis(**h) for h in extracted_item.get("hypotheses", []) or []]
        variables = [Variable(**v) for v in extracted_item.get("variables", []) or []]
        conclusion_block = extracted_item.get("conclusion") or None
        conclusion = Conclusion(**conclusion_block) if isinstance(conclusion_block, dict) else None

        proof_block = extracted_item.get("proof")
        proof = Proof(**proof_block) if isinstance(proof_block, dict) else None

        # --- Raw locator + content
        reg_ctx_raw = raw_item.get("_registry_context") or {}
        registry_context = RegistryContext(
            stage=reg_ctx_raw.get("stage"),
            document_id=reg_ctx_raw.get("document_id"),
            chapter_index=reg_ctx_raw.get("chapter_index"),
            chapter_file=reg_ctx_raw.get("chapter_file"),
            section_id=reg_ctx_raw.get("section_id"),
        )

        content = raw_item.get("content")
        raw_directive = raw_item.get("raw_directive")
        if clean_numbered_content:
            content = cls._strip_numbered_prefixes(content)
            raw_directive = cls._strip_numbered_prefixes(raw_directive)

        raw_locator = RawLocator(
            section=raw_item.get("section"),
            start_line=raw_item.get("start_line"),
            end_line=raw_item.get("end_line"),
            header_lines=raw_item.get("header_lines"),
            content_start=raw_item.get("content_start"),
            content_end=raw_item.get("content_end"),
            content=content,
            raw_directive=raw_directive,
            references=raw_item.get("references"),
            registry_context=registry_context,
        )

        # --- Merge tags (raw registry usually has none; we union to be safe)
        tags = list(
            dict.fromkeys((extracted_item.get("tags") or []) + (raw_item.get("tags") or []))
        )

        return cls(
            type=extracted_item.get("type") or raw_item.get("directive_type") or "proposition",
            label=raw_label,
            title=title,
            nl_statement=extracted_item.get("nl_statement"),
            equations=equations,
            hypotheses=hypotheses,
            conclusion=conclusion,
            variables=variables,
            implicit_assumptions=extracted_item.get("implicit_assumptions"),
            local_refs=extracted_item.get("local_refs"),
            proof=proof,
            tags=tags,
            raw=raw_locator,
        )


# ---------- Build + CLI helpers ----------


def _build_unified_propositions(
    directive_payload: dict[str, Any],
    extracted_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not extracted_items:
        return []

    [item for item in directive_payload.get("items", []) or [] if isinstance(item, dict)]
    raw_lookup = directive_lookup(directive_payload)
    matched_labels: set[str] = set()
    missing: list[str] = []
    unified: list[dict[str, Any]] = []

    for extracted in extracted_items:
        if not isinstance(extracted, dict):
            continue
        label = extracted.get("label")
        if not isinstance(label, str) or not label:
            logger.warning("Skipping extracted proposition without a label: %s", extracted)
            continue

        raw_item = raw_lookup.get(label)
        if raw_item is None:
            missing.append(label)
            continue

        try:
            unified_obj = PropositionModel.from_instances(raw_item, extracted)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to merge proposition %s: %s", label, exc)
            continue

        matched_labels.add(label)
        unified.append(unified_obj.model_dump(mode="json"))

    unmatched = set(raw_lookup) - matched_labels
    if missing:
        logger.warning(
            "Skipped %s extracted proposition(s) without directive matches: %s",
            len(missing),
            ", ".join(sorted(set(missing))),
        )
    if unmatched:
        logger.info(
            "Directive propositions without extraction counterparts: %s",
            ", ".join(sorted(unmatched)),
        )

    return unified


def preprocess_document_propositions(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> Path:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    directives_path = registry_dir / "directives" / "proposition.json"
    if not directives_path.exists():
        raise FileNotFoundError(f"Missing directives/proposition.json in {registry_dir}")

    extract_dir = resolve_extract_directory(registry_dir)
    extracted_candidates = [
        extract_dir / "proposition.json",
        extract_dir / "proposition_extracted.json",
    ]
    extracted_path = select_existing_file(extracted_candidates)

    directive_payload = load_directive_payload(directives_path)
    extracted_items = load_extracted_items(extracted_path)
    unified_payload = _build_unified_propositions(directive_payload, extracted_items)

    destination = output_path
    if destination is None:
        preprocess_dir = registry_dir / "preprocess"
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        destination = preprocess_dir / "propositions.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s unified proposition(s) to %s",
        len(unified_payload),
        destination,
    )
    return destination


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge directives/proposition.json and extract/proposition*.json into "
            "registry/preprocess/propositions.json for a single document."
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
        help="Optional custom destination for the generated propositions.json file.",
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
        output_path = preprocess_document_propositions(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
