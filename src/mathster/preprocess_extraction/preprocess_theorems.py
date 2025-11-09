from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import re


# ---------- Leaf models for the extracted representation ----------

class Equation(BaseModel):
    label: Optional[str] = None
    latex: str

class Hypothesis(BaseModel):
    text: Optional[str] = None
    latex: Optional[str] = None

class Conclusion(BaseModel):
    text: Optional[str] = None
    latex: Optional[str] = None

class Variable(BaseModel):
    symbol: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class Assumption(BaseModel):
    text: str
    confidence: Optional[float] = None

class ProofStep(BaseModel):
    kind: Optional[str] = None
    text: Optional[str] = None
    latex: Optional[str] = None

class Proof(BaseModel):
    availability: Optional[str] = None
    steps: List[ProofStep] = Field(default_factory=list)


# ---------- Leaf models for the directive (raw) representation ----------

class Span(BaseModel):
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content_start: Optional[int] = None
    content_end: Optional[int] = None
    header_lines: List[int] = Field(default_factory=list)


# ---------- Unified model ----------

class UnifiedTheorem(BaseModel):
    """
    A unified theorem object that merges one directive-style item (from theorem.json)
    with one structured/extracted item (from theorem_extracted.json).
    """

    # Core identity
    label: str
    title: Optional[str] = None
    type: Optional[str] = Field(default="theorem", description="Type from the extracted object when available.")

    # Readable text / content
    nl_statement: Optional[str] = Field(
        default=None, description="Natural-language summary from the extracted object."
    )
    content_markdown: Optional[str] = Field(
        default=None,
        description="Cleaned markdown/plaintext of the raw directive content (line numbers stripped).",
    )
    raw_directive: Optional[str] = Field(
        default=None, description="Full raw directive block from theorem.json, if present."
    )

    # Structured math content
    equations: List[Equation] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    conclusion: Optional[Conclusion] = None
    variables: List[Variable] = Field(default_factory=list)
    implicit_assumptions: List[Assumption] = Field(default_factory=list)
    local_refs: List[str] = Field(default_factory=list)
    proof: Optional[Proof] = None
    tags: List[str] = Field(default_factory=list)

    # Positioning + provenance
    document_id: Optional[str] = None
    section: Optional[str] = None
    span: Optional[Span] = None
    references: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registry_context: Dict[str, Any] = Field(default_factory=dict)
    generated_at: Optional[str] = None

    # Helpful: keep both original labels if they differed
    alt_labels: List[str] = Field(default_factory=list)

    # ------------------------ Construction ------------------------

    @classmethod
    def from_instances(cls, directive: Dict[str, Any], extracted: Dict[str, Any]) -> "UnifiedTheorem":
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
        content_markdown = cls._strip_line_numbers(raw_content) if isinstance(raw_content, str) else None

        # Span
        span = Span(
            start_line=directive_item.get("start_line"),
            end_line=directive_item.get("end_line"),
            content_start=directive_item.get("content_start"),
            content_end=directive_item.get("content_end"),
            header_lines=directive_item.get("header_lines", []) or [],
        ) if directive_item else None

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
        concl = (Conclusion(**extracted["conclusion"])
                 if isinstance(extracted.get("conclusion"), dict)
                 else None)
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
    def _strip_line_numbers(text: Optional[str]) -> Optional[str]:
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
