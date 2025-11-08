#!/usr/bin/env python3

"""
Agent for parsing `::{prf:axiom}` directives into structured JSON output.

Usage
-----
python extract_axioms.py \
  --doc docs/source/1_euclidean_gas/01_fragile_gas_framework.md \
  --lm openai/gpt-4o-mini \
  --passes 5 \
  --threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import dspy

from mathster.agents.core import (
    DirectiveAgentPaths,
    LATEX_FENCE_PATTERN,
    METADATA_PATTERN,
    run_directive_extraction_loop,
    URI_PATTERN,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# DSPy Signature
# --------------------------------------------------------------------------------------


class ParseAxiomDirectiveSplit(dspy.Signature):
    """
    Convert a `::{prf:axiom}` directive into structured semantic content.

    INPUT
    -----
    - directive_text (str): Raw directive body (header/title/body/closing fence).
    - context_hints (str, optional): Nearby prose snippet for hints.

    OUTPUT FIELDS
    -------------
    - label_str (str): Axiom label (`def-axiom-*` or `axiom-*` per registry conventions).
    - title_str (str): Human-facing title if present.
    - axiom_class_str (str): Category (structural/regularity/probabilistic/etc.).
    - nl_summary_str (str): Natural-language restatement (1–3 sentences).

    - core_statement_json (json object):
        {"text": <string|null>, "latex": <string|null>}

    - hypotheses_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...]

    - implications_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...]

    - parameters_json (json array):
        [{"symbol": <string>, "description": <string|null>, "constraints": [<string>, ...]}, ...]

    - references_json (json array of str):
        Labels cited in the axiom (definitions, theorems, figures, etc.).

    - failure_modes_json (json array):
        [{"description": <string|null>, "impact": <string|null>} ...]

    Rules: no metadata (line numbers, timestamps); keep LaTeX fence-free.
    """

    directive_text = dspy.InputField(desc="Raw axiom directive text.")
    context_hints = dspy.InputField(desc="Optional nearby context snippet.", optional=True)

    label_str = dspy.OutputField(desc="Directive label.")
    title_str = dspy.OutputField(desc="Title if present.")
    axiom_class_str = dspy.OutputField(desc="Classification (structural, regularity, etc.).")
    nl_summary_str = dspy.OutputField(desc="Concise natural-language summary.")

    core_statement_json = dspy.OutputField(
        desc='JSON object {"text": str|null, "latex": str|null}'
    )
    hypotheses_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    implications_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    parameters_json = dspy.OutputField(
        desc='JSON array [{"symbol": str, "description": str|null, "constraints": [str,...]}, ...]'
    )
    references_json = dspy.OutputField(desc='JSON array of strings ["def-...", "thm-...", ...]')
    failure_modes_json = dspy.OutputField(
        desc='JSON array [{"description": str|null, "impact": str|null}, ...]'
    )
    tags_json = dspy.OutputField(
        desc='JSON array of 3-10 keyword strings for search (e.g., ["regularity","sobolev","geometric"]).'
    )


# --------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------

_URI_PAT = URI_PATTERN
_META_PAT = METADATA_PATTERN
_FENCE_PAT = LATEX_FENCE_PATTERN
_LABEL_PAT = re.compile(r"^(def-axiom|axiom)-[a-z0-9-]+$")


def _json_loads(payload: str | None, default):
    if not payload or not payload.strip():
        return default
    try:
        return json.loads(payload)
    except Exception:
        return default


def _nonempty(value: str | None) -> bool:
    return bool(value and value.strip())


def _no_fences(latex: str | None) -> bool:
    return not _FENCE_PAT.search(latex or "")


def _reasonable_text_len(text: str | None, min_words: int = 5, max_chars: int = 600) -> bool:
    if not _nonempty(text):
        return False
    words = len(text.strip().split())
    return (words >= min_words) and (len(text) <= max_chars)


def _has_dups(seq) -> bool:
    seen = set()
    for item in seq or []:
        if item in seen:
            return True
        seen.add(item)
    return False


def axiom_reward(args: dict[str, Any], pred) -> float:
    """
    Reward function for ParseAxiomDirectiveSplit outputs.
    """

    label_str = (getattr(pred, "label_str", None) or "").strip()
    title_str = (getattr(pred, "title_str", None) or "").strip()
    axiom_class_str = (getattr(pred, "axiom_class_str", None) or "").strip().lower()
    nl_summary_str = (getattr(pred, "nl_summary_str", None) or "").strip()

    core_statement = _json_loads(getattr(pred, "core_statement_json", None), {})
    hypotheses = _json_loads(getattr(pred, "hypotheses_json", None), [])
    implications = _json_loads(getattr(pred, "implications_json", None), [])
    parameters = _json_loads(getattr(pred, "parameters_json", None), [])
    references = _json_loads(getattr(pred, "references_json", None), [])
    failure_modes = _json_loads(getattr(pred, "failure_modes_json", None), [])
    tags = _json_loads(getattr(pred, "tags_json", None), [])

    score = 0.0

    # 1) Label/title/classification (0.20)
    part = 0.0
    if _LABEL_PAT.match(label_str):
        part += 0.10
    if _nonempty(title_str):
        part += 0.05
    if _nonempty(axiom_class_str):
        part += 0.05
    score += min(part, 0.20)

    # 2) Summary (0.20)
    part = 0.0
    if _reasonable_text_len(nl_summary_str):
        part += 0.15
    if not _META_PAT.search(nl_summary_str):
        part += 0.05
    score += min(part, 0.20)

    # 3) Core statement (0.15)
    part = 0.0
    if isinstance(core_statement, dict):
        if _nonempty(core_statement.get("text")) or _nonempty(core_statement.get("latex")):
            part += 0.10
            latex = core_statement.get("latex")
            if latex is None or _no_fences(latex):
                part += 0.05
    score += min(part, 0.15)

    # 4) Hypotheses + Implications (0.20)
    part = 0.0
    if isinstance(hypotheses, list):
        part += 0.05
        valid = True
        for entry in hypotheses[:12]:
            if not isinstance(entry, dict):
                valid = False
                break
            text = entry.get("text")
            latex = entry.get("latex")
            if not (_nonempty(text) or _nonempty(latex)):
                valid = False
                break
            if latex and not _no_fences(latex):
                valid = False
                break
        if valid:
            part += 0.05
    if isinstance(implications, list):
        part += 0.05
        valid = True
        for entry in implications[:12]:
            if not isinstance(entry, dict):
                valid = False
                break
            text = entry.get("text")
            latex = entry.get("latex")
            if not (_nonempty(text) or _nonempty(latex)):
                valid = False
                break
            if latex and not _no_fences(latex):
                valid = False
                break
        if valid:
            part += 0.05
    score += min(part, 0.20)

    # 5) Parameters (0.10)
    part = 0.0
    if isinstance(parameters, list):
        part += 0.03
        valid = True
        for param in parameters[:12]:
            if not isinstance(param, dict) or not _nonempty(param.get("symbol")):
                valid = False
                break
            constraints = param.get("constraints")
            if constraints is not None and not isinstance(constraints, list):
                valid = False
                break
        if valid:
            part += 0.07
    score += min(part, 0.10)

    # 6) References (0.05)
    part = 0.0
    if isinstance(references, list) and all(isinstance(ref, str) for ref in references):
        part += 0.03
        if not _has_dups(references):
            part += 0.02
    score += min(part, 0.05)

    # 7) Failure modes (0.05)
    part = 0.0
    if isinstance(failure_modes, list):
        part += 0.02
        if all(
            isinstance(entry, dict) and _nonempty(entry.get("description"))
            for entry in failure_modes[:8]
        ):
            part += 0.03
    score += min(part, 0.05)

    # 8) Tags / keywords (0.05)
    part = 0.0
    if isinstance(tags, list):
        part += 0.02
        if 3 <= len(tags) <= 10 and all(isinstance(tag, str) and tag.strip() for tag in tags):
            part += 0.03
    score += min(part, 0.05)

    # 9) Anti-metadata bonus (0.05)
    part = 0.05
    text_blobs = [label_str, title_str, axiom_class_str, nl_summary_str]
    for bucket in (hypotheses, implications, failure_modes):
        for obj in bucket if isinstance(bucket, list) else []:
            if isinstance(obj, dict):
                for key in ("text", "latex", "description", "impact"):
                    val = obj.get(key)
                    if isinstance(val, str):
                        text_blobs.append(val)
            elif isinstance(obj, str):
                text_blobs.append(obj)
    for tag in tags if isinstance(tags, list) else []:
        if isinstance(tag, str):
            text_blobs.append(tag)
    for blob in text_blobs:
        if _URI_PAT.search(blob or "") or _META_PAT.search(blob or ""):
            part -= 0.01
    part = max(0.0, part)
    score += min(part, 0.05)

    return max(0.0, min(score, 1.0))


# --------------------------------------------------------------------------------------
# Assemblers
# --------------------------------------------------------------------------------------


def assemble_output(res) -> dict[str, Any]:
    """Convert DSPy result into JSON-friendly dict."""

    def as_json(field, default):
        try:
            return json.loads(field) if field else default
        except Exception:
            return default

    return {
        "label": (res.label_str or "").strip() or None,
        "title": (res.title_str or "").strip() or None,
        "axiom_class": (res.axiom_class_str or "").strip() or None,
        "nl_summary": (res.nl_summary_str or "").strip() or None,
        "core_statement": as_json(res.core_statement_json, {}),
        "hypotheses": as_json(res.hypotheses_json, []),
        "implications": as_json(res.implications_json, []),
        "parameters": as_json(res.parameters_json, []),
        "references": as_json(res.references_json, []),
        "failure_modes": as_json(res.failure_modes_json, []),
        "tags": as_json(res.tags_json, []),
    }


# --------------------------------------------------------------------------------------
# Main agent
# --------------------------------------------------------------------------------------


def run_agent(
    document_path: str | Path,
    lm_spec: str = "openai/gpt-4o-mini",
    passes: int = 5,
    threshold: float = 0.9,
    max_tokens: int = 16000,
) -> None:
    """
    Execute the axiom extraction pipeline for a specific document.
    """

    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=max_tokens)

    program = dspy.Refine(
        module=dspy.Predict(ParseAxiomDirectiveSplit),
        N=passes,
        reward_fn=axiom_reward,
        threshold=threshold,
    )

    fallback_predict = dspy.Predict(ParseAxiomDirectiveSplit)
    doc_path = Path(document_path)
    paths = DirectiveAgentPaths.build(doc_path, directive_basename="axiom")

    run_directive_extraction_loop(
        paths=paths,
        program_call=program,
        assemble_output=assemble_output,
        directive_type_fallback="axiom",
        fallback_call=fallback_predict,
        logger_obj=logger,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse axiom directives via DSPy Refine.")
    parser.add_argument("--doc", required=True, help="Path to the document markdown file.")
    parser.add_argument(
        "--lm",
        default="gemini/gemini-flash-lite-latest",
        help="LM spec for DSPy inference.",
    )
    parser.add_argument("--passes", type=int, default=5, help="Number of refine passes.")
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="Reward threshold for early stop."
    )
    args = parser.parse_args(argv)

    run_agent(
        document_path=args.doc,
        lm_spec=args.lm,
        passes=args.passes,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
