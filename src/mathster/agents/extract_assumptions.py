#!/usr/bin/env python3

"""
Agent for parsing `::{prf:assumption}` directives into structured JSON objects.

Usage
-----
python extract_assumptions.py \
  --doc docs/source/2_geometric_gas/16_convergence_mean_field.md \
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


class ParseAssumptionDirectiveSplit(dspy.Signature):
    """
    Convert a `::{prf:assumption}` directive into structured analysis artifacts.

    INPUT
    -----
    - directive_text (str): Exact directive body (header/title/body/closing fence).
    - context_hints (str, optional): Tiny snippet of nearby prose (≤ ~320 chars).

    OUTPUT FIELDS
    -------------
    - label_str (str):         Directive label (assump-*).
    - title_str (str):         Short title/headline if present.
    - scope_str (str):         Scope classification (global/local/model/regime/etc.).
    - nl_summary_str (str):    A concise natural-language summary (1–3 sentences).

    - bullet_items_json (json array):
        [{"name": <string|null>, "text": <string|null>, "latex": <string|null>} ...]
        representing enumerated assumptions or clauses.

    - conditions_json (json array):
        [{"type": <string|null>, "text": <string|null>, "latex": <string|null>} ...]
        for inequalities, bounds, or cases.

    - parameters_json (json array):
        [{"symbol": <string>, "description": <string|null>, "constraints": [<string>, ...]}, ...]

    - references_json (json array of str):
        Labels cited in the assumption (definitions, theorems, axioms, etc.).

    - notes_json (json array):
        [{"type": <string|null>, "text": <string|null>} ...] capturing remarks, motivations,
        or qualitative commentary embedded in the directive.

    Rules: omit metadata, keep LaTeX fence-free, and only include content present
    in the directive or the supplied context window.
    """

    directive_text = dspy.InputField(desc="Raw assumption directive text.")
    context_hints = dspy.InputField(desc="Optional nearby prose.", optional=True)

    label_str = dspy.OutputField(desc="Directive label (`assump-*`).")
    title_str = dspy.OutputField(desc="Title/heading if present.")
    scope_str = dspy.OutputField(desc="Scope classification (global/local/model/etc.).")
    nl_summary_str = dspy.OutputField(desc="Concise natural-language summary.")

    bullet_items_json = dspy.OutputField(
        desc='JSON array [{"name": str|null, "text": str|null, "latex": str|null}, ...]'
    )
    conditions_json = dspy.OutputField(
        desc='JSON array [{"type": str|null, "text": str|null, "latex": str|null}, ...]'
    )
    parameters_json = dspy.OutputField(
        desc='JSON array [{"symbol": str, "description": str|null, "constraints": [str,...]}, ...]'
    )
    references_json = dspy.OutputField(desc='JSON array of strings ["ax-1.2","def-fit",...]')
    notes_json = dspy.OutputField(desc='JSON array [{"type": str|null, "text": str|null}, ...]')
    tags_json = dspy.OutputField(
        desc='JSON array of 3-10 keyword strings for search (e.g., ["mass","confiment","ldp"]).'
    )


# --------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------

_URI_PAT = URI_PATTERN
_META_PAT = METADATA_PATTERN
_FENCE_PAT = LATEX_FENCE_PATTERN
_LABEL_PAT = re.compile(r"^assump-[a-z0-9-]+$")


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


def assumption_reward(args: dict[str, Any], pred) -> float:
    """
    Reward scoring for ParseAssumptionDirectiveSplit outputs.
    """

    label_str = (getattr(pred, "label_str", None) or "").strip()
    title_str = (getattr(pred, "title_str", None) or "").strip()
    scope_str = (getattr(pred, "scope_str", None) or "").strip().lower()
    nl_summary_str = (getattr(pred, "nl_summary_str", None) or "").strip()

    bullet_items = _json_loads(getattr(pred, "bullet_items_json", None), [])
    conditions = _json_loads(getattr(pred, "conditions_json", None), [])
    parameters = _json_loads(getattr(pred, "parameters_json", None), [])
    references = _json_loads(getattr(pred, "references_json", None), [])
    notes = _json_loads(getattr(pred, "notes_json", None), [])
    tags = _json_loads(getattr(pred, "tags_json", None), [])

    score = 0.0

    # 1) Label/title/scope (0.20)
    part = 0.0
    if _LABEL_PAT.match(label_str):
        part += 0.10
    if _nonempty(title_str):
        part += 0.05
    if _nonempty(scope_str):
        part += 0.05
    score += min(part, 0.20)

    # 2) Summary (0.20)
    part = 0.0
    if _reasonable_text_len(nl_summary_str):
        part += 0.15
    if not _META_PAT.search(nl_summary_str):
        part += 0.05
    score += min(part, 0.20)

    # 3) Bullet items (0.20)
    part = 0.0
    if isinstance(bullet_items, list):
        part += 0.05
        valid = True
        for bullet in bullet_items[:12]:
            if not isinstance(bullet, dict):
                valid = False
                break
            text = bullet.get("text")
            latex = bullet.get("latex")
            if not (_nonempty(text) or _nonempty(latex)):
                valid = False
                break
            if latex and not _no_fences(latex):
                valid = False
                break
        if valid:
            part += 0.15
    score += min(part, 0.20)

    # 4) Conditions (0.15)
    part = 0.0
    if isinstance(conditions, list):
        part += 0.05
        valid = True
        for cond in conditions[:12]:
            if not isinstance(cond, dict):
                valid = False
                break
            text = cond.get("text")
            latex = cond.get("latex")
            if not (_nonempty(text) or _nonempty(latex)):
                valid = False
                break
            if latex and not _no_fences(latex):
                valid = False
                break
        if valid:
            part += 0.10
    score += min(part, 0.15)

    # 5) Parameters (0.10)
    part = 0.0
    if isinstance(parameters, list):
        part += 0.03
        valid = True
        for entry in parameters[:12]:
            if not isinstance(entry, dict) or not _nonempty(entry.get("symbol")):
                valid = False
                break
            constraints = entry.get("constraints")
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

    # 7) Notes (0.05)
    part = 0.0
    if isinstance(notes, list):
        part += 0.02
        if all(
            (isinstance(note, dict) and _nonempty(note.get("text")))
            or (isinstance(note, str) and _nonempty(note))
            for note in notes[:8]
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
    text_blobs = [label_str, title_str, scope_str, nl_summary_str]
    for bucket in (bullet_items, conditions, notes):
        for obj in bucket if isinstance(bucket, list) else []:
            if isinstance(obj, dict):
                for key in ("name", "text", "latex", "description"):
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
    """
    Convert DSPy outputs into the final JSON-friendly dictionary.
    """

    def as_json(field, default):
        try:
            return json.loads(field) if field else default
        except Exception:
            return default

    return {
        "label": (res.label_str or "").strip() or None,
        "title": (res.title_str or "").strip() or None,
        "scope": (res.scope_str or "").strip() or None,
        "nl_summary": (res.nl_summary_str or "").strip() or None,
        "bullet_items": as_json(res.bullet_items_json, []),
        "conditions": as_json(res.conditions_json, []),
        "parameters": as_json(res.parameters_json, []),
        "references": as_json(res.references_json, []),
        "notes": as_json(res.notes_json, []),
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
    Execute the Refine pipeline over assumption directives in the target document.
    """

    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=max_tokens)

    program = dspy.Refine(
        module=dspy.Predict(ParseAssumptionDirectiveSplit),
        N=passes,
        reward_fn=assumption_reward,
        threshold=threshold,
    )

    fallback_predict = dspy.Predict(ParseAssumptionDirectiveSplit)
    doc_path = Path(document_path)
    paths = DirectiveAgentPaths.build(doc_path, directive_basename="assumption")

    run_directive_extraction_loop(
        paths=paths,
        program_call=program,
        assemble_output=assemble_output,
        directive_type_fallback="assumption",
        fallback_call=fallback_predict,
        logger_obj=logger,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse assumption directives via DSPy Refine.")
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
