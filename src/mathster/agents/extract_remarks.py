#!/usr/bin/env python3

"""
Agent for extracting structured data from `::{prf:remark}` directives.

Usage
-----
python extract_remarks.py \
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


class ParseRemarkDirectiveSplit(dspy.Signature):
    """
    Convert a `::{prf:remark}` directive into structured content.

    INPUT
    -----
    - directive_text (str): directive block (header → closing fence).
    - context_hints (str, optional): small nearby snippet for extra cues.

    OUTPUT FIELDS
    -------------
    - label_str (str): Remark label (e.g., `rem-margin-stability`).
    - title_str (str): Heading if present.
    - remark_type_str (str): Classification (motivation/design choice/caveat/etc.).
    - nl_summary_str (str): concise summary (1–3 sentences).

    - key_points_json (json array):
        [{"text": <string|null>, "latex": <string|null>, "importance": <string|null>} ...]

    - quantitative_notes_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...] capturing bounds/equations.

    - references_json (json array of str):
        Labels cited inside the remark.

    - recommendations_json (json array):
        [{"text": <string|null>, "severity": <string|null>} ...] for actionable advice.

    - dependencies_json (json array of str):
        Concepts/sections the remark depends on or clarifies.
    """

    directive_text = dspy.InputField(desc="Raw remark directive text.")
    context_hints = dspy.InputField(desc="Optional nearby prose.", optional=True)

    label_str = dspy.OutputField(desc="Remark label.")
    title_str = dspy.OutputField(desc="Title if present.")
    remark_type_str = dspy.OutputField(desc="Remark classification.")
    nl_summary_str = dspy.OutputField(desc="Natural-language summary.")

    key_points_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null, "importance": str|null}, ...]'
    )
    quantitative_notes_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    references_json = dspy.OutputField(desc='JSON array of strings ["thm-...", "def-...", ...]')
    recommendations_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "severity": str|null}, ...]'
    )
    dependencies_json = dspy.OutputField(desc='JSON array of strings ["sec-2.1","rem-other",...]')
    tags_json = dspy.OutputField(
        desc='JSON array of 3-10 keyword strings for search (e.g., ["design-choice","stability"]).'
    )


# --------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------

_URI_PAT = URI_PATTERN
_META_PAT = METADATA_PATTERN
_FENCE_PAT = LATEX_FENCE_PATTERN
_LABEL_PAT = re.compile(r"^rem-[a-z0-9-]+$")


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


def remark_reward(args: dict[str, Any], pred) -> float:
    """
    Reward function for ParseRemarkDirectiveSplit outputs.
    """

    label_str = (getattr(pred, "label_str", None) or "").strip()
    title_str = (getattr(pred, "title_str", None) or "").strip()
    remark_type_str = (getattr(pred, "remark_type_str", None) or "").strip().lower()
    nl_summary_str = (getattr(pred, "nl_summary_str", None) or "").strip()

    key_points = _json_loads(getattr(pred, "key_points_json", None), [])
    quantitative_notes = _json_loads(getattr(pred, "quantitative_notes_json", None), [])
    references = _json_loads(getattr(pred, "references_json", None), [])
    recommendations = _json_loads(getattr(pred, "recommendations_json", None), [])
    dependencies = _json_loads(getattr(pred, "dependencies_json", None), [])
    tags = _json_loads(getattr(pred, "tags_json", None), [])

    score = 0.0

    # 1) Label/title/type (0.20)
    part = 0.0
    if _LABEL_PAT.match(label_str):
        part += 0.10
    if _nonempty(title_str):
        part += 0.05
    if _nonempty(remark_type_str):
        part += 0.05
    score += min(part, 0.20)

    # 2) Summary (0.20)
    part = 0.0
    if _reasonable_text_len(nl_summary_str):
        part += 0.15
    if not _META_PAT.search(nl_summary_str):
        part += 0.05
    score += min(part, 0.20)

    # 3) Key points (0.20)
    part = 0.0
    if isinstance(key_points, list):
        part += 0.05
        valid = True
        for item in key_points[:10]:
            if not isinstance(item, dict):
                valid = False
                break
            text = item.get("text")
            latex = item.get("latex")
            if not (_nonempty(text) or _nonempty(latex)):
                valid = False
                break
            if latex and not _no_fences(latex):
                valid = False
                break
        if valid:
            part += 0.15
    score += min(part, 0.20)

    # 4) Quantitative notes (0.15)
    part = 0.0
    if isinstance(quantitative_notes, list):
        part += 0.05
        valid = True
        for note in quantitative_notes[:10]:
            if not isinstance(note, dict):
                valid = False
                break
            text = note.get("text")
            latex = note.get("latex")
            if not (_nonempty(text) or _nonempty(latex)):
                valid = False
                break
            if latex and not _no_fences(latex):
                valid = False
                break
        if valid:
            part += 0.10
    score += min(part, 0.15)

    # 5) Recommendations (0.10)
    part = 0.0
    if isinstance(recommendations, list):
        part += 0.03
        if all(
            (isinstance(rec, dict) and _nonempty(rec.get("text")))
            or (isinstance(rec, str) and _nonempty(rec))
            for rec in recommendations[:8]
        ):
            part += 0.07
    score += min(part, 0.10)

    # 6) References (0.05)
    part = 0.0
    if isinstance(references, list) and all(isinstance(ref, str) for ref in references):
        part += 0.03
        if not _has_dups(references):
            part += 0.02
    score += min(part, 0.05)

    # 7) Dependencies (0.05)
    part = 0.0
    if isinstance(dependencies, list) and all(isinstance(dep, str) for dep in dependencies):
        part += 0.05
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
    text_blobs = [label_str, title_str, remark_type_str, nl_summary_str]
    for bucket in (key_points, quantitative_notes, recommendations):
        for obj in bucket if isinstance(bucket, list) else []:
            if isinstance(obj, dict):
                for key in ("text", "latex"):
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
    Recombine split outputs into the JSON blob stored on disk.
    """

    def as_json(field, default):
        try:
            return json.loads(field) if field else default
        except Exception:
            return default

    return {
        "label": (res.label_str or "").strip() or None,
        "title": (res.title_str or "").strip() or None,
        "remark_type": (res.remark_type_str or "").strip() or None,
        "nl_summary": (res.nl_summary_str or "").strip() or None,
        "key_points": as_json(res.key_points_json, []),
        "quantitative_notes": as_json(res.quantitative_notes_json, []),
        "references": as_json(res.references_json, []),
        "recommendations": as_json(res.recommendations_json, []),
        "dependencies": as_json(res.dependencies_json, []),
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
    Execute the remark extraction pipeline.
    """

    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=max_tokens)

    program = dspy.Refine(
        module=dspy.Predict(ParseRemarkDirectiveSplit),
        N=passes,
        reward_fn=remark_reward,
        threshold=threshold,
    )

    fallback_predict = dspy.Predict(ParseRemarkDirectiveSplit)
    doc_path = Path(document_path)
    paths = DirectiveAgentPaths.build(doc_path, directive_basename="remark")

    run_directive_extraction_loop(
        paths=paths,
        program_call=program,
        assemble_output=assemble_output,
        directive_type_fallback="remark",
        fallback_call=fallback_predict,
        logger_obj=logger,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse remark directives via DSPy Refine.")
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
