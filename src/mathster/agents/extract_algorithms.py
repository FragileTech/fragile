#!/usr/bin/env python3

"""
Agent for extracting structured content from `::{prf:algorithm}` directives.

Usage
-----
python extract_algorithms.py \
  --doc docs/source/1_euclidean_gas/03_cloning.md \
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


class ParseAlgorithmDirectiveSplit(dspy.Signature):
    """
    Convert a `::{prf:algorithm}` directive into a structured representation.

    INPUT
    -----
    - directive_text (str): Algorithm directive text (header/title/body/fence).
    - context_hints (str, optional): Nearby prose snippet for added context.

    OUTPUT FIELDS
    -------------
    - label_str (str): Algorithm label (`alg-*`).
    - title_str (str): Algorithm name/title.
    - complexity_str (str): Claimed complexity class if mentioned (O(n log n), etc.).
    - nl_summary_str (str): Brief natural-language description.

    - signature_json (json object):
        {"input": [<string>, ...], "output": [<string>, ...], "parameters": [<string>, ...]}

    - steps_json (json array):
        [{"order": <int|null>, "text": <string|null>, "latex": <string|null>, "comment": <string|null>} ...]

    - guard_conditions_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...] (preconditions, invariants).

    - references_json (json array of str):
        Labels cited in the algorithm.

    - failure_modes_json (json array):
        [{"description": <string|null>, "impact": <string|null>} ...]
    """

    directive_text = dspy.InputField(desc="Raw algorithm directive text.")
    context_hints = dspy.InputField(desc="Optional context window.", optional=True)

    label_str = dspy.OutputField(desc="Algorithm label.")
    title_str = dspy.OutputField(desc="Title if present.")
    complexity_str = dspy.OutputField(desc="Complexity classification.")
    nl_summary_str = dspy.OutputField(desc="Concise summary of the algorithm.")

    signature_json = dspy.OutputField(
        desc='JSON object {"input": [str,...], "output": [str,...], "parameters": [str,...]}'
    )
    steps_json = dspy.OutputField(
        desc='JSON array [{"order": int|null, "text": str|null, "latex": str|null, "comment": str|null}, ...]'
    )
    guard_conditions_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    references_json = dspy.OutputField(desc='JSON array of strings ["def-...", "thm-...", ...]')
    failure_modes_json = dspy.OutputField(
        desc='JSON array [{"description": str|null, "impact": str|null}, ...]'
    )


# --------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------

_URI_PAT = URI_PATTERN
_META_PAT = METADATA_PATTERN
_FENCE_PAT = LATEX_FENCE_PATTERN
_LABEL_PAT = re.compile(r"^alg-[a-z0-9-]+$")


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


def algorithm_reward(args: dict[str, Any], pred) -> float:
    """
    Reward scoring for ParseAlgorithmDirectiveSplit outputs.
    """

    label_str = (getattr(pred, "label_str", None) or "").strip()
    title_str = (getattr(pred, "title_str", None) or "").strip()
    complexity_str = (getattr(pred, "complexity_str", None) or "").strip().lower()
    nl_summary_str = (getattr(pred, "nl_summary_str", None) or "").strip()

    signature = _json_loads(getattr(pred, "signature_json", None), {})
    steps = _json_loads(getattr(pred, "steps_json", None), [])
    guards = _json_loads(getattr(pred, "guard_conditions_json", None), [])
    references = _json_loads(getattr(pred, "references_json", None), [])
    failure_modes = _json_loads(getattr(pred, "failure_modes_json", None), [])

    score = 0.0

    # 1) Label/title/complexity (0.20)
    part = 0.0
    if _LABEL_PAT.match(label_str):
        part += 0.10
    if _nonempty(title_str):
        part += 0.05
    if _nonempty(complexity_str):
        part += 0.05
    score += min(part, 0.20)

    # 2) Summary (0.15)
    part = 0.0
    if _reasonable_text_len(nl_summary_str):
        part += 0.12
    if not _META_PAT.search(nl_summary_str):
        part += 0.03
    score += min(part, 0.15)

    # 3) Signature (0.10)
    part = 0.0
    if isinstance(signature, dict):
        inputs = signature.get("input")
        outputs = signature.get("output")
        params = signature.get("parameters")
        if all(isinstance(x, list) for x in (inputs, outputs, params)):
            part += 0.10
    score += min(part, 0.10)

    # 4) Steps (0.25)
    part = 0.0
    if isinstance(steps, list):
        part += 0.05
        valid = True
        for entry in steps[:50]:
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
            part += 0.20
    score += min(part, 0.25)

    # 5) Guard conditions (0.10)
    part = 0.0
    if isinstance(guards, list):
        part += 0.03
        valid = True
        for entry in guards[:10]:
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

    # 8) Anti-metadata bonus (0.10)
    part = 0.10
    text_blobs = [label_str, title_str, nl_summary_str]
    for bucket in (steps, guards, failure_modes):
        for obj in bucket if isinstance(bucket, list) else []:
            if isinstance(obj, dict):
                for key in ("text", "latex", "comment", "description", "impact"):
                    val = obj.get(key)
                    if isinstance(val, str):
                        text_blobs.append(val)
            elif isinstance(obj, str):
                text_blobs.append(obj)
    for blob in text_blobs:
        if _URI_PAT.search(blob or "") or _META_PAT.search(blob or ""):
            part -= 0.02
    part = max(0.0, part)
    score += min(part, 0.10)

    return max(0.0, min(score, 1.0))


# --------------------------------------------------------------------------------------
# Assemblers
# --------------------------------------------------------------------------------------


def assemble_output(res) -> dict[str, Any]:
    """Recombine DSPy outputs into the JSON we persist."""

    def as_json(field, default):
        try:
            return json.loads(field) if field else default
        except Exception:
            return default

    return {
        "label": (res.label_str or "").strip() or None,
        "title": (res.title_str or "").strip() or None,
        "complexity": (res.complexity_str or "").strip() or None,
        "nl_summary": (res.nl_summary_str or "").strip() or None,
        "signature": as_json(res.signature_json, {}),
        "steps": as_json(res.steps_json, []),
        "guard_conditions": as_json(res.guard_conditions_json, []),
        "references": as_json(res.references_json, []),
        "failure_modes": as_json(res.failure_modes_json, []),
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
    Execute the algorithm extraction pipeline.
    """

    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=max_tokens)

    program = dspy.Refine(
        module=dspy.Predict(ParseAlgorithmDirectiveSplit),
        N=passes,
        reward_fn=algorithm_reward,
        threshold=threshold,
    )

    fallback_predict = dspy.Predict(ParseAlgorithmDirectiveSplit)
    doc_path = Path(document_path)
    paths = DirectiveAgentPaths.build(doc_path, directive_basename="algorithm")

    run_directive_extraction_loop(
        paths=paths,
        program_call=program,
        assemble_output=assemble_output,
        directive_type_fallback="algorithm",
        fallback_call=fallback_predict,
        logger_obj=logger,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse algorithm directives via DSPy Refine.")
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
