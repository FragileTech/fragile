#!/usr/bin/env python3

"""
Agent for capturing structured data from `::{prf:conjecture}` directives.

Usage
-----
python extract_conjectures.py \
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


class ParseConjectureDirectiveSplit(dspy.Signature):
    """
    Convert a `::{prf:conjecture}` directive into a structured representation.

    INPUT
    -----
    - directive_text (str): directive body (header→closing fence).
    - context_hints (str, optional): short snippet of nearby prose.

    OUTPUT FIELDS
    -------------
    - label_str (str): Conjecture label (e.g., `conj-ldp-mean-field`).
    - title_str (str): Heading/title if present.
    - conjecture_type_str (str): Nature of conjecture (analytic/probabilistic/etc.).
    - status_str (str): Stated status (open/folklore/partial progress/etc.).
    - nl_summary_str (str): concise natural-language summary (1–3 sentences).

    - statement_json (json object):
        {"text": <string|null>, "latex": <string|null>}

    - evidence_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...] capturing heuristic support.

    - obstacles_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...] for known difficulties.

    - parameters_json (json array):
        [{"symbol": <string>, "description": <string|null>, "constraints": [<string>, ...]}, ...]

    - references_json (json array of str):
        Labels cited in the conjecture.

    - recommended_paths_json (json array):
        [{"text": <string|null>, "priority": <string|null>} ...] lines of attack if suggested.
    """

    directive_text = dspy.InputField(desc="Raw conjecture directive text.")
    context_hints = dspy.InputField(desc="Optional nearby prose.", optional=True)

    label_str = dspy.OutputField(desc="Conjecture label.")
    title_str = dspy.OutputField(desc="Title if present.")
    conjecture_type_str = dspy.OutputField(desc="Conjecture classification.")
    status_str = dspy.OutputField(desc="Status (open/partial/etc.).")
    nl_summary_str = dspy.OutputField(desc="Concise summary.")

    statement_json = dspy.OutputField(desc='JSON object {"text": str|null, "latex": str|null}')
    evidence_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    obstacles_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    parameters_json = dspy.OutputField(
        desc='JSON array [{"symbol": str, "description": str|null, "constraints": [str,...]}, ...]'
    )
    references_json = dspy.OutputField(desc='JSON array of strings ["thm-...", "assump-...", ...]')
    recommended_paths_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "priority": str|null}, ...]'
    )
    tags_json = dspy.OutputField(
        desc='JSON array of 3-10 keyword strings for search (e.g., ["lsi","ldp","mean-field"]).'
    )


# --------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------

_URI_PAT = URI_PATTERN
_META_PAT = METADATA_PATTERN
_FENCE_PAT = LATEX_FENCE_PATTERN
_LABEL_PAT = re.compile(r"^conj-[a-z0-9-]+$")


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


def conjecture_reward(args: dict[str, Any], pred) -> float:
    """
    Reward scoring for ParseConjectureDirectiveSplit outputs.
    """

    label_str = (getattr(pred, "label_str", None) or "").strip()
    title_str = (getattr(pred, "title_str", None) or "").strip()
    conj_type_str = (getattr(pred, "conjecture_type_str", None) or "").strip().lower()
    status_str = (getattr(pred, "status_str", None) or "").strip().lower()
    nl_summary_str = (getattr(pred, "nl_summary_str", None) or "").strip()

    statement = _json_loads(getattr(pred, "statement_json", None), {})
    evidence = _json_loads(getattr(pred, "evidence_json", None), [])
    obstacles = _json_loads(getattr(pred, "obstacles_json", None), [])
    parameters = _json_loads(getattr(pred, "parameters_json", None), [])
    references = _json_loads(getattr(pred, "references_json", None), [])
    recommended_paths = _json_loads(getattr(pred, "recommended_paths_json", None), [])
    tags = _json_loads(getattr(pred, "tags_json", None), [])

    score = 0.0

    # 1) Labels / type / status (0.20)
    part = 0.0
    if _LABEL_PAT.match(label_str):
        part += 0.10
    if _nonempty(conj_type_str):
        part += 0.05
    if _nonempty(status_str):
        part += 0.05
    score += min(part, 0.20)

    # 2) Summary (0.15)
    part = 0.0
    if _reasonable_text_len(nl_summary_str):
        part += 0.12
    if not _META_PAT.search(nl_summary_str):
        part += 0.03
    score += min(part, 0.15)

    # 3) Statement (0.15)
    part = 0.0
    if isinstance(statement, dict):
        text = statement.get("text")
        latex = statement.get("latex")
        if _nonempty(text) or _nonempty(latex):
            part += 0.10
            if latex is None or _no_fences(latex):
                part += 0.05
    score += min(part, 0.15)

    # 4) Evidence (0.15)
    part = 0.0
    if isinstance(evidence, list):
        part += 0.05
        valid = True
        for entry in evidence[:10]:
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
            part += 0.10
    score += min(part, 0.15)

    # 5) Obstacles (0.10)
    part = 0.0
    if isinstance(obstacles, list):
        part += 0.03
        if all(
            isinstance(entry, dict)
            and (_nonempty(entry.get("text")) or _nonempty(entry.get("latex")))
            for entry in obstacles[:10]
        ):
            part += 0.07
    score += min(part, 0.10)

    # 6) Parameters (0.10)
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

    # 7) References (0.05)
    part = 0.0
    if isinstance(references, list) and all(isinstance(ref, str) for ref in references):
        part += 0.03
        if not _has_dups(references):
            part += 0.02
    score += min(part, 0.05)

    # 8) Recommended paths (0.05)
    part = 0.0
    if isinstance(recommended_paths, list):
        part += 0.02
        if all(
            (isinstance(path, dict) and _nonempty(path.get("text")))
            or (isinstance(path, str) and _nonempty(path))
            for path in recommended_paths[:6]
        ):
            part += 0.03
    score += min(part, 0.05)

    # 9) Tags / keywords (0.05)
    part = 0.0
    if isinstance(tags, list):
        part += 0.02
        if 3 <= len(tags) <= 10 and all(isinstance(tag, str) and tag.strip() for tag in tags):
            part += 0.03
    score += min(part, 0.05)

    # 10) Anti-metadata bonus (0.05)
    part = 0.05
    text_blobs = [label_str, title_str, conj_type_str, status_str, nl_summary_str]
    for bucket in (evidence, obstacles, recommended_paths):
        for obj in bucket if isinstance(bucket, list) else []:
            if isinstance(obj, dict):
                for key in ("text", "latex", "priority"):
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
    """Recombine DSPy outputs into the final JSON object."""

    def as_json(field, default):
        try:
            return json.loads(field) if field else default
        except Exception:
            return default

    return {
        "label": (res.label_str or "").strip() or None,
        "title": (res.title_str or "").strip() or None,
        "conjecture_type": (res.conjecture_type_str or "").strip() or None,
        "status": (res.status_str or "").strip() or None,
        "nl_summary": (res.nl_summary_str or "").strip() or None,
        "statement": as_json(res.statement_json, {}),
        "evidence": as_json(res.evidence_json, []),
        "obstacles": as_json(res.obstacles_json, []),
        "parameters": as_json(res.parameters_json, []),
        "references": as_json(res.references_json, []),
        "recommended_paths": as_json(res.recommended_paths_json, []),
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
    Execute the conjecture extraction pipeline for a given document.
    """

    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=max_tokens)

    program = dspy.Refine(
        module=dspy.Predict(ParseConjectureDirectiveSplit),
        N=passes,
        reward_fn=conjecture_reward,
        threshold=threshold,
    )

    fallback_predict = dspy.Predict(ParseConjectureDirectiveSplit)
    doc_path = Path(document_path)
    paths = DirectiveAgentPaths.build(doc_path, directive_basename="conjecture")

    run_directive_extraction_loop(
        paths=paths,
        program_call=program,
        assemble_output=assemble_output,
        directive_type_fallback="conjecture",
        fallback_call=fallback_predict,
        logger_obj=logger,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse conjecture directives via DSPy Refine.")
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
