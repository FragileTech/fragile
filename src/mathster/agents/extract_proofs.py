#!/usr/bin/env python3

"""
Usage
-----
python extract_proofs.py \
  --doc docs/source/1_euclidean_gas/07_mean_field.md \
  --lm openai/gpt-4o-mini \
  --passes 5 \
  --threshold 0.9

Input format
------------
proof.json is produced by the directive registry and contains entries shaped like:
{
  "directive_type": "proof",
  "label": "proof-thm-main",
  "start_line": 200, "end_line": 260,
  "content": "202: :label: proof-thm-main\\n203: **Proof.** ...",
  "raw_directive": "200: :::{prf:proof} ... 260: :::",
  ...
}

This agent mirrors mathster/agents/extract_theorems.py but targets `::{prf:proof}`
blocks. It produces structured JSON capturing strategy, steps, assumptions, equations,
and references for downstream verification.
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


class ParseProofDirectiveSplit(dspy.Signature):
    """
    Convert a `::{prf:proof}` directive into a structured description.

    INPUT
    -----
    - directive_text (str): Raw directive (header/body/closing fence) with no unrelated text.
    - context_hints (str, optional): Tiny snippet of nearby prose for inferring missing cues.

    OUTPUT FIELDS
    -------------
    - label_str (str): Proof label (expected format `proof-*`).
    - proves_label_str (str): Label of the statement being proved (e.g., `thm-...`, `lem-...`).
    - proof_type_str (str): Dominant technique (direct, contradiction, induction, reference, other).
    - proof_status_str (str): `complete`, `sketch`, `omitted`, or `by-reference`.
    - strategy_summary_str (str): One- or two-sentence plan overview.

    - conclusion_json (json object):
        {"text": <string|null>, "latex": <string|null>}

    - assumptions_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...]

    - steps_json (json array):
        [{"order": <int|null>, "kind": <string|null>, "text": <string|null>,
          "latex": <string|null>, "references": [<string>, ...],
          "derived_statement": <string|null>} ...]

    - key_equations_json (json array):
        [{"label": <string|null>, "latex": <string>, "role": <string|null>} ...]

    - references_json (json array of strings):
        ["thm-aux", "lem-helper", ...]

    - cases_json (json array):
        [{"name": <string|null>, "condition": <string|null>, "summary": <string|null>} ...]

    - remarks_json (json array):
        [{"type": <string|null>, "text": <string|null>} ...]

    - gaps_json (json array):
        [{"description": <string>, "severity": <string|null>, "location_hint": <string|null>} ...]

    All LaTeX fragments must be fence-free; omit metadata like line numbers.
    """

    directive_text = dspy.InputField(desc="Raw proof directive text (header/body).")
    context_hints = dspy.InputField(desc="Optional nearby prose window.", optional=True)

    label_str = dspy.OutputField(desc="Proof label (`proof-*`).")
    proves_label_str = dspy.OutputField(desc="Label proved (e.g., `thm-main`).")
    proof_type_str = dspy.OutputField(
        desc="Technique: direct/contradiction/induction/reference/construction/variational/probabilistic/other."
    )
    proof_status_str = dspy.OutputField(desc="Status: complete/sketch/omitted/by-reference.")
    strategy_summary_str = dspy.OutputField(desc="Short strategy description (1-2 sentences).")

    conclusion_json = dspy.OutputField(desc='JSON object {"text": str|null, "latex": str|null}.')
    assumptions_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...].'
    )
    steps_json = dspy.OutputField(
        desc='JSON array [{"order": int|null, "kind": str|null, "text": str|null, "latex": str|null, "references": [str,...], "derived_statement": str|null}, ...].'
    )
    key_equations_json = dspy.OutputField(
        desc='JSON array [{"label": str|null, "latex": str, "role": str|null}, ...].'
    )
    references_json = dspy.OutputField(desc='JSON array of labels ["thm-main", "lem-x-bound"].')
    cases_json = dspy.OutputField(
        desc='JSON array [{"name": str|null, "condition": str|null, "summary": str|null}, ...].'
    )
    remarks_json = dspy.OutputField(desc='JSON array [{"type": str|null, "text": str|null}, ...].')
    gaps_json = dspy.OutputField(
        desc='JSON array [{"description": str, "severity": str|null, "location_hint": str|null}, ...].'
    )
    tags_json = dspy.OutputField(
        desc='JSON array of 3-10 keyword strings for search (e.g., ["mass-conservation","flux"]).'
    )


# --------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------

_URI_PAT = URI_PATTERN
_META_PAT = METADATA_PATTERN
_LABEL_PAT = re.compile(r"^proof-[a-z0-9-]+$")
_RESULT_LABEL_PAT = re.compile(r"^(thm|lem|prop|cor|claim|def|axiom|obs|remark|eq)-[a-z0-9-]+$")
_FENCE_PAT = LATEX_FENCE_PATTERN

_ALLOWED_PROOF_TYPES = {
    "direct",
    "contradiction",
    "induction",
    "construction",
    "variational",
    "probabilistic",
    "reference",
    "energy",
    "symmetry",
    "other",
}

_ALLOWED_STATUSES = {"complete", "sketch", "omitted", "by-reference"}

_STEP_KINDS = {
    "assume",
    "apply",
    "derive",
    "compute",
    "estimate",
    "case-start",
    "case-end",
    "induction-base",
    "induction-step",
    "contradiction",
    "conclude",
    "remark",
    "other",
}


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


def proof_reward(args: dict[str, Any], pred) -> float:
    """
    Reward function for ParseProofDirectiveSplit outputs.
    """

    label_str = (getattr(pred, "label_str", None) or "").strip()
    proves_label_str = (getattr(pred, "proves_label_str", None) or "").strip()
    proof_type_str = (getattr(pred, "proof_type_str", None) or "").strip().lower()
    proof_status_str = (getattr(pred, "proof_status_str", None) or "").strip().lower()
    strategy_summary_str = (getattr(pred, "strategy_summary_str", None) or "").strip()

    conclusion = _json_loads(getattr(pred, "conclusion_json", None), {})
    assumptions = _json_loads(getattr(pred, "assumptions_json", None), [])
    steps = _json_loads(getattr(pred, "steps_json", None), [])
    key_equations = _json_loads(getattr(pred, "key_equations_json", None), [])
    references = _json_loads(getattr(pred, "references_json", None), [])
    cases = _json_loads(getattr(pred, "cases_json", None), [])
    remarks = _json_loads(getattr(pred, "remarks_json", None), [])
    gaps = _json_loads(getattr(pred, "gaps_json", None), [])

    score = 0.0

    # 1) Labels/type/status (0.25)
    part = 0.0
    if _LABEL_PAT.match(label_str):
        part += 0.1
    if _RESULT_LABEL_PAT.match(proves_label_str):
        part += 0.08
    if proof_type_str in _ALLOWED_PROOF_TYPES:
        part += 0.04
    if proof_status_str in _ALLOWED_STATUSES:
        part += 0.03
    score += min(part, 0.25)

    # 2) Strategy summary (0.15)
    part = 0.0
    if _reasonable_text_len(strategy_summary_str):
        part += 0.12
    if not _META_PAT.search(strategy_summary_str):
        part += 0.03
    score += min(part, 0.15)

    # 3) Steps (0.20)
    part = 0.0
    if isinstance(steps, list):
        part += 0.05
        valid_steps = True
        for step in steps[:25]:
            if not isinstance(step, dict):
                valid_steps = False
                break
            kind = (step.get("kind") or "").strip().lower()
            if kind and kind not in _STEP_KINDS:
                valid_steps = False
                break
            latex = step.get("latex")
            if latex and not _no_fences(latex):
                valid_steps = False
                break
            refs = step.get("references")
            if refs is not None and not isinstance(refs, list):
                valid_steps = False
                break
        if valid_steps:
            part += 0.10
        if steps:
            part += 0.05
    score += min(part, 0.20)

    # 4) Key equations (0.10)
    part = 0.0
    if isinstance(key_equations, list):
        part += 0.03
        eq_ok = True
        for eq in key_equations[:15]:
            if not isinstance(eq, dict) or not _nonempty(eq.get("latex")):
                eq_ok = False
                break
            if not _no_fences(eq.get("latex")):
                eq_ok = False
                break
        if eq_ok:
            part += 0.07
    score += min(part, 0.10)

    # 5) Assumptions + conclusion (0.10)
    part = 0.0
    if isinstance(assumptions, list):
        part += 0.04
        if all(
            isinstance(a, dict) and (_nonempty(a.get("text")) or _nonempty(a.get("latex")))
            for a in assumptions[:10]
        ):
            part += 0.02
    if isinstance(conclusion, dict) and (
        _nonempty(conclusion.get("text")) or _nonempty(conclusion.get("latex"))
    ):
        part += 0.04
    score += min(part, 0.10)

    # 6) Cases + remarks (0.05)
    part = 0.0
    if isinstance(cases, list):
        part += 0.02
        if all(isinstance(case, dict) for case in cases[:6]):
            part += 0.01
    if isinstance(remarks, list):
        part += 0.01
        if all(isinstance(r, dict) and _nonempty(r.get("text")) for r in remarks[:6]):
            part += 0.01
    score += min(part, 0.05)

    # 7) References (0.05)
    part = 0.0
    if isinstance(references, list) and all(isinstance(r, str) for r in references):
        part += 0.03
        if not _has_dups(references):
            part += 0.02
    score += min(part, 0.05)

    # 8) Gaps (0.05)
    part = 0.0
    if isinstance(gaps, list):
        part += 0.02
        if all(isinstance(g, dict) and _nonempty(g.get("description")) for g in gaps[:6]):
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
    text_blobs = [label_str, proves_label_str, strategy_summary_str]
    for bucket in (assumptions, steps, key_equations, cases, remarks, gaps):
        for obj in bucket if isinstance(bucket, list) else []:
            if isinstance(obj, dict):
                for key in ("text", "latex", "description", "summary", "condition"):
                    val = obj.get(key)
                    if isinstance(val, str):
                        text_blobs.append(val)
            elif isinstance(obj, str):
                text_blobs.append(obj)
    if isinstance(conclusion, dict):
        for key in ("text", "latex"):
            val = conclusion.get(key)
            if isinstance(val, str):
                text_blobs.append(val)
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
# Utilities
# --------------------------------------------------------------------------------------
def assemble_output(res) -> dict[str, Any]:
    def as_json(payload, default):
        try:
            return json.loads(payload) if payload else default
        except Exception:
            return default

    return {
        "label": (res.label_str or "").strip() or None,
        "proves": (res.proves_label_str or "").strip() or None,
        "proof_type": (res.proof_type_str or "").strip() or None,
        "proof_status": (res.proof_status_str or "").strip() or None,
        "strategy_summary": (res.strategy_summary_str or "").strip() or None,
        "conclusion": as_json(res.conclusion_json, {}),
        "assumptions": as_json(res.assumptions_json, []),
        "steps": as_json(res.steps_json, []),
        "key_equations": as_json(res.key_equations_json, []),
        "references": as_json(res.references_json, []),
        "cases": as_json(res.cases_json, []),
        "remarks": as_json(res.remarks_json, []),
        "gaps": as_json(res.gaps_json, []),
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
    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=max_tokens)

    program = dspy.Refine(
        module=dspy.Predict(ParseProofDirectiveSplit),
        N=passes,
        reward_fn=proof_reward,
        threshold=threshold,
    )

    doc_path = Path(document_path)
    fallback_predict = dspy.Predict(ParseProofDirectiveSplit)
    paths = DirectiveAgentPaths.build(doc_path, directive_basename="proof")

    run_directive_extraction_loop(
        paths=paths,
        program_call=program,
        assemble_output=assemble_output,
        directive_type_fallback="proof",
        fallback_call=fallback_predict,
        logger_obj=logger,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    import flogging

    flogging.setup(level="INFO")
    parser = argparse.ArgumentParser(description="Parse proof directives via DSPy Refine.")
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
