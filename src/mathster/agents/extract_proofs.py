#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import dspy
from dotenv import load_dotenv

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
    proof_status_str = dspy.OutputField(
        desc="Status: complete/sketch/omitted/by-reference."
    )
    strategy_summary_str = dspy.OutputField(
        desc="Short strategy description (1-2 sentences)."
    )

    conclusion_json = dspy.OutputField(
        desc='JSON object {"text": str|null, "latex": str|null}.'
    )
    assumptions_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...].'
    )
    steps_json = dspy.OutputField(
        desc='JSON array [{"order": int|null, "kind": str|null, "text": str|null, "latex": str|null, "references": [str,...], "derived_statement": str|null}, ...].'
    )
    key_equations_json = dspy.OutputField(
        desc='JSON array [{"label": str|null, "latex": str, "role": str|null}, ...].'
    )
    references_json = dspy.OutputField(
        desc='JSON array of labels ["thm-main", "lem-x-bound"].'
    )
    cases_json = dspy.OutputField(
        desc='JSON array [{"name": str|null, "condition": str|null, "summary": str|null}, ...].'
    )
    remarks_json = dspy.OutputField(
        desc='JSON array [{"type": str|null, "text": str|null}, ...].'
    )
    gaps_json = dspy.OutputField(
        desc='JSON array [{"description": str, "severity": str|null, "location_hint": str|null}, ...].'
    )


# --------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------

_URI_PAT = re.compile(r"(https?://|file://|s3://|gs://)")
_META_PAT = re.compile(r"\b(line|page|timestamp|uuid|sha256)\b", re.I)
_LABEL_PAT = re.compile(r"^proof-[a-z0-9-]+$")
_RESULT_LABEL_PAT = re.compile(r"^(thm|lem|prop|cor|claim|def|axiom|obs|remark|eq)-[a-z0-9-]+$")
_LINENO = re.compile(r"^\s*\d+:\s?")
_FENCE_PAT = re.compile(r"(\$\$|\\\[|\\\]|\\begin\{equation\}|\\end\{equation\})")

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


def proof_reward(args: Dict[str, Any], pred) -> float:
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
            isinstance(a, dict)
            and (_nonempty(a.get("text")) or _nonempty(a.get("latex")))
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

    # 9) Anti-metadata bonus (0.05)
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
    for blob in text_blobs:
        if _URI_PAT.search(blob or "") or _META_PAT.search(blob or ""):
            part -= 0.01
    part = max(0.0, part)
    score += min(part, 0.05)

    return max(0.0, min(score, 1.0))


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def strip_line_numbers(text: str) -> str:
    return "\n".join(_LINENO.sub("", line) for line in (text or "").splitlines())


def synthesize_directive_text(obj: Dict[str, Any]) -> str:
    dtype = (obj.get("directive_type") or "proof").strip()
    title = (obj.get("title") or "").strip()
    label = (obj.get("label") or obj.get("metadata", {}).get("label") or "").strip()

    header = f"::{{prf:{dtype}}} {title}".rstrip()
    label_line = f":label: {label}" if label else ""
    body = strip_line_numbers(obj.get("content") or "")

    parts = [header]
    if label_line:
        parts.append(label_line)
    if body:
        parts.append(body)
    parts.append(":::")
    return "\n\n".join(parts)


def extract_directive_text(obj: Dict[str, Any]) -> str:
    raw = obj.get("raw_directive")
    if isinstance(raw, str) and raw.strip():
        return strip_line_numbers(raw)
    return synthesize_directive_text(obj)


def tiny_context_hints(obj: Dict[str, Any], doc_text: str, window: int = 320) -> str:
    content = strip_line_numbers(obj.get("content") or "").strip()
    if content:
        if len(content) > window:
            head = content[: window // 2]
            tail = content[-window // 2 :]
            return head + "\n...\n" + tail
        return content

    key = (obj.get("label") or obj.get("title") or "").strip()
    if not key:
        return ""
    idx = doc_text.find(key)
    if idx == -1 and key:
        idx = doc_text.find(key[: min(len(key), 60)])
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(doc_text), idx + len(key) + window)
    return doc_text[start:end]


def load_json_or_jsonl(path: Path) -> Iterable[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
    else:
        obj = json.loads(text)
        obj = obj.get("items", obj)
        if isinstance(obj, list):
            for item in obj:
                yield item
        else:
            yield obj


def assemble_output(res) -> Dict[str, Any]:
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
    document_folder = doc_path.parent / str(doc_path.stem)
    proofs_path = document_folder / "registry" / "directives" / "proof.json"
    if not proofs_path.exists():
        logger.warning(f"No proof directives found at {proofs_path}")
        return

    doc_text = doc_path.read_text(encoding="utf-8")
    out_dir = document_folder / "extract"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "proof.json"
    outputs = []

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, obj in enumerate(load_json_or_jsonl(proofs_path), start=1):
            if not isinstance(obj, dict):
                continue

            directive_text = extract_directive_text(obj)
            context_hints = tiny_context_hints(obj, doc_text, window=320)

            try:
                res = program(directive_text=directive_text, context_hints=context_hints)
                logger.info(f"✓ Refine succeeded for proof #{idx}: {obj.get('label')}")
            except Exception:
                logger.warning(
                    "Refine failed for proof #%s (%s); falling back to single-pass Predict.",
                    idx,
                    obj.get("label"),
                )
                res = dspy.Predict(ParseProofDirectiveSplit)(
                    directive_text=directive_text, context_hints=context_hints
                )

            outputs.append(assemble_output(res))

        out_f.write(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n")

    logger.info(f"Wrote {len(outputs)} proofs to {out_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
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

