#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage
-----
python extract_definitions.py \
  --doc docs/source/1_euclidean_gas/07_mean_field.md \
  --lm openai/gpt-4o-mini \
  --passes 5 \
  --threshold 0.9

Input format
------------
definition.json comes from the directive registry (see
docs/source/<chapter>/<document>/registry/directives/definition.json) and
contains entries like:
{
  "directive_type": "definition",
  "label": "def-mean-field-phase-space",
  "title": "Phase Space",
  "start_line": 37, "end_line": 48,
  "content": "40: :label: def-mean-field-phase-space\\n41: ...",
  "raw_directive": "37: ...",
  ...
}

This script mirrors mathster/agents/extract_theorems.py but targets
`::{prf:definition}` directives. It assembles a compact structured object per
definition using a DSPy Refine program.
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
import flogging

flogging.setup(level="INFO")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------------------
# DSPy Signature
# --------------------------------------------------------------------------------------


class ParseDefinitionDirectiveSplit(dspy.Signature):
    """
    Transform a raw `::{prf:definition}` directive into a structured bundle.

    INPUT
    -----
    - directive_text (str): Verbatim directive block (header/title/body + fences).
    - context_hints (str, optional): Small snippet of nearby prose helpful for
      inferring the scope or motivation of the definition.

    OUTPUT FIELDS
    -------------
    - label_str (str):      Definition label (must match def-* pattern if present).
    - term_str (str):       Canonical term being defined.
    - object_type_str (str):Category of object (set/function/operator/process/other).
    - nl_definition_str (str): Short natural-language paraphrase.

    - formal_conditions_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...] capturing displayed
        equations or bulletized criteria.

    - properties_json (json array):
        [{"name": <string|null>, "description": <string|null>}, ...] summarizing
        qualitative properties or requirements.

    - parameters_json (json array):
        [{"symbol": <string>, "description": <string|null>, "constraints": [<string>, ...]}, ...]
        listing symbols introduced or constrained by the definition.

    - examples_json (json array):
        [{"text": <string|null>, "latex": <string|null>} ...] illustrating canonical
        instances if the directive provides them.

    - related_refs_json (json array of str):
        ["def-other", "thm-foo", ...] capturing labels cited in the directive.

    - notes_json (json array):
        [{"type": <string|null>, "text": <string|null>} ...] for remarks, intuition,
        or usage guidance explicitly stated.

    Rules: omit metadata, keep LaTeX fence-free, only include content actually
    present in the directive or tiny context window.
    """

    directive_text = dspy.InputField(desc="Raw definition directive text (header/body).")
    context_hints = dspy.InputField(
        desc="Optional nearby prose for scope/motivation.", optional=True
    )

    label_str = dspy.OutputField(desc="Definition label (def-*).")
    term_str = dspy.OutputField(desc="Term being defined.")
    object_type_str = dspy.OutputField(
        desc="Category of mathematical object (set/function/operator/process/other)."
    )
    nl_definition_str = dspy.OutputField(desc="Concise natural-language paraphrase.")

    formal_conditions_json = dspy.OutputField(
        desc='JSON array [{"text": str|null, "latex": str|null}, ...]'
    )
    properties_json = dspy.OutputField(
        desc='JSON array [{"name": str|null, "description": str|null}, ...]'
    )
    parameters_json = dspy.OutputField(
        desc='JSON array [{"symbol": str, "description": str|null, "constraints": [str,...]}, ...]'
    )
    examples_json = dspy.OutputField(desc='JSON array [{"text": str|null, "latex": str|null}, ...]')
    related_refs_json = dspy.OutputField(desc='JSON array of label strings ["def-foo","thm-bar",...]')
    notes_json = dspy.OutputField(desc='JSON array [{"type": str|null, "text": str|null}, ...]')


# --------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------

_URI_PAT = re.compile(r"(https?://|file://|s3://|gs://)")
_META_PAT = re.compile(r"\b(line|page|timestamp|uuid|sha256)\b", re.I)
_LABEL_PAT = re.compile(r"^def-[a-z0-9-]+$")
_LINENO = re.compile(r"^\s*\d+:\s?")
_FENCE_PAT = re.compile(r"(\$\$|\\\[|\\\]|\\begin\{equation\}|\\end\{equation\})")
_ALLOWED_OBJECT_TYPES = {
    "set",
    "function",
    "operator",
    "measure",
    "process",
    "space",
    "vector",
    "scalar",
    "tensor",
    "manifold",
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


def definition_reward(args: Dict[str, Any], pred) -> float:
    """
    Reward function for ParseDefinitionDirectiveSplit outputs.
    Returns a float in [0, 1].
    """

    label_str = (getattr(pred, "label_str", None) or "").strip()
    term_str = (getattr(pred, "term_str", None) or "").strip()
    object_type_str = (getattr(pred, "object_type_str", None) or "").strip().lower()
    nl_definition_str = (getattr(pred, "nl_definition_str", None) or "").strip()

    formal_conditions = _json_loads(getattr(pred, "formal_conditions_json", None), [])
    properties = _json_loads(getattr(pred, "properties_json", None), [])
    parameters = _json_loads(getattr(pred, "parameters_json", None), [])
    examples = _json_loads(getattr(pred, "examples_json", None), [])
    related_refs = _json_loads(getattr(pred, "related_refs_json", None), [])
    notes = _json_loads(getattr(pred, "notes_json", None), [])

    score = 0.0
    max_score = 0.0

    # 1) Label + term + type (0.20)
    max_score += 0.20
    s1 = 0.0
    if _LABEL_PAT.match(label_str):
        s1 += 0.10
    if _nonempty(term_str):
        s1 += 0.06
    if object_type_str in _ALLOWED_OBJECT_TYPES:
        s1 += 0.04
    score += min(s1, 0.20)

    # 2) NL definition (0.20)
    max_score += 0.20
    s2 = 0.0
    if _reasonable_text_len(nl_definition_str):
        s2 += 0.15
    if not _META_PAT.search(nl_definition_str):
        s2 += 0.05
    score += min(s2, 0.20)

    # 3) Formal conditions (0.15)
    max_score += 0.15
    s3 = 0.0
    if isinstance(formal_conditions, list):
        s3 += 0.05
        if all(isinstance(cond, dict) and (_nonempty(cond.get("text")) or _nonempty(cond.get("latex"))) for cond in formal_conditions[:10]):
            s3 += 0.05
        if all(_no_fences(cond.get("latex")) for cond in formal_conditions if isinstance(cond, dict)):
            s3 += 0.05
    score += min(s3, 0.15)

    # 4) Parameters (0.15)
    max_score += 0.15
    s4 = 0.0
    if isinstance(parameters, list):
        s4 += 0.05
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
            s4 += 0.10
    score += min(s4, 0.15)

    # 5) Properties (0.10)
    max_score += 0.10
    s5 = 0.0
    if isinstance(properties, list):
        s5 += 0.04
        if all(
            isinstance(prop, dict)
            and (_nonempty(prop.get("name")) or _nonempty(prop.get("description")))
            for prop in properties[:10]
        ):
            s5 += 0.06
    score += min(s5, 0.10)

    # 6) Examples (0.05)
    max_score += 0.05
    s6 = 0.0
    if isinstance(examples, list):
        s6 += 0.02
        if all(
            isinstance(ex, dict)
            and (_nonempty(ex.get("text")) or _nonempty(ex.get("latex")))
            for ex in examples[:5]
        ):
            s6 += 0.02
        if all(_no_fences(ex.get("latex")) for ex in examples if isinstance(ex, dict)):
            s6 += 0.01
    score += min(s6, 0.05)

    # 7) Related references (0.05)
    max_score += 0.05
    s7 = 0.0
    if isinstance(related_refs, list) and all(isinstance(ref, str) for ref in related_refs):
        s7 += 0.03
        if not _has_dups(related_refs):
            s7 += 0.02
    score += min(s7, 0.05)

    # 8) Notes (0.05)
    max_score += 0.05
    s8 = 0.0
    if isinstance(notes, list):
        s8 += 0.02
        if all(
            (isinstance(note, dict) and (_nonempty(note.get("text"))))
            or (isinstance(note, str) and _nonempty(note))
            for note in notes[:6]
        ):
            s8 += 0.03
    score += min(s8, 0.05)

    # 9) Anti-metadata bonus (0.05)
    max_score += 0.05
    s9 = 0.05
    text_blobs = [label_str, term_str, nl_definition_str]
    for bucket in (formal_conditions, properties, parameters, examples, notes):
        for obj in bucket if isinstance(bucket, list) else []:
            if isinstance(obj, dict):
                for key in ("text", "latex", "description", "name"):
                    val = obj.get(key)
                    if isinstance(val, str):
                        text_blobs.append(val)
            elif isinstance(obj, str):
                text_blobs.append(obj)
    for blob in text_blobs:
        if _URI_PAT.search(blob or "") or _META_PAT.search(blob or ""):
            s9 -= 0.01
    s9 = max(0.0, s9)
    score += min(s9, 0.05)

    return max(0.0, min(score, 1.0))


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def strip_line_numbers(text: str) -> str:
    """Remove 'NNN: ' prefixes from line-numbered snippets."""
    return "\n".join(_LINENO.sub("", line) for line in (text or "").splitlines())


def synthesize_directive_text(obj: Dict[str, Any]) -> str:
    """
    Build a directive block if raw_directive is missing.
    """
    dtype = (obj.get("directive_type") or "definition").strip()
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
    """
    Provide a short snippet of nearby prose for additional hints.
    """
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
    """Recombine split outputs into a single dictionary."""

    def as_json(payload, default):
        try:
            return json.loads(payload) if payload else default
        except Exception:
            return default

    return {
        "label": (res.label_str or "").strip() or None,
        "term": (res.term_str or "").strip() or None,
        "object_type": (res.object_type_str or "").strip() or None,
        "nl_definition": (res.nl_definition_str or "").strip() or None,
        "formal_conditions": as_json(res.formal_conditions_json, []),
        "properties": as_json(res.properties_json, []),
        "parameters": as_json(res.parameters_json, []),
        "examples": as_json(res.examples_json, []),
        "related_refs": as_json(res.related_refs_json, []),
        "notes": as_json(res.notes_json, []),
    }


# --------------------------------------------------------------------------------------
# Main agent
# --------------------------------------------------------------------------------------


def run_agent(
    document_path: str | Path,
    lm_spec: str = "openai/gpt-4o-mini",
    passes: int = 5,
    threshold: float = 0.9,
    max_tokens: int = 12000,
) -> None:
    """
    Run DSPy refine loop over `definition.json`.
    """

    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=max_tokens)

    program = dspy.Refine(
        module=dspy.Predict(ParseDefinitionDirectiveSplit),
        N=passes,
        reward_fn=definition_reward,
        threshold=threshold,
    )

    doc_path = Path(document_path)
    document_folder = doc_path.parent / str(doc_path.stem)
    definitions_path = document_folder / "registry" / "directives" / "definition.json"
    if not definitions_path.exists():
        logger.warning(f"No definition directives found at {definitions_path}")
        return

    doc_text = doc_path.read_text(encoding="utf-8")
    out_dir = document_folder / "extract"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "definition.json"
    outputs = []

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, obj in enumerate(load_json_or_jsonl(definitions_path), start=1):
            if not isinstance(obj, dict):
                continue

            directive_text = extract_directive_text(obj)
            context_hints = tiny_context_hints(obj, doc_text, window=320)

            try:
                res = program(directive_text=directive_text, context_hints=context_hints)
                logger.info(f"✓ Refine succeeded for definition #{idx}: {obj.get('label')}")
            except Exception:
                logger.warning(
                    "Refine failed for definition #%s (%s); falling back to single-pass Predict.",
                    idx,
                    obj.get("label"),
                )
                res = dspy.Predict(ParseDefinitionDirectiveSplit)(
                    directive_text=directive_text, context_hints=context_hints
                )

            outputs.append(assemble_output(res))

        out_f.write(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n")

    logger.info(f"Wrote {len(outputs)} definitions to {out_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parse definition directives via DSPy Refine."
    )
    parser.add_argument(
        "--doc", required=True, help="Path to the document markdown file."
    )
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

