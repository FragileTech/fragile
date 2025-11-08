#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage
-----
python run_theorem_agent.py \
  --doc full_document.md \
  --out parsed_theorems.jsonl \
  --lm openai/gpt-4o-mini \
  --passes 5 \
  --threshold 0.85

Input format
------------
theorems.json can be a JSON array or JSONL of objects shaped like the example the user provided:
{
  "directive_type": "theorem",
  "label": "thm-mean-field-equation",
  "title": "The Mean-Field Equations for the Euclidean Gas",
  "start_line": 614, "end_line": 651,
  "content": "617: ...",                     # with line-number prefixes
  "raw_directive": "614: :::{prf:theorem} ... 651: :::",
  "section": "## ...",
  ...
}

This script extracts/normalizes a directive block and runs dspy.Refine on the
ParseTheoremDirectiveSplit signature to emit *one* consolidated JSON object per theorem.
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import dspy
from dotenv import load_dotenv


# --------------------------------------------------------------------------------------
# 1) Your split-output signature (exactly as you defined it earlier)
#    If you already import it from your own module, delete this class and import instead.
# --------------------------------------------------------------------------------------
class ParseTheoremDirectiveSplit(dspy.Signature):
    """
    Transform one raw theorem directive (e.g., a `::{prf:theorem}` block) into
    a compact structured representation with SEPARATE outputs per top-level key.

    INPUT
    -----
    - directive_text (str): Verbatim directive body (header/title -> closing fence).
      Do NOT include unrelated context lines or other directives.
    - context_hints (str, optional): Small nearby text to help infer *implicit assumptions*.
      Keep short; do not pass whole documents.

    OUTPUT FIELDS (one per top-level key)
    -------------------------------------
    - type_str (str):           A single string, e.g., "theorem", "lemma", "proposition".
    - label_str (str):          The directive label if present (e.g., "thm-mean-field-equation").
    - title_str (str):          Human-facing title present in the directive, if any.
    - nl_statement_str (str):   Concise natural-language statement (no commentary).

    - equations_json (json):    JSON ARRAY of objects. Each object MUST be:
        [{"label": <string|null>, "latex": <string>}, ...]
      * Keep equations in their original order.
      * Remove math fences; preserve LaTeX faithfully.

    - hypotheses_json (json):   JSON ARRAY of objects:
        [{"text": <string>, "latex": <string|null>}, ...]
      * Only explicit hypotheses in the directive; omit if unknown.

    - conclusion_json (json):   JSON OBJECT:
        {"text": <string|null>, "latex": <string|null>}
      * Core claim/goal only.

    - variables_json (json):    JSON ARRAY of objects:
        [{"symbol": <string>, "role": <string|null>, "constraints": [<string>, ...]}, ...]
      * Include only if clearly stated in the directive.

    - implicit_assumptions_json (json): JSON ARRAY of objects:
        [{"text": <string>, "confidence": <number|null>}, ...]
      * Add only if directive wording or context_hints strongly suggests them.

    - local_refs_json (json):   JSON ARRAY of strings:
        ["lem-xyz", "eq-transport", ...]
      * Labels cited INSIDE this directive; do not resolve or expand.

    - proof_json (json):        JSON OBJECT:
        {
          "availability": "present" | "sketch" | "omitted" | "by-reference",
          "steps": [
            {"kind": <string>, "text": <string|null>, "latex": <string|null>}, ...
          ]
        }
      * Steps: concise micro-steps; strip fences from any LaTeX.

    STRICT RULES
    ------------
    - Do NOT include computable metadata (pages, line numbers, timestamps, tool scores, URIs, etc.).
    - Omit fields if truly unknown (or emit empty JSON arrays/objects as appropriate).
    - Preserve LaTeX faithfully in 'latex' fields; strip $$, \\[\\], \\begin{equation} ... \\end{equation}.
    - Keep 'equations' ordered as they appear in the directive.
    """

    # Inputs (directive-only; no global metadata)
    directive_text = dspy.InputField(desc="Raw theorem directive text (header + body).")
    context_hints  = dspy.InputField(desc="Tiny local context to infer implicit assumptions.", optional=True)

    # Scalar strings
    type_str         = dspy.OutputField(desc="One of: 'theorem','lemma','proposition',…")
    label_str        = dspy.OutputField(desc="Directive label if present, else empty.")
    title_str        = dspy.OutputField(desc="Human-facing title if present, else empty.")
    nl_statement_str = dspy.OutputField(desc="Concise natural-language statement only.")

    # JSON fragments (stringified JSON)
    equations_json              = dspy.OutputField(desc='JSON array: [{"label": string|null, "latex": string}, ...]')
    hypotheses_json             = dspy.OutputField(desc='JSON array: [{"text": string, "latex": string|null}, ...]')
    conclusion_json             = dspy.OutputField(desc='JSON object: {"text": string|null, "latex": string|null}')
    variables_json              = dspy.OutputField(desc='JSON array: [{"symbol": string, "role": string|null, "constraints": [string,...]}, ...]')
    implicit_assumptions_json   = dspy.OutputField(desc='JSON array: [{"text": string, "confidence": number|null}, ...]')
    local_refs_json             = dspy.OutputField(desc='JSON array of strings: ["lem-3.4","eq-main",...]')
    proof_json                  = dspy.OutputField(desc='JSON object: {"availability": "...", "steps":[{"kind": "...", "text": "...", "latex": "..."}]}')

# --------------------------------------------------------------------------------------
# 2) Reward function (import yours if you already have it; fallback kept here)
# --------------------------------------------------------------------------------------

# reward_theorem_split.py
# Reward function for ParseTheoremDirectiveSplit outputs
import json, re
from typing import Any, Dict, List, Tuple

# Allowed enums for gentle validation
_ALLOWED_TYPES = {"theorem", "lemma", "proposition", "corollary", "claim"}
_ALLOWED_AVAIL = {"present", "sketch", "omitted", "by-reference"}
_ALLOWED_STEP_KINDS = {
    "assume","apply","rewrite","calculation","case-start","case-end",
    "induction-base","induction-step","construction","contradiction",
    "conclude","remark","other"
}

# Simple helpers
_FENCE_PAT = re.compile(r"(\$\$|\\\[|\\\]|\\begin\{equation\}|\\end\{equation\})")
_URI_PAT   = re.compile(r"(https?://|file://|s3://|gs://)")
_META_PAT  = re.compile(r"\b(line|page|timestamp|uuid|sha256)\b", re.I)

def _json_loads(s: str | None, default):
    if not s or not s.strip():
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

def _nonempty(s: str | None) -> bool:
    return bool(s and s.strip())

def _no_fences(latex: str) -> bool:
    return not _FENCE_PAT.search(latex or "")

def _reasonable_text_len(text: str | None, min_words=5, max_chars=600) -> bool:
    if not _nonempty(text):
        return False
    words = len(text.strip().split())
    return (words >= min_words) and (len(text) <= max_chars)

def _has_dups(seq) -> bool:
    seen = set()
    for x in seq or []:
        if x in seen:
            return True
        seen.add(x)
    return False


def theorem_reward(args: Dict[str, Any], pred) -> float:
    """
    Reward for DSPy BestOfN / Refine.
    Expects pred to have the fields produced by ParseTheoremDirectiveSplit:
      type_str, label_str, title_str, nl_statement_str,
      equations_json, hypotheses_json, conclusion_json,
      variables_json, implicit_assumptions_json, local_refs_json, proof_json.
    Returns float in [0, 1].
    """
    # Defensive defaults
    type_str         = (getattr(pred, "type_str", None) or "").strip().lower()
    label_str        = (getattr(pred, "label_str", "") or "").strip()
    title_str        = (getattr(pred, "title_str", "") or "").strip()
    nl_statement_str = (getattr(pred, "nl_statement_str", "") or "").strip()

    equations  = _json_loads(getattr(pred, "equations_json", None), [])
    hypotheses = _json_loads(getattr(pred, "hypotheses_json", None), [])
    conclusion = _json_loads(getattr(pred, "conclusion_json", None), {})
    variables  = _json_loads(getattr(pred, "variables_json", None), [])
    impls      = _json_loads(getattr(pred, "implicit_assumptions_json", None), [])
    local_refs = _json_loads(getattr(pred, "local_refs_json", None), [])
    proof      = _json_loads(getattr(pred, "proof_json", None), {})

    # We’ll accumulate partial points toward 1.0
    score = 0.0
    max_score = 0.0

    # 1) Type / title / label (0.15)
    max_score += 0.15
    s1 = 0.0
    if type_str in _ALLOWED_TYPES:                          s1 += 0.08
    if _nonempty(title_str):                                s1 += 0.02
    # label: allowed but optional; if present, prefer short and label-like
    if _nonempty(label_str) and len(label_str) <= 80 \
       and not _URI_PAT.search(label_str):                  s1 += 0.05
    score += min(s1, 0.15)

    # 2) NL statement (concise, no metadata) (0.15)
    max_score += 0.15
    s2 = 0.0
    if _reasonable_text_len(nl_statement_str):              s2 += 0.12
    if not _META_PAT.search(nl_statement_str):              s2 += 0.03
    score += min(s2, 0.15)

    # 3) Equations (JSON parse, required fields, no fences) (0.20)
    max_score += 0.20
    s3 = 0.0
    if isinstance(equations, list):
        s3 += 0.05
        ok_all = True
        no_fences_all = True
        for e in equations[:12]:  # check first few; long proofs can be large
            if not isinstance(e, dict) or "latex" not in e:
                ok_all = False
                break
            if not _nonempty(e["latex"]):
                ok_all = False
                break
            if not _no_fences(e["latex"]):
                no_fences_all = False
        if ok_all:            s3 += 0.10
        if no_fences_all:     s3 += 0.05
    score += min(s3, 0.20)

    # 4) Hypotheses + Conclusion (structure, minimal presence) (0.20)
    max_score += 0.20
    s4 = 0.0
    # Hypotheses array of objects with at least "text" or "latex"
    if isinstance(hypotheses, list):
        s4 += 0.05
        good_hyps = True
        for h in hypotheses[:10]:
            if not isinstance(h, dict) or not ( _nonempty(h.get("text")) or _nonempty(h.get("latex")) ):
                good_hyps = False
                break
        if good_hyps: s4 += 0.05
    # Conclusion object with text or latex
    if isinstance(conclusion, dict):
        s4 += 0.05
        if _nonempty(conclusion.get("text")) or _nonempty(conclusion.get("latex")):
            s4 += 0.05
    score += min(s4, 0.20)

    # 5) Variables (optional but structured) (0.06)
    max_score += 0.06
    s5 = 0.0
    if isinstance(variables, list):
        s5 += 0.02
        ok_vars = True
        for v in variables[:15]:
            if not isinstance(v, dict) or not _nonempty(v.get("symbol")):
                ok_vars = False
                break
            if "constraints" in v and not isinstance(v.get("constraints"), list):
                ok_vars = False
                break
        if ok_vars:
            s5 += 0.04
    score += min(s5, 0.06)

    # 6) Implicit assumptions (optional; small, plausible) (0.08)
    #    Penalize overuse without context (heuristic).
    max_score += 0.08
    s6 = 0.0
    if isinstance(impls, list):
        s6 += 0.02
        plausible = True
        if len(impls) > 6:
            plausible = False
        for a in impls[:10]:
            if not isinstance(a, dict) or not _nonempty(a.get("text", "")):
                plausible = False
                break
        if plausible:
            s6 += 0.06
    score += min(s6, 0.08)

    # 7) Local refs (array[str], no dups) (0.04)
    max_score += 0.04
    s7 = 0.0
    if isinstance(local_refs, list) and all(isinstance(x, str) for x in local_refs):
        s7 += 0.03
        if not _has_dups(local_refs):
            s7 += 0.01
    score += min(s7, 0.04)

    # 8) Proof (availability + steps structure, fence-free LaTeX) (0.12)
    max_score += 0.12
    s8 = 0.0
    if isinstance(proof, dict):
        s8 += 0.02
        avail = (proof.get("availability") or "").strip().lower()
        if avail in _ALLOWED_AVAIL: s8 += 0.03
        steps = proof.get("steps", [])
        if isinstance(steps, list):
            s8 += 0.03
            ok_steps = True
            no_fence_steps = True
            for st in steps[:20]:
                if not isinstance(st, dict): ok_steps = False; break
                kind = (st.get("kind") or "").strip().lower()
                if kind and kind not in _ALLOWED_STEP_KINDS:
                    ok_steps = False; break
                latex = st.get("latex")
                if latex and not _no_fences(latex):
                    no_fence_steps = False
            if ok_steps:        s8 += 0.02
            if no_fence_steps:  s8 += 0.02
    score += min(s8, 0.12)

    # 9) Anti‑metadata check across main strings (0.0–0.10 bonus)
    #    Encourage ignoring computable metadata.
    max_score += 0.10
    s9 = 0.10
    texts_to_check = [label_str, title_str, nl_statement_str]
    for arr in (hypotheses, variables, impls):
        for obj in arr if isinstance(arr, list) else []:
            if isinstance(obj, dict):
                for k in ("text","latex"):
                    val = obj.get(k)
                    if isinstance(val, str):
                        texts_to_check.append(val)
    # proof step texts/latex
    for st in proof.get("steps", []) if isinstance(proof, dict) else []:
        if isinstance(st, dict):
            if isinstance(st.get("text"), str): texts_to_check.append(st["text"])
            if isinstance(st.get("latex"), str): texts_to_check.append(st["latex"])

    for t in texts_to_check:
        if _URI_PAT.search(t or "") or _META_PAT.search(t or ""):
            s9 -= 0.02  # small penalty per occurrence
    s9 = max(0.0, s9)
    score += min(s9, 0.10)

    # Normalize (just in case weights drift)
    score = max(0.0, min(score, 1.0))
    return score


# --------------------------------------------------------------------------------------
# 3) Utilities
# --------------------------------------------------------------------------------------
_LINENO = re.compile(r"^\s*\d+:\s?")   # matches "123: " prefix

def strip_line_numbers(text: str) -> str:
    """Remove 'NNN: ' prefixes from each line."""
    return "\n".join(_LINENO.sub("", ln) for ln in (text or "").splitlines())

def synthesize_directive_text(obj: Dict[str, Any]) -> str:
    """
    Build a well-formed directive block if 'raw_directive' is missing.
    Uses directive_type/title/label and the 'content' body.
    """
    dtype = (obj.get("directive_type") or "theorem").strip()
    title = (obj.get("title") or "").strip()
    label = (obj.get("label") or obj.get("metadata", {}).get("label") or "").strip()

    header = f"::{{prf:{dtype}}} {title}".rstrip()
    labelln = f":label: {label}" if label else ""
    body = strip_line_numbers(obj.get("content") or "")
    parts = [header]
    if labelln: parts.append(labelln)
    if body: parts.append(body)
    parts.append(":::")
    return "\n\n".join(parts)

def extract_directive_text(obj: Dict[str, Any]) -> str:
    """Prefer raw_directive; otherwise synthesize from fields."""
    raw = obj.get("raw_directive")
    return strip_line_numbers(raw) if isinstance(raw, str) and raw.strip() else synthesize_directive_text(obj)

def tiny_context_hints(obj: Dict[str, Any], doc_text: str, window: int = 320) -> str:
    """
    Build a very small context window:
    - primary: the 'content' field (already local to the directive).
    - fallback: locate label/title in doc_text and return a short window.
    """
    content = strip_line_numbers(obj.get("content") or "").strip()
    if content:
        # Clamp to a short window from the content itself
        c = content
        if len(c) > window:
            # take head+tail
            head = c[:window//2]
            tail = c[-window//2:]
            return head + "\n...\n" + tail
        return c

    # fallback: find in doc
    key = (obj.get("label") or obj.get("title") or "").strip()
    if not key:
        return ""
    idx = doc_text.find(key)
    if idx == -1:
        key2 = key[:60]
        idx = doc_text.find(key2) if key2 else -1
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end   = min(len(doc_text), idx + len(key) + window)
    return doc_text[start:end]

def load_json_or_jsonl(path: Path) -> Iterable[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln: continue
            try: yield json.loads(ln)
            except Exception: pass
    else:
        obj = json.loads(text)
        obj = obj.get("items", obj)
        if isinstance(obj, list):
            for it in obj: yield it
        else:
            yield obj

def assemble_output(res) -> Dict[str, Any]:
    """Turn split fields into the unified object."""
    def j(x, default):
        try: return json.loads(x) if x else default
        except Exception: return default

    return {
        "type":                 (res.type_str or "").strip() or None,
        "label":                (res.label_str or "").strip() or None,
        "title":                (res.title_str or "").strip() or None,
        "nl_statement":         (res.nl_statement_str or "").strip() or None,
        "equations":            j(res.equations_json, []),
        "hypotheses":           j(res.hypotheses_json, []),
        "conclusion":           j(res.conclusion_json, {}),
        "variables":            j(res.variables_json, []),
        "implicit_assumptions": j(res.implicit_assumptions_json, []),
        "local_refs":           j(res.local_refs_json, []),
        "proof":                j(res.proof_json, {}),
    }

# --------------------------------------------------------------------------------------
# 4) Data loading and output
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# 5) Main refine loop
# --------------------------------------------------------------------------------------
def run_agent(
    document_path: str | Path,
    theorems_path: str | Path | None=None,
    out_path: str | Path | None=None,
    lm_spec: str = "openai/gpt-4o-mini",
    passes: int = 5,
    threshold: float = 0.95,
    max_tokens: int = 16000,
) -> None:
    # LM config
    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=max_tokens)

    # Program: Refine over the split signature
    program = dspy.Refine(
        module=dspy.Predict(ParseTheoremDirectiveSplit),
        N=passes,
        reward_fn=theorem_reward,
        threshold=threshold,
    )

    # Load doc (for tiny context only)
    doc_path = Path(document_path)
    document_folder = doc_path.parent / str(doc_path.stem)
    if out_path is None:
        out_path = document_folder / "extract"
        out_path.mkdir(exist_ok=True)
        out_path = out_path / "theorem.json"
    if theorems_path is None:
        theorems_path = document_folder / "registry" / "directives" / "theorem.json"
    # Iterate theorems and write outputs
    doc_text = doc_path.read_text(encoding="utf-8")
    out_f = Path(out_path).open("w", encoding="utf-8")
    outputs = []
    for idx, obj in enumerate(load_json_or_jsonl(Path(theorems_path)), start=1):
        if not isinstance(obj, dict):
            continue

        directive_text = extract_directive_text(obj)
        context_hints  = tiny_context_hints(obj, doc_text, window=320)
        print(idx)

        try:
            res = program(directive_text=obj, context_hints=context_hints)
            print("✓ Refine succeeded")
        except Exception:
            # Fallback: one-pass predict
            res = dspy.Predict(ParseTheoremDirectiveSplit)(
                directive_text=directive_text, context_hints=context_hints
            )

        assembled = assemble_output(res)
        outputs.append(assembled)
    out_f.write(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n")

    out_f.close()


# --------------------------------------------------------------------------------------
# 5) CLI
# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Parse theorem directives (structured objects as input) via DSPy Refine.")
    ap.add_argument("--doc", required=True, help="Path to the full document (used only for tiny context hints).")
    ap.add_argument("--theorems", required=False, default=None, help="Path to JSON/JSONL of theorem objects (like the provided example).")
    ap.add_argument("--out", required=False, default=None, help="Output JSONL; one structured theorem per line.")
    ap.add_argument("--lm", default="gemini/gemini-flash-lite-latest", help="LM spec for DSPy.")
    ap.add_argument("--passes", type=int, default=5, help="Refine attempts (N).")
    ap.add_argument("--threshold", type=float, default=0.95, help="Refine early-stop threshold.")
    args = ap.parse_args(argv)

    run_agent(
        theorems_path=args.theorems,
        document_path=args.doc,
        out_path=args.out,
        lm_spec=args.lm,
        passes=args.passes,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
