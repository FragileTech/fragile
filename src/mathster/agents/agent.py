#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage
-----
python run_theorem_agent.py \
  --theorems theorems.json \
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
    Split-output parser for one theorem directive.
    See your previous definition; unchanged here.
    """
    directive_text = dspy.InputField(desc="Raw theorem directive text (header + body).")
    context_hints  = dspy.InputField(desc="Tiny local context to infer implicit assumptions.", optional=True)

    type_str         = dspy.OutputField(desc="theorem|lemma|proposition|...")
    label_str        = dspy.OutputField(desc="Directive label if present.")
    title_str        = dspy.OutputField(desc="Human-facing title if present.")
    nl_statement_str = dspy.OutputField(desc="Concise natural-language statement.")

    equations_json              = dspy.OutputField(desc='JSON array: [{"label": string|null, "latex": string}, ...]')
    hypotheses_json             = dspy.OutputField(desc='JSON array: [{"text": string, "latex": string|null}, ...]')
    conclusion_json             = dspy.OutputField(desc='JSON object: {"text": string|null, "latex": string|null}')
    variables_json              = dspy.OutputField(desc='JSON array: [{"symbol": string, "role": string|null, "constraints": [string,...]}, ...]')
    implicit_assumptions_json   = dspy.OutputField(desc='JSON array: [{"text": string, "confidence": number|null}, ...]')
    local_refs_json             = dspy.OutputField(desc='JSON array of strings')
    proof_json                  = dspy.OutputField(desc='JSON object: {"availability": "...", "steps":[...]}')

# --------------------------------------------------------------------------------------
# 2) Reward function (import yours if you already have it; fallback kept here)
# --------------------------------------------------------------------------------------
try:
    from reward_theorem_split import theorem_reward  # your stronger reward
except Exception:
    import re
    _URI_PAT   = re.compile(r"(https?://|file://|s3://|gs://)")
    _META_PAT  = re.compile(r"\b(line|page|timestamp|uuid|sha256)\b", re.I)
    def _loads(x, default):
        import json
        try: return json.loads(x) if x else default
        except Exception: return default
    def theorem_reward(args: Dict[str, Any], pred) -> float:
        # Very lightweight fallback (schema-ish + anti-metadata); returns [0,1]
        try:
            s = 0.0
            t  = (getattr(pred, "type_str", "") or "").strip().lower()
            ns = (getattr(pred, "nl_statement_str", "") or "")
            eq = _loads(getattr(pred, "equations_json", None), [])
            pf = _loads(getattr(pred, "proof_json", None), {})
            if t in {"theorem","lemma","proposition","corollary","claim"}: s += 0.2
            if ns and len(ns.split()) >= 5 and len(ns) <= 600 and not _META_PAT.search(ns): s += 0.2
            if isinstance(eq, list) and all(isinstance(x, dict) and "latex" in x and x["latex"] for x in eq[:8]): s += 0.3
            if isinstance(pf, dict) and isinstance(pf.get("steps", []), list): s += 0.2
            # small anti-metadata/URI bonus
            txts = [getattr(pred, "label_str", ""), getattr(pred, "title_str", ""), ns]
            if not any(_URI_PAT.search(t or "") or _META_PAT.search(t or "") for t in txts): s += 0.1
            return max(0.0, min(1.0, s))
        except Exception:
            return 0.0

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
# 4) Main refine loop
# --------------------------------------------------------------------------------------
def run_agent(
    theorems_path: str,
    document_path: str,
    out_path: str,
    lm_spec: str = "openai/gpt-4o-mini",
    passes: int = 5,
    threshold: float = 0.85
) -> None:
    # LM config
    load_dotenv()
    dspy.settings.configure(lm=dspy.LM(lm_spec), max_tokens=16000,)

    # Program: Refine over the split signature
    program = dspy.Refine(
        module=dspy.Predict(ParseTheoremDirectiveSplit),
        N=passes,
        reward_fn=theorem_reward,
        threshold=threshold,
    )

    # Load doc (for tiny context only)
    doc_text = Path(document_path).read_text(encoding="utf-8")

    # Iterate theorems and write outputs
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
    ap.add_argument("--theorems", required=True, help="Path to JSON/JSONL of theorem objects (like the provided example).")
    ap.add_argument("--doc", required=True, help="Path to the full document (used only for tiny context hints).")
    ap.add_argument("--out", required=True, help="Output JSONL; one structured theorem per line.")
    ap.add_argument("--lm", default="gemini/gemini-flash-lite-latest", help="LM spec for DSPy.")
    ap.add_argument("--passes", type=int, default=5, help="Refine attempts (N).")
    ap.add_argument("--threshold", type=float, default=0.85, help="Refine early-stop threshold.")
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
