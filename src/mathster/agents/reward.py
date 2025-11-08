# reward_theorem_split.py
# Reward function for ParseTheoremDirectiveSplit outputs

from __future__ import annotations
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
