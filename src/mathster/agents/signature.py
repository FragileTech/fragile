import dspy

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
