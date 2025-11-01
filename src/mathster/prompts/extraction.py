"""
Stage 1 Extraction Prompts for Mathematical Paper Processing.

This module contains prompt templates for the raw extraction phase of the
Extract-then-Enrich pipeline. These prompts guide LLMs to transcribe mathematical
entities from markdown sections with minimal interpretation.

Design Principles:
- Verbatim extraction (minimize interpretation)
- Preserve original LaTeX and mathematical notation exactly
- Extract all context (before/after text)
- Assign temporary IDs for later cross-referencing
- Flag ambiguities rather than resolving them

Usage:
    from fragile.mathster.prompts.extraction import MAIN_EXTRACTION_PROMPT

    prompt = MAIN_EXTRACTION_PROMPT.format(
        section_text=section_content,
        section_id="§2.1"
    )

Maps to Lean:
    namespace ExtractionPrompts
      def main_extraction_template : String
      def entity_examples : String
    end ExtractionPrompts
"""

# =============================================================================
# MAIN EXTRACTION PROMPT
# =============================================================================

MAIN_EXTRACTION_PROMPT = """
You are a precise mathematical text extraction assistant. Your task is to extract
mathematical entities from a section of a mathematical paper written in markdown.

**CRITICAL INSTRUCTIONS:**
1. Extract entities VERBATIM - preserve exact LaTeX, notation, and wording
2. Do NOT interpret, simplify, or rephrase mathematical content
3. Extract ALL context (text before/after) to aid later semantic understanding
4. Assign temporary IDs for tracking (e.g., "raw-thm-001", "raw-def-001")
5. When uncertain about entity type or reference, FLAG it in the extraction
6. Extract EVERY equation, even if unnumbered (assign unique temp ID)
7. Preserve markdown formatting within extracted text

**INPUT:**
Section ID: {section_id}

Section Content:
```markdown
{section_text}
```

**OUTPUT FORMAT:**
Return a JSON object with the following structure:

{{
  "section_id": "{section_id}",
  "definitions": [...],      // List of RawDefinition objects
  "theorems": [...],         // List of RawTheorem objects
  "mathster": [...],           // List of RawProof objects
  "axioms": [...],           // List of RawAxiom objects
  "citations": [...],        // List of RawCitation objects
  "equations": [...],        // List of RawEquation objects
  "parameters": [...],       // List of RawParameter objects
  "remarks": [...]           // List of RawRemark objects
}}

---

## 1. DEFINITIONS (RawDefinition)

Extract formal definitions of terms, concepts, or notation.

**Schema:**
{{
  "temp_id": "raw-def-NNN",          // Sequential: raw-def-001, raw-def-002, ...
  "term_being_defined": "string",    // The term (e.g., "walker", "potential")
  "full_text": "string",             // Complete definition text (verbatim)
  "parameters_mentioned": ["..."],   // Symbols defined/mentioned (e.g., ["x", "v"])
  "source_section": "{section_id}"   // Section where found
}}

**Example:**
If the markdown contains:
```
**Definition 2.1** (Walker State). A *walker* is a tuple $w := (x, v, s)$ where
$x \\in \\mathcal{{X}}$ is the position, $v \\in \\mathbb{{R}}^d$ is the velocity, and
$s \\in \\{{\\text{{alive}}, \\text{{dead}}\\}}$ is the status.
```

Extract as:
{{
  "temp_id": "raw-def-001",
  "term_being_defined": "Walker State",
  "full_text": "A *walker* is a tuple $w := (x, v, s)$ where $x \\in \\mathcal{{X}}$ is the position, $v \\in \\mathbb{{R}}^d$ is the velocity, and $s \\in \\{{\\text{{alive}}, \\text{{dead}}\\}}$ is the status.",
  "parameters_mentioned": ["w", "x", "v", "s"],
  "source_section": "{section_id}"
}}

---

## 2. THEOREMS (RawTheorem)

Extract theorems, lemmas, propositions, and corollaries.

**Schema:**
{{
  "temp_id": "raw-thm-NNN",                    // Sequential: raw-thm-001, ...
  "label_text": "string",                      // e.g., "Theorem 3.1", "Lemma 2.5"
  "statement_type": "theorem|lemma|proposition|corollary",
  "context_before": "string or null",          // Paragraph before (if clarifying)
  "full_statement_text": "string",            // Complete statement (verbatim)
  "conclusion_formula_latex": "string or null",// Main formula if present
  "equation_label": "string or null",          // e.g., "(3.1)" if numbered
  "explicit_definition_references": ["..."],   // Referenced definitions
  "source_section": "{section_id}"
}}

**Example:**
If the markdown contains:
```
The following result establishes convergence.

**Theorem 3.1** (Keystone Principle). Let $v > 0$ and assume the potential $U$
is Lipschitz. Then the Euclidean Gas converges exponentially:
$$
d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}}
$$ {{(3.1)}}
for constants $C, \\lambda > 0$.
```

Extract as:
{{
  "temp_id": "raw-thm-001",
  "label_text": "Theorem 3.1",
  "statement_type": "theorem",
  "context_before": "The following result establishes convergence.",
  "full_statement_text": "Let $v > 0$ and assume the potential $U$ is Lipschitz. Then the Euclidean Gas converges exponentially: $$ d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}} $$ for constants $C, \\lambda > 0$.",
  "conclusion_formula_latex": "d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}}",
  "equation_label": "(3.1)",
  "explicit_definition_references": ["Euclidean Gas", "potential U"],
  "source_section": "{section_id}"
}}

---

## 3. PROOFS (RawProof)

Extract proof content for theorems, lemmas, etc.

**Schema:**
{{
  "temp_id": "raw-proof-NNN",              // Sequential: raw-proof-001, ...
  "proves_label_text": "string",          // e.g., "Theorem 3.1", "Lemma 2.5"
  "strategy_text": "string or null",      // High-level strategy (if stated)
  "steps": ["..."] or null,               // Explicit numbered steps (if present)
  "full_body_text": "string or null",     // Complete proof text (verbatim)
  "explicit_theorem_references": ["..."], // Other theorems cited
  "citations_in_text": ["..."],           // Citation keys (e.g., ["han2016"])
  "source_section": "{section_id}"
}}

**Example:**
If the markdown contains:
```
**Proof of Theorem 3.1.** We proceed in two steps:
1. Establish Lipschitz continuity of the transition kernel
2. Apply Grönwall's inequality to obtain exponential decay

*Step 1:* By assumption, $U$ is Lipschitz with constant $L_U$...

(Full proof text continues...)

The result follows from [Han & Chen 2016].
```

Extract as:
{{
  "temp_id": "raw-proof-001",
  "proves_label_text": "Theorem 3.1",
  "strategy_text": "We proceed in two steps: 1. Establish Lipschitz continuity of the transition kernel 2. Apply Grönwall's inequality to obtain exponential decay",
  "steps": [
    "Establish Lipschitz continuity of the transition kernel",
    "Apply Grönwall's inequality to obtain exponential decay"
  ],
  "full_body_text": "(Complete verbatim proof text here)",
  "explicit_theorem_references": ["Grönwall's inequality"],
  "citations_in_text": ["Han & Chen 2016"],
  "source_section": "{section_id}"
}}

---

## 4. AXIOMS (RawAxiom)

Extract foundational axioms that establish the framework's core assumptions.

**Schema:**
{{
  "temp_id": "raw-axiom-NNN",                 // Sequential: raw-axiom-001, ...
  "label_text": "string",                     // e.g., "Axiom of Bounded Displacement"
  "name": "string",                           // Title of the axiom
  "core_assumption_text": "string",          // The fundamental claim (verbatim)
  "parameters_text": ["string", ...],        // List of parameter definitions (may be empty)
  "condition_text": "string",                // Formal condition statement (empty string if not stated)
  "failure_mode_analysis_text": "string or null", // Analysis of violations (null if not present)
  "source_section": "{section_id}"
}}

**Example:**
If the markdown contains:
```
**Axiom 1.1** (Axiom of Bounded Displacement). All walkers satisfy a fundamental
displacement bound that ensures physical realizability.

**Parameters:**
- Let $\\varepsilon > 0$ be the displacement scale
- Let $\\Delta t$ be the time step

**Core Assumption:**
For any walker state $w$ and time step $\\Delta t$, the displacement satisfies
$$
|x(t + \\Delta t) - x(t)| \\leq \\varepsilon \\sqrt{{\\Delta t}}
$$

**Condition:**
This holds when $\\Delta t < \\varepsilon^2$.

**Failure Mode:**
If violated, walkers can exhibit unphysical teleportation behavior.
```

Extract as:
{{
  "temp_id": "raw-axiom-001",
  "label_text": "Axiom 1.1",
  "name": "Axiom of Bounded Displacement",
  "core_assumption_text": "For any walker state $w$ and time step $\\Delta t$, the displacement satisfies $|x(t + \\Delta t) - x(t)| \\leq \\varepsilon \\sqrt{{\\Delta t}}$",
  "parameters_text": [
    "Let $\\varepsilon > 0$ be the displacement scale",
    "Let $\\Delta t$ be the time step"
  ],
  "condition_text": "This holds when $\\Delta t < \\varepsilon^2$.",
  "failure_mode_analysis_text": "If violated, walkers can exhibit unphysical teleportation behavior.",
  "source_section": "{section_id}"
}}

**Notes:**
- If parameters section is missing, use empty list: `"parameters_text": []`
- If condition is not stated, use empty string: `"condition_text": ""`
- If failure mode is not discussed, use null: `"failure_mode_analysis_text": null`
- Extract the core assumption verbatim, including all LaTeX

---

## 5. CITATIONS (RawCitation)

Extract bibliographic references from the text or bibliography section.

**Schema:**
{{
  "key_in_text": "string",       // Citation key (e.g., "han2016")
  "full_entry_text": "string"    // Complete BibTeX entry or formatted citation
}}

**Example:**
If the markdown contains:
```
[Han & Chen 2016] Han, T. and Chen, X. "Convergence of particle systems"
J. Math. Anal. 42(3), 2016.
```

Extract as:
{{
  "key_in_text": "han2016",
  "full_entry_text": "Han, T. and Chen, X. \\"Convergence of particle systems\\" J. Math. Anal. 42(3), 2016."
}}

---

## 6. EQUATIONS (RawEquation)

Extract ALL display equations, both numbered and unnumbered.

**Schema:**
{{
  "temp_id": "raw-eq-NNN",            // Sequential: raw-eq-001, raw-eq-002, ...
  "equation_label": "string or null", // e.g., "(2.3)" if numbered, null otherwise
  "latex_content": "string",          // LaTeX between $$ delimiters (verbatim)
  "context_before": "string or null", // Sentence introducing the equation
  "context_after": "string or null",  // Sentence explaining the equation
  "source_section": "{section_id}"
}}

**Example 1 (numbered equation):**
If the markdown contains:
```
The kinetic operator is defined by the Langevin dynamics:
$$
dx_t = v_t dt, \\quad dv_t = -\\gamma v_t dt + \\sqrt{{2\\gamma}} dW_t
$$ {{(2.1)}}
where $\\gamma > 0$ is the friction coefficient.
```

Extract as:
{{
  "temp_id": "raw-eq-001",
  "equation_label": "(2.1)",
  "latex_content": "dx_t = v_t dt, \\quad dv_t = -\\gamma v_t dt + \\sqrt{{2\\gamma}} dW_t",
  "context_before": "The kinetic operator is defined by the Langevin dynamics:",
  "context_after": "where $\\gamma > 0$ is the friction coefficient.",
  "source_section": "{section_id}"
}}

**Example 2 (unnumbered equation):**
```
The potential energy is given by:
$$
U(x) = \\sum_{{i=1}}^k e^{{-\\|x - \\mu_i\\|^2 / \\sigma^2}}
$$
This defines a mixture of Gaussians.
```

Extract as:
{{
  "temp_id": "raw-eq-002",
  "equation_label": null,
  "latex_content": "U(x) = \\sum_{{i=1}}^k e^{{-\\|x - \\mu_i\\|^2 / \\sigma^2}}",
  "context_before": "The potential energy is given by:",
  "context_after": "This defines a mixture of Gaussians.",
  "source_section": "{section_id}"
}}

---

## 7. PARAMETERS (RawParameter)

Extract parameter definitions (constants, variables, symbols).

**Schema:**
{{
  "temp_id": "raw-param-NNN",        // Sequential: raw-param-001, ...
  "symbol": "string",                // Symbol (e.g., "γ", "N", "h")
  "meaning": "string",               // Brief meaning (verbatim)
  "full_text": "string",             // Complete context where defined
  "scope": "global|local",           // Document-wide or section-specific
  "source_section": "{section_id}"
}}

**Example:**
If the markdown contains:
```
Let $N \\in \\mathbb{{N}}$ denote the number of walkers, and let $h > 0$ be
a small discretization parameter.
```

Extract TWO parameters:
{{
  "temp_id": "raw-param-001",
  "symbol": "N",
  "meaning": "number of walkers",
  "full_text": "Let $N \\in \\mathbb{{N}}$ denote the number of walkers",
  "scope": "global",
  "source_section": "{section_id}"
}}

{{
  "temp_id": "raw-param-002",
  "symbol": "h",
  "meaning": "small discretization parameter",
  "full_text": "let $h > 0$ be a small discretization parameter",
  "scope": "local",
  "source_section": "{section_id}"
}}

---

## 8. REMARKS (RawRemark)

Extract informal notes, observations, examples, or commentary.

**Schema:**
{{
  "temp_id": "raw-remark-NNN",                // Sequential: raw-remark-001, ...
  "remark_type": "note|remark|observation|comment|example",
  "full_text": "string",                      // Complete text (verbatim)
  "source_section": "{section_id}"
}}

**Example:**
If the markdown contains:
```
**Remark 2.3.** The condition $v > 0$ is essential. Without kinetic energy,
the walkers cannot explore the state space efficiently.
```

Extract as:
{{
  "temp_id": "raw-remark-001",
  "remark_type": "remark",
  "full_text": "The condition $v > 0$ is essential. Without kinetic energy, the walkers cannot explore the state space efficiently.",
  "source_section": "{section_id}"
}}

---

## EXTRACTION GUIDELINES

1. **Numbering:** Use sequential IDs within each entity type (raw-def-001, raw-def-002, ...)
2. **Verbatim:** Preserve exact LaTeX, markdown formatting, and wording
3. **Context:** Include surrounding text to aid semantic understanding
4. **Ambiguity:** When unsure (e.g., is it a lemma or theorem?), make your best guess
5. **Completeness:** Extract EVERY entity, even informal ones
6. **References:** List explicit references to other entities (definitions, theorems, citations)
7. **Equations:** Assign temp IDs to ALL equations, including unnumbered ones
8. **Markdown:** Preserve markdown syntax like **bold**, *italic*, `code`, etc.

---

## OUTPUT FORMAT

Return ONLY a valid JSON object matching the StagingDocument schema:

{{
  "section_id": "{section_id}",
  "definitions": [...],
  "theorems": [...],
  "mathster": [...],
  "citations": [...],
  "equations": [...],
  "parameters": [...],
  "remarks": [...]
}}

Do NOT include any explanatory text before or after the JSON.
Do NOT wrap the JSON in markdown code blocks.
Return pure JSON only.
"""


# =============================================================================
# ALTERNATIVE PROMPTS (for specific extraction tasks)
# =============================================================================

DEFINITION_ONLY_PROMPT = """
Extract ONLY definitions from the following markdown section.

Section: {section_id}

Content:
```markdown
{section_text}
```

Return a JSON array of RawDefinition objects:
[
  {{
    "temp_id": "raw-def-001",
    "term_being_defined": "...",
    "full_text": "...",
    "parameters_mentioned": [...],
    "source_section": "{section_id}"
  }},
  ...
]
"""


THEOREM_ONLY_PROMPT = """
Extract ONLY theorems, lemmas, propositions, and corollaries from the following section.

Section: {section_id}

Content:
```markdown
{section_text}
```

Return a JSON array of RawTheorem objects following the schema:
[
  {{
    "temp_id": "raw-thm-001",
    "label_text": "...",
    "statement_type": "theorem|lemma|proposition|corollary",
    "context_before": "...",
    "full_statement_text": "...",
    "conclusion_formula_latex": "...",
    "equation_label": "...",
    "explicit_definition_references": [...],
    "source_section": "{section_id}"
  }},
  ...
]
"""


EQUATION_ONLY_PROMPT = """
Extract ALL display equations (both numbered and unnumbered) from the following section.

Section: {section_id}

Content:
```markdown
{section_text}
```

Return a JSON array of RawEquation objects:
[
  {{
    "temp_id": "raw-eq-001",
    "equation_label": "(2.1)" or null,
    "latex_content": "...",
    "context_before": "...",
    "context_after": "...",
    "source_section": "{section_id}"
  }},
  ...
]
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_extraction_prompt(section_text: str, section_id: str) -> str:
    """
    Get the main extraction prompt with section content substituted.

    Args:
        section_text: Markdown content of the section
        section_id: Section identifier (e.g., "§2.1")

    Returns:
        Formatted prompt string ready for LLM

    Examples:
        >>> prompt = get_extraction_prompt(
        ...     section_text="# Section 2\\n\\n**Theorem 2.1**...", section_id="§2"
        ... )
    """
    return MAIN_EXTRACTION_PROMPT.format(section_text=section_text, section_id=section_id)


def get_focused_extraction_prompt(entity_type: str, section_text: str, section_id: str) -> str:
    """
    Get a focused extraction prompt for a specific entity type.

    Args:
        entity_type: One of "definitions", "theorems", "equations"
        section_text: Markdown content
        section_id: Section identifier

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If entity_type is not recognized
    """
    prompts = {
        "definitions": DEFINITION_ONLY_PROMPT,
        "theorems": THEOREM_ONLY_PROMPT,
        "equations": EQUATION_ONLY_PROMPT,
    }

    if entity_type not in prompts:
        raise ValueError(f"Unknown entity type: {entity_type}")

    return prompts[entity_type].format(section_text=section_text, section_id=section_id)
