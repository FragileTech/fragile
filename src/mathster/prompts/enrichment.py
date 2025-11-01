"""
Stage 2 Enrichment Prompts for Mathematical Paper Processing.

This module contains prompt templates for the enrichment phase of the
Extract-then-Enrich pipeline. These prompts guide LLMs through focused
semantic parsing tasks to transform raw extractions into fully-structured
mathematical objects.

Design Principles:
- Focused, single-task prompts (decompose, parse, link, resolve)
- Semantic understanding (vs. verbatim transcription in Stage 1)
- Cross-referencing and relationship extraction
- SymPy-compatible symbolic representation
- Validation against framework conventions

Usage:
    from fragile.mathster.prompts.enrichment import (
        DECOMPOSE_THEOREM_PROMPT,
        PARSE_LATEX_TO_DUAL_PROMPT
    )

    prompt = DECOMPOSE_THEOREM_PROMPT.format(
        theorem_statement="Let v > 0 and assume U is Lipschitz. Then..."
    )

Maps to Lean:
    namespace EnrichmentPrompts
      def decompose_theorem_template : String
      def parse_latex_template : String
      def analyze_proof_structure_template : String
      def resolve_reference_template : String
    end EnrichmentPrompts
"""

# =============================================================================
# THEOREM DECOMPOSITION
# =============================================================================

DECOMPOSE_THEOREM_PROMPT = """
Decompose the following theorem statement into assumptions and conclusion.

**Theorem Statement:**
{theorem_statement}

**Task:**
Separate the statement into:
1. **Assumptions**: All hypotheses, conditions, or "given" clauses
2. **Conclusion**: The main claim or "then" clause

Return a JSON object with this structure:

{{
  "assumptions": [
    "assumption 1 (verbatim LaTeX preserved)",
    "assumption 2 (verbatim LaTeX preserved)",
    ...
  ],
  "conclusion": "conclusion statement (verbatim LaTeX preserved)"
}}

**Guidelines:**
- Preserve LaTeX notation exactly (e.g., $v > 0$, $U \\in C^1$)
- Each assumption should be a complete, self-contained statement
- The conclusion should be the main result being claimed
- Do NOT simplify or rephrase mathematical expressions
- Include quantifiers in assumptions (e.g., "Let $v > 0$" is an assumption)

**Example:**

Input: "Let $v > 0$ and assume the potential $U$ is Lipschitz with constant $L_U \\leq 1$. Then the Euclidean Gas converges exponentially: $d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}}$."

Output:
{{
  "assumptions": [
    "Let $v > 0$",
    "assume the potential $U$ is Lipschitz with constant $L_U \\leq 1$"
  ],
  "conclusion": "the Euclidean Gas converges exponentially: $d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}}$"
}}

Return ONLY the JSON object, no additional text.
"""


# =============================================================================
# LATEX TO DUAL STATEMENT PARSING
# =============================================================================

PARSE_LATEX_TO_DUAL_PROMPT = """
Parse the following mathematical statement into both LaTeX and SymPy representations.

**Statement:**
{statement_text}

**Task:**
Create a DualStatement with parallel LaTeX and SymPy representations.

Return a JSON object with this structure:

{{
  "latex": "...",              // LaTeX representation (verbatim)
  "sympy_expr": "...",         // SymPy-parseable string
  "free_symbols": [...],       // List of symbol names used
  "assumptions": {{...}},        // Symbol assumptions (e.g., {{"v": "positive"}})
  "natural_language": "..."    // Brief English description
}}

**Guidelines:**
1. **LaTeX**: Preserve exactly as given
2. **SymPy**: Convert to valid SymPy expression string
   - Use SymPy functions: `exp`, `log`, `sqrt`, `Abs`, etc.
   - Use `**` for exponentiation (not `^`)
   - Use `*` for multiplication explicitly
3. **Free Symbols**: List all mathematical symbols (excluding constants like π, e)
4. **Assumptions**: Specify properties of symbols (e.g., "positive", "real", "integer")
5. **Natural Language**: One-sentence plain English description

**Example:**

Input: "$d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}}$"

Output:
{{
  "latex": "d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}}",
  "sympy_expr": "d_W <= C * exp(-lambda_param * t)",
  "free_symbols": ["d_W", "C", "lambda_param", "t"],
  "assumptions": {{
    "C": "positive",
    "lambda_param": "positive",
    "t": "real"
  }},
  "natural_language": "The Wasserstein distance is bounded by an exponentially decaying function of time."
}}

**Common SymPy Conversions:**
- $\\|x\\|$ → `Abs(x)` or `sqrt(x**2 + y**2)` (depending on context)
- $e^x$ → `exp(x)`
- $\\log x$ → `log(x)`
- $\\sqrt{{x}}$ → `sqrt(x)`
- $\\sum_{{i=1}}^n$ → `Sum(expr, (i, 1, n))`
- $\\int_a^b$ → `Integral(expr, (x, a, b))`
- $\\nabla f$ → Use symbol like `grad_f` (SymPy doesn't have direct gradient notation)
- $\\partial_x f$ → Use `Derivative(f, x)`

Return ONLY the JSON object, no additional text.
"""


# =============================================================================
# PROOF STRUCTURE ANALYSIS
# =============================================================================

ANALYZE_PROOF_STRUCTURE_PROMPT = """
Analyze the structure of the following proof and decompose it into logical steps.

**Proof Text:**
{proof_text}

**Theorem Being Proven:**
{theorem_label}

**Task:**
Break the proof into logical steps with natural language descriptions and mathematical justifications.

Return a JSON object with this structure:

{{
  "strategy": "...",           // High-level proof strategy (1-2 sentences)
  "steps": [
    {{
      "step_number": 1,
      "natural_language_description": "...",
      "mathematical_claim": "...",          // LaTeX if present
      "justification": "...",               // How is this step justified?
      "uses_theorems": [...],               // Referenced theorems/lemmas
      "uses_definitions": [...],            // Referenced definitions
      "citations": [...]                    // Citations used
    }},
    ...
  ],
  "conclusion": "..."          // Final concluding statement
}}

**Guidelines:**
1. **Strategy**: Summarize the overall approach (e.g., "proof by contradiction", "direct construction", "induction on n")
2. **Steps**: Each logical block in the proof
   - Natural language: What is being shown in this step?
   - Mathematical claim: Key equation or inequality (LaTeX)
   - Justification: Why is this step valid? (e.g., "by Lemma 2.3", "by definition of convergence")
3. **References**: Track all external dependencies (theorems, definitions, citations)
4. **Conclusion**: The final statement tying everything together

**Example:**

Input:
```
Proof of Theorem 3.1. We proceed in two steps.

Step 1: By Lemma 2.5, the transition kernel $P_t$ is Lipschitz:
$$
|P_t f(x) - P_t f(y)| \\leq L e^{{-\\gamma t}} |x - y|
$$

Step 2: Applying Grönwall's inequality [Han 2016] to the distance evolution yields:
$$
d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}}
$$

This completes the proof.
```

Output:
{{
  "strategy": "Two-step proof: first establish Lipschitz property of the transition kernel, then apply Grönwall's inequality to obtain exponential convergence.",
  "steps": [
    {{
      "step_number": 1,
      "natural_language_description": "Establish Lipschitz continuity of the transition kernel",
      "mathematical_claim": "|P_t f(x) - P_t f(y)| \\leq L e^{{-\\gamma t}} |x - y|",
      "justification": "by Lemma 2.5",
      "uses_theorems": ["Lemma 2.5"],
      "uses_definitions": ["transition kernel"],
      "citations": []
    }},
    {{
      "step_number": 2,
      "natural_language_description": "Apply Grönwall's inequality to obtain exponential decay bound",
      "mathematical_claim": "d_W(\\mu_N^t, \\pi) \\leq C e^{{-\\lambda t}}",
      "justification": "Applying Grönwall's inequality to the distance evolution",
      "uses_theorems": ["Grönwall's inequality"],
      "uses_definitions": [],
      "citations": ["Han 2016"]
    }}
  ],
  "conclusion": "This completes the proof of exponential convergence."
}}

Return ONLY the JSON object, no additional text.
"""


# =============================================================================
# REFERENCE RESOLUTION
# =============================================================================

RESOLVE_REFERENCE_PROMPT = """
Resolve the following ambiguous reference to a specific entity.

**Reference Text:**
{reference_text}

**Context:**
{context}

**Available Entities:**
{available_entities}

**Task:**
Identify which entity the reference points to.

Return a JSON object with this structure:

{{
  "resolved_label": "...",          // The entity label (e.g., "thm-keystone")
  "entity_type": "...",             // "theorem", "definition", "lemma", etc.
  "confidence": "high|medium|low",  // Your confidence in the resolution
  "reasoning": "..."                // Brief explanation of your reasoning
}}

**Guidelines:**
1. Match reference text against entity names/labels
2. Use context to disambiguate if multiple matches
3. Be conservative: mark "low" confidence if uncertain
4. Provide clear reasoning for your choice

**Example:**

Input:
- Reference text: "by the Keystone Principle"
- Context: "We apply the Keystone Principle to obtain exponential convergence"
- Available entities: [
    {{"label": "thm-keystone", "name": "Keystone Principle", "type": "theorem"}},
    {{"label": "def-keystone-property", "name": "Keystone Property", "type": "definition"}}
  ]

Output:
{{
  "resolved_label": "thm-keystone",
  "entity_type": "theorem",
  "confidence": "high",
  "reasoning": "The context mentions 'apply the Keystone Principle to obtain exponential convergence', which indicates using a theorem (not just a definition). The label 'thm-keystone' matches the name 'Keystone Principle'."
}}

Return ONLY the JSON object, no additional text.
"""


# =============================================================================
# DEFINITION LINKING
# =============================================================================

LINK_DEFINITION_TO_OBJECT_PROMPT = """
Determine which mathematical object type this definition applies to.

**Definition:**
{definition_text}

**Term Being Defined:**
{term}

**Available Object Types:**
- walker
- swarm
- potential
- metric
- operator
- distribution
- (or "none" if not applicable)

**Task:**
Return a JSON object with this structure:

{{
  "object_type": "...",             // The object type or "none"
  "reasoning": "...",               // Why this object type?
  "parameters": [...]               // Key parameters in the definition
}}

**Guidelines:**
1. Match the term to the most specific object type
2. If the definition describes a general concept (not an object instance), use "none"
3. List the key parameters/symbols introduced in the definition

**Example:**

Input:
- Term: "Walker State"
- Definition: "A walker is a tuple $w := (x, v, s)$ where $x \\in \\mathcal{{X}}$ is the position, $v \\in \\mathbb{{R}}^d$ is the velocity, and $s \\in \\{{\\text{{alive}}, \\text{{dead}}\\}}$ is the status."

Output:
{{
  "object_type": "walker",
  "reasoning": "The definition explicitly defines a 'walker' as a mathematical object with specific components (position, velocity, status).",
  "parameters": ["w", "x", "v", "s"]
}}

Return ONLY the JSON object, no additional text.
"""


# =============================================================================
# SYMPY CONTEXT EXTRACTION
# =============================================================================

EXTRACT_SYMPY_CONTEXT_PROMPT = """
Extract symbolic context (symbols, assumptions, notation) from the following text.

**Text:**
{text}

**Task:**
Identify all mathematical symbols and their properties for SymPy context.

Return a JSON object with this structure:

{{
  "symbols": [
    {{
      "name": "...",                    // Symbol name (e.g., "N", "gamma")
      "latex": "...",                   // LaTeX representation (e.g., "\\gamma")
      "domain": "...",                  // "real", "positive", "integer", etc.
      "description": "..."              // Brief meaning
    }},
    ...
  ],
  "assumptions": [
    "...",                              // Global assumptions (e.g., "N >= 1")
    ...
  ],
  "notation_conventions": {{
    "symbol": "meaning"                 // e.g., {{"||.||": "Euclidean norm"}}
  }}
}}

**Guidelines:**
1. **Symbols**: Extract all parameters, variables, constants
2. **Domain**: Specify the mathematical domain (real, positive, integer, etc.)
3. **Assumptions**: List global constraints (e.g., "N >= 1", "h -> 0")
4. **Notation**: Document special notation used throughout

**Example:**

Input: "Let $N \\in \\mathbb{{N}}$ denote the number of walkers, with $N \\geq 2$. The friction coefficient $\\gamma > 0$ controls the damping. Throughout, $\\|\\cdot\\|$ denotes the Euclidean norm."

Output:
{{
  "symbols": [
    {{
      "name": "N",
      "latex": "N",
      "domain": "positive_integer",
      "description": "number of walkers"
    }},
    {{
      "name": "gamma",
      "latex": "\\gamma",
      "domain": "positive",
      "description": "friction coefficient controlling damping"
    }}
  ],
  "assumptions": [
    "N >= 2",
    "gamma > 0"
  ],
  "notation_conventions": {{
    "||.||": "Euclidean norm"
  }}
}}

Return ONLY the JSON object, no additional text.
"""


# =============================================================================
# EQUATION LABELING AND LINKING
# =============================================================================

LINK_EQUATION_TO_ENTITIES_PROMPT = """
Identify which mathematical entities (theorems, definitions, mathster) reference this equation.

**Equation:**
{equation_latex}

**Equation Label:**
{equation_label}

**Context:**
{context}

**Available Entities:**
{available_entities}

**Task:**
Determine which entities explicitly reference this equation.

Return a JSON object with this structure:

{{
  "referenced_by": [
    {{
      "entity_label": "...",        // e.g., "thm-main-convergence"
      "entity_type": "...",         // "theorem", "proof", "definition"
      "reference_type": "..."       // "statement", "proof_step", "definition"
    }},
    ...
  ],
  "introduces_symbols": [...],      // New symbols defined in this equation
  "uses_symbols": [...]             // Symbols used from earlier definitions
}}

**Guidelines:**
1. Look for explicit references like "Equation (2.3)" or "by (2.3)"
2. Check if the equation appears in theorem statements
3. Identify proof steps that cite this equation
4. List symbols first defined vs. used in this equation

Return ONLY the JSON object, no additional text.
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_decompose_theorem_prompt(theorem_statement: str) -> str:
    """
    Get prompt for decomposing theorem into assumptions + conclusion.

    Args:
        theorem_statement: Full theorem statement text

    Returns:
        Formatted prompt string
    """
    return DECOMPOSE_THEOREM_PROMPT.format(theorem_statement=theorem_statement)


def get_parse_latex_prompt(statement_text: str) -> str:
    """
    Get prompt for parsing LaTeX into DualStatement.

    Args:
        statement_text: LaTeX mathematical statement

    Returns:
        Formatted prompt string
    """
    return PARSE_LATEX_TO_DUAL_PROMPT.format(statement_text=statement_text)


def get_analyze_proof_prompt(proof_text: str, theorem_label: str) -> str:
    """
    Get prompt for analyzing proof structure.

    Args:
        proof_text: Full proof text
        theorem_label: Label of theorem being proven

    Returns:
        Formatted prompt string
    """
    return ANALYZE_PROOF_STRUCTURE_PROMPT.format(
        proof_text=proof_text, theorem_label=theorem_label
    )


def get_resolve_reference_prompt(
    reference_text: str, context: str, available_entities: list
) -> str:
    """
    Get prompt for resolving ambiguous reference.

    Args:
        reference_text: The ambiguous reference (e.g., "the main theorem")
        context: Surrounding text for disambiguation
        available_entities: List of candidate entities (as dicts)

    Returns:
        Formatted prompt string
    """
    import json

    entities_json = json.dumps(available_entities, indent=2)

    return RESOLVE_REFERENCE_PROMPT.format(
        reference_text=reference_text, context=context, available_entities=entities_json
    )


def get_link_definition_prompt(definition_text: str, term: str) -> str:
    """
    Get prompt for linking definition to object type.

    Args:
        definition_text: Full definition text
        term: Term being defined

    Returns:
        Formatted prompt string
    """
    return LINK_DEFINITION_TO_OBJECT_PROMPT.format(definition_text=definition_text, term=term)


def get_extract_sympy_context_prompt(text: str) -> str:
    """
    Get prompt for extracting SymPy context from text.

    Args:
        text: Text containing mathematical notation and symbols

    Returns:
        Formatted prompt string
    """
    return EXTRACT_SYMPY_CONTEXT_PROMPT.format(text=text)


def get_link_equation_prompt(
    equation_latex: str, equation_label: str, context: str, available_entities: list
) -> str:
    """
    Get prompt for linking equation to entities.

    Args:
        equation_latex: LaTeX content of equation
        equation_label: Equation label (e.g., "(2.3)")
        context: Surrounding text
        available_entities: List of entities that might reference this equation

    Returns:
        Formatted prompt string
    """
    import json

    entities_json = json.dumps(available_entities, indent=2)

    return LINK_EQUATION_TO_ENTITIES_PROMPT.format(
        equation_latex=equation_latex,
        equation_label=equation_label,
        context=context,
        available_entities=entities_json,
    )
