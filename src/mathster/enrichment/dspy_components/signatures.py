"""
DSPy Signature definitions for semantic validation of extracted entities.

Defines input/output specifications for DSPy agents that validate extracted
mathematical entities to ensure text content matches stated type and definition.
"""

import dspy


class ValidateEntityText(dspy.Signature):
    """
    Validate that extracted text matches entity definition and line ranges are correct.

    This signature performs semantic validation to ensure:
    1. **Type correctness**: Extracted text is actually the stated entity type
    2. **Content accuracy**: Text matches the entity's metadata (term, symbol, etc.)
    3. **Completeness**: Line ranges capture the full entity (not truncated)
    4. **Precision**: Line ranges don't include extra unrelated text

    **Critical for parameters**: Unlike definitions/theorems which have directive markers
    ({prf:definition}, {prf:theorem}), parameters lack structural markers. Validation
    ensures the text actually defines the parameter symbol.

    The agent should analyze:
    - For **definitions**: Does text define the stated term?
    - For **theorems**: Does text contain a theorem statement?
    - For **parameters**: Does text define the parameter symbol?
    - For **proofs**: Does text prove the stated theorem?

    Validation approach:
    1. Read the extracted_text
    2. Check if it matches the entity_type
    3. For parameters, verify symbol is defined in text
    4. Assess if line_range is optimal (not too short/long)
    5. Provide confidence score and specific errors
    """

    entity_type: str = dspy.InputField(
        desc="Entity type: definition, theorem, lemma, proof, parameter, axiom, etc."
    )

    entity_label: str = dspy.InputField(
        desc="Unique entity label (e.g., def-lipschitz, thm-convergence, param-tau)"
    )

    entity_metadata: str = dspy.InputField(
        desc="JSON of entity metadata: {term: '...', symbol: '...', statement_type: '...', etc.}"
    )

    extracted_text: str = dspy.InputField(
        desc="Full text content extracted from the line ranges (this is what we're validating)"
    )

    line_range: str = dspy.InputField(
        desc="Line ranges used for extraction (JSON: [[start, end], ...]) - verify these are correct"
    )

    is_valid: bool = dspy.OutputField(
        desc="True if extracted text matches entity type and definition"
    )

    confidence: str = dspy.OutputField(
        desc="Confidence in validation: 'high' (definitely correct), 'medium' (probably correct), 'low' (issues found)"
    )

    validation_errors: list[str] = dspy.OutputField(
        desc="Specific issues found (empty list if is_valid=True and high confidence)"
    )

    suggestions: str = dspy.OutputField(
        desc="Suggestions for improvement if line ranges are incorrect (e.g., 'Extend to line 300 to include full statement')"
    )
