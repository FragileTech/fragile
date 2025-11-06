"""
DSPy agents for semantic validation of extracted mathematical entities.

Provides agents that verify extracted text matches entity type and definition,
with special focus on parameters which lack structural directive markers.
"""

import dspy

from mathster.enrichment.dspy_components.signatures import ValidateEntityText


class SemanticValidator(dspy.Module):
    """
    DSPy agent for semantic validation of extracted entities.

    Uses ChainOfThought reasoning to verify that extracted text content
    matches the entity's stated type and metadata. This is especially
    important for parameters which don't have directive markers like
    {prf:definition} or {prf:theorem}.

    The agent receives:
    - Entity type (definition, theorem, parameter, etc.)
    - Entity metadata (term, symbol, label)
    - Extracted full_text from line ranges
    - Line ranges that were used

    The agent returns:
    - is_valid: Whether text matches entity
    - confidence: high/medium/low
    - validation_errors: Specific issues
    - suggestions: Corrections if line ranges wrong
    """

    def __init__(self):
        super().__init__()
        # Use ChainOfThought - validation is reasoning task
        self.prog = dspy.ChainOfThought(ValidateEntityText)

    def forward(
        self,
        entity_type: str,
        entity_label: str,
        entity_metadata: dict,
        extracted_text: str,
        line_range: list[list[int]],
    ):
        """
        Validate that extracted text matches entity definition.

        Args:
            entity_type: Type of entity (definition, theorem, parameter, etc.)
            entity_label: Unique label
            entity_metadata: Entity-specific metadata
            extracted_text: Text content to validate
            line_range: Line ranges used for extraction

        Returns:
            Dict with is_valid, confidence, validation_errors, suggestions
        """
        import json

        # Call agent with reasoning
        result = self.prog(
            entity_type=entity_type,
            entity_label=entity_label,
            entity_metadata=json.dumps(entity_metadata),
            extracted_text=extracted_text,
            line_range=json.dumps(line_range),
        )

        return {
            "is_valid": result.is_valid,
            "confidence": result.confidence,
            "validation_errors": result.validation_errors if hasattr(result, "validation_errors") else [],
            "suggestions": result.suggestions if hasattr(result, "suggestions") else "",
        }
