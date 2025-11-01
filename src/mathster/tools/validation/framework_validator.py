"""Framework consistency validation using Gemini."""

from pathlib import Path
from typing import Optional

from mathster.tools.validation.base_validator import BaseValidator, ValidationResult


class FrameworkValidator(BaseValidator):
    """Validates framework consistency using Gemini 2.5 Pro."""

    def __init__(self, strict: bool = False, glossary_path: Path | None = None):
        """Initialize framework validator.

        Args:
            strict: If True, warnings are treated as errors
            glossary_path: Path to docs/glossary.md for notation reference
        """
        super().__init__(strict=strict)
        self.glossary_path = glossary_path
        self.glossary_content: str | None = None

        if glossary_path and glossary_path.exists():
            with open(glossary_path) as f:
                self.glossary_content = f.read()

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate entity against framework standards.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # This is a placeholder for Gemini-based validation
        # In actual implementation, this would:
        # 1. Extract entity statement/definition
        # 2. Query Gemini with framework context from glossary
        # 3. Check for notation consistency
        # 4. Verify axiom usage correctness
        # 5. Validate definition alignment

        # For now, add metadata indicating this validator needs MCP integration
        result.metadata["framework_validation"] = "requires_gemini_mcp"
        result.metadata["glossary_available"] = self.glossary_content is not None

        # Basic checks without Gemini
        self._check_notation_basics(data, file_name, result)

        return result

    def _check_notation_basics(self, data: dict, file_name: str, result: ValidationResult) -> None:
        """Perform basic notation checks without LLM.

        Args:
            data: Entity data dictionary
            file_name: Name of file being validated
            result: ValidationResult to populate
        """
        # Check for common notation issues
        statement = data.get("statement", "")
        mathematical_expression = data.get("mathematical_expression", "")

        # Check for backticks (should be LaTeX)
        if "`" in statement or "`" in mathematical_expression:
            result.add_warning(
                file=file_name,
                message="Found backticks in mathematical content - should use LaTeX $...$ notation",
                suggestion="Replace `x` with $x$, etc.",
            )

        # Check for missing LaTeX delimiters
        if "\\mathcal" in statement and "$" not in statement:
            result.add_warning(
                file=file_name,
                message="Found LaTeX commands without $ delimiters",
                suggestion="Wrap LaTeX in $...$ for inline or $$...$$ for display",
            )

    def validate_axiom_usage(self, theorem_data: dict, file_path: Path) -> ValidationResult:
        """Validate that axioms are used correctly in theorem.

        This is a placeholder for Gemini-based axiom usage validation.

        Args:
            theorem_data: Theorem entity data
            file_path: Path to theorem JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Placeholder - would query Gemini to verify axiom usage
        result.metadata["axiom_validation"] = "requires_gemini_mcp"

        # Basic check: verify input_axioms exist
        input_axioms = theorem_data.get("input_axioms", [])
        if not input_axioms:
            result.add_warning(
                file=file_name,
                field="input_axioms",
                message="No axioms specified for theorem",
                suggestion="Identify which axioms this theorem relies on",
            )

        return result

    def validate_definition_consistency(
        self, object_data: dict, file_path: Path
    ) -> ValidationResult:
        """Validate that definition aligns with framework.

        This is a placeholder for Gemini-based definition consistency validation.

        Args:
            object_data: Object entity data
            file_path: Path to object JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Placeholder - would query Gemini with glossary context
        result.metadata["definition_validation"] = "requires_gemini_mcp"

        # Basic check: mathematical_expression should be non-empty
        if not object_data.get("mathematical_expression"):
            result.add_error(
                file=file_name,
                field="mathematical_expression",
                message="Missing mathematical expression for object",
            )

        return result

    def validate_with_gemini(
        self, data: dict, file_path: Path, validation_type: str
    ) -> ValidationResult:
        """Validate entity using Gemini 2.5 Pro (MCP integration required).

        This method would be implemented using the mcp__gemini-cli__ask-gemini tool
        when called from Claude Code.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file
            validation_type: Type of validation ('notation', 'axiom_usage', 'definition')

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # This is a placeholder - actual implementation would use MCP
        result.metadata["gemini_validation"] = "not_implemented"
        result.metadata["validation_type"] = validation_type

        result.add_warning(
            file=file_name,
            message="Gemini validation not yet implemented in standalone mode",
            suggestion="Use mcp__gemini-cli__ask-gemini from Claude Code for LLM validation",
        )

        return result
