"""Entity-specific validators for mathematical entities."""

from pathlib import Path

from mathster.core.enriched_types import EquationBox, ParameterBox, RemarkBox
from mathster.core.math_types import Axiom, MathematicalObject, Parameter, TheoremBox
from mathster.core.proof_system import ProofBox
from mathster.tools.validation.base_validator import BaseValidator, ValidationResult


class TheoremValidator(BaseValidator):
    """Validates TheoremBox entities (theorems, lemmas, propositions, corollaries)."""

    REQUIRED_FIELDS = ["label", "name", "statement"]
    VALID_OUTPUT_TYPES = [
        "property",
        "bound",
        "convergence",
        "existence",
        "uniqueness",
        "equivalence",
        "characterization",
    ]

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate a theorem/lemma/proposition/corollary.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Required fields
        self.validate_required_fields(data, self.REQUIRED_FIELDS, file_name, result)

        # Validate against Pydantic schema
        theorem = self.validate_pydantic_schema(data, TheoremBox, file_name, result)
        if not theorem:
            return result

        # Label format
        label = data.get("label", "")
        expected_prefixes = ["thm-", "lem-", "prop-", "cor-"]
        if label and not any(label.startswith(p) for p in expected_prefixes):
            result.add_warning(
                file=file_name,
                field="label",
                message=f"Label '{label}' does not start with expected prefix (thm-, lem-, prop-, cor-)",
                suggestion="Use appropriate prefix for statement type",
            )

        # Validate output_type
        output_type = data.get("output_type")
        if output_type and output_type not in self.VALID_OUTPUT_TYPES:
            result.add_warning(
                file=file_name,
                field="output_type",
                message=f"Unusual output_type '{output_type}'",
                suggestion=f"Consider using one of: {', '.join(self.VALID_OUTPUT_TYPES)}",
            )

        # Check for dependencies
        if not data.get("input_objects") and not data.get("input_axioms"):
            result.add_warning(
                file=file_name,
                message="No input_objects or input_axioms specified - theorem has no explicit dependencies",
                suggestion="Add input_objects and input_axioms to track dependencies",
            )

        # Check for properties_required if input_objects exist
        input_objects = data.get("input_objects", [])
        properties_required = data.get("properties_required", {})
        if input_objects and not properties_required:
            result.add_warning(
                file=file_name,
                field="properties_required",
                message="input_objects specified but properties_required is empty",
                suggestion="Specify which properties of each object are required",
            )

        # Check tags
        tags = data.get("tags", [])
        if not tags:
            result.add_warning(
                file=file_name,
                field="tags",
                message="No tags specified",
                suggestion="Add descriptive tags for discoverability",
            )

        return result


class AxiomValidator(BaseValidator):
    """Validates Axiom entities."""

    REQUIRED_FIELDS = ["label", "name", "statement"]

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate an axiom.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Required fields
        self.validate_required_fields(data, self.REQUIRED_FIELDS, file_name, result)

        # Validate against Pydantic schema
        axiom = self.validate_pydantic_schema(data, Axiom, file_name, result)
        if not axiom:
            return result

        # Label format
        label = data.get("label", "")
        self.validate_label_format(label, "ax-", file_name, result)

        # Check for foundational_framework
        if not data.get("foundational_framework"):
            result.add_warning(
                file=file_name,
                field="foundational_framework",
                message="No foundational_framework specified",
                suggestion="Specify which framework this axiom belongs to",
            )

        # Check for core_assumption
        if not data.get("core_assumption"):
            result.add_warning(
                file=file_name,
                field="core_assumption",
                message="No core_assumption specified",
                suggestion="Describe the core assumption this axiom makes",
            )

        # Check parameters
        parameters = data.get("parameters", [])
        if not parameters:
            result.add_warning(
                file=file_name,
                field="parameters",
                message="No parameters listed",
                suggestion="List parameters that appear in this axiom",
            )

        return result


class ObjectValidator(BaseValidator):
    """Validates MathematicalObject entities (from definitions)."""

    REQUIRED_FIELDS = ["label", "name", "mathematical_expression"]
    VALID_OBJECT_TYPES = [
        "SPACE",
        "OPERATOR",
        "MEASURE",
        "FUNCTION",
        "SET",
        "METRIC",
        "DISTRIBUTION",
        "PROCESS",
        "ALGORITHM",
        "CONSTANT",
    ]

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate a mathematical object.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Required fields
        self.validate_required_fields(data, self.REQUIRED_FIELDS, file_name, result)

        # Validate against Pydantic schema
        obj = self.validate_pydantic_schema(data, MathematicalObject, file_name, result)
        if not obj:
            return result

        # Label format
        label = data.get("label", "")
        self.validate_label_format(label, "obj-", file_name, result)

        # Validate object_type
        object_type = data.get("object_type")
        if object_type and object_type not in self.VALID_OBJECT_TYPES:
            result.add_warning(
                file=file_name,
                field="object_type",
                message=f"Unusual object_type '{object_type}'",
                suggestion=f"Consider using one of: {', '.join(self.VALID_OBJECT_TYPES)}",
            )

        # Check current_attributes
        attributes = data.get("current_attributes", [])
        if not attributes:
            result.add_warning(
                file=file_name,
                field="current_attributes",
                message="No current_attributes listed",
                suggestion="List properties/attributes this object possesses",
            )

        # Check tags
        tags = data.get("tags", [])
        if not tags:
            result.add_warning(
                file=file_name,
                field="tags",
                message="No tags specified",
                suggestion="Add descriptive tags for discoverability",
            )

        return result


class ParameterValidator(BaseValidator):
    """Validates Parameter/ParameterBox entities."""

    REQUIRED_FIELDS = ["symbol"]

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate a parameter.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Required fields
        self.validate_required_fields(data, self.REQUIRED_FIELDS, file_name, result)

        # Determine if enriched (ParameterBox) or pipeline (Parameter)
        is_enriched = "scope" in data or "dependencies" in data
        model_class = ParameterBox if is_enriched else Parameter

        # Validate against Pydantic schema
        param = self.validate_pydantic_schema(data, model_class, file_name, result)
        if not param:
            return result

        # Check domain
        if not data.get("domain"):
            result.add_warning(
                file=file_name,
                field="domain",
                message="No domain specified",
                suggestion="Specify the mathematical domain of this parameter",
            )

        # For enriched parameters, check scope
        if is_enriched and not data.get("scope"):
            result.add_warning(
                file=file_name,
                field="scope",
                message="No scope specified for enriched parameter",
                suggestion="Specify scope: global, local, or universal",
            )

        # Check constraints
        if not data.get("constraints"):
            result.add_warning(
                file=file_name,
                field="constraints",
                message="No constraints specified",
                suggestion="Add constraints if parameter has restrictions (e.g., γ > 0)",
            )

        return result


class ProofValidator(BaseValidator):
    """Validates ProofBox entities."""

    REQUIRED_FIELDS = ["proof_id", "theorem"]

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate a proof.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Required fields
        self.validate_required_fields(data, self.REQUIRED_FIELDS, file_name, result)

        # Validate against Pydantic schema
        proof = self.validate_pydantic_schema(data, ProofBox, file_name, result)
        if not proof:
            return result

        # Check theorem linkage
        theorem_ref = data.get("theorem")
        if theorem_ref and isinstance(theorem_ref, dict):
            theorem_label = theorem_ref.get("label")
            if not theorem_label:
                result.add_error(
                    file=file_name,
                    field="theorem.label",
                    message="Theorem back-reference has no label",
                )
        elif not theorem_ref:
            result.add_error(
                file=file_name,
                field="theorem",
                message="No theorem back-reference",
            )

        # Check steps
        steps = data.get("steps", [])
        if not steps:
            result.add_warning(
                file=file_name,
                field="steps",
                message="Proof has no steps",
                suggestion="Add proof steps with step_number, content, justification",
            )
        else:
            # Validate step structure
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    result.add_error(
                        file=file_name,
                        field=f"steps[{i}]",
                        message="Step is not a dictionary",
                    )
                    continue

                # Check step_number
                if "step_number" not in step:
                    result.add_warning(
                        file=file_name,
                        field=f"steps[{i}].step_number",
                        message="Step missing step_number",
                        suggestion="Add sequential step numbers",
                    )

                # Check content
                if not step.get("content"):
                    result.add_warning(
                        file=file_name,
                        field=f"steps[{i}].content",
                        message="Step has empty content",
                        suggestion="Add mathematical derivation or explanation",
                    )

                # Check justification
                if not step.get("justification"):
                    result.add_warning(
                        file=file_name,
                        field=f"steps[{i}].justification",
                        message="Step has no justification",
                        suggestion="Reference axiom/lemma/theorem used in this step",
                    )

        # Check proof_status
        proof_status = data.get("proof_status")
        if not proof_status:
            result.add_warning(
                file=file_name,
                field="proof_status",
                message="No proof_status specified",
                suggestion="Set to: unproven, sketched, expanded, or verified",
            )

        return result


class RemarkValidator(BaseValidator):
    """Validates RemarkBox entities."""

    REQUIRED_FIELDS = ["label", "content"]

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate a remark.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Required fields
        self.validate_required_fields(data, self.REQUIRED_FIELDS, file_name, result)

        # Validate against Pydantic schema
        remark = self.validate_pydantic_schema(data, RemarkBox, file_name, result)
        if not remark:
            return result

        # Label format
        label = data.get("label", "")
        self.validate_label_format(label, "remark-", file_name, result)

        # Check remark_type
        remark_type = data.get("remark_type")
        valid_types = ["note", "observation", "intuition", "example", "warning", "historical"]
        if remark_type and remark_type not in valid_types:
            result.add_warning(
                file=file_name,
                field="remark_type",
                message=f"Unusual remark_type '{remark_type}'",
                suggestion=f"Consider using one of: {', '.join(valid_types)}",
            )

        # Check related_entities
        if not data.get("related_entities"):
            result.add_warning(
                file=file_name,
                field="related_entities",
                message="No related_entities specified",
                suggestion="Link to relevant theorems/definitions this remark discusses",
            )

        return result


class EquationValidator(BaseValidator):
    """Validates EquationBox entities."""

    REQUIRED_FIELDS = ["label", "latex_content"]

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate an equation.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Required fields
        self.validate_required_fields(data, self.REQUIRED_FIELDS, file_name, result)

        # Validate against Pydantic schema
        equation = self.validate_pydantic_schema(data, EquationBox, file_name, result)
        if not equation:
            return result

        # Label format
        label = data.get("label", "")
        self.validate_label_format(label, "eq-", file_name, result)

        # Check equation_type
        equation_type = data.get("equation_type")
        valid_types = ["definition", "identity", "evolution", "constraint", "property"]
        if equation_type and equation_type not in valid_types:
            result.add_warning(
                file=file_name,
                field="equation_type",
                message=f"Unusual equation_type '{equation_type}'",
                suggestion=f"Consider using one of: {', '.join(valid_types)}",
            )

        # Check symbol tracking
        if not data.get("introduces_symbols") and not data.get("references_symbols"):
            result.add_warning(
                file=file_name,
                message="No symbol tracking (introduces_symbols or references_symbols)",
                suggestion="Track which symbols this equation introduces or uses",
            )

        return result
