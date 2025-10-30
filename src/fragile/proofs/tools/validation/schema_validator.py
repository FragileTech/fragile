"""Schema validation using Pydantic models."""

from pathlib import Path
from typing import Literal, Optional, Type

from pydantic import BaseModel

from fragile.proofs.core.enriched_types import (
    EnrichedAxiom,
    EnrichedDefinition,
    EnrichedObject,
    EnrichedTheorem,
    EquationBox,
    ParameterBox,
    RemarkBox,
)
from fragile.proofs.core.math_types import Axiom, MathematicalObject, Parameter, TheoremBox
from fragile.proofs.core.proof_system import ProofBox
from fragile.proofs.tools.validation.base_validator import BaseValidator, ValidationResult


# Type for validation mode
ValidationMode = Literal["refined", "pipeline"]


class SchemaValidator(BaseValidator):
    """Validates entities against Pydantic schemas.

    Supports two validation modes:
    - "refined": Validates refined_data/ using enriched schemas (more permissive)
    - "pipeline": Validates pipeline_data/ using pipeline schemas (strict)
    """

    # Schema map for refined_data (enriched, transitional format)
    REFINED_SCHEMA_MAP: dict[str, type[BaseModel]] = {
        "theorem": EnrichedTheorem,
        "axiom": EnrichedAxiom,
        "definition": EnrichedDefinition,
        "object": EnrichedObject,
        "parameter": Parameter,  # Parameters same in both stages
        "parameter_box": ParameterBox,
        "proof": ProofBox,
        "remark": RemarkBox,
        "equation": EquationBox,
    }

    # Schema map for pipeline_data (strict, execution-ready format)
    PIPELINE_SCHEMA_MAP: dict[str, type[BaseModel]] = {
        "theorem": TheoremBox,
        "axiom": Axiom,
        "object": MathematicalObject,
        "parameter": Parameter,
        "parameter_box": ParameterBox,
        "proof": ProofBox,
        "remark": RemarkBox,
        "equation": EquationBox,
    }

    def __init__(self, strict: bool = False, mode: ValidationMode = "pipeline"):
        """Initialize schema validator.

        Args:
            strict: If True, warnings are treated as errors
            mode: Validation mode - "refined" for refined_data/, "pipeline" for pipeline_data/
        """
        super().__init__(strict=strict)
        self.mode = mode
        self.schema_map = (
            self.REFINED_SCHEMA_MAP if mode == "refined" else self.PIPELINE_SCHEMA_MAP
        )

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate entity against its Pydantic schema.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Determine entity type from file path or label
        entity_type = self._infer_entity_type(file_path, data)
        if not entity_type:
            result.add_error(
                file=file_name,
                message="Could not infer entity type from file path or label",
                severity="critical",
            )
            return result

        # Get corresponding Pydantic model from mode-specific schema map
        model_class = self.schema_map.get(entity_type)
        if not model_class:
            result.add_error(
                file=file_name,
                message=f"No schema mapping for entity type '{entity_type}'",
                severity="critical",
            )
            return result

        # Validate against schema
        validated_model = self.validate_pydantic_schema(data, model_class, file_name, result)

        if validated_model:
            result.metadata["entity_type"] = entity_type
            result.metadata["model_class"] = model_class.__name__

        return result

    def _infer_entity_type(self, file_path: Path, data: dict) -> str | None:
        """Infer entity type from file path and/or data.

        Args:
            file_path: Path to entity JSON file
            data: Entity data dictionary

        Returns:
            Entity type string, or None if cannot infer
        """
        # Check parent directory name first
        parent_dir = file_path.parent.name
        if parent_dir in {
            "theorems",
            "axioms",
            "objects",
            "definitions",
            "parameters",
            "proofs",
            "remarks",
            "equations",
        }:
            if parent_dir == "theorems":
                # For theorems directory, need to distinguish theorem/lemma/proposition
                return self._infer_theorem_subtype(file_path, data)
            if parent_dir == "definitions":
                return "definition"
            if parent_dir == "parameters":
                # Check if it's enriched ParameterBox or pipeline Parameter
                if "scope" in data or "dependencies" in data:
                    return "parameter_box"
                return "parameter"
            return parent_dir.rstrip("s")  # Remove trailing 's'

        # Check label prefix
        label = data.get("label", "")
        if label:
            if label.startswith(("thm-", "lem-", "prop-", "cor-")):
                return "theorem"
            if label.startswith(("ax-", "axiom-")):
                return "axiom"
            if label.startswith("def-"):
                return "definition"
            if label.startswith("obj-"):
                return "object"
            if label.startswith("param-"):
                return "parameter_box" if "scope" in data else "parameter"
            if label.startswith("proof-"):
                return "proof"
            if label.startswith("remark-"):
                return "remark"
            if label.startswith("eq-"):
                return "equation"

        return None

    def _infer_theorem_subtype(self, file_path: Path, data: dict) -> str:
        """All theorem-like entities use TheoremBox schema.

        Args:
            file_path: Path to entity JSON file
            data: Entity data dictionary

        Returns:
            Always returns 'theorem' (TheoremBox handles all subtypes)
        """
        # TheoremBox handles theorems, lemmas, propositions, corollaries
        # Differentiation happens via statement_type field or label prefix
        return "theorem"

    def validate_directory_by_type(self, directory: Path, entity_type: str) -> ValidationResult:
        """Validate all entities of a specific type in a directory.

        Args:
            directory: Directory to scan
            entity_type: Entity type to validate

        Returns:
            Aggregated ValidationResult
        """
        return self.validate_directory(directory, pattern="*.json")
