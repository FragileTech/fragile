"""Cross-reference and dependency validation."""

import json
from pathlib import Path
from typing import Optional

from mathster.tools.validation.base_validator import BaseValidator, ValidationResult


class RelationshipValidator(BaseValidator):
    """Validates cross-references and dependencies between entities."""

    def __init__(self, strict: bool = False, refined_dir: Path | None = None):
        """Initialize relationship validator.

        Args:
            strict: If True, warnings are treated as errors
            refined_dir: Path to refined_data directory (for loading entity registry)
        """
        super().__init__(strict=strict)
        self.refined_dir = refined_dir
        self.entity_registry: dict[str, dict] = {}

        if refined_dir:
            self._build_entity_registry()

    def _build_entity_registry(self) -> None:
        """Build registry of all entities from refined_data directory."""
        if not self.refined_dir or not self.refined_dir.exists():
            return

        # Scan all entity types
        entity_dirs = [
            "theorems",
            "axioms",
            "objects",
            "parameters",
            "mathster",
            "remarks",
            "equations",
        ]

        for entity_type in entity_dirs:
            entity_dir = self.refined_dir / entity_type
            if not entity_dir.exists():
                continue

            for json_file in entity_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    label = data.get("label") or data.get("proof_id")
                    if label:
                        self.entity_registry[label] = {
                            "type": entity_type.rstrip("s"),
                            "file": json_file.name,
                            "data": data,
                        }
                except Exception:
                    # Silently skip files that can't be loaded
                    pass

    def validate_entity(self, data: dict, file_path: Path) -> ValidationResult:
        """Validate cross-references in a single entity.

        Args:
            data: Entity data dictionary
            file_path: Path to entity JSON file

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, entity_count=1)
        file_name = file_path.name

        # Check input_objects references
        input_objects = data.get("input_objects", [])
        for obj_label in input_objects:
            if obj_label not in self.entity_registry:
                result.add_error(
                    file=file_name,
                    field="input_objects",
                    message=f"Referenced object '{obj_label}' not found in registry",
                )
            elif self.entity_registry[obj_label]["type"] != "object":
                result.add_warning(
                    file=file_name,
                    field="input_objects",
                    message=f"Referenced '{obj_label}' is not an object (type: {self.entity_registry[obj_label]['type']})",
                )

        # Check input_axioms references
        input_axioms = data.get("input_axioms", [])
        for axiom_label in input_axioms:
            if axiom_label not in self.entity_registry:
                result.add_error(
                    file=file_name,
                    field="input_axioms",
                    message=f"Referenced axiom '{axiom_label}' not found in registry",
                )
            elif self.entity_registry[axiom_label]["type"] != "axiom":
                result.add_warning(
                    file=file_name,
                    field="input_axioms",
                    message=f"Referenced '{axiom_label}' is not an axiom (type: {self.entity_registry[axiom_label]['type']})",
                )

        # Check input_parameters references
        input_parameters = data.get("input_parameters", [])
        for param_label in input_parameters:
            if param_label not in self.entity_registry:
                result.add_warning(
                    file=file_name,
                    field="input_parameters",
                    message=f"Referenced parameter '{param_label}' not found in registry",
                    suggestion="Parameter may be defined inline or globally",
                )

        # Check properties_required references
        properties_required = data.get("properties_required", {})
        for obj_label, properties in properties_required.items():
            if obj_label not in self.entity_registry:
                result.add_error(
                    file=file_name,
                    field="properties_required",
                    message=f"Object '{obj_label}' in properties_required not found in registry",
                )
            else:
                # Check if properties exist on the object
                obj_data = self.entity_registry[obj_label]["data"]
                obj_attributes = obj_data.get("current_attributes", [])
                for prop in properties:
                    if prop not in obj_attributes:
                        result.add_warning(
                            file=file_name,
                            field="properties_required",
                            message=f"Property '{prop}' not found in object '{obj_label}' attributes",
                            suggestion="Verify property name or add to object attributes",
                        )

        # For mathster, check theorem back-reference
        if "theorem" in data:
            theorem_ref = data.get("theorem")
            if isinstance(theorem_ref, dict):
                theorem_label = theorem_ref.get("label")
                if theorem_label and theorem_label not in self.entity_registry:
                    result.add_error(
                        file=file_name,
                        field="theorem",
                        message=f"Theorem back-reference '{theorem_label}' not found in registry",
                    )

        # For remarks, check related_entities
        related_entities = data.get("related_entities", [])
        for entity_label in related_entities:
            if entity_label not in self.entity_registry:
                result.add_warning(
                    file=file_name,
                    field="related_entities",
                    message=f"Related entity '{entity_label}' not found in registry",
                    suggestion="Check if label is correct",
                )

        return result

    def validate_bidirectional_consistency(self) -> ValidationResult:
        """Validate bidirectional consistency of relationships.

        For example, if theorem T references object O, check if any relationships
        document this connection.

        Returns:
            ValidationResult with consistency checks
        """
        result = ValidationResult(is_valid=True, entity_count=len(self.entity_registry))

        # This would require loading relationship files
        # For now, just return basic result
        result.metadata["bidirectional_check"] = "not_implemented"

        return result

    def generate_dependency_graph(self) -> dict[str, list[str]]:
        """Generate dependency graph from entity registry.

        Returns:
            Dictionary mapping entity labels to list of dependencies
        """
        graph: dict[str, list[str]] = {}

        for label, entity_info in self.entity_registry.items():
            data = entity_info["data"]
            dependencies: list[str] = []

            # Collect all dependencies
            dependencies.extend(data.get("input_objects", []))
            dependencies.extend(data.get("input_axioms", []))
            dependencies.extend(data.get("input_parameters", []))
            dependencies.extend(data.get("depends_on_objects", []))

            if dependencies:
                graph[label] = dependencies

        return graph

    def detect_circular_dependencies(self) -> list[list[str]]:
        """Detect circular dependencies in the entity graph.

        Returns:
            List of circular dependency chains
        """
        graph = self.generate_dependency_graph()
        cycles: list[list[str]] = []

        def dfs_cycle_detect(node: str, path: list[str], visited: set[str]) -> None:
            """DFS to detect cycles."""
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                dfs_cycle_detect(neighbor, path.copy(), visited)

        visited_global: set[str] = set()
        for node in graph:
            if node not in visited_global:
                dfs_cycle_detect(node, [], visited_global)

        return cycles
